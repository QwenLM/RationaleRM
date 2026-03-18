#!/usr/bin/env python3
"""
Meta Evaluation Inference Script
Evaluates how well model-generated checklists match human-annotated checklists

Core Features:
1. Read data containing human-checklist and model-checklist
2. Use LLM to evaluate semantic matching between two checklists
3. Output matching scores for subsequent Recall/Precision calculation
"""

import argparse
import json
import os
import time
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openai
except ImportError:
    openai = None


# ============================================================================
# Evaluation Prompt Template
# ============================================================================

EVAL_PROMPT_TEMPLATE = """You are a rigorous alignment achievement analyst. Given an original evaluation list and a reference evaluation list (both are lists of critique points), determine to what extent each item in the "original evaluation list" expresses the "expected purpose/improvement objective of each critique in the reference evaluation list," and assign achievement scores (0–1) based on semantic importance. Different phrasings with the same meaning are considered equivalent, but mere semantic mention without achieving the objective should be considered not achieved. Abstract or vague descriptions of shortcomings or problems should be considered not achieved.

Response 1 and Response A refer to the same thing, and Response 2 and Response B refer to the same thing.

[Original evaluation list start]
{source_list}
[Original evaluation list end]

[Reference evaluation list start]
{target_list}
[Reference evaluation list end]


For each item in the reference evaluation list, find the best-matching single item in the original evaluation list (if there is no match, consider it not achieved; matching R0 is acceptable), compute the achievement determination criterion (value of c), match as strictly as possible, and prioritize lower-score matches.
The objective is considered achieved only when specific issues are clearly pointed out in detail (e.g., a particular passage lacks coherence), rather than merely providing vague evaluations such as stating whether response A or B is better or worse. Especially when comparing responses A and B, if the evaluations only agree on which one is better but the reasons differ, it should not be considered as meeting the objective.

- Not achieved/contradictory: 0.0 (the detailed process did not address the goal of the critique, or provided an opposite conclusion/failure, or merely listed elements without achieving the objective, or described shortcomings/problems abstractly and vaguely without accurately pinpointing the issue—for example, only evaluating which is better without stating the reason, or saying the logic is incoherent without specifying where exactly)
- Slightly involved: 0.25 (only mentioned some elements; not implemented or no results, unable to prove the objective was achieved)
- Partially achieved: 0.5 (took measures or conducted analysis related to the objective, but multiple key steps are missing or no verifiable result/conclusion was formed)
- Mostly achieved: 0.75 (the main objective is basically realized, key conclusions align, but secondary conditions, boundaries, or a small amount of supporting detail are missing)
- Fully achieved: 1.0 (the detailed process clearly shows the expected purpose of the critique was achieved; includes necessary execution steps, evidence, and results; key conditions and constraints are all satisfied)



Output format (fixed; ensure the scores can be extracted; Rx@Sy indicates that item x in reference list R best matches item y in original list S):
(think)
<RESULT_START>
Scores corresponding to claims:
- R1@Sx: a decimal between 0 and 1, keep at least two decimal places, e.g., 0.75
- R2@Sx: a decimal between 0 and 1, keep at least two decimal places, e.g., 0.75
- R3@S0: 0 (indicates there is no matching content)
... list in sequence
<RESULT_END>

Notes:

- Only use the "retained claims/critique points" in the brief summary as the objects of evaluation; do not count newly added content from the detailed process toward achievement.
- For quantitative claims, check numbers, ranges, thresholds, and conditions; if key constraints are not satisfied, do not judge as fully achieved."""


# ============================================================================
# Result Extraction Module
# ============================================================================

def extract_result_scores(response: str) -> dict:
    """
    Extract scoring information between RESULT_START and RESULT_END tags
    
    Args:
        response: Response text containing scoring results
        
    Returns:
        dict: Dictionary containing extraction results:
        {
            'scores': [(R_num, S_num, score), ...],  # List of tuples
            'is_ordered': bool,  # Whether Rx are in sequence
            'errors': []  # List of error messages
        }
    """
    result = {
        'scores': [],
        'is_ordered': True,
        'errors': []
    }
    
    # Extract content between RESULT_START and RESULT_END
    result_pattern = r'<RESULT_START>(.*?)<RESULT_END>'
    match = re.search(result_pattern, response, re.DOTALL)
    
    if not match:
        result['errors'].append("RESULT_START and RESULT_END tags not found")
        return result
    
    result_content = match.group(1)
    
    # Extract score lines in format: - Rx@Sy: score
    score_pattern = r'- R(\d+)@S(\d+): (\d+(?:\.\d+)?)'
    matches = re.findall(score_pattern, result_content)
    
    if not matches:
        result['errors'].append("No score lines matching the expected format found")
        return result
    
    # Parse matched results
    r_numbers = []
    for m in matches:
        try:
            r_num = int(m[0])
            s_num = int(m[1])
            score = float(m[2])
            
            # Check score range
            if not (0 <= score <= 1):
                result['errors'].append(f"Score {score} for R{r_num}@S{s_num} is not in range 0-1")
            
            result['scores'].append((r_num, s_num, score))
            r_numbers.append(r_num)
            
        except ValueError as e:
            result['errors'].append(f"Parse error: {m} - {str(e)}")
    
    # Check if Rx are in sequence
    if len(r_numbers) > 1:
        for i in range(1, len(r_numbers)):
            if r_numbers[i] != r_numbers[i-1] + 1:
                result['is_ordered'] = False
                result['errors'].append(f"R numbers not consecutive: R{r_numbers[i-1]} -> R{r_numbers[i]}")
                break
    
    # Check if sequence starts from R1
    if r_numbers and r_numbers[0] != 1:
        result['is_ordered'] = False
        result['errors'].append(f"R numbers do not start from 1, starts from R{r_numbers[0]}")
    
    return result


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


# ============================================================================
# Data Processing Module
# ============================================================================

class DataProcessor:
    """Module for data reading, processing, and saving"""
    
    def __init__(self, input_file: str, output_file: str, model_be_evaluated: str):
        self.input_file = input_file
        self.output_file = output_file
        self.model_be_evaluated = model_be_evaluated
        
        self.raw_data = self._load_jsonl(input_file)
        self.deal_data = self._process_raw_data(self.raw_data)
        
        # Load existing results for checkpoint resumption
        if os.path.exists(output_file):
            self.existing_results = self._load_jsonl(output_file)
        else:
            self.existing_results = []
        
        self.deal_data = self._filter_completed(self.deal_data, self.existing_results)
    
    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON parse failed at line {line_num}: {e}")
        logging.info(f"Successfully loaded {len(data)} items from {filepath}")
        return data
    
    def _process_raw_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw data and generate evaluation prompts"""
        deal_data = []
        model_checklist_key = f"{self.model_be_evaluated}-checklist"
        
        for item in raw_data:
            # Check required fields
            if 'human-checklist' not in item:
                logging.warning(f"Data missing human-checklist field, skipping")
                continue
            
            # Case-insensitive search for model checklist
            found_key = None
            for key in item.keys():
                if key.lower() == model_checklist_key.lower():
                    found_key = key
                    break
            
            if not found_key:
                logging.warning(f"Data missing {model_checklist_key} field, skipping")
                continue
            
            human_checklist = item['human-checklist']
            model_checklist = item[found_key]
            
            # Build checklist strings
            human_checklist_str = ""
            for idx, point in enumerate(human_checklist):
                human_checklist_str += f"R{idx+1}: {point.strip()}\n"
            
            model_checklist_str = ""
            for idx, point in enumerate(model_checklist):
                model_checklist_str += f"S{idx+1}: {point.strip()}\n"
            
            # Generate prompt
            prompt = EVAL_PROMPT_TEMPLATE.format(
                source_list=model_checklist_str.strip(),
                target_list=human_checklist_str.strip()
            )
            
            deal_data.append({
                'prompt': prompt,
                'metadata': item
            })
        
        logging.info(f"Processed {len(deal_data)} items for evaluation")
        return deal_data
    
    def _create_prompt_id(self, data: Dict) -> str:
        """Create unique ID for prompt"""
        return data.get('prompt', '')
    
    def _filter_completed(self, deal_data: List[Dict], existing_results: List[Dict]) -> List[Dict]:
        """Filter completed data for checkpoint resumption"""
        processed_ids = set()
        for result in existing_results:
            if result.get('status') == 'success' and 'prompt' in result:
                processed_ids.add(result['prompt'])
        
        filtered = [d for d in deal_data if self._create_prompt_id(d) not in processed_ids]
        logging.info(f"Filtered to {len(filtered)} items pending processing")
        return filtered
    
    def get_deal_data(self) -> List[Dict]:
        return self.deal_data
    
    def save_results(self, results: List[Dict], mode: str = 'a'):
        """Save results to file"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, mode, encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')


# ============================================================================
# Inference Engine Module
# ============================================================================

class OpenAIInferenceEngine:
    """OpenAI API Inference Engine"""
    
    def __init__(self, api_key: str, api_base: str, model: str, 
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _call_api_single(self, item: Dict, inference_kwargs: Dict) -> Dict:
        """Single API call with format validation and retry"""
        messages = [{"role": "user", "content": item['prompt']}]
        
        generation_params = {
            'model': self.model,
            'messages': messages,
            'max_tokens': inference_kwargs.get('max_tokens', 8192),
            'temperature': inference_kwargs.get('temperature', 0.6),
        }
        
        # API call retry loop
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**generation_params)
                
                # Format validation retry loop
                for format_attempt in range(self.max_retries):
                    generated_text = response.choices[0].message.content
                    extraction_result = extract_result_scores(generated_text)
                    
                    if not extraction_result['errors']:
                        return {
                            'success': True,
                            'response': response,
                            'extraction_result': extraction_result,
                            'format_retry_count': format_attempt
                        }
                    else:
                        logging.warning(f"Format check failed (attempt {format_attempt + 1}/{self.max_retries}): {extraction_result['errors']}")
                        if format_attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            response = self.client.chat.completions.create(**generation_params)
                        else:
                            return {
                                'success': True,
                                'response': response,
                                'extraction_result': extraction_result,
                                'format_retry_count': format_attempt + 1,
                                'format_error': True
                            }
                
            except Exception as e:
                logging.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return {
                        'success': False,
                        'response': None,
                        'error': str(e),
                        'extraction_result': None
                    }
    
    def infer_batch(self, deal_data: List[Dict], concurrent_requests: int = 10,
                   rate_limit_delay: float = 0.1, inference_kwargs: Dict = None) -> List[Dict]:
        """Batch inference"""
        results = []
        inference_kwargs = inference_kwargs or {}
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            future_to_idx = {}
            
            for i, data in enumerate(deal_data):
                future = executor.submit(self._call_api_single, data, inference_kwargs)
                future_to_idx[future] = i
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
            
            # Collect results
            idx_results = [None] * len(deal_data)
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    idx_results[idx] = result
                    completed += 1
                    if result['success']:
                        logging.info(f"Completed {completed}/{len(deal_data)}")
                except Exception as e:
                    idx_results[idx] = {
                        'success': False,
                        'response': None,
                        'error': str(e)
                    }
                    logging.error(f"Exception: {e}")
        
        return self._process_results(idx_results, deal_data, inference_kwargs)
    
    def _process_results(self, api_results: List[Dict], deal_data: List[Dict],
                        inference_kwargs: Dict) -> List[Dict]:
        """Process API response results"""
        batch_results = []
        
        for api_result, data in zip(api_results, deal_data):
            if api_result['success']:
                response = api_result['response']
                samples = []
                
                for choice in response.choices:
                    if choice.message.content:
                        sample = {
                            "generated_text": choice.message.content.strip(),
                            "finish_reason": choice.finish_reason,
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            },
                        }
                        samples.append(sample)
                
                result = {
                    "prompt": data["prompt"],
                    "samples": samples,
                    "status": "success" if samples else "error",
                    "metadata": data["metadata"],
                }
                
                if api_result.get('extraction_result'):
                    result['extraction_result'] = api_result['extraction_result']
                    if api_result.get('format_error'):
                        result['format_error'] = True
                        result['status'] = 'format_error'
            else:
                result = {
                    "prompt": data["prompt"],
                    "samples": [],
                    "status": "error",
                    "metadata": data["metadata"],
                    "error_message": api_result.get('error', 'Unknown error'),
                }
            
            batch_results.append(result)
        
        return batch_results


# ============================================================================
# Main Flow
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Meta Evaluation Inference Script')
    
    # Basic parameters
    parser.add_argument('--input-file', type=str, required=True,
                       help='Input file path (JSONL format with human-checklist and model-checklist)')
    parser.add_argument('--output-file', type=str, required=True,
                       help='Output file path')
    parser.add_argument('--model-be-evaluated', type=str, required=True,
                       help='Name of the model being evaluated (used to find corresponding checklist field)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    
    # API configuration
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (can also be set via OPENAI_API_KEY environment variable)')
    parser.add_argument('--api-base', type=str, default=None,
                       help='API base URL (can also be set via OPENAI_BASE_URL, default: https://api.openai.com/v1)')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='Evaluation model name')
    
    # Inference parameters
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum generation tokens')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--max-retries', type=int, default=5,
                       help='Maximum retry attempts')
    parser.add_argument('--concurrent-requests', type=int, default=10,
                       help='Number of concurrent requests')
    parser.add_argument('--rate-limit-delay', type=float, default=1.0,
                       help='Delay between requests (seconds)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Get API configuration from environment variables if not specified via command line
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    api_base = args.api_base or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    
    if not api_key:
        print("Error: Please provide API key via --api-key or OPENAI_API_KEY environment variable")
        return
    
    # Setup logging
    log_file = args.log_file or f'meta_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_logging(log_file)
    logging.info(f"Starting Meta Evaluation inference task")
    logging.info(f"Arguments: {vars(args)}")
    
    # Create data processor
    data_processor = DataProcessor(
        input_file=args.input_file,
        output_file=args.output_file,
        model_be_evaluated=args.model_be_evaluated
    )
    
    # Create inference engine
    inference_engine = OpenAIInferenceEngine(
        api_key=api_key,
        api_base=api_base,
        model=args.model,
        max_retries=args.max_retries
    )
    
    # Get data to process
    deal_data = data_processor.get_deal_data()
    
    if not deal_data:
        logging.info("All data has been processed, no new inference needed")
        return
    
    # Inference parameters
    inference_kwargs = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
    }
    
    # Batch inference
    batch_size = args.batch_size
    total_batches = (len(deal_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(deal_data))
        batch_data = deal_data[start_idx:end_idx]
        
        logging.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_data)} items)")
        
        batch_results = inference_engine.infer_batch(
            batch_data,
            concurrent_requests=args.concurrent_requests,
            rate_limit_delay=args.rate_limit_delay,
            inference_kwargs=inference_kwargs
        )
        
        # Save batch results
        data_processor.save_results(batch_results, mode='a')
        
        success_count = sum(1 for r in batch_results if r.get('status') == 'success')
        logging.info(f"Batch {batch_idx + 1} completed: {success_count}/{len(batch_results)} succeeded")
    
    logging.info("Meta Evaluation inference task completed")


if __name__ == "__main__":
    main()
