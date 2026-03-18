import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset
import argparse
import logging
from typing import Dict, List, Optional
import os

# ============================================================================
# Hybrid Reward Trainer
# ============================================================================

class HybridRewardTrainer(RewardTrainer):
    """
    Custom Reward Trainer that implements Hybrid Reward Training.
    Combines standard preference loss with rationale consistency rewards.
    """
    def __init__(self, *args, rationale_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rationale_weight = rationale_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the hybrid loss: Preference Loss + rationale_weight * Rationale Loss
        Note: In GenRMs, the 'reward' is often the log-probability of a specific token
        or a sequence-level score. This implementation assumes a classification-style RM.
        """
        # Standard RewardTrainer loss (Bradley-Terry preference loss)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Add Rationale Consistency Signal if available in inputs
        # In a real implementation, 'rationale_scores' would be pre-computed AP scores
        # or a differentiable approximation of rationale alignment.
        if "rationale_scores" in inputs:
            rationale_reward = inputs["rationale_scores"].mean()
            # We want to maximize rationale_reward, so we subtract it from the loss
            loss = loss - self.rationale_weight * rationale_reward
            
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# Data Preparation
# ============================================================================

def preprocess_function(examples, tokenizer, max_length):
    """
    Preprocess the RationaleRM dataset into preference pairs.
    """
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(prompt + chosen, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(prompt + rejected, truncation=True, max_length=max_length)
        
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        
    return new_examples

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RationaleRM using Hybrid Reward")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model for RM")
    parser.add_argument("--dataset_name", type=str, default="Qwen/RationaleRM", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="./output/rm_training", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--rationale_weight", type=float, default=0.1, help="Weight for rationale consistency loss")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=1, torch_dtype=torch.float16, device_map="auto"
    )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)
    
    train_dataset = dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
        gradient_accumulation_steps=4,
        report_to="none",
    )

    trainer = HybridRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        rationale_weight=args.rationale_weight
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
