<div align="center">

<p align="right">
  <strong>English</strong> | <a href="./README_zh.md">中文</a>
</p>

<h1>Outcome Accuracy is Not Enough:<br/> Aligning the Reasoning Process of Reward Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.04649"><img src="https://img.shields.io/badge/arXiv-2602.04649-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Qwen/RationaleRM"><img src="https://img.shields.io/badge/🤗%20Dataset-RationaleRM-yellow" alt="Dataset"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/legalcode.en"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.04649"><strong>[📄 Paper]</strong></a> •
  <a href="#dataset"><strong>[🤗 Dataset]</strong></a> •
  <a href="#citation"><strong>[📜 Citation]</strong></a>
</p>

<p align="center">
  <img src="images/overall_compare.png" alt="Outcome Accuracy vs Rationale Consistency" width="70%">
</p>

<p align="center"><em>Outcome Accuracy vs Rationale Consistency: Rationale Consistency effectively distinguishes frontier models and detects deceptive alignment</em></p>

</div>


---

## 📖 Overview

**RationaleRM** is a research project that investigates how to align not just the *outcomes* but also the *reasoning processes* of reward models with human judgments. We discover that generative reward models (GenRMs) and LLM-as-a-Judge exhibit **Deceptive Alignment** issues — models may reach the same final result as humans through superficial or even incorrect reasoning processes.

To address this, we propose the **Rationale Consistency** metric, which measures the alignment between the model's reasoning process and human judgment rationales. We also design the **MetaJudge** framework to compute this metric: it decomposes human and model rationales into atomic units, then performs strict one-to-one semantic matching to precisely quantify their consistency.

**Core Contributions:**

- 🔍 **MetaJudge Framework**: Decomposes human rationales into atomic units and uses LLMs for strict one-to-one semantic matching
- 📊 **Rationale Consistency Metric**: Effectively detects deceptive alignment and distinguishes frontier models (e.g., GPT-5 or Gemini 3 Pro)
- 🛠️ **Hybrid Reward Training**: Combines rationale reward (Average Precision) and outcome reward to prevent "rationale degeneration"
- 🏆 **SOTA Performance**: Achieves best results on RM-Bench (87.1%) and JudgeBench (82.0%)

---

## 🚨 Key Finding: The Deceptive Alignment Trap

We evaluated 19 frontier models and found two critical flaws when relying solely on outcome accuracy:

### Outcome Accuracy Cannot Distinguish Frontier Models

In the green region, although multiple models achieve similar outcome accuracy, rationale consistency clearly distinguishes stronger models (such as GPT-5, o3, Gemini 3 Pro) from weaker ones (such as Claude 3.5, GPT-4.1).

### Outcome Accuracy Cannot Detect Deceptive Alignment

The most typical example is the comparison between **o3 and o3-mini**: both have similar outcome accuracy, but o3-mini's rationale consistency is nearly 50% lower. o3-mini relies on surface cues (such as formatting, emojis) to make judgments, while o3 performs rigorous fact-checking like humans do.

> 💡 **Key Insight**: Models can make correct choices for wrong reasons. Outcome accuracy alone cannot detect this deceptive alignment.

---

## 📉 Training Finding: Outcome-Only Supervision Leads to Rationale Degeneration

<p align="center">
  <img src="images/reward_compare.png" alt="Training Dynamics" width="70%">
</p>

<p align="center"><em>Training dynamics comparison: Similar outcome rewards, but significantly different rationale rewards</em></p>

The figure above shows a key finding during training: **outcome-only supervision leads to continuous decline in model-human reasoning process consistency**.

- **Left**: Both methods achieve nearly identical outcome rewards, indicating models can learn to select correct answers
- **Right**: Rationale rewards show significant divergence — without rationale consistency constraints, model rationale rewards continuously decline, ultimately **24.2%** lower than our method

This reveals the **Rationale Degeneration** phenomenon: when intermediate reasoning processes are not incentivized, models abandon high-cost evidence verification and instead rely on cheaper surface cues to achieve similar outcome rewards.

---

## 🏆 Main Results

We evaluate on two challenging benchmarks:

- **RM-Bench**: Evaluates model ability to distinguish subtle differences and style biases
- **JudgeBench**: Emphasizes deep judgment and logical reasoning

| Model                                  |    RM-Bench    |   JudgeBench   |      Avg      |
| :------------------------------------- | :------------: | :------------: | :-----------: |
| **Generative Reward Models**           |                |                |               |
| RM-R1-Distilled-Qwen-32B               |      83.9      |      78.8      |     81.4      |
| RRM-32B                                |      73.1      |      75.7      |     74.4      |
| Nemotron-Super-49B                     |      82.7      |      77.2      |     80.0      |
| RewardAnything-8B-v1                   |      83.1      |      62.6      |     72.9      |
| GRAM-R²                                |      85.7      |      81.0      |     83.4      |
| **Outcome-Only Baselines**             |                |                |               |
| Qwen3-14B (Outcome-Only)               |      83.6      |      70.0      |     76.8      |
| Qwen3-30B-A3B (Outcome-Only)           |      84.9      |      75.7      |     80.3      |
| **Our Method (Outcome + Rationale)**   |                |                |               |
| Qwen3-14B (Ours)                       |      86.7      |      79.1      |     82.9      |
| **Qwen3-30B-A3B (Ours)**               | **87.1** | **82.0** | **84.6** |

> 💡 Our method effectively reverses the rationale consistency decline observed during outcome-only training (from 25% to 37%).

---


## 🚀 Quick Start

### Project Structure

```
RationaleRM/
├── metajudge_infer.py              # Semantic matching inference script
├── metajudge_infer.sh              # Shell script for running inference
├── metajudge_analysis.py           # Analysis script for computing metrics
├── images/                         # Images
│   ├── overall_compare.png
│   └── reward_compare.png
├── data/                           # Datasets
│   ├── helpsteer3_test_1000.jsonl      # Test set: 1000 samples
│   └── helpsteer3_human_checklist.jsonl # Full dataset (22,116 samples)
└── example/                   # Example data for testing
    ├── infer_input_10samples.jsonl
    ├── model-low_deceptive_alignment.jsonl
    └── model-high_deceptive_alignment.jsonl
```

### Step 1: Prepare Data

Input data should be in JSONL format with the following fields:
- `human-checklist`: List of human atomic rationales (reference)
- `{model}-checklist`: List of model-generated atomic rationales to be evaluated

Example:
```json
{
  "domain": "general",
  "context": [...],
  "response1": "...",
  "response2": "...",
  "human-checklist": [
    "Response 1 lacks polysyllabic rhymes",
    "Response 2's meter is inconsistent"
  ],
  "model-low_deceptive_alignment-checklist": [
    "Response A's rhyme scheme is forced",
    "Response B's rhythm feels awkward"
  ]
}
```

### Step 2: Run Inference

The inference script evaluates how well each model-generated checklist item matches the human checklist:

```bash
# Set environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI

# Run inference
python metajudge_infer.py \
    --input-file data/helpsteer3_test_1000.jsonl \
    --output-file output/results.jsonl \
    --model gpt-4o \
    --model-be-evaluated model-low_deceptive_alignment \
    --concurrent-requests 5
```

Or use the shell script:
```bash
bash metajudge_infer.sh
```

Key parameters:
- `--input-file`: Path to input JSONL file
- `--output-file`: Path for output results
- `--model`: LLM model for semantic matching (e.g., gpt-4o, qwen-plus)
- `--model-be-evaluated`: The critic model whose checklist will be evaluated
- `--concurrent-requests`: Number of parallel API requests

API configuration (via environment variables or command line):
- `OPENAI_API_KEY` or `--api-key`: API key for the LLM service
- `OPENAI_BASE_URL` or `--api-base`: API base URL (default: https://api.openai.com/v1)

### Step 3: Analyze Results

Compute Precision, Recall, F1, and Average Precision:

```bash
# Analyze single file
python metajudge_analysis.py \
    --input-file example/low_deceptive_alignment_infer_output.jsonl \
    --model-be-evaluated model-low_deceptive_alignment

# Analyze all files in a directory
python metajudge_analysis.py \
    --input-dir example/ \
    --sort-by recall
```

Output example:

```text
====================================================================================================
Results Sorted by RECALL
====================================================================================================
Model                                         Precision    Recall       F1           AP           Valid   
----------------------------------------------------------------------------------------------------
model-low_deceptive_alignment                 0.3300       0.4297       0.3684       0.3991       10      
model-high_deceptive_alignment                0.1850       0.2242       0.1985       0.2376       10      
====================================================================================================
```

---

## 📊 Metrics

MetaJudge computes the following metrics:

| Metric | Description |
|--------|-------------|
| **Recall** | Proportion of human rationales matched by model rationales |
| **Precision** | Proportion of model rationales that match human rationales (for evaluation) |
| **F1** | Harmonic mean of Precision and Recall |
| **Average Precision (AP)** | Used for training in this paper |

---

<a id="dataset"></a>

## 📂 Dataset

We provide two datasets:

### 1. HelpSteer3 Human Checklist (Full Dataset)

**`helpsteer3_human_checklist.jsonl`** contains the complete HelpSteer3 dataset with human-annotated atomic rationales, suitable for training.

### 2. Test Set (with Model Checklists)

**`helpsteer3_test_1000.jsonl`** contains 1000 selected test samples used for testing in the paper. We provide two model checklists representing different levels of deceptive alignment:

| Field | Description |
|-------|-------------|
| `human-checklist` | Human-annotated atomic rationales (reference) |
| `model-low_deceptive_alignment-checklist` | Low deceptive alignment model checklist (corresponds to high Rationale Consistency in the paper) |
| `model-low_deceptive_alignment-label` | Low deceptive alignment model preference label |
| `model-low_deceptive_alignment-generated_text` | Low deceptive alignment model full generated text |
| `model-high_deceptive_alignment-checklist` | High deceptive alignment model checklist (corresponds to low Rationale Consistency in the paper) |
| `model-high_deceptive_alignment-label` | High deceptive alignment model preference label |
| `model-high_deceptive_alignment-generated_text` | High deceptive alignment model full generated text |

> **Note:** 
> - Atomic rationales were generated using GPT-5 for research purposes only.
> - The `model-high_deceptive_alignment` and `model-low_deceptive_alignment` data are provided for testing/evaluation purposes only and were not used for training.

---

<a id="citation"></a>

## 📜 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{wang2026outcome,
  title={Outcome Accuracy is Not Enough: Aligning the Reasoning Process of Reward Models},
  author={Wang, Binghai and Liu, Yantao and Liu, Yuxuan and Tang, Tianyi and Wang, Shenzhi and Gao, Chang and Zheng, Chujie and Zhang, Yichang and Yu, Le and Liu, Shixuan and Gui, Tao and Zhang, Qi and Huang, Xuanjing and Yu, Bowen and Huang, Fei and Lin, Junyang},
  journal={arXiv preprint arXiv:2602.04649},
  year={2026}
}
```

---

<div align="center">

**Developed by Qwen Team in collaboration with Fudan University**

</div>
