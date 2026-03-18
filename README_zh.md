<div align="center">

<p align="right">
  <a href="./README.md">English</a> | <strong>中文</strong>
</p>

<h1>Outcome Accuracy is Not Enough:<br/> Aligning the Reasoning Process of Reward Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.04649"><img src="https://img.shields.io/badge/arXiv-2602.04649-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Qwen/RationaleRM"><img src="https://img.shields.io/badge/🤗%20Dataset-RationaleRM-yellow" alt="Dataset"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/legalcode.en"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.04649"><strong>[📄 论文]</strong></a> •
  <a href="#dataset-zh"><strong>[🤗 数据集]</strong></a> •
  <a href="#citation-zh"><strong>[📜 引用]</strong></a>
</p>

<p align="center">
  <img src="images/overall_compare.png" alt="Outcome Accuracy vs Rationale Consistency" width="70%">
</p>

<p align="center"><em>结果准确率 vs 理由一致性：理由一致性能够有效区分前沿模型并检测欺骗性对齐</em></p>

</div>


---

## 📖 项目简介

**RationaleRM** 是一个研究项目，旨在探索如何将奖励模型的*推理过程*（而不仅仅是*结果*）与人类判断对齐。我们发现，生成式奖励模型（GenRMs）和 LLM-as-a-Judge 存在**欺骗性对齐（Deceptive Alignment）**问题——模型可能通过肤浅甚至错误的判断过程达到与人类相同的最终结果。

为此，我们提出了 **理由一致性（Rationale Consistency）** 指标，用于衡量模型的推理过程与人类判断依据之间的对齐程度。同时，我们设计了 **MetaJudge** 框架来计算该指标：它将人类和模型的判别理由分解为原子单元，通过严格的一一语义匹配，精确量化两者的一致性。

**核心贡献：**

- 🔍 **MetaJudge 框架**：将人类理由分解为原子单元，使用 LLM 进行严格的一对一语义匹配
- 📊 **理由一致性指标**：能够有效发现欺骗性对齐，并区分前沿模型（如 GPT-5 或 Gemini 3 Pro）
- 🛠️ **混合奖励训练**：结合理由奖励（Average Precision）和结果奖励，防止"理由退化"
- 🏆 **SOTA 性能**：在 RM-Bench（87.1%）和 JudgeBench（82.0%）上取得最佳结果

---

## 🚨 核心发现：欺骗性对齐陷阱

我们对 19 个前沿模型进行评估，发现仅依赖结果准确率存在两个关键缺陷：

### 结果准确率无法区分前沿模型

在绿色区域中，尽管多个模型达到相似的结果准确率，但理由一致性能够清晰区分更强的模型（如 GPT-5、o3、Gemini 3 Pro）和较弱的模型（如 Claude 3.5、GPT-4.1）。

### 结果准确率无法检测欺骗性对齐

最典型的例子是 **o3 与 o3-mini** 的对比：两者结果准确率相近，但 o3-mini 的理由一致性低了近 50%。o3-mini 依赖表面线索（如格式、表情符号）做出判断，而 o3 会像人类一样进行严格的事实核查。

> 💡 **关键洞察**：模型可以因为错误的理由而做出正确的选择，仅靠结果准确率无法检测这种欺骗性对齐。

---

## 📉 训练发现：仅结果监督导致理由退化

<p align="center">
  <img src="images/reward_compare.png" alt="Training Dynamics" width="70%">
</p>

<p align="center"><em>训练动态对比：结果奖励相近，但理由奖励差异显著</em></p>

上图展示了训练过程中的关键发现：**仅结果监督会导致模型与人类判别过程的一致性持续下降**。

- **左图**：两种方法的结果奖励（Outcome Reward）几乎相同，说明模型都能学会选择正确答案
- **右图**：理由奖励（Rationale Reward）出现显著分化——没有理由一致性约束时，模型的理由奖励持续下降，最终比我们的方法低约 **24.2%**

这揭示了**理由退化（Rationale Degeneration）**现象：当中间判断过程不被激励时，模型会放弃高成本的证据核查，转而依赖更廉价的表面线索来达成相似的结果奖励。

---

## 🏆 主要结果

我们在两个具有挑战性的基准上进行评估：

- **RM-Bench**：评估模型区分细微差异和风格偏差的能力
- **JudgeBench**：强调深度判断和逻辑推理

| 模型                                  |    RM-Bench    |   JudgeBench   |      平均      |
| :------------------------------------ | :------------: | :------------: | :------------: |
| **生成式奖励模型**              |                |                |                |
| RM-R1-Distilled-Qwen-32B              |      83.9      |      78.8      |      81.4      |
| RRM-32B                               |      73.1      |      75.7      |      74.4      |
| Nemotron-Super-49B                    |      82.7      |      77.2      |      80.0      |
| RewardAnything-8B-v1                  |      83.1      |      62.6      |      72.9      |
| GRAM-R²                              |      85.7      |      81.0      |      83.4      |
| **结果监督基线**                |                |                |                |
| Qwen3-14B (Outcome-Only)              |      83.6      |      70.0      |      76.8      |
| Qwen3-30B-A3B (Outcome-Only)          |      84.9      |      75.7      |      80.3      |
| **我们的方法（结果+理由监督）** |                |                |                |
| Qwen3-14B (Ours)                      |      86.7      |      79.1      |      82.9      |
| **Qwen3-30B-A3B (Ours)**        | **87.1** | **82.0** | **84.6** |

> 💡 我们的方法有效逆转了仅结果监督训练时观察到的理由一致性下降（从 25% 提升至 37%）。

---


## 🚀 快速开始

### 项目结构

```
RationaleRM/
├── metajudge_infer.py              # 语义匹配推理脚本
├── metajudge_infer.sh              # 运行推理的 Shell 脚本
├── metajudge_analysis.py           # 计算指标的分析脚本
├── images/                         # 图片
│   ├── overall_compare.png
│   └── reward_compare.png
├── data/                           # 数据集
│   ├── helpsteer3_test_1000.jsonl      # 测试集：1000 条样本
│   └── helpsteer3_human_checklist.jsonl # 全集（22,116 条样本）
└── example/                   # 测试用示例数据
    ├── infer_input_10samples.jsonl
    ├── model-low_deceptive_alignment.jsonl
    └── model-high_deceptive_alignment.jsonl
```

### 第一步：准备数据

输入数据应为 JSONL 格式，包含以下字段：
- `human-checklist`：人类原子理由列表（参考标准）
- `{model}-checklist`：待评估的模型生成原子理由列表

示例：
```json
{
  "domain": "general",
  "context": [...],
  "response1": "...",
  "response2": "...",
  "human-checklist": [
    "回复1缺少多音节押韵",
    "回复2的韵律不一致"
  ],
  "model-low_deceptive_alignment-checklist": [
    "回复A的押韵方案显得牵强",
    "回复B的节奏感觉生硬"
  ]
}
```

### 第二步：运行推理

推理脚本评估每个模型生成的 checklist 条目与人类 checklist 的匹配程度：

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选，默认为 OpenAI

# 运行推理
python metajudge_infer.py \
    --input-file data/helpsteer3_test_1000.jsonl \
    --output-file output/results.jsonl \
    --model gpt-4o \
    --model-be-evaluated model-low_deceptive_alignment \
    --concurrent-requests 5
```

或使用 Shell 脚本：
```bash
bash metajudge_infer.sh
```

关键参数：
- `--input-file`：输入 JSONL 文件路径
- `--output-file`：输出结果路径
- `--model`：用于语义匹配的 LLM 模型（如 gpt-4o、qwen-plus）
- `--model-be-evaluated`：待评估的 critic 模型（其 checklist 将被评估）
- `--concurrent-requests`：并行 API 请求数

API 配置（通过环境变量或命令行参数）：
- `OPENAI_API_KEY` 或 `--api-key`：LLM 服务的 API 密钥
- `OPENAI_BASE_URL` 或 `--api-base`：API 基础 URL（默认：https://api.openai.com/v1）

### 第三步：分析结果

计算 Precision、Recall、F1 和 Average Precision：

```bash
# 分析单个文件
python metajudge_analysis.py \
    --input-file example/model-low_deceptive_alignment.jsonl \
    --model-be-evaluated model-low_deceptive_alignment

# 分析目录中的所有文件
python metajudge_analysis.py \
    --input-dir example/ \
    --sort-by recall
```

输出示例：

```text
====================================================================================================
按 RECALL 排序的结果
====================================================================================================
Model                                         Precision    Recall       F1           AP           Valid   
----------------------------------------------------------------------------------------------------
model-low_deceptive_alignment                 0.3300       0.4297       0.3684       0.3991       10      
model-high_deceptive_alignment                0.1850       0.2242       0.1985       0.2376       10      
====================================================================================================
```

---

## 📊 评估指标

MetaJudge 计算以下指标：

| 指标 | 描述 |
|------|------|
| **Recall（召回率）** | 被模型理由匹配的人类理由比例 |
| **Precision（精确率）** | 匹配人类理由的模型理由比例（用于评估） |
| **F1** | Precision 和 Recall 的调和平均值 |
| **Average Precision (AP)** | 用于本文当中的训练 |

---

<a id="dataset-zh"></a>

## 📂 数据集

我们提供两个数据集：

### 1. HelpSteer3 人类 Checklist（全集）

**`helpsteer3_human_checklist.jsonl`** 包含完整的 HelpSteer3 数据集及人工标注的原子理由，可用于训练。

### 2. 测试集（包含模型 Checklist）

**`helpsteer3_test_1000.jsonl`** 包含 1000 条经过挑选的测试样本，在论文当中用于测试，我们提供了两个代表不同欺骗性对齐程度的模型 checklist：

| 字段 | 描述 |
|------|------|
| `human-checklist` | 人工标注的原子理由（参考标准） |
| `model-low_deceptive_alignment-checklist` | 低欺骗性对齐模型的 checklist（对应论文中高 Rationale Consistency） |
| `model-low_deceptive_alignment-label` | 低欺骗性对齐模型的偏好标签 |
| `model-low_deceptive_alignment-generated_text` | 低欺骗性对齐模型的完整生成文本 |
| `model-high_deceptive_alignment-checklist` | 高欺骗性对齐模型的 checklist（对应论文中低 Rationale Consistency） |
| `model-high_deceptive_alignment-label` | 高欺骗性对齐模型的偏好标签 |
| `model-high_deceptive_alignment-generated_text` | 高欺骗性对齐模型的完整生成文本 |

> **注意：**
> - Atomic rationales were generated using GPT-5 for research purposes only.
> - `model-high_deceptive_alignment` 和 `model-low_deceptive_alignment` 数据仅用于测试/评估，未用于训练。

---

<a id="citation-zh"></a>

## 📜 引用

如果您觉得这项工作有帮助，请引用我们的论文：

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

**由 Qwen Team与复旦大学联合开发**

</div>
