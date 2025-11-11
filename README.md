✅ Note: All five Colab notebooks (Full Finetuning, LoRA, GRPO Reasoning, Continued Pretraining, and DPO) have been consolidated into a single unified notebook — Unsloth_ai_colab.ipynb — for ease of execution and review.


# 🧠 Unsloth AI Assignment

This repository contains six Colab notebooks demonstrating modern fine-tuning and alignment techniques using **[Unsloth](https://github.com/unslothai/unsloth)** — an optimized library for efficient large language model (LLM) training on Google Colab.

Each notebook focuses on a different AI training paradigm such as full fine-tuning, continued pretraining, LoRA / QLoRA adaptation, and alignment through DPO or GRPO.  
All experiments were performed using **`smollm2-135M`**, a compact open model suitable for Colab GPUs.

---

## 📁 Repository Structure

```
unsloth_assignment/
│
├── 1_full_finetuning.ipynb
├── 2_continued_pretraining_hindi.ipynb
├── 3_qlora_finetuning.ipynb
├── 4_grpo_reasoning.ipynb
├── 5_dpo_alignment.ipynb
├── 6_rlaif_reward_model.ipynb
│
└── README.md
```

---

## 📚 Overview of Colabs

| **#** | **Colab Title** | **Technique / Trainer** | **Objective** |
|:--:|:-----------------|:------------------------|:--------------|
| **1** | `1_full_finetuning.ipynb` | Full fine-tuning (SFTTrainer) | Trains the base model end-to-end on an English instruction dataset to learn general instruction following. |
| **2** | `2_continued_pretraining_hindi.ipynb` | Continued pretraining (SFTTrainer) | Performs domain adaptation on a **custom Hindi text corpus**, extending the model’s multilingual capabilities. |
| **3** | `3_qlora_finetuning.ipynb` | QLoRA / 4-bit fine-tuning | Demonstrates parameter-efficient training using low-rank adapters and quantized weights for faster training. |
| **4** | `4_grpo_reasoning.ipynb` | GRPOTrainer (Reasoning Optimization) | Enhances reasoning and chain-of-thought behavior using a small reasoning / math dataset. |
| **5** | `5_dpo_alignment.ipynb` | DPOTrainer (Direct Preference Optimization) | Aligns model responses with human preferences using a dataset of paired “chosen” vs “rejected” answers. |
| **6** | `6_rlaif_reward_model.ipynb` | Reward model training (optional) | Trains a small reward model on synthetic “good / bad” responses, demonstrating self-alignment (RLAIF). |

---

## 🧩 Datasets Used

| **Colab #** | **Dataset** | **Language** | **Purpose** |
|:-------------|:-------------|:--------------|:-------------|
| 1 | Alpaca / Instruction dataset (`yahma/alpaca-cleaned`) | English | Teaches base model instruction following. |
| 2 | Custom Hindi corpus (Wikipedia / news / text file) | Hindi | Expands vocabulary and contextual fluency in Hindi. |
| 3 | Same Alpaca-style dataset | English | Tests QLoRA efficiency versus full finetuning. |
| 4 | Reasoning dataset (math / logic Q&A) | English | Improves step-by-step reasoning and consistency. |
| 5 | Preference pairs (helpful vs harmful answers) | English | Enables alignment without a reward model (DPO). |
| 6 | Synthetic reward dataset | English | Optional demonstration of reward modeling. |

> **Note:** Large model checkpoints (`pytorch_model.bin`, `*.safetensors`, etc.) are intentionally **not included** in this repository to keep it lightweight.  
> All trained weights were saved separately to Google Drive during experiments.

---

## ⚙️ Environment Setup

Each notebook automatically installs required dependencies inside Colab:

```bash
!pip install -q "unsloth==2025.11.2" "trl" "transformers" "datasets" "peft" "accelerate"
```

Runtime requirements:
- GPU: T4 / A100 (Colab)
- Python: 3.10 or higher
- Libraries: Unsloth, TRL, PEFT, Transformers, Accelerate

---

## 💾 Saving & Checkpoints

Every notebook includes a consistent saving pattern:

```python
SAVE_DIR = "experiment_name_final"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved to:", SAVE_DIR)
```

Optionally, models were archived or moved to Google Drive for permanent storage.

---

## ✅ Deliverables Summary

| Component | Description |
|:-----------|:-------------|
| **Colab notebooks** | Complete training and evaluation code for all six use-cases. |
| **Images (optional)** | Output screenshots and comparison results. |
| **README.md** | Full documentation of tasks, datasets, and objectives. |

---

### 📄 License
Open-source educational submission under SJSU academic guidelines.  
All datasets used are open or publicly available for research.

