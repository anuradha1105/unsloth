✅ Note: All five Colab notebooks (Full Finetuning, LoRA, GRPO Reasoning, Continued Pretraining, and DPO) have been consolidated into a single unified notebook — Unsloth_ai_colab.ipynb — for ease of execution and review.


# Unsloth Modern AI Suite 
 
**Goal:** Implement five Unsloth-driven LLM workflows, each demonstrated in a **Colab notebook** and a **YouTube walkthrough**. Provide clear configs, evaluation, and export paths (GGUF / Ollama).

> **Quick links**
> - Unsloth notebooks (reference): https://github.com/unslothai/notebooks
> - Finetuning guide: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
> - RL guide + GRPO: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
> - Reasoning blog (R1/GRPO): https://unsloth.ai/blog/r1-reasoning
> - Continued pretraining basics: https://docs.unsloth.ai/basics/continued-pretraining
> - Example Kaggle NB: https://www.kaggle.com/code/kingabzpro/fine-tuning-llms-using-unsloth

---

## 🗂️ Repository Layout

```
unsloth/
├─ requirements.txt                    # minimal pinned deps
├─ configs/
│  ├─ base.yaml                        # common knobs (model, data, train)
│  ├─ lora.yaml                        # LoRA/QLoRA settings
│  ├─ rl.yaml                          # RLHF/SFT/GRPO knobs
│  └─ cpt.yaml                         # continued pretraining knobs
├─ data/
│  ├─ raw/                             # source datasets (gitignored)
│  └─ processed/                       # tokenized/Cleaned JSONL splits
├─ colabs/                             # each assignment has its own Colab
│  ├─ colab1_full_ft_smollm2_135m.ipynb
│  ├─ colab2_lora_smollm2_135m.ipynb
│  ├─ colab3_rl_preference_data.ipynb
│  ├─ colab4_rl_grpo_reasoning.ipynb
│  └─ colab5_continued_pretraining.ipynb
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train.py                         # FT/SFT with/without LoRA
│  ├─ evaluate.py                      # loss/ppl + quick win-rate
│  ├─ infer.py                         # prompt chat/infer
│  ├─ export.py                        # merge adapters, export gguf/ollama
│  └─ utils.py
├─ experiments/
│  ├─ runs/                            # TB/accelerate logs (gitignored)
│  └─ results/                         # metrics, artifacts, screenshots
├─ models/                             # adapters/merged weights (gitignored)
└─ docs/

```


---

## 🔧 Environment & Installation

- Recommended Python: **3.10–3.11**
- Colab: **T4/A100**, CUDA ≥ 12.x works; enable GPU in *Runtime → Change runtime type*.  
- Minimal install:
  ```bash
  pip install "unsloth" "transformers>=4.44" "accelerate>=1.0.0"               "datasets>=3.0.0" "trl>=0.11.0" "peft>=0.13.0"               "evaluate" "scikit-learn" "tensorboard" "pyyaml"               "bitsandbytes"  # Linux GPU only
  ```
- If you see **“unsloth ... does not provide the extra 'all'”**, just use `unsloth` (extras were removed).
- GPU wheels: install a PyTorch CUDA build matching your driver; then reinstall `bitsandbytes`.

---

## 🎯 Assignment Overview 

For each task:
1. **Cloned/created a Colab**, ran **end-to-end without errors**, and **recorded a YouTube walkthrough** explaining: dataset, input format, code path, hyperparameters, training, evaluation, and inference.
2. Produced **metrics** (loss & perplexity, and task-specific evals), **screenshots**, and **short demo outputs**.
3. Committed configs, results tables, and a short **demo_script.md** for the talk track.



---

## a) Colab 1 — **Full Finetuning** on a Small Model (SmolLM2-135M)

- **Objective:** Show *full finetuning* (`full_finetuning: true`) on the smallest model to minimize compute while demonstrating full-weight updates.
- **Base models considered:** `HuggingFaceTB/SmolLM2-135M` (primary), plus notes on `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` and other small open weights.
- **Task type:** Instruction-following (chat) or lightweight coding Q&A.
- **Input format (SFT):** JSONL with `instruction`, `input`, `output` (or chat template format). Example:
  ```json
  {"instruction":"Write a Python function to sum a list.","input":"","output":"def sum_list(xs): return sum(xs)"}
  ```
- **Chat templates:** I tested multiple chat model templates and documented which prompt format/tokenizer special tokens are required.
- **Config notes:**
  - `configs/base.yaml`: set `model_name: "HuggingFaceTB/SmolLM2-135M"`
  - `configs/rl.yaml`: not used here
  - In code/notebook, pass `full_finetuning=True`, disable LoRA modules.
  - Short epochs (e.g., 1–3) and small seq len for Colab T4.
- **Colab:** `colabs/colab1_full_ft_smollm2_135m.ipynb`
- **Video:** *(link to upload)*

**Run card**
| Run | Full FT | Model | Dataset | Seq Len | Batch x Accum | Epochs | LR | Val PPL ↓ | Notes | Video |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| colab1-ft-s135m | ✅ | SmolLM2‑135M | (name here) | 2048 | 4 × 4 | 2 | 2e‑4 | **X.XX** | Baseline full‑FT | [YouTube]() |

---

## b) Colab 2 — **LoRA / QLoRA** on the Same Model & Dataset

- **Objective:** Repeat Colab 1 with **parameter-efficient finetuning** (LoRA/QLoRA).
- **Why:** Demonstrate memory savings, faster training, and similar task quality.
- **Config notes:**
  - `configs/lora.yaml`: `use_lora: true`, set `r`, `alpha`, `target_modules`, `bnb_4bit: true` for QLoRA.
  - In notebook, ensure `full_finetuning=False` (or omit) and PEFT is active.
- **Colab:** `colabs/colab2_lora_smollm2_135m.ipynb`
- **Video:** *(link to upload)*

**Run card**
| Run | LoRA | Model | Dataset | r / α | 4‑bit | Epochs | LR | Val PPL ↓ | Δ vs Full FT | Notes | Video |
|---|---|---|---|---|---|---:|---:|---:|---|---|---|
| colab2-lora-s135m | ✅ | SmolLM2‑135M | (name) | 16 / 32 | nf4 | 2 | 2e‑4 | **X.XX** | (↑/↓) | Memory‑efficient | [YouTube]() |

---

## c) Colab 3 — **Reinforcement Learning (Preference Data)**

- **Objective:** Use a dataset containing **prompt + preferred + rejected** responses to run an RL pipeline (e.g., DPO/ORPO/standard preference objectives as supported by TRL/Unsloth flow).
- **Input format:** JSONL with `prompt`, `chosen`, `rejected` (or the format required by the chosen trainer).
- **Config:** `configs/rl.yaml` (num_steps, kl_coef/β where applicable, reward/preference objective).
- **Outputs:** Policy checkpoint (adapters if LoRA), reward curves, offline eval on held-out prompts.
- **Colab:** `colabs/colab3_rl_preference_data.ipynb`
- **Video:** *(link to upload)*

**Run card**
| Run | Objective | Model | Dataset | Steps | β / KL | Val Metric | Notes | Video |
|---|---|---|---|---:|---:|---:|---|---|
| colab3-rl-pref | DPO/ORPO | SmolLM2‑135M | (name) | N | 0.1 | **X.XX** | Pref‑tuning | [YouTube]() |

---

## d) Colab 4 — **Reinforcement Learning with GRPO (Reasoning)**

- **Objective:** Train a **reasoning‑style** model using **GRPO** as per Unsloth’s guide.
- **Dataset:** Problem statements with verifiable answers (math/logic). Model generates solutions; reward based on correctness & reasoning signals.
- **Key notes:**
  - Smaller models can be used for demonstration; watch context length and reward computation latency.
  - Log both **pass@k** and **reward curves**.
- **Colab:** `colabs/colab4_rl_grpo_reasoning.ipynb`
- **Video:** *(link to upload)*

**Run card**
| Run | Algo | Model | Task | Steps | Pass@1 ↑ | Reward ↑ | Notes | Video |
|---|---|---|---|---:|---:|---:|---|---|
| colab4-grpo | GRPO | SmolLM2‑135M (or 2B) | reasoning | N | **X%** | **X.XX** | R1‑style demo | [YouTube]() |

---

## e) Colab 5 — **Continued Pretraining** (Teach a New Domain/Language)

- **Objective:** Use Unsloth for **continued pretraining (CPT)** on raw text to extend vocabulary/domain knowledge (e.g., a low‑resource language or domain docs).
- **Data:** Unstructured text (tokenized). Create `data/processed/cpt_train.txt` (or HF dataset).
- **Config:** `configs/cpt.yaml` with tokenizer updates (if needed), MLM/causal LM flags, and schedule.
- **Artifacts:** Updated tokenizer (optional), continued‑pretrained model, sample losses.
- **Colab:** `colabs/colab5_continued_pretraining.ipynb`
- **Video:** *(link to upload)*

**Run card**
| Run | Mode | Base | Tokens (M) | Epochs | LR | Val Loss ↓ | Notes | Video |
|---|---|---|---:|---:|---:|---:|---|---|
| colab5-cpt | CPT (causal) | SmolLM2‑135M | X | 1‑3 | 2e‑4 | **X.XX** | New language/domain | [YouTube]() |

---

## 📦 Data & Formats

- **SFT (full‑FT / LoRA):** JSONL with `instruction`, `input`, `output`. For chat‑template models, convert to the model’s template (documented in each notebook).
- **Preference RL:** JSONL with `prompt`, `chosen`, `rejected` (or trainer‑specific schema).
- **GRPO:** Problem → answer pair; reward function checks correctness (and optionally solution structure).
- **CPT:** Raw text corpus (UTF‑8).

Data prep entrypoint:
```bash
python scripts/prepare_data.py --config configs/base.yaml
```

---

## 🧪 Evaluation

- **SFT/FT:** train/val loss, **perplexity**; small qualitative **win‑rate** vs base model on N prompts.
- **RL (pref):** objective metric loss; human/automatic win‑rate on held‑out prompts.
- **GRPO:** **pass@k**, reward‑per‑step curves; sample chain‑of‑thought style outputs (without exposing training-only content).
- **CPT:** pretrain loss over steps; downstream quick probe (few prompts).

Results saved under `experiments/results/<run_name>/` with a `metrics.json` and a markdown `summary.md`. Screenshots go in `docs/figures/`.

---

## 💬 Inference & Chat Templates

- Inference script:
  ```bash
  python scripts/infer.py --adapters models/<run>/adapters     --prompt "Explain CRISP‑DM phases briefly for a grad class."
  ```
- I tested **multiple chat templates** (Llama/Gemma/Mistral‑style) and noted the required BOS/EOS/special tokens in the notebooks.
- Sample prompts: `docs/demo_scripts/*`

---

## 🚢 Export (GGUF & Ollama)

- Merge LoRA adapters:
  ```bash
  python scripts/export.py --adapters models/<run>/adapters --merge-to models/<run>/merged
  ```
- Export GGUF (for llama.cpp):
  ```bash
  python scripts/export.py --to-gguf models/<run>/gguf
  ```
- **Ollama**:
  1. Export merged/safetensors as required by the guide.
  2. Create `Modelfile` and run `ollama create my-unsloth -f Modelfile`
  3. `ollama run my-unsloth`
- I included a short **inference UI** note in the notebooks (text box → response).

---

## 🧰 Troubleshooting (What I hit & fixed)

- **Pip extras warning:** `unsloth[all]` → just `unsloth`.
- **CUDA/Torch mismatch:** installed the matching PyTorch CUDA wheel; reinstalled `bitsandbytes`.
- **OOM on T4:** reduced `max_seq_length`, batch size; enabled QLoRA (`bnb_4bit: true`).
- **`generate()` crash:** set `use_cache=False` during training; shorter prompts; verified tokenizer/model pairing.
- **Accelerate config:** `accelerate config` to set mixed precision & GPU selection.

---

## 📑 Report Template


| Colab | Mode | Model | Dataset | Key HPs | Main Metric | Result | Artifacts |
|---|---|---|---|---|---|---|---|
| 1 | Full FT | SmolLM2‑135M | (name) | seq=2048, lr=2e‑4, ep=2 | Val PPL | **X.XX** | adapters/merged, sample gens |
| 2 | LoRA | SmolLM2‑135M | (same) | r=16, nf4, ep=2 | Val PPL | **X.XX** | adapters, merged |
| 3 | RL (pref) | SmolLM2‑135M | (pref set) | steps=N, β=0.1 | Win‑rate | **X%** | policy, curves |
| 4 | GRPO | SmolLM2‑135M/2B | reasoning | steps=N | Pass@1 | **X%** | policy, curves |
| 5 | CPT | SmolLM2‑135M | (corpus) | tokens=M, ep=1‑3 | Val loss | **X.XX** | cpt weights |

---

