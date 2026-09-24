# OLORIC Model Training Guide

This document describes the complete training workflow for the OLORIC adaptive tutoring model, covering environment setup, data preparation, QLoRA fine-tuning, and evaluation.

---

## 1. Prerequisites & Environment

### Hardware
- **Target GPU**: 1× NVIDIA L4 24GB (Ada Lovelace, compute capability 8.9)
- **Minimum VRAM**: 16 GB (24 GB recommended for sequence length 2048)
- **Storage**: Minimum 20 GB free disk space

### Software
- **Python**: 3.10+ (recommended 3.10 or 3.11)
- **CUDA Runtime**: CUDA 12.1 or higher
- **PyTorch**: 2.1.0+ with CUDA support

### Dependency Installation
```bash
pip install -r requirements.txt
pip install -e .
```

Verify setup:
```bash
python scripts/inspect_environment.py
```

---

## 2. Model Architecture & Fine-Tuning Strategy

OLORIC uses Parameter-Efficient Fine-Tuning (PEFT) with **QLoRA** (4-bit quantized Low-Rank Adaptation):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | `Qwen/Qwen3-4B-Instruct-2507` | Instruction-tuned open-weight transformer |
| Quantization | 4-bit NormalFloat4 (NF4) | BitsAndBytes double quantization |
| Compute Dtype | `torch.bfloat16` | Native bfloat16 computation on L4 |
| LoRA Rank ($r$) | 16 | Adapter dimension |
| LoRA Alpha ($\alpha$) | 32 | Scaling factor ($\alpha / r = 2$) |
| LoRA Dropout | 0.05 | Regularization |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All linear projection layers |
| Optimizer | `paged_adamw_8bit` | 8-bit pageable optimizer to minimize VRAM |
| Micro-Batch Size | 1 | Fits within 24GB VRAM |
| Gradient Accumulation | 8 | Effective batch size = 8 |
| Max Sequence Length | 2048 | Document context + conversation history |

Configurations are managed in `configs/model.yaml` and `configs/qlora.yaml`.

---

## 3. Data Preparation & Validation

### Data Format
Training data consists of JSONL files conforming to the schema defined in `src/oloric/schemas/dataset.py`. Each example contains:
- `id`: unique identifier
- `category`: tutoring category (e.g. `simple_explanation`, `analogy`, `strategy_switching`)
- `domain`: subject area (e.g. `economics`, `science`, `mathematics`)
- `instruction`: prompt instruction
- `context`: document context, learner state, conversation history, current goal
- `target_response`: structured model output (action, explanation, diagnosis, memory candidate)

### Validation
Validate dataset files before training:
```bash
# Validate individual files
python scripts/validate_dataset.py --train data/train.jsonl --validation data/validation.jsonl

# Validate all partitions (train, validation, test)
python scripts/validate_dataset.py --all
```

---

## 4. Gate 1 — Verification Before Full Training

> [!CRITICAL]
> Do NOT begin full training until Gate 1 empirical verification has passed.

### Step 4.1: Establish Base-Model Baseline
Run the 20-scenario benchmark on the un-finetuned base model:
```bash
python scripts/run_baseline_inference.py --mode both --output-dir evaluation/baseline
```
Score automated compliance:
```bash
python scripts/score_baseline_automated.py --responses evaluation/baseline/responses.jsonl --output-dir evaluation/baseline
```

### Step 4.2: Run L4 GPU Smoke Test
Validate memory consumption and training stability:
```bash
python scripts/l4_smoke_test.py --test both --output-dir evaluation/smoke_test
```
Confirm:
- Peak VRAM is under 20 GB
- Checkpoint save and reload succeed
- Post-reload inference generates valid tokens

---

## 5. Running QLoRA Training

Execute the training script:
```bash
python scripts/train_qlora.py \
    --train-file ./data/train.jsonl \
    --val-file ./data/validation.jsonl \
    --output-dir ./checkpoints \
    --logging-steps 5 \
    --save-steps 50
```

### Training Options
- `--train-file`: Path to training dataset (JSONL)
- `--val-file`: Path to validation dataset (JSONL)
- `--output-dir`: Checkpoint output directory (default: `./checkpoints`)
- `--model-name`: Override base model identifier
- `--resume-from`: Path to a checkpoint to resume training
- `--max-train-samples`: Limit number of training samples (for smoke testing)
- `--max-val-samples`: Limit number of validation samples

---

## 6. Evaluation & Export

### Evaluate Trained Model
```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/final_model \
    --benchmark-file ./evaluation/benchmark.jsonl \
    --output-dir ./evaluation/reports
```

### Export Adapter
```bash
python scripts/export_adapter.py \
    --model-path ./checkpoints/final_model \
    --output-path ./exported_model
```
