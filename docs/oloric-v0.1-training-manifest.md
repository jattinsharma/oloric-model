# OLORIC v0.1 — Final Training Manifest & Configuration Freeze

**Status:** FROZEN  
**Freeze Date:** 2026-09-25  
**Target Architecture:** Qwen/Qwen3-4B-Instruct-2507 + QLoRA (NF4 4-bit)  
**Execution Environment:** Lightning AI Studio (1× NVIDIA L4 24GB, CUDA 12.8, PyTorch 2.8.0+cu128, Transformers 5.17.0, PEFT 0.21.0, bitsandbytes 0.50.2)

---

## 1. Dataset Integrity & Materialized Splits

| Attribute | Specification | Verification / Provenance |
| :--- | :--- | :--- |
| **Source Dataset Path** | `data/generated/enhanced_seed_data.jsonl` | Audited authoritative dataset |
| **Dataset SHA-256 (LF normalized)** | `9DA9F5E9B45CA85F718B3E71FD8FB81881D46101D9815C7323411C30C7625921` | Verified unchanged; matches commit `6277bf8` |
| **Total Examples** | 480 | Exactly 20 categories × 24 examples; 6 domains × 80 examples |
| **Split Algorithm** | Deterministic PRNG shuffle: `random.Random(42).shuffle(range(480))` | Implemented in `scripts/materialize_splits.py` |
| **Split Ratio** | 80% Train / 10% Validation / 10% Test | Configured in `configs/dataset.yaml` (`seed: 42`) |
| **Train Examples** | **384** (`data/splits/train.jsonl`) | Verified disjoint; 0 shared records with val/test |
| **Validation Examples** | **48** (`data/splits/validation.jsonl`) | Verified disjoint; 0 shared records with train/test |
| **Test Examples** | **48** (`data/splits/test.jsonl`) | Verified disjoint; 0 shared records with train/val |
| **Benchmark Count (Held Out)** | **20** (`evaluation/benchmark.jsonl`) | Fully held-out evaluation set (Hash: `92aa1cc5...`) |
| **Benchmark Leakage Audit** | **0** full record overlap across all splits | 0 identical task instances shared with training data |

---

## 2. Base Model Specification & Pinned Revision

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model ID** | `Qwen/Qwen3-4B-Instruct-2507` | Official HuggingFace repository |
| **Model Revision (Commit SHA)** | `cdbee75f17c01a7cc42f958dc650907174af0554` | Pinned immutable Git commit hash on HuggingFace Hub |
| **Max Sequence Length** | 2048 | Truncation right; conservative for L4 24GB |
| **Base Model Memory Footprint** | 2.49 GB (in 4-bit NF4) | Empirically verified on L4 during baseline generation |

---

## 3. Quantization Configuration (QLoRA NF4)

| Parameter | Setting | Notes |
| :--- | :--- | :--- |
| **Load in 4-bit** | `True` | `load_in_4bit = True` |
| **Quantization Type** | `nf4` | Normalized Float 4 |
| **Double Quantization** | `True` | `bnb_4bit_use_double_quant = True` |
| **Compute Dtype** | `bfloat16` | Native support on NVIDIA L4 (Ada Lovelace) |

---

## 4. LoRA Adapter Architecture

| Parameter | Setting |
| :--- | :--- |
| **LoRA Rank ($r$)** | 16 |
| **LoRA Alpha ($\alpha$)** | 32 |
| **LoRA Dropout** | 0.05 |
| **LoRA Bias** | `none` |
| **Task Type** | `CAUSAL_LM` |
| **Target Modules** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| **Trainable Parameters** | ~20M parameters (0.5% of total base model) |

---

## 5. Hyperparameters & Optimization Schedule

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | `paged_adamw_8bit` | Prevents OOM spikes during memory allocation |
| **Learning Rate** | `2.0e-4` (0.0002, float) | Explicit float notation for YAML 1.1 / PyYAML compatibility |
| **Weight Decay** | `0.01` | L2 regularization |
| **Max Gradient Norm** | `0.3` | Gradient clipping threshold |
| **Learning Rate Scheduler** | `cosine` | Cosine decay to zero |
| **Number of Epochs** | 3 | Full passes over the 384 training examples |
| **Per-Device Micro-Batch Size** | 1 | Fits comfortably within L4 VRAM at seq_len 2048 |
| **Gradient Accumulation Steps** | 4 | **Authoritative setting**; empirically verified in L4 smoke test |
| **Effective Batch Size** | **4** | $1 \text{ (micro-batch)} \times 4 \text{ (grad accum)} = 4$ |
| **Steps Per Epoch** | 96 | $384 \text{ train examples} / 4 = 96$ optimizer steps/epoch |
| **Total Optimizer Steps** | **288** | $96 \text{ steps/epoch} \times 3 \text{ epochs} = 288$ steps |
| **Warmup Ratio / Steps** | 3% → **8 steps** | $\lfloor 288 \times 0.03 \rfloor = 8$ warmup steps (`warmup_steps=8`) |
| **Checkpoint Frequency** | **Every 50 steps** | Saves checkpoints at steps 50, 100, 150, 200, 250, 288 |
| **Evaluation Frequency** | **Every 50 steps** | Evaluates validation loss on the 48 validation examples |
| **Logging Frequency** | **Every 5 steps** | Real-time loss tracking |
| **Max Checkpoints Kept** | 3 | `save_total_limit = 3` |

---

## 6. Category Taxonomy & Domain Balance

Authoritative Category Count: **20 Categories** (24 examples each = 480 total):
1. `simple_explanation`
2. `simplification`
3. `analogy`
4. `concrete_example`
5. `numerical_example`
6. `prerequisite_detection`
7. `misconception_detection`
8. `follow_up_questions`
9. `multi_turn_tutoring`
10. `repeated_confusion`
11. `strategy_switching`
12. `diagnostic_questions`
13. `hint_based_teaching`
14. `practice_questions`
15. `error_correction`
16. `partial_understanding`
17. `understanding_confirmation`
18. `memory_generation`
19. `document_grounded`
20. `context_retention`

Academic Domains: **6 Domains** (80 examples each = 480 total):
- `accountancy`
- `economics`
- `general_academic`
- `mathematics`
- `nutrition_food_science`
- `science`

---

## 7. Exact Training Command

To execute the authoritative Oloric v0.1 training run on the NVIDIA L4 environment:

```bash
python scripts/train_qlora.py \
  --train-file data/splits/train.jsonl \
  --val-file data/splits/validation.jsonl \
  --output-dir ./checkpoints \
  --save-steps 50 \
  --logging-steps 5
```

---

TRAINING_CONFIGURATION_FROZEN = YES
