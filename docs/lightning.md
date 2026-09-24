# Lightning AI Studio Deployment Guide — OLORIC

This guide provides an exact, step-by-step reproducible workflow for running the OLORIC model on a **Lightning AI Studio** with **1× NVIDIA L4 24GB**.

---

## 1. Hardware & Studio Setup

### 1.1 Studio Specification
- **Platform**: Lightning AI Studio
- **GPU**: 1× NVIDIA L4 (24GB VRAM, Ada Lovelace architecture, compute capability 8.9)
- **Host OS**: Linux (Ubuntu 22.04 LTS)
- **Python**: 3.10 or 3.11 (standard in Lightning GPU studios)
- **CUDA Runtime**: CUDA 12.1+ pre-installed

### 1.2 Verify Studio GPU Environment
Open a terminal in your Studio and verify the GPU:
```bash
nvidia-smi
```
Expected output:
- GPU: `NVIDIA L4`
- Total Memory: `22528 MiB` (~24GB)
- Driver & CUDA Version: `CUDA Version: 12.x`

Verify PyTorch sees CUDA:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))"
```
Expected output:
```
CUDA Available: True
Device: NVIDIA L4
VRAM (GB): 22.5
```

---

## 2. Repository Setup & Dependency Installation

### 2.1 Clone Repository
```bash
git clone https://github.com/jattinsharma/oloric-model.git
cd oloric-model
```

### 2.2 Install Dependencies
> [!IMPORTANT]
> Do NOT install a CPU-only PyTorch wheel. Lightning Studio comes with a CUDA-enabled PyTorch pre-installed. The `requirements.txt` specifies `torch>=2.1.0` so pip will keep the pre-installed CUDA build.

```bash
pip install -r requirements.txt
pip install -e .
```

### 2.3 Inspect Environment
Run the project inspection script to verify all prerequisites:
```bash
python scripts/inspect_environment.py
```
Ensure all items display `[OK]` or `[PASS]` and overall assessment is `System Ready for OLORIC: [YES]`.

---

## 3. Base Model Configuration

- **Model Identifier**: `Qwen/Qwen3-4B-Instruct-2507`
- **Architecture**: Qwen3 instruction-tuned causal language model (~4 billion parameters)
- **Quantization**: 4-bit NormalFloat4 (NF4) with double quantization and `bfloat16` compute dtype
- **Chat Template**: ChatML format (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant`)
- **Adapter Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

Hugging Face authentication (if accessing gated/private models):
```bash
huggingface-cli login
```

---

## 4. Gate 1 — Baseline Empirical Measurement

Before training, establish the empirical baseline on the raw base model.

### 4.1 Run Baseline Inference
Runs all 20 benchmark scenarios across both modes:
- **Mode A (General)**: standard tutoring prompt without schema constraints
- **Mode B (Structured)**: Oloric contract prompt with strict JSON schema instructions

```bash
python scripts/run_baseline_inference.py --mode both --benchmark evaluation/benchmark.jsonl --output-dir evaluation/baseline
```

Outputs generated:
- `evaluation/baseline/responses.jsonl` — all raw model responses, token counts, and latency
- `evaluation/baseline/manifest.json` — library versions, hardware details, generation config

### 4.2 Automated Structural Scoring
Evaluates deterministic criteria (JSON validity, schema compliance, required fields, enum validity):
```bash
python scripts/score_baseline_automated.py --responses evaluation/baseline/responses.jsonl --output-dir evaluation/baseline
```

Outputs generated:
- `evaluation/baseline/automated_scores.json`
- `evaluation/baseline/automated_score_summary.md`

### 4.3 Semantic Rubric Evaluation
Evaluate qualitative dimensions using the 11-dimension rubric:
- Open `evaluation/baseline/semantic_rubric.md`
- Score the responses across pedagogy, ground truth fidelity, and strategy switching
- Identify and document failure cases

---

## 5. Gate 1 — L4 GPU Smoke Test

Verify 4-bit QLoRA training stability and measure exact VRAM consumption:

```bash
python scripts/l4_smoke_test.py --test both --output-dir evaluation/smoke_test
```

This runs:
1. **Test A (Synthetic)**: 5 steps, sequence length 256, micro-batch size 1
2. **Test B (Real Oloric Config)**: 10 steps, sequence length 2048, micro-batch size 1
3. **Checkpoint Lifecycle**: Save adapter checkpoint, reload from disk, run test inference

Outputs generated:
- `evaluation/smoke_test/l4_smoke_test_report.json`
- `evaluation/smoke_test/l4_smoke_test_summary.md`

Verify the gate criteria:
- Actual peak VRAM < 20 GB (safety margin on 24 GB L4)
- Checkpoint reload: PASS
- Post-reload inference: PASS

---

## 6. Dataset Validation

Before initiating full training, validate all dataset partitions:
```bash
python scripts/validate_dataset.py --train data/train.jsonl --validation data/validation.jsonl
```
Ensure all examples conform to schemas, contain non-empty responses, and pass Pydantic validation.

---

## 7. QLoRA Training

Run the 4-bit QLoRA fine-tuning pipeline on the L4 GPU:

```bash
python scripts/train_qlora.py \
    --train-file ./data/train.jsonl \
    --val-file ./data/validation.jsonl \
    --output-dir ./checkpoints \
    --logging-steps 5 \
    --save-steps 50
```

Configuration highlights (from `configs/qlora.yaml`):
- Quantization: 4-bit NF4, double quant, `bnb_4bit_compute_dtype="bfloat16"`
- Optimizer: `paged_adamw_8bit`
- Micro-batch size: 1
- Gradient accumulation steps: 8 (effective batch size: 8)
- Sequence length: 2048
- Precision: `bf16=True`, `tf32=True`
- Checkpoint directory: `./checkpoints/final_model`

---

## 8. Post-Training Evaluation

Evaluate the fine-tuned adapter against the benchmark dataset:

```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/final_model \
    --benchmark-file ./evaluation/benchmark.jsonl \
    --output-dir ./evaluation/reports
```

To compare fine-tuned model performance against the base model:
```bash
python scripts/evaluate.py \
    --model-path ./checkpoints/final_model \
    --compare-baseline \
    --benchmark-file ./evaluation/benchmark.jsonl
```

---

## 9. Export Adapter for Deployment

Export and verify the trained adapter:
```bash
python scripts/export_adapter.py \
    --model-path ./checkpoints/final_model \
    --output-path ./exported_model
```
Optional: Push directly to Hugging Face Hub:
```bash
python scripts/export_adapter.py \
    --model-path ./checkpoints/final_model \
    --output-path ./exported_model \
    --push-to-hub \
    --hub-model-name your-org/oloric-4b-qlora
```
