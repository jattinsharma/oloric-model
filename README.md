# OLORIC Model Project

An adaptive teaching model designed to help students move from confusion to understanding when studying documents.

## Overview

OLORIC is a specialized tutoring model that learns HOW TO TEACH, not what to teach. It works with document context, learner state, and conversation history to determine the most appropriate tutoring action and produce useful responses that adapt to the student's needs.

## Key Features

- Specialized for tutoring behavior (explanation, simplification, analogy, etc.)
- Designed to work with document-grounded contexts
- Learns to switch teaching strategies when initial explanations fail
- Generates structured "Oloric Memory" candidates for knowledge retention
- Built on open-weight instruction-tuned Transformer base models
- Optimized for LoRA/QLoRA fine-tuning on consumer GPUs (NVIDIA L4 24GB)

## Project Structure

```
oloric-model/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── model.yaml
│   ├── dataset.yaml
│   ├── qlora.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── schemas/
│   ├── model_input.py
│   ├── model_output.py
│   └── dataset.py
├── src/
│   └── oloric/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── formatting.py
│       ├── training.py
│       ├── inference.py
│       ├── evaluation.py
│       ├── validators.py
│       └── utils.py
├── scripts/
│   ├── inspect_environment.py
│   ├── validate_dataset.py
│   ├── run_baseline_inference.py
│   ├── score_baseline_automated.py
│   ├── l4_smoke_test.py
│   ├── train_qlora.py
│   ├── evaluate.py
│   └── export_adapter.py
├── evaluation/
│   ├── benchmark.jsonl
│   ├── baseline/
│   │   └── semantic_rubric.md
│   ├── smoke_test/
│   └── reports/
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── training.md
    ├── evaluation.md
    └── lightning.md
```

## Quickstart: Lightning AI Studio (1× NVIDIA L4 24GB)

### 1. Setup & Environment Inspection
```bash
# Clone and enter repository
git clone https://github.com/jattinsharma/oloric-model.git
cd oloric-model

# Install dependencies (retaining CUDA PyTorch)
pip install -r requirements.txt
pip install -e .

# Verify environment readiness
python scripts/inspect_environment.py
```

### 2. Gate 1 — Empirical Baseline & GPU Smoke Test
```bash
# Run baseline inference on Qwen/Qwen3-4B-Instruct-2507
python scripts/run_baseline_inference.py --mode both --output-dir evaluation/baseline

# Score automated structural compliance
python scripts/score_baseline_automated.py --responses evaluation/baseline/responses.jsonl --output-dir evaluation/baseline

# Execute L4 GPU smoke test (measure VRAM, test checkpoint save/reload)
python scripts/l4_smoke_test.py --test both --output-dir evaluation/smoke_test
```

### 3. Dataset Validation & QLoRA Training
```bash
# Validate dataset files
python scripts/validate_dataset.py --train data/train.jsonl --validation data/validation.jsonl

# Run 4-bit QLoRA training
python scripts/train_qlora.py \
    --train-file ./data/train.jsonl \
    --val-file ./data/validation.jsonl \
    --output-dir ./checkpoints
```

### 4. Evaluation
```bash
# Evaluate trained adapter
python scripts/evaluate.py \
    --model-path ./checkpoints/final_model \
    --benchmark-file ./evaluation/benchmark.jsonl
```

## Documentation

- [docs/lightning.md](docs/lightning.md) — Comprehensive Lightning AI Studio deployment guide
- [docs/training.md](docs/training.md) — Training setup, parameters, and workflow
- [docs/evaluation.md](docs/evaluation.md) — Evaluation framework, automated scoring, and semantic rubric
- [docs/architecture.md](docs/architecture.md) — System architecture and design principles
- [docs/dataset.md](docs/dataset.md) — Dataset design, schemas, and taxonomy

## License

[MIT License](LICENSE)

