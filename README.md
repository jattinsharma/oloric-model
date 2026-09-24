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
│   ├── generated/
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
│   ├── generate_seed_data.py
│   ├── run_baseline.py
│   ├── smoke_test.py
│   ├── train_qlora.py
│   ├── evaluate.py
│   └── export_adapter.py
├── evaluation/
│   ├── benchmark.jsonl
│   ├── rubric.md
│   └── reports/
├── notebooks/
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── training.md
    ├── evaluation.md
    └── lightning.md
```

## Getting Started

See [docs/training.md](docs/training.md) for setup instructions and [docs/lightning.md](docs/lightning.md) for Lightning AI Studio deployment.

## License

[MIT License](LICENSE)
