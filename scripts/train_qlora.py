#!/usr/bin/env python3
"""
QLoRA training script for OLORIC.
Implements the training pipeline using QLoRA for efficient fine-tuning.
"""
import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# Ensure src directory is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from oloric.training import OloricTrainer
from oloric.data import data_manager
from oloric.schemas.dataset import TrainingExample

def load_dataset(file_path: str) -> List[TrainingExample]:
    """Load dataset from JSONL file."""
    examples = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                example = TrainingExample(**data)
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error parsing example on line {line_num}: {e}")
    return examples


def examples_to_dicts(examples: List[TrainingExample]) -> List[dict]:
    """Convert TrainingExample Pydantic objects to plain dicts for tokenization."""
    return [ex.dict() for ex in examples]

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train OLORIC with QLoRA")
    parser.add_argument("--train-file", type=str, default="./data/splits/train.jsonl",
                        help="Path to training dataset")
    parser.add_argument("--val-file", type=str, default="./data/splits/validation.jsonl",
                        help="Path to validation dataset")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Base model name (overrides config)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Maximum number of training samples to use")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Maximum number of validation samples to use")
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OLORIC QLoRA Training")
    print("=" * 60)
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_examples = load_dataset(args.train_file)
        val_examples = load_dataset(args.val_file) if os.path.exists(args.val_file) else None
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return 1
    
    # Limit samples if specified
    if args.max_train_samples and args.max_train_samples < len(train_examples):
        train_examples = train_examples[:args.max_train_samples]
        print(f"Limited training to {args.max_train_samples} samples")
    
    if args.max_val_samples and val_examples and args.max_val_samples < len(val_examples):
        val_examples = val_examples[:args.max_val_samples]
        print(f"Limited validation to {args.max_val_samples} samples")
    
    if len(train_examples) == 0:
        print("Error: No training examples loaded")
        return 1

    # Convert Pydantic objects to plain dicts for tokenization
    print("Converting examples to dicts...")
    train_dicts = examples_to_dicts(train_examples)
    val_dicts   = examples_to_dicts(val_examples) if val_examples else None

    # Initialize trainer
    print("Initializing trainer...")
    try:
        trainer = OloricTrainer()

        # Override config if specified
        if args.model_name:
            print(f"Overriding model name to: {args.model_name}")
            trainer.model_config.setdefault("base_model", {})["name"] = args.model_name

        # Override training args if specified
        if args.logging_steps is not None:
            trainer.training_config["logging_steps"] = args.logging_steps
        if args.save_steps is not None:
            trainer.training_config["save_steps"] = args.save_steps
            trainer.training_config["eval_steps"] = args.save_steps

    except Exception as e:
        print(f"Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Setup model & tokenizer FIRST so formatter is available for tokenization
    print("Setting up model and tokenizer...")
    try:
        trainer.setup_model_and_tokenizer()
    except Exception as e:
        print(f"Error setting up model/tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Tokenize datasets using the live formatter (model must be loaded first)
    print("Tokenizing datasets...")
    try:
        train_dataset = trainer.prepare_dataset(train_dicts)
        val_dataset   = trainer.prepare_dataset(val_dicts) if val_dicts else None
        print(f"Train dataset: {len(train_dataset)} examples tokenized")
        if val_dataset:
            print(f"Val dataset:   {len(val_dataset)} examples tokenized")
    except Exception as e:
        print(f"Error tokenizing datasets: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Start training
    print("Starting training...")
    try:
        start_time = time.time()
        trainer_obj = trainer.train(train_dataset, val_dataset)
        end_time = time.time()

        training_time = end_time - start_time
        print(f"Training completed in {training_time/3600:.2f} hours")

        # Save final model
        final_output_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_output_dir)
        print(f"Final model saved to: {final_output_dir}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
