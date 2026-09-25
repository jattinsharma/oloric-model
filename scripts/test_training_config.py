#!/usr/bin/env python3
"""
Lightweight training configuration and initialization test for OLORIC v0.1.

Verifies:
1. Loading the frozen YAML configs (configs/model.yaml, configs/qlora.yaml)
2. Resolving all training parameters
3. Strict type validation for every parameter (especially numeric fields)
4. Construction of TrainingArguments without any string/float type errors
5. Trainer initialization and optimizer creation (without running full training)
"""
import os
import sys

# Ensure repository root and src directory are on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch
from transformers import TrainingArguments, Trainer
from oloric.config import config
from oloric.training import (
    validate_training_parameters,
    parse_numeric_param,
    OloricTrainer
)


def test_training_configuration():
    print("=" * 70)
    print("STEP 1: Load Frozen Configurations")
    print("=" * 70)
    m_cfg = config.get_model_config()
    q_cfg = config.get_qlora_config()
    bm_cfg = m_cfg.get("base_model", {})
    t_cfg = m_cfg.get("training", {})

    print(f"Model ID:              {bm_cfg.get('name')}")
    print(f"Model Revision:        {bm_cfg.get('revision')}")
    print(f"Raw learning_rate:     {t_cfg.get('learning_rate')!r} (type: {type(t_cfg.get('learning_rate')).__name__})")
    print(f"Raw gradient_accum:    {t_cfg.get('gradient_accumulation_steps')!r} (type: {type(t_cfg.get('gradient_accumulation_steps')).__name__})")
    print(f"Raw warmup_ratio:      {t_cfg.get('warmup_ratio')!r} (type: {type(t_cfg.get('warmup_ratio')).__name__})")
    print(f"Raw weight_decay:      {t_cfg.get('weight_decay')!r} (type: {type(t_cfg.get('weight_decay')).__name__})")

    print("\n" + "=" * 70)
    print("STEP 2: Validate and Resolve All Parameters")
    print("=" * 70)
    resolved = validate_training_parameters(t_cfg, q_cfg, bm_cfg)

    # Print every parameter value and exact Python type
    print(f"{'PARAMETER':30s} {'RESOLVED VALUE':20s} {'PYTHON TYPE':15s}")
    print("-" * 70)
    for k, v in resolved.items():
        print(f"{k:30s} {str(v):20s} {type(v).__name__:15s}")

    # Explicit assertions for required types
    assert isinstance(resolved["learning_rate"], float), f"learning_rate must be float, got {type(resolved['learning_rate'])}"
    assert isinstance(resolved["weight_decay"], float), f"weight_decay must be float, got {type(resolved['weight_decay'])}"
    assert isinstance(resolved["max_grad_norm"], float), f"max_grad_norm must be float, got {type(resolved['max_grad_norm'])}"
    assert isinstance(resolved["warmup_ratio"], float), f"warmup_ratio must be float, got {type(resolved['warmup_ratio'])}"
    assert isinstance(resolved["warmup_steps"], int), f"warmup_steps must be int, got {type(resolved['warmup_steps'])}"
    assert isinstance(resolved["num_train_epochs"], int), f"num_train_epochs must be int, got {type(resolved['num_train_epochs'])}"
    assert isinstance(resolved["per_device_train_batch_size"], int), f"per_device_train_batch_size must be int, got {type(resolved['per_device_train_batch_size'])}"
    assert isinstance(resolved["gradient_accumulation_steps"], int), f"gradient_accumulation_steps must be int, got {type(resolved['gradient_accumulation_steps'])}"
    assert isinstance(resolved["logging_steps"], int), f"logging_steps must be int, got {type(resolved['logging_steps'])}"
    assert isinstance(resolved["save_steps"], int), f"save_steps must be int, got {type(resolved['save_steps'])}"
    assert isinstance(resolved["eval_steps"], int), f"eval_steps must be int, got {type(resolved['eval_steps'])}"
    assert isinstance(resolved["save_total_limit"], int), f"save_total_limit must be int, got {type(resolved['save_total_limit'])}"
    assert isinstance(resolved["max_seq_length"], int), f"max_seq_length must be int, got {type(resolved['max_seq_length'])}"
    print("\n[PASS] All type assertions verified successfully.")

    print("\n" + "=" * 70)
    print("STEP 3: Construct TrainingArguments")
    print("=" * 70)
    trainer_instance = OloricTrainer()
    training_args = trainer_instance.get_training_arguments()
    print("TrainingArguments constructed successfully:")
    print(f"  training_args.learning_rate: {training_args.learning_rate} (type: {type(training_args.learning_rate).__name__})")
    print(f"  training_args.warmup_steps:  {training_args.warmup_steps} (type: {type(training_args.warmup_steps).__name__})")
    print(f"  training_args.gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")

    print("\n" + "=" * 70)
    print("STEP 4: Optimizer Initialization Test (Reproduces and Verifies '<=' Fix)")
    print("=" * 70)
    dummy_model = torch.nn.Linear(16, 4)
    trainer = Trainer(
        model=dummy_model,
        args=training_args,
    )
    # create_optimizer verifies that 0.0 <= lr passes without TypeError: '<=' not supported
    trainer.create_optimizer()
    print("Optimizer created successfully:")
    print(f"  Optimizer class: {trainer.optimizer.__class__.__name__}")
    for group in trainer.optimizer.param_groups:
        print(f"  param_group lr: {group['lr']} (type: {type(group['lr']).__name__})")
        assert isinstance(group['lr'], float), f"Optimizer lr must be float, got {type(group['lr'])}"
    print("[PASS] Optimizer initialization succeeded with 0.0 <= lr check passing!")

    print("\n" + "=" * 70)
    print("STEP 5: Validation Error Handling Test")
    print("=" * 70)
    # Test that invalid strings fail with clear messages
    invalid_cases = [
        ("learning_rate", "not_a_number", "Invalid float value"),
        ("num_train_epochs", "three", "Invalid integer value"),
        ("gradient_accumulation_steps", 0, "must be >= 1"),
        ("warmup_ratio", 1.5, "must be <= 1.0"),
    ]
    for field, val, err_substr in invalid_cases:
        try:
            bad_cfg = dict(t_cfg)
            bad_cfg[field] = val
            validate_training_parameters(bad_cfg, q_cfg, bm_cfg)
            raise AssertionError(f"Expected failure for {field}={val!r}, but passed!")
        except (ValueError, TypeError) as e:
            assert err_substr in str(e), f"Expected '{err_substr}' in error message, got: {e}"
            print(f"  Correctly caught invalid input [{field}={val!r}]: {e}")
    print("[PASS] Error handling correctly catches invalid parameters with clear messages.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: Training configuration is fully verified and type-safe.")
    print("=" * 70)


if __name__ == "__main__":
    test_training_configuration()
