#!/usr/bin/env python3
"""
Dataset validation script for OLORIC.
Validates training, validation, and test datasets.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Ensure src directory is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from oloric.validators import DatasetValidator

def validate_dataset_file(file_path: str) -> bool:
    """
    Validate a dataset file.

    Args:
        file_path: Path to JSONL dataset file

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False

    validator = DatasetValidator()
    is_valid, errors, warnings = validator.validate_jsonl_file(file_path)

    print(f"Validating: {file_path}")
    print("-" * 50)

    if warnings:
        print("Warnings:")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"  [WARN] {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
        print()

    if errors:
        print("Errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  [FAIL] {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        print()
        print(f"VALIDATION FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    else:
        print(f"VALIDATION PASSED: {len(warnings)} warning(s)")
        return True

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate OLORIC dataset files")
    parser.add_argument(
        "--train",
        type=str,
        default="./data/generated/enhanced_seed_data.jsonl",
        help="Path to training dataset (default: data/generated/enhanced_seed_data.jsonl)"
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="./data/validation.jsonl",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="./data/test.jsonl",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all datasets (train, validation, test)"
    )

    args = parser.parse_args()

    files_to_validate = []
    if args.all:
        files_to_validate = [args.train, args.validation, args.test]
    else:
        # Validate at least one file
        if os.path.exists(args.train):
            files_to_validate.append(args.train)
        if os.path.exists(args.validation):
            files_to_validate.append(args.validation)
        if os.path.exists(args.test):
            files_to_validate.append(args.test)

        if not files_to_validate:
            files_to_validate = [args.train]  # Default to train

    all_valid = True
    for file_path in files_to_validate:
        if not validate_dataset_file(file_path):
            all_valid = False
        print()  # Empty line between files

    if all_valid:
        print("[PASS] All dataset validations passed!")
        return 0
    else:
        print("[FAIL] Some dataset validations failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
