"""
Tests for OLORIC validators.
"""
import json
import tempfile
import os
import sys
from unittest.mock import MagicMock

# Mock the problematic modules before importing oloric
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['oloric.formatting'] = MagicMock()

# Add the src directory to the path so we can import oloric modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oloric.validators import validator
from oloric.schemas.model_input import OloricModelInput
from oloric.schemas.model_output import OloricModelOutput


def test_valid_jsonl():
    """Test validation of valid JSONL file."""
    # Create a valid example
    valid_example = {
        "id": "test_001",
        "category": "simple_explanation",
        "domain": "economics",
        "instruction": "Explain the concept of supply and demand.",
        "context": {
            "task": "resolve_confusion",
            "document_context": {
                "document_id": "econ101_chapter2",
                "title": "Principles of Economics",
                "page": 25,
                "section": "Supply and Demand",
                "selected_text": "The law of demand states that...",
                "surrounding_context": "In economics, the law of demand is a fundamental principle...",
                "retrieved_evidence": [
                    "Demand curve shows relationship between price and quantity demanded",
                    "As price increases, quantity demanded decreases"
                ]
            },
            "learner_state": {
                "level": "beginner",
                "concept": "law of demand",
                "mastery": 0.3,
                "known_prerequisites": ["basic algebra"],
                "weak_prerequisites": ["graph interpretation"],
                "known_misconceptions": []
            },
            "conversation_context": [
                {
                    "role": "student",
                    "content": "I don't understand why demand decreases when price increases."
                }
            ],
            "current_goal": "Understand the law of demand well enough to explain it to others"
        },
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "The law of demand states that when the price of a good increases, the quantity demanded decreases, assuming all other factors remain constant.",
            "understanding_check": {
                "required": True,
                "question": "What happens to quantity demanded when price increases according to the law of demand?",
                "expected_answer": "Quantity demanded decreases"
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that higher prices always mean higher demand"
            },
            "memory": {
                "candidate": True,
                "title": "Law of Demand Definition",
                "content": "The law of demand states that when price increases, quantity demanded decreases (ceteris paribus).",
                "anchor_concept": "law of demand",
                "memory_type": "definition",
                "confidence": 0.8
            }
        }
    }

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(valid_example) + '\n')
        temp_file = f.name

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)

        # Check results
        assert is_valid, f"Valid file should pass validation. Errors: {errors}"
        assert len(errors) == 0, f"Valid file should have no errors. Got: {errors}"
        print("[PASS] Valid JSONL file test passed")

    finally:
        # Clean up
        os.unlink(temp_file)


def test_invalid_jsonl():
    """Test validation of invalid JSONL file."""
    # Create an invalid example (missing required field)
    invalid_example = {
        "id": "test_002",
        "category": "simple_explanation",
        # Missing domain
        "instruction": "Explain the concept of supply and demand.",
        "context": {
            "task": "resolve_confusion",
            "document_context": {
                "document_id": "econ101_chapter2",
                "title": "Principles of Economics",
                "page": 25,
                "section": "Supply and Demand",
                "selected_text": "The law of demand states that...",
                "surrounding_context": "In economics, the law of demand is a fundamental principle...",
                "retrieved_evidence": [
                    "Demand curve shows relationship between price and quantity demanded",
                    "As price increases, quantity demanded decreases"
                ]
            },
            "learner_state": {
                "level": "beginner",
                "concept": "law of demand",
                "mastery": 0.3,
                "known_prerequisites": ["basic algebra"],
                "weak_prerequisites": ["graph interpretation"],
                "known_misconceptions": []
            },
            "conversation_context": [
                {
                    "role": "student",
                    "content": "I don't understand why demand decreases when price increases."
                }
            ],
            "current_goal": "Understand the law of demand well enough to explain it to others"
        },
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "The law of demand states that when the price of a good increases, the quantity demanded decreases, assuming all other factors remain constant.",
            "understanding_check": {
                "required": True,
                "question": "What happens to quantity demanded when price increases according to the law of demand?",
                "expected_answer": "Quantity demanded decreases"
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that higher prices always mean higher demand"
            },
            "memory": {
                "candidate": True,
                "title": "Law of Demand Definition",
                "content": "The law of demand states that when price increases, quantity demanded decreases (ceteris paribus).",
                "anchor_concept": "law of demand",
                "memory_type": "definition",
                "confidence": 0.8
            }
        }
    }

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(invalid_example) + '\n')
        temp_file = f.name

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)

        # Check results
        assert not is_valid, "Invalid file should fail validation"
        assert len(errors) > 0, "Invalid file should have errors"
        assert any("domain" in error.lower() for error in errors), "Should have error about missing domain"
        print("[PASS] Invalid JSONL file test passed")

    finally:
        # Clean up
        os.unlink(temp_file)


def test_duplicate_ids():
    """Test validation catches duplicate IDs."""
    # Create two examples with same ID
    example1 = {
        "id": "duplicate_id",
        "category": "simple_explanation",
        "domain": "economics",
        "instruction": "Explain the concept of supply and demand.",
        "context": {
            "task": "resolve_confusion",
            "document_context": {
                "document_id": "econ101_chapter2",
                "title": "Principles of Economics",
                "page": 25,
                "section": "Supply and Demand",
                "selected_text": "The law of demand states that...",
                "surrounding_context": "In economics, the law of demand is a fundamental principle...",
                "retrieved_evidence": [
                    "Demand curve shows relationship between price and quantity demanded",
                    "As price increases, quantity demanded decreases"
                ]
            },
            "learner_state": {
                "level": "beginner",
                "concept": "law of demand",
                "mastery": 0.3,
                "known_prerequisites": ["basic algebra"],
                "weak_prerequisites": ["graph interpretation"],
                "known_misconceptions": []
            },
            "conversation_context": [
                {
                    "role": "student",
                    "content": "I don't understand why demand decreases when price increases."
                }
            ],
            "current_goal": "Understand the law of demand well enough to explain it to others"
        },
        "target": {
            "task": "resolve_confusion",
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "The law of demand states that when the price of a good increases, the quantity demanded decreases, assuming all other factors remain constant.",
            "understanding_check": {
                "required": True,
                "question": "What happens to quantity demanded when price increases according to the law of demand?",
                "expected_answer": "Quantity demanded decreases"
            },
            "diagnosis": {
                "confusion_type": "conceptual",
                "severity": "medium",
                "misconception_addressed": "Belief that higher prices always mean higher demand"
            },
            "memory": {
                "candidate": True,
                "title": "Law of Demand Definition",
                "content": "The law of demand states that when price increases, quantity demanded decreases (ceteris paribus).",
                "anchor_concept": "law of demand",
                "memory_type": "definition",
                "confidence": 0.8
            }
        }
    }

    example2 = example1.copy()
    example2["instruction"] = "Explain elasticity of demand."

    # Create temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps(example1) + '\n')
        f.write(json.dumps(example2) + '\n')
        temp_file = f.name

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)

        # Check results
        assert not is_valid, "File with duplicate IDs should fail validation"
        assert len(errors) > 0, "File with duplicate IDs should have errors"
        assert any("duplicate" in error.lower() for error in errors), "Should have error about duplicate ID"
        print("[PASS] Duplicate IDs test passed")

    finally:
        # Clean up
        os.unlink(temp_file)


if __name__ == "__main__":
    test_valid_jsonl()
    test_invalid_jsonl()
    test_duplicate_ids()
    print("\nAll tests passed! [PASS]")