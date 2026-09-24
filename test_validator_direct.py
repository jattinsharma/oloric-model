#!/usr/bin/env python3
"""
Direct test of validators.py functionality by copying the essential parts.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

# Mock the schema classes for testing
class MockOloricModelInput:
    def __init__(self, **kwargs):
        # Just store the kwargs for testing
        self.data = kwargs

    @classmethod
    def __get_validators__(cls):
        # Return empty list to bypass validation in test
        return []

    class Config:
        extra = "forbid"

class MockOloricModelOutput:
    def __init__(self, **kwargs):
        # Just store the kwargs for testing
        self.data = kwargs

    @classmethod
    def __get_validators__(cls):
        # Return empty list to bypass validation in test
        return []

    class Config:
        extra = "forbid"

class DatasetValidator:
    """Validates OLORIC training dataset."""

    def __init__(self):
        """Initialize validator."""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_jsonl_file(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            self.validation_errors.append(f"File not found: {file_path}")
            return False, self.validation_errors, self.validation_warnings
        except Exception as e:
            self.validation_errors.append(f"Error reading file: {e}")
            return False, self.validation_errors, self.validation_warnings

        # Validate each line
        seen_ids = set()
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                is_valid, errors, warnings = self.validate_example(data, i)
                if not is_valid:
                    self.validation_errors.extend(errors)
                self.validation_warnings.extend(warnings)

                # Check for duplicate IDs
                example_id = data.get("id")
                if example_id:
                    if example_id in seen_ids:
                        self.validation_errors.append(f"Line {i}: Duplicate ID '{example_id}'")
                    else:
                        seen_ids.add(example_id)

            except json.JSONDecodeError as e:
                self.validation_errors.append(f"Line {i}: Invalid JSON - {e}")
            except Exception as e:
                self.validation_errors.append(f"Line {i}: Unexpected error - {e}")

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def validate_example(self, data: Dict[str, Any], line_num: int) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single training example.

        Args:
            data: Example data dictionary
            line_num: Line number for error reporting

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Validate required fields
        required_fields = ["id", "category", "domain", "instruction", "context", "target"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Line {line_num}: Missing required field '{field}'")

        if errors:
            return False, errors, warnings

        # Validate ID format
        example_id = data["id"]
        if not isinstance(example_id, str) or not example_id:
            errors.append(f"Line {line_num}: ID must be a non-empty string")

        # Validate category
        valid_categories = [
            "simple_explanation", "simplification", "analogy", "concrete_example",
            "numerical_example", "prerequisite_detection", "misconception_detection",
            "follow_up_questions", "multi_turn_tutoring", "repeated_confusion",
            "strategy_switching", "diagnostic_questions", "hint_based_teaching",
            "practice_questions", "error_correction", "partial_understanding",
            "understanding_confirmation", "memory_generation", "document_grounded",
            "context_retention"
        ]
        if data["category"] not in valid_categories:
            errors.append(f"Line {line_num}: Invalid category '{data['category']}'")

        # Validate domain
        valid_domains = [
            "economics", "accountancy", "mathematics", "science",
            "nutrition_food_science", "general_academic"
        ]
        if data["domain"] not in valid_domains:
            errors.append(f"Line {line_num}: Invalid domain '{data['domain']}'")

        # Validate context
        try:
            context = MockOloricModelInput(**data["context"])
        except Exception as e:
            errors.append(f"Line {line_num}: Invalid context - {e}")

        # Validate target
        try:
            target = MockOloricModelOutput(**data["target"])
        except Exception as e:
            errors.append(f"Line {line_num}: Invalid target - {e}")

        # Additional validations
        if "conversation" in data:
            conv_errors, conv_warnings = self._validate_conversation(data["conversation"], line_num)
            errors.extend(conv_errors)
            warnings.extend(conv_warnings)

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_conversation(self, conversation: List[Dict[str, Any]], line_num: int) -> Tuple[List[str], List[str]]:
        """
        Validate conversation history.

        Args:
            conversation: Conversation list
            line_num: Line number for error reporting

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        if not isinstance(conversation, list):
            errors.append(f"Line {line_num}: Conversation must be a list")
            return errors, warnings

        valid_roles = {"student", "tutor"}
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                errors.append(f"Line {line_num}: Conversation turn {i} must be a dictionary")
                continue

            if "role" not in turn or "content" not in turn:
                errors.append(f"Line {line_num}: Conversation turn {i} missing role or content")
                continue

            if turn["role"] not in valid_roles:
                errors.append(f"Line {line_num}: Conversation turn {i} has invalid role '{turn['role']}'")

            if not isinstance(turn["content"], str) or not turn["content"].strip():
                warnings.append(f"Line {line_num}: Conversation turn {i} has empty content")

        return errors, warnings

    def validate_schema_compatibility(self, context: MockOloricModelInput,
                                    target: MockOloricModelOutput) -> Tuple[bool, List[str]]:
        """
        Validate that context and target are compatible.

        Args:
            context: Input context
            target: Expected output

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        # Check that the task makes sense
        if getattr(context, 'data', {}).get("task") != "resolve_confusion":
            # Allow other tasks but warn
            pass

        # Check that action is valid
        valid_actions = {
            "diagnose", "explain", "simplify", "analogy", "example",
            "prerequisite", "misconception_correction", "hint", "practice",
            "recap", "confirm_understanding", "memory_candidate"
        }
        target_action = getattr(target, 'data', {}).get("action", "")
        if target_action not in valid_actions:
            errors.append(f"Invalid action '{target_action}'. Must be one of {valid_actions}")

        # Check difficulty levels
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        target_difficulty = getattr(target, 'data', {}).get("difficulty", "")
        if target_difficulty not in valid_difficulties:
            errors.append(f"Invalid difficulty '{target_difficulty}'. Must be one of {valid_difficulties}")

        # Check that if memory is candidate, required fields are present
        memory_data = getattr(target, 'data', {}).get("memory", {})
        if memory_data.get("candidate"):
            if not memory_data.get("title"):
                errors.append("Memory candidate must have title when candidate is True")
            if not memory_data.get("content"):
                errors.append("Memory candidate must have content when candidate is True")
            if not memory_data.get("anchor_concept"):
                errors.append("Memory candidate must have anchor_concept when candidate is True")

        # Check understanding check consistency
        understanding_check = getattr(target, 'data', {}).get("understanding_check", {})
        if understanding_check.get("required") and not understanding_check.get("question"):
            errors.append("Understanding check is required but no question provided")

        is_valid = len(errors) == 0
        return is_valid, errors


# Global validator instance
validator = DatasetValidator()


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
                "weak_prerequisites": ["understanding of slopes"],
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
    with open("test_valid.jsonl", "w") as f:
        f.write(json.dumps(valid_example) + '\n')

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file("test_valid.jsonl")

        # Check results
        print(f"Valid file test: is_valid={is_valid}, errors={errors}, warnings={warnings}")
        assert is_valid, f"Valid file should pass validation. Errors: {errors}"
        assert len(errors) == 0, f"Valid file should have no errors. Got: {errors}"
        print("[PASS] Valid JSONL file test passed")

    finally:
        # Clean up
        import os
        if os.path.exists("test_valid.jsonl"):
            os.unlink("test_valid.jsonl")


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
    with open("test_invalid.jsonl", "w") as f:
        f.write(json.dumps(invalid_example) + '\n')

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file("test_invalid.jsonl")

        # Check results
        print(f"Invalid file test: is_valid={is_valid}, errors={errors}, warnings={warnings}")
        assert not is_valid, "Invalid file should fail validation"
        assert len(errors) > 0, "Invalid file should have errors"
        assert any("domain" in error.lower() for error in errors), "Should have error about missing domain"
        print("[PASS] Invalid JSONL file test passed")

    finally:
        # Clean up
        import os
        if os.path.exists("test_invalid.jsonl"):
            os.unlink("test_invalid.jsonl")


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
    with open("test_duplicate.jsonl", "w") as f:
        f.write(json.dumps(example1) + '\n')
        f.write(json.dumps(example2) + '\n')

    try:
        # Validate the file
        is_valid, errors, warnings = validator.validate_jsonl_file("test_duplicate.jsonl")

        # Check results
        print(f"Duplicate IDs test: is_valid={is_valid}, errors={errors}, warnings={warnings}")
        assert not is_valid, "File with duplicate IDs should fail validation"
        assert len(errors) > 0, "File with duplicate IDs should have errors"
        assert any("duplicate" in error.lower() for error in errors), "Should have error about duplicate ID"
        print("[PASS] Duplicate IDs test passed")

    finally:
        # Clean up
        import os
        if os.path.exists("test_duplicate.jsonl"):
            os.unlink("test_duplicate.jsonl")


if __name__ == "__main__":
    test_valid_jsonl()
    test_invalid_jsonl()
    test_duplicate_ids()
    print("\nAll tests passed! [PASS]")