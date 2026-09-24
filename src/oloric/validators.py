"""
Validators for OLORIC dataset and schemas.
"""
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .schemas.dataset import TrainingExample, VALID_CATEGORIES, VALID_DOMAINS
from .schemas.model_input import OloricModelInput
from .schemas.model_output import OloricModelOutput
import logging

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates OLORIC training dataset."""

    @staticmethod
    def get_required_fields() -> List[str]:
        """Get required fields from the TrainingExample schema."""
        if hasattr(TrainingExample, "model_fields"):
            # Pydantic v2
            return [
                name for name, field in TrainingExample.model_fields.items()
                if field.is_required()
            ]
        elif hasattr(TrainingExample, "__fields__"):
            # Pydantic v1
            return [
                name for name, field in TrainingExample.__fields__.items()
                if getattr(field, "required", False)
            ]
        return ["id", "category", "domain", "instruction", "context", "target"]

    def __init__(self):
        """Initialize validator."""
        self.validation_errors = []
        self.validation_warnings = []
        self.required_fields = self.get_required_fields()

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

        # Validate required fields defined by the TrainingExample schema
        required_fields = getattr(self, "required_fields", None) or self.get_required_fields()
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
        if data["category"] not in VALID_CATEGORIES:
            errors.append(f"Line {line_num}: Invalid category '{data['category']}'")

        # Validate domain
        if data["domain"] not in VALID_DOMAINS:
            errors.append(f"Line {line_num}: Invalid domain '{data['domain']}'")

        # Validate context
        try:
            context = OloricModelInput(**data["context"])
        except Exception as e:
            errors.append(f"Line {line_num}: Invalid context - {e}")

        # Validate target
        try:
            target = OloricModelOutput(**data["target"])
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

            role = turn.get("role")
            content = turn.get("content")
            if role is None or content is None:
                errors.append(f"Line {line_num}: Conversation turn {i} missing role or content")
                continue

            if role not in valid_roles:
                errors.append(f"Line {line_num}: Conversation turn {i} has invalid role '{role}'")

            if not isinstance(content, str) or not content.strip():
                warnings.append(f"Line {line_num}: Conversation turn {i} has empty content")

        return errors, warnings

    def validate_schema_compatibility(self, context: OloricModelInput,
                                    target: OloricModelOutput) -> Tuple[bool, List[str]]:
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
        if context.task != "resolve_confusion":
            # Allow other tasks but warn
            pass

        # Check that action is valid
        valid_actions = {
            "diagnose", "explain", "simplify", "analogy", "example",
            "prerequisite", "misconception_correction", "hint", "practice",
            "recap", "confirm_understanding", "memory_candidate"
        }
        if target.action not in valid_actions:
            errors.append(f"Invalid action '{target.action}'. Must be one of {valid_actions}")

        # Check difficulty levels
        valid_difficulties = {"beginner", "intermediate", "advanced"}
        if target.difficulty not in valid_difficulties:
            errors.append(f"Invalid difficulty '{target.difficulty}'. Must be one of {valid_difficulties}")

        # Check that if memory is candidate, required fields are present
        if target.memory.candidate:
            if not target.memory.title:
                errors.append("Memory candidate must have title when candidate is True")
            if not target.memory.content:
                errors.append("Memory candidate must have content when candidate is True")
            if not target.memory.anchor_concept:
                errors.append("Memory candidate must have anchor_concept when candidate is True")

        # Check understanding check consistency
        if target.understanding_check.required and not target.understanding_check.question:
            errors.append("Understanding check is required but no question provided")

        is_valid = len(errors) == 0
        return is_valid, errors


# Global validator instance
validator = DatasetValidator()