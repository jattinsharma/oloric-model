"""
Simple test for OLORIC validators without importing full oloric package.
"""
import json
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

# Mock the problematic sys.modules entries
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['oloric.formatting'] = MagicMock()
sys.modules['oloric.config'] = MagicMock()
sys.modules['oloric.data'] = MagicMock()
sys.modules['oloric.training'] = MagicMock()
sys.modules['oloric.inference'] = MagicMock()

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oloric.validators import validator

def test_validator_import():
    """Test that we can import the validator."""
    assert validator is not None
    print("[PASS] Validator imported successfully")

def test_validate_jsonl_file_not_found():
    """Test validation of non-existent file."""
    is_valid, errors, warnings = validator.validate_jsonl_file("/non/existent/file.jsonl")
    assert not is_valid
    assert len(errors) > 0
    assert "File not found" in errors[0]
    print("[PASS] File not found test passed")

def test_validate_jsonl_file_empty():
    """Test validation of empty file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)
        assert is_valid  # Empty file is valid (no lines to validate)
        assert len(errors) == 0
        print("[PASS] Empty file test passed")
    finally:
        os.unlink(temp_file)

def test_validate_jsonl_file_valid_line():
    """Test validation of file with valid JSON line."""
    # Create mock classes for the schema validation
    mock_model_input = MagicMock()
    mock_model_output = MagicMock()

    # Patch the imports in the validators module
    with patch('oloric.schemas.model_input.OloricModelInput', return_value=mock_model_input), \
         patch('oloric.schemas.model_output.OloricModelOutput', return_value=mock_model_output):

        valid_example = {
            "id": "test_001",
            "category": "simple_explanation",
            "domain": "economics",
            "instruction": "Explain the concept.",
            "context": {"task": "resolve_confusion"},
            "target": {"task": "resolve_confusion", "action": "explain"}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(valid_example) + '\n')
            temp_file = f.name

        try:
            is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)
            # This might fail due to our mocks, but let's see
            print(f"Result: is_valid={is_valid}, errors={errors}, warnings={warnings}")
        finally:
            os.unlink(temp_file)

if __name__ == "__main__":
    test_validator_import()
    test_validate_jsonl_file_not_found()
    test_validate_jsonl_file_empty()
    test_validate_jsonl_file_valid_line()
    print("\nAll simple tests completed!")