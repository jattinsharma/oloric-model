#!/usr/bin/env python3
"""
Test the actual validators.py file by handling imports properly.
"""
import json
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the modules that might cause issues
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['oloric.formatting'] = MagicMock()
sys.modules['oloric.config'] = MagicMock()
sys.modules['oloric.data'] = MagicMock()
sys.modules['oloric.training'] = MagicMock()
sys.modules['oloric.inference'] = MagicMock()

# Now import the validators module
import importlib.util

def load_validators_module():
    """Load the validators module with mocked dependencies."""
    # Mock the schema modules
    schema_mock = MagicMock()
    dataset_mock = MagicMock()
    model_input_mock = MagicMock()
    model_output_mock = MagicMock()

    # Create mock classes for the schemas
    mock_training_example = MagicMock()
    mock_model_input_class = MagicMock()
    mock_model_output_class = MagicMock()

    # Set up the mocks to return instances when called
    mock_model_input_class.return_value = mock_model_input_class
    mock_model_output_class.return_value = mock_model_output_class

    # Configure the mocks
    dataset_mock.TrainingExample = mock_training_example
    model_input_mock.OloricModelInput = mock_model_input_class
    model_output_mock.OloricModelOutput = mock_model_output_class

    # Patch the imports
    with patch.dict('sys.modules', {
        'oloric.schemas.dataset': dataset_mock,
        'oloric.schemas.model_input': model_input_mock,
        'oloric.schemas.model_output': model_output_mock
    }):
        # Load and execute the validators module
        spec = importlib.util.spec_from_file_location(
            "validators",
            os.path.join(os.path.dirname(__file__), 'src', 'oloric', 'validators.py')
        )
        validators_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validators_module)
        return validators_module

def test_validator_functionality():
    """Test the validator functionality with mocked schemas."""
    validators_module = load_validators_module()

    # Create a simple test
    is_valid, errors, warnings = validators_module.validator.validate_jsonl_file("nonexistent.jsonl")
    assert not is_valid
    assert "File not found" in errors[0]
    print("[PASS] File not found test passed")

    # Test with empty file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_file = f.name

    try:
        is_valid, errors, warnings = validators_module.validator.validate_jsonl_file(temp_file)
        assert is_valid  # Empty file should be valid
        assert len(errors) == 0
        print("[PASS] Empty file test passed")
    finally:
        os.unlink(temp_file)

    print("All validator tests passed!")

if __name__ == "__main__":
    test_validator_functionality()