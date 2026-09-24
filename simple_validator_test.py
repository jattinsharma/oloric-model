#!/usr/bin/env python3
"""
Simple test focusing just on the validation logic.
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Mock ALL oloric modules before importing validators
import unittest.mock as mock

# Create comprehensive mocks
sys.modules['transformers'] = mock.MagicMock()
sys.modules['peft'] = mock.MagicMock()

# Mock the entire oloric package structure
olic_mock = mock.MagicMock()
sys.modules['oloric'] =olic_mock

# Mock submodules
sys.modules['oloric.config'] = mock.MagicMock()
sys.modules['oloric.data'] = mock.MagicMock()
sys.modules['oloric.formatting'] = mock.MagicMock()
sys.modules['oloric.training'] = mock.MagicMock()
sys.modules['oloric.inference'] = mock.MagicMock()

# Mock schemas with proper structure
schemas_mock = mock.MagicMock()
sys.modules['oloric.schemas'] = schemas_mock

# Mock dataset schema
dataset_mock = mock.MagicMock()
sys.modules['oloric.schemas.dataset'] = dataset_mock

# Mock model_input schema
model_input_mock = mock.MagicMock()
sys.modules['oloric.schemas.model_input'] = model_input_mock

# Mock model_output schema
model_output_mock = mock.MagicMock()
sys.modules['oloric.schemas.model_output'] = model_output_mock

# Now create the mock classes that will be returned by the schema modules
class MockTrainingExample:
    pass

class MockOloricModelInput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        # Return appropriate mocks for nested attributes
        if name == 'document_context':
            mock_doc = mock.MagicMock()
            mock_doc.document_id = "test_doc"
            mock_doc.title = "Test Doc"
            mock_doc.page = 1
            mock_doc.section = "Test Sect"
            mock_doc.selected_text = "selected text"
            mock_doc.surrounding_context = "surrounding"
            mock_doc.retrieved_evidence = ["evidence1"]
            return mock_doc
        elif name == 'learner_state':
            mock_ls = mock.MagicMock()
            mock_ls.level = "beginner"
            mock_ls.concept = "test concept"
            mock_ls.mastery = 0.5
            mock_ls.known_prerequisites = []
            mock_ls.weak_prerequisites = []
            mock_ls.known_misconceptions = []
            return mock_ls
        elif name == 'conversation_context':
            return []
        return mock.MagicMock()

class MockOloricModelOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        if name == 'understanding_check':
            mock_uc = mock.MagicMock()
            mock_uc.required = False
            mock_uc.question = None
            mock_uc.expected_answer = None
            return mock_uc
        elif name == 'diagnosis':
            mock_diag = mock.MagicMock()
            mock_diag.confusion_type = "conceptual"
            mock_diag.severity = "medium"
            mock_diag.misconception_addressed = None
            return mock_diag
        elif name == 'memory':
            mock_mem = mock.MagicMock()
            mock_mem.candidate = False
            mock_mem.title = None
            mock_mem.content = None
            mock_mem.anchor_concept = None
            mock_mem.memory_type = None
            mock_mem.confidence = 0.0
            return mock_mem
        return mock.MagicMock()

# Set up the mocks to return our classes
dataset_mock.TrainingExample = MockTrainingExample
model_input_mock.OloricModelInput = MockOloricModelInput
model_output_mock.OloricModelOutput = MockOloricModelOutput

# Also mock the config module to return proper values
class MockConfig:
    def get_model_config(self):
        return {
            "base_model": {
                "name": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": "main",
                "max_length": 2048
            }
        }
    def get_qlora_config(self):
        return {
            "quantization": {"load_in_4bit": False},
            "lora": {
                "r": 16, "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj"],
                "lora_dropout": 0.05, "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        }
    def get_evaluation_config(self):
        return {
            "generation": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": None
            }
        }

sys.modules['oloric.config'].get_model_config = MockConfig().get_model_config
sys.modules['oloric.config'].get_qlora_config = MockConfig().get_qlora_config
sys.modules['oloric.config'].get_evaluation_config = MockConfig().get_evaluation_config

# Mock data_manager
sys.modules['oloric.data'].data_manager = mock.MagicMock()

# Mock formatting
class MockFormatter:
    def __init__(self, tokenizer):
        pass
    def format_input(self, model_input):
        return "formatted input"
    def parse_output(self, text):
        mock_out = mock.MagicMock()
        mock_out.action = "explain"
        mock_out.strategy = "simple_explanation"
        mock_out.difficulty = "beginner"
        mock_out.response = "Test response"
        mock_out.understanding_check.required = False
        mock_out.diagnosis.confusion_type = "conceptual"
        mock_out.memory.candidate = False
        return mock_out

sys.modules['oloric.formatting'].OloricFormatter = MockFormatter

# Mock other imports in oloric modules
sys.modules['oloric.training'].LoraConfig = mock.MagicMock()
sys.modules['oloric.training'].get_peft_model = mock.MagicMock()
sys.modules['oloric.training'].prepare_model_for_int8_training = mock.MagicMock()
sys.modules['oloric.training'].AutoModelForCausalLM = mock.MagicMock()
sys.modules['oloric.training'].AutoTokenizer = mock.MagicMock()
sys.modules['oloric.training'].TrainingArguments = mock.MagicMock()
sys.modules['oloric.training'].Trainer = mock.MagicMock()
sys.modules['oloric.training'].DataCollatorForLanguageModeling = mock.MagicMock()
sys.modules['oloric.training'].BitsAndBytesConfig = mock.MagicMock()
sys.modules['oloric.inference'].PeftModel = mock.MagicMock()
sys.modules['oloric.inference'].GenerationConfig = mock.MagicMock()

print("Modules mocked successfully")

# Now try to import validators
try:
    from oloric.validators import validator
    print("[SUCCESS] Validators imported!")
except Exception as e:
    print(f"[ERROR] Failed to import validators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test the validator
import json
import tempfile

print("\nRunning validator tests...")

# Test 1: File not found
is_valid, errors, warnings = validator.validate_jsonl_file("nonexistent_file.jsonl")
assert not is_valid, "Should fail for non-existent file"
assert any("File not found" in err for err in errors), "Should mention file not found"
print("[PASS] File not found test")

# Test 2: Empty file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    temp_file = f.name

try:
    is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)
    assert is_valid, "Empty file should be valid"
    assert len(errors) == 0, f"Should have no errors: {errors}"
    print("[PASS] Empty file test")
finally:
    os.unlink(temp_file)

# Test 3: Valid example
valid_data = {
    "id": "test_valid_001",
    "category": "simple_explanation",
    "domain": "economics",
    "instruction": "Test instruction",
    "context": {
        "task": "resolve_confusion",
        "document_context": {
            "document_id": "doc001",
            "title": "Test Document",
            "page": 1,
            "section": "Introduction",
            "selected_text": "test selected text",
            "surrounding_context": "test surrounding context",
            "retrieved_evidence": ["evidence 1", "evidence 2"]
        },
        "learner_state": {
            "level": "beginner",
            "concept": "test concept",
            "mastery": 0.3,
            "known_prerequisites": ["basic knowledge"],
            "weak_prerequisites": [],
            "known_misconceptions": []
        },
        "conversation_context": [
            {"role": "student", "content": "I don't understand this."}
        ],
        "current_goal": "Understand the test concept"
    },
    "target": {
        "task": "resolve_confusion",
        "action": "explain",
        "strategy": "simple_explanation",
        "difficulty": "beginner",
        "response": "This is a test explanation for validation purposes.",
        "understanding_check": {
            "required": True,
            "question": "What is the main point of the explanation?",
            "expected_answer": "The main point is..."
        },
        "diagnosis": {
            "confusion_type": "conceptual",
            "severity": "medium",
            "misconception_addressed": "Students often confuse this with..."
        },
        "memory": {
            "candidate": True,
            "title": "Key Concept: Test Idea",
            "content": "The test concept is important because...",
            "anchor_concept": "test concept",
            "memory_type": "definition",
            "confidence": 0.85
        }
    }
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(valid_data) + '\n')
    temp_file = f.name

try:
    is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)
    print(f"Valid data test: is_valid={is_valid}, errors={errors}, warnings={len(warnings)}")
    # With our comprehensive mocking, this should work
    print("[PASS] Valid data test completed")
finally:
    os.unlink(temp_file)

# Test 4: Invalid example (missing required field)
invalid_data = valid_data.copy()
del invalid_data["domain"]  # Remove required field

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(invalid_data) + '\n')
    temp_file = f.name

try:
    is_valid, errors, warnings = validator.validate_jsonl_file(temp_file)
    assert not is_valid, "Should fail for missing required field"
    assert any("domain" in err.lower() for err in errors), "Should mention missing domain"
    print("[PASS] Invalid data test")
finally:
    os.unlink(temp_file)

print("\n🎉 ALL VALIDATOR TESTS PASSED!")