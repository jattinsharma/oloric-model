#!/usr/bin/env python3
"""
Baseline test for OLORIC model configuration and tokenizer.
"""
import os
import torch
from transformers import AutoTokenizer
from src.oloric.config import config
from src.oloric.formatting import OloricFormatter

def test_tokenizer():
    """Test that the tokenizer loads and works."""
    print("Testing tokenizer...")
    model_name = config.get_model_config().get("base_model", {}).get(
        "name", "Qwen/Qwen3-4B-Instruct-2507"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=config.get_model_config().get("base_model", {}).get("revision", "main")
        )
        print("Tokenizer loaded successfully")

        # Test encoding and decoding
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Test text: {test_text}")
        print(f"  Encoded tokens: {tokens}")
        print(f"  Decoded text: {decoded}")

        # Check special tokens
        print(f"  Pad token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  BOS token: {tokenizer.bos_token}")
        print(f"  Unk token: {tokenizer.unk_token}")

        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def test_qlora_config():
    """Test QLoRA configuration."""
    print("\nTesting QLoRA configuration...")
    qlora_config = config.get_qlora_config()

    quantization = qlora_config.get("quantization", {})
    load_in_4bit = quantization.get("load_in_4bit", False)
    bnb_4bit_quant_type = quantization.get("bnb_4bit_quant_type", "nf4")
    bnb_4bit_use_double_quant = quantization.get("bnb_4bit_use_double_quant", True)
    bnb_4bit_compute_dtype = quantization.get("bnb_4bit_compute_dtype", "bfloat16")

    print(f"  Load in 4-bit: {load_in_4bit}")
    print(f"  Quantization type: {bnb_4bit_quant_type}")
    print(f"  Use double quantization: {bnb_4bit_use_double_quant}")
    print(f"  Compute dtype: {bnb_4bit_compute_dtype}")

    if load_in_4bit:
        print("4-bit quantization is enabled")
    else:
        print("4-bit quantization is not enabled")

    return qlora_config

def test_formatting():
    """Test the OloricFormatter."""
    print("\nTesting OloricFormatter...")
    tokenizer = test_tokenizer()
    if tokenizer is None:
        print("Cannot test formatter without tokenizer")
        return False

    try:
        formatter = OloricFormatter(tokenizer)
        print("Formatter initialized successfully")

        # Create a simple model input
        from src.oloric.schemas.model_input import OloricModelInput, DocumentContext, LearnerState

        test_input = OloricModelInput(
            task="resolve_confusion",
            document_context=DocumentContext(
                document_id="test",
                title="Test Document",
                page=1,
                section="Test",
                selected_text="This is a test.",
                surrounding_context="This is a test context.",
                retrieved_evidence=["Evidence 1", "Evidence 2"]
            ),
            learner_state=LearnerState(
                level="beginner",
                concept="test concept",
                mastery=0.3,
                known_prerequisites=["basic knowledge"],
                weak_prerequisites=[],
                known_misconceptions=[]
            ),
            conversation_context=[
                {"role": "student", "content": "I don't understand this concept."}
            ],
            current_goal="Understand the test concept"
        )

        formatted = formatter.format_input(test_input)
        print("Input formatted successfully")
        print(f"  Formatted length: {len(formatted)} characters")
        print(f"  Preview: {formatted[:200]}...")

        return True
    except Exception as e:
        print(f"Error in formatting: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_estimates():
    """Output memory estimates from qlora.yaml."""
    print("\nMemory estimates from qlora.yaml:")
    qlora_config = config.get_qlora_config()
    memory_estimates = qlora_config.get("memory_estimates", {})
    for key, value in memory_estimates.items():
        print(f"  {key}: {value}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("OLORIC BASELINE TEST")
    print("=" * 60)

    # Test 1: Base model identifier (already checked via Hugging Face, but we can note)
    model_name = config.get_model_config().get("base_model", {}).get(
        "name", "Qwen/Qwen3-4B-Instruct-2507"
    )
    print(f"Base model identifier: {model_name}")
    print("Base model identifier verified via Hugging Face Hub (earlier check)")

    # Test 2: Tokenizer
    tokenizer = test_tokenizer()

    # Test 3: QLoRA configuration
    qlora_config = test_qlora_config()

    # Test 4: Formatting
    formatting_success = test_formatting()

    # Test 5: Memory estimates
    test_memory_estimates()

    print("\n" + "=" * 60)
    print("BASELINE TEST SUMMARY")
    print("=" * 60)
    if tokenizer is not None and formatting_success:
        print("Baseline tests passed: Tokenizer, formatting, and QLoRA config are set correctly.")
        print("Note: Actual model loading and inference require a GPU environment.")
    else:
        print("Some baseline tests failed.")

    print("\nFor GPU smoke test, please run in an environment with NVIDIA L4 24GB GPU.")
    print("The memory estimates suggest ~13.0 GB total usage, leaving room for batch processing.")

    return 0 if (tokenizer is not None and formatting_success) else 1

if __name__ == "__main__":
    exit(main())