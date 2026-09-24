#!/usr/bin/env python3
"""
Test script to verify the OLORIC model loads correctly.
"""
import os
import torch
from src.oloric.inference import OloricInference

def test_model_loading():
    """Test that the model loads correctly."""
    print("Testing OLORIC model loading...")

    try:
        # Initialize inference engine (will load base model)
        inference = OloricInference()

        print("✓ Model loaded successfully")
        print(f"  Model type: {type(inference.model)}")
        print(f"  Tokenizer type: {type(inference.tokenizer)}")

        # Check if model is quantized
        if hasattr(inference.model, 'is_loaded_in_8bit') or hasattr(inference.model, 'is_loaded_in_4bit'):
            print(f"  Model is quantized: 4-bit = {getattr(inference.model, 'is_loaded_in_4bit', False)}")

        # Test simple generation
        from src.oloric.schemas.model_input import OloricModelInput, DocumentContext, LearnerState

        # Create a simple test input
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

        print("✓ Test input created successfully")

        # Try to generate a response (this will take a moment)
        print("Generating test response...")
        response = inference.generate_response(test_input)

        print("✓ Response generated successfully")
        print(f"  Response action: {response.action}")
        print(f"  Response strategy: {response.strategy}")
        print(f"  Response preview: {response.response[:100]}...")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)