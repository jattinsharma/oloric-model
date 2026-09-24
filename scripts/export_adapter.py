#!/usr/bin/env python3
"""
Adapter export script for OLORIC.
Exports trained LoRA adapters for deployment.
"""
import os
import sys
import json
import argparse
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def export_adapter(model_path: str, 
                  output_path: str,
                  base_model_name: str = None,
                  push_to_hub: bool = False,
                  hub_model_name: str = None) -> None:
    """
    Export LoRA adapter for deployment.
    
    Args:
        model_path: Path to the trained model with adapter
        output_path: Path to save the exported adapter
        base_model_name: Name of the base model (if not in config)
        push_to_hub: Whether to push to Hugging Face Hub
        hub_model_name: Name for the model on Hugging Face Hub
    """
    print(f"Loading adapter from: {model_path}")
    
    # Load the adapter config to get base model name
    try:
        peft_config = PeftConfig.from_pretrained(model_path)
        if base_model_name is None:
            base_model_name = peft_config.base_model_name_or_path
        print(f"Base model: {base_model_name}")
    except Exception as e:
        print(f"Error loading adapter config: {e}")
        # Try to infer from path or use default
        if base_model_name is None:
            base_model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Default
            print(f"Using default base model: {base_model_name}")
    
    # Load base model
    print("Loading base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,  # Use float16 for export
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return 1
    
    # Load the PEFT model
    print("Loading PEFT model...")
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        return 1
    
    # Merge adapter with base model (optional)
    print("Merging adapter with base model...")
    try:
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        print("Adapter merged successfully")
    except Exception as e:
        print(f"Warning: Could not merge adapter: {e}")
        print("Saving adapter separately instead")
        merged_model = model  # Keep as PEFT model
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model
    print(f"Saving model to: {output_path}")
    try:
        merged_model.save_pretrained(output_path)
        # Also save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print("Model and tokenizer saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
        return 1
    
    # Save adapter configuration info
    adapter_info = {
        "base_model": base_model_name,
        "adapter_type": "lora",
        "adapter_path": model_path if isinstance(model, PeftModel) else "merged",
        "export_timestamp": str(torch.datetime.now() if hasattr(torch, 'datetime') else "unknown"),
        "model_type": type(merged_model).__name__
    }
    
    info_file = os.path.join(output_path, "adapter_info.json")
    with open(info_file, 'w') as f:
        json.dump(adapter_info, f, indent=2)
    print(f"Adapter info saved to: {info_file}")
    
    # Push to Hugging Face Hub if requested
    if push_to_hub:
        if not hub_model_name:
            print("Error: hub_model_name required for pushing to hub")
            return 1
        
        try:
            print(f"Pushing to Hugging Face Hub: {hub_model_name}")
            merged_model.push_to_hub(hub_model_name)
            tokenizer.push_to_hub(hub_model_name)
            print("Successfully pushed to Hugging Face Hub")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            # Don't fail the entire process for hub errors
            print("Model saved locally despite hub push failure")
    
    return 0

def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export OLORIC LoRA adapter")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model with adapter")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the exported adapter")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name (if not specified in adapter)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to Hugging Face Hub after export")
    parser.add_argument("--hub-model-name", type=str, default=None,
                        help="Model name for Hugging Face Hub")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OLORIC Adapter Export")
    print("=" * 60)
    
    try:
        result = export_adapter(
            model_path=args.model_path,
            output_path=args.output_path,
            base_model_name=args.base_model,
            push_to_hub=args.push_to_hub,
            hub_model_name=args.hub_model_name
        )
        
        if result == 0:
            print("=" * 60)
            print("Adapter export completed successfully!")
            print("=" * 60)
            return 0
        else:
            print("=" * 60)
            print("Adapter export failed!")
            print("=" * 60)
            return result
    except Exception as e:
        print(f"Unexpected error during export: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
