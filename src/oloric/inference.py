"""
Inference interface for OLORIC model.
"""
import torch
from typing import Dict, Any, Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import PeftModel
from .schemas.model_input import OloricModelInput
from .schemas.model_output import OloricModelOutput
from .config import config
from .formatting import OloricFormatter
import logging

logger = logging.getLogger(__name__)


class OloricInference:
    """Handles inference for OLORIC model."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize OLORIC inference.

        Args:
            model_path: Path to fine-tuned model (if None, uses base model)
        """
        self.model_config = config.get_model_config()
        self.qlora_config = config.get_qlora_config()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.formatter = None
        self.generation_config = None

        self._setup_model()

    def _setup_model(self) -> None:
        """Setup model and tokenizer for inference."""
        model_name = self.model_config.get("base_model", {}).get(
            "name", "Qwen/Qwen3-4B-Instruct-2507"
        )

        logger.info(f"Loading model: {model_name}")
        if self.model_path:
            logger.info(f"Loading adapter from: {self.model_path}")

        cuda_available = torch.cuda.is_available()
        device_map = "auto" if cuda_available else "cpu"

        # Setup quantization config if using 4-bit on CUDA
        quantization_config = None
        if cuda_available and self.qlora_config.get("quantization", {}).get("load_in_4bit", False):
            quant_config = self.qlora_config["quantization"]
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", True),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            revision=self.model_config.get("base_model", {}).get("revision", "main"),
            quantization_config=quantization_config
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=self.model_config.get("base_model", {}).get("revision", "main")
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load adapter if provided
        if self.model_path:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model.eval()

        # Setup formatter
        self.formatter = OloricFormatter(self.tokenizer)

        # Setup generation config
        eval_config = config.get_evaluation_config().get("generation", {})
        self.generation_config = GenerationConfig(
            max_new_tokens=eval_config.get("max_new_tokens", 512),
            temperature=eval_config.get("temperature", 0.7),
            top_p=eval_config.get("top_p", 0.9),
            do_sample=eval_config.get("do_sample", True),
            pad_token_id=self.tokenizer.eos_token_id
        )

    def generate_response(self, model_input: OloricModelInput) -> OloricModelOutput:
        """
        Generate response for given input.

        Args:
            model_input: OloricModelInput object

        Returns:
            OloricModelOutput object
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized")

        # Format input
        input_text = self.formatter.format_input(model_input)

        # Tokenize and ensure on model device
        device = self.model.device
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config.get("base_model", {}).get("max_length", 2048)
        ).to(device)

        # Generate using torch.inference_mode()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )

        # Decode
        input_len = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        # Parse output
        try:
            model_output = self.formatter.parse_output(generated_text)
        except Exception as e:
            e.raw_response = generated_text
            raise e

        return model_output

    def batch_generate(self, model_inputs: List[OloricModelInput]) -> List[OloricModelOutput]:
        """
        Generate responses for multiple inputs.

        Args:
            model_inputs: List of OloricModelInput objects

        Returns:
            List of OloricModelOutput objects
        """
        return [self.generate_response(model_input) for model_input in model_inputs]


# Global inference instance with lazy loading
class _LazyInferenceEngine:
    def __init__(self):
        self._engine = None

    def set_engine(self, engine: OloricInference) -> None:
        """Explicitly set the underlying inference engine to avoid reloading."""
        self._engine = engine

    def __getattr__(self, name):
        if self._engine is None:
            self._engine = OloricInference()
        return getattr(self._engine, name)


inference_engine = _LazyInferenceEngine()