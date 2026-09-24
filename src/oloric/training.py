"""
Training pipeline for OLORIC model.
"""
import os
import torch
from typing import Dict, Any, Optional, Tuple, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .config import config
from .data import data_manager
from .formatting import OloricFormatter
import logging

logger = logging.getLogger(__name__)


class OloricTrainer:
    """Handles training of OLORIC model."""
    
    def __init__(self):
        """Initialize OLORIC trainer."""
        self.model_config = config.get_model_config()
        self.qlora_config = config.get_qlora_config()
        self.training_config = self.model_config.get("training", {})
        self.model = None
        self.tokenizer = None
        self.formatter = None
        
    def setup_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        Setup model and tokenizer for training.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = self.model_config.get("base_model", {}).get(
            "name", "Qwen/Qwen3-4B-Instruct-2507"
        )
        
        logger.info(f"Loading base model: {model_name}")
        
        # Setup quantization config if using 4-bit
        quantization_config = None
        if self.qlora_config.get("quantization", {}).get("load_in_4bit", False):
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
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            revision=self.model_config.get("base_model", {}).get("revision", "main")
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
        
        # Prepare model for training
        if self.qlora_config.get("quantization", {}).get("load_in_4bit", False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        lora_config = self.qlora_config.get("lora", {})
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM")
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Setup formatter
        self.formatter = OloricFormatter(self.tokenizer)
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, examples: List[Dict[str, Any]]) -> Any:
        """
        Prepare dataset for training.
        
        Args:
            examples: List of training examples
            
        Returns:
            Processed dataset
        """
        # Convert examples to format expected by formatter
        formatted_examples = []
        
        for example in examples:
            # This would convert structured examples to input/target pairs
            # For now, returning placeholder
            formatted_examples.append(example)
        
        # In a real implementation, we would:
        # 1. Convert each example to input text using formatter.format_input
        # 2. Convert target to output text using formatter.format_output
        # 3. Concatenate input and target for language modeling
        # 4. Tokenize the result
        
        return formatted_examples
    
    def train(self, train_dataset: Any, eval_dataset: Optional[Any] = None) -> Trainer:
        """
        Train the OLORIC model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            
        Returns:
            Trainer object
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.get("output_dir", "./checkpoints"),
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 4),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            optim=self.training_config.get("optim", "paged_adamw_8bit"),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            max_grad_norm=self.training_config.get("max_grad_norm", 0.3),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 100),
            eval_steps=self.training_config.get("eval_steps", 100),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            fp16=self.training_config.get("fp16", False),
            bf16=self.training_config.get("bf16", True),
            tf32=self.training_config.get("tf32", True),
            dataloader_pin_memory=self.training_config.get("dataloader_pin_memory", False),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 0),
            remove_unused_columns=False,
            report_to="none"  # Disable wandb/comet logging for simplicity
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save model to
        """
        if self.model is not None:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")


# Global trainer instance
trainer = OloricTrainer()
