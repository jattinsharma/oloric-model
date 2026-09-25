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


def parse_numeric_param(
    field_name: str,
    val: Any,
    target_type: type,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Any:
    """
    Validate and convert numeric parameter strictly.
    Fails with clear descriptive error on invalid strings or mismatched types.
    """
    if isinstance(val, bool):
        raise TypeError(f"Invalid boolean value for numeric field '{field_name}': {val!r}")

    if target_type is float:
        if isinstance(val, (int, float)):
            f_val = float(val)
        elif isinstance(val, str):
            val_str = val.strip()
            try:
                f_val = float(val_str)
            except ValueError:
                raise ValueError(
                    f"Invalid float value for '{field_name}': '{val}'. Expected a valid numeric float."
                )
        else:
            raise TypeError(
                f"Invalid type for '{field_name}': {type(val).__name__} (value: {val!r}). Expected float or int."
            )

        if min_val is not None and f_val < min_val:
            raise ValueError(f"Value for '{field_name}' ({f_val}) must be >= {min_val}")
        if max_val is not None and f_val > max_val:
            raise ValueError(f"Value for '{field_name}' ({f_val}) must be <= {max_val}")
        return f_val

    elif target_type is int:
        if isinstance(val, int):
            i_val = val
        elif isinstance(val, float):
            if val.is_integer():
                i_val = int(val)
            else:
                raise ValueError(f"Invalid non-integer float for '{field_name}': {val}. Expected an integer.")
        elif isinstance(val, str):
            val_str = val.strip()
            try:
                f = float(val_str)
                if f.is_integer():
                    i_val = int(f)
                else:
                    raise ValueError(f"Invalid non-integer string for '{field_name}': '{val}'. Expected an integer.")
            except ValueError:
                raise ValueError(
                    f"Invalid integer value for '{field_name}': '{val}'. Expected a valid integer."
                )
        else:
            raise TypeError(
                f"Invalid type for '{field_name}': {type(val).__name__} (value: {val!r}). Expected int."
            )

        if min_val is not None and i_val < min_val:
            raise ValueError(f"Value for '{field_name}' ({i_val}) must be >= {min_val}")
        if max_val is not None and i_val > max_val:
            raise ValueError(f"Value for '{field_name}' ({i_val}) must be <= {max_val}")
        return i_val

    raise ValueError(f"Unsupported target_type: {target_type}")


def validate_training_parameters(
    training_config: Dict[str, Any],
    qlora_config: Optional[Dict[str, Any]] = None,
    base_model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Strictly validate and type-cast all training arguments before TrainingArguments is created.
    Returns a dictionary of strictly typed parameters.
    """
    qlora_cfg = qlora_config or {}
    bm_cfg = base_model_config or {}
    q_training = qlora_cfg.get("training", {})

    # Gradient accumulation: training_config takes priority, then qlora.yaml, default 4
    ga_raw = training_config.get("gradient_accumulation_steps", q_training.get("gradient_accumulation_steps", 4))
    gradient_accumulation_steps = parse_numeric_param("gradient_accumulation_steps", ga_raw, int, min_val=1)

    # Learning rate: training_config takes priority, then qlora.yaml (authoritative: 2.0e-4)
    lr_raw = training_config.get("learning_rate", q_training.get("learning_rate", 2.0e-4))
    learning_rate = parse_numeric_param("learning_rate", lr_raw, float, min_val=1e-8, max_val=1.0)

    # Weight decay
    wd_raw = training_config.get("weight_decay", q_training.get("weight_decay", 0.01))
    weight_decay = parse_numeric_param("weight_decay", wd_raw, float, min_val=0.0)

    # Max grad norm
    mgn_raw = training_config.get("max_grad_norm", q_training.get("max_grad_norm", 0.3))
    max_grad_norm = parse_numeric_param("max_grad_norm", mgn_raw, float, min_val=0.01)

    # Warmup ratio
    wr_raw = training_config.get("warmup_ratio", q_training.get("warmup_ratio", 0.03))
    warmup_ratio = parse_numeric_param("warmup_ratio", wr_raw, float, min_val=0.0, max_val=1.0)

    # Epochs
    nte_raw = training_config.get("num_train_epochs", q_training.get("num_train_epochs", 3))
    num_train_epochs = parse_numeric_param("num_train_epochs", nte_raw, int, min_val=1)

    # Batch sizes
    bs_raw = training_config.get("per_device_train_batch_size", q_training.get("per_device_train_batch_size", 1))
    per_device_train_batch_size = parse_numeric_param("per_device_train_batch_size", bs_raw, int, min_val=1)

    ebs_raw = training_config.get("per_device_eval_batch_size", 1)
    per_device_eval_batch_size = parse_numeric_param("per_device_eval_batch_size", ebs_raw, int, min_val=1)

    # Dataset size
    ds_raw = training_config.get("dataset_size", 384)
    dataset_size = parse_numeric_param("dataset_size", ds_raw, int, min_val=1)

    # Compute steps and warmup steps
    steps_per_epoch = max(1, dataset_size // (per_device_train_batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    # Cadence
    ls_raw = training_config.get("logging_steps", q_training.get("logging_steps", 5))
    logging_steps = parse_numeric_param("logging_steps", ls_raw, int, min_val=1)

    ss_raw = training_config.get("save_steps", q_training.get("save_steps", 50))
    save_steps = parse_numeric_param("save_steps", ss_raw, int, min_val=1)

    es_raw = training_config.get("eval_steps", q_training.get("eval_steps", 50))
    eval_steps = parse_numeric_param("eval_steps", es_raw, int, min_val=1)

    stl_raw = training_config.get("save_total_limit", q_training.get("save_total_limit", 3))
    save_total_limit = parse_numeric_param("save_total_limit", stl_raw, int, min_val=1)

    dnw_raw = training_config.get("dataloader_num_workers", q_training.get("dataloader_num_workers", 0))
    dataloader_num_workers = parse_numeric_param("dataloader_num_workers", dnw_raw, int, min_val=0)

    msl_raw = bm_cfg.get("max_length", qlora_cfg.get("sequence", {}).get("max_length", 2048))
    max_seq_length = parse_numeric_param("max_seq_length", msl_raw, int, min_val=1)

    # String parameters
    output_dir = str(training_config.get("output_dir", "./checkpoints"))
    optim = str(q_training.get("optim", training_config.get("optim", "paged_adamw_8bit")))
    lr_scheduler_type = str(q_training.get("lr_scheduler_type", training_config.get("lr_scheduler_type", "cosine")))

    # Boolean parameters
    gradient_checkpointing = bool(q_training.get("gradient_checkpointing", training_config.get("gradient_checkpointing", True)))
    fp16 = bool(q_training.get("fp16", training_config.get("fp16", False)))
    bf16 = bool(q_training.get("bf16", training_config.get("bf16", True)))
    tf32 = bool(q_training.get("tf32", training_config.get("tf32", True)))
    dataloader_pin_memory = bool(q_training.get("dataloader_pin_memory", training_config.get("dataloader_pin_memory", False)))
    remove_unused_columns = bool(q_training.get("remove_unused_columns", training_config.get("remove_unused_columns", False)))

    return {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": save_total_limit,
        "max_seq_length": max_seq_length,
        "dataset_size": dataset_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "dataloader_num_workers": dataloader_num_workers,
        "optim": optim,
        "lr_scheduler_type": lr_scheduler_type,
        "gradient_checkpointing": gradient_checkpointing,
        "fp16": fp16,
        "bf16": bf16,
        "tf32": tf32,
        "dataloader_pin_memory": dataloader_pin_memory,
        "remove_unused_columns": remove_unused_columns,
    }


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
        formatted_examples = []
        for example in examples:
            formatted_examples.append(example)
        return formatted_examples

    def get_training_arguments(self) -> TrainingArguments:
        """
        Validate all training parameters strictly, print their resolved values and types,
        and construct TrainingArguments.
        """
        params = validate_training_parameters(
            self.training_config,
            self.qlora_config,
            self.model_config.get("base_model", {})
        )

        print("=" * 60)
        print("OLORIC TRAINING PARAMETER TYPE VALIDATION:")
        print("=" * 60)
        numeric_keys = [
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
            "max_seq_length",
            "lr_scheduler_type",
            "optim",
        ]
        for k in numeric_keys:
            val = params[k]
            print(f"  {k} = {val}")
            print(f"  type = {type(val)}")
        print("=" * 60)

        cuda_available = torch.cuda.is_available()
        bf16_flag = params["bf16"] if cuda_available else False
        tf32_flag = params["tf32"] if cuda_available else False
        optim_choice = params["optim"] if cuda_available else "adamw_torch"

        training_args = TrainingArguments(
            output_dir=params["output_dir"],
            num_train_epochs=params["num_train_epochs"],
            per_device_train_batch_size=params["per_device_train_batch_size"],
            per_device_eval_batch_size=params["per_device_eval_batch_size"],
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            gradient_checkpointing=params["gradient_checkpointing"],
            optim=optim_choice,
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            max_grad_norm=params["max_grad_norm"],
            warmup_steps=params["warmup_steps"],
            lr_scheduler_type=params["lr_scheduler_type"],
            logging_steps=params["logging_steps"],
            save_steps=params["save_steps"],
            eval_steps=params["eval_steps"],
            save_total_limit=params["save_total_limit"],
            fp16=params["fp16"],
            bf16=bf16_flag,
            tf32=tf32_flag,
            dataloader_pin_memory=params["dataloader_pin_memory"],
            dataloader_num_workers=params["dataloader_num_workers"],
            remove_unused_columns=params["remove_unused_columns"],
            report_to="none",
        )
        return training_args
    
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
        
        training_args = self.get_training_arguments()
        
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
