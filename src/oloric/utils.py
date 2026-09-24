"""
Utility functions for OLORIC.
"""
import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed}")


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        Torch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_json_safely(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed object or default
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_vram_usage(model: torch.nn.Module, batch_size: int = 1, 
                        sequence_length: int = 2048) -> Dict[str, float]:
    """
    Estimate VRAM usage for model.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        Dictionary with VRAM usage estimates in GB
    """
    # This is a simplified estimation
    # In practice, would need to actually run the model to measure
    
    # Get model size in parameters
    total_params, _ = count_parameters(model)
    
    # Rough estimation:
    # Model weights: 4 bytes per parameter (FP32) or 2 bytes (FP16) or 1 byte (int8)
    # Activations: roughly proportional to batch_size * sequence_length * hidden_size
    # For estimation, we'll use a simplified formula
    
    # Assume model is in 4-bit quantization for OLORIC
    weights_gb = (total_params * 0.5) / (1024**3)  # 0.5 bytes per parameter (4-bit)
    
    # Rough activation estimation (very simplified)
    # This would depend heavily on model architecture
    activation_gb = (batch_size * sequence_length * 4096 * 2) / (1024**3)  # Assuming hidden size 4096, 2 bytes per activation
    
    # Gradient storage (if training)
    # For 4-bit quantization with paged optimizer, this is complex to estimate
    
    return {
        "model_weights_gb": round(weights_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "estimated_total_gb": round(weights_gb + activation_gb, 2)
    }


# Convenience functions
def get_optimized_dtype() -> torch.dtype:
    """Get optimal dtype for current hardware."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    else:
        return torch.float32


def is_main_process() -> bool:
    """Check if this is the main process (for distributed training)."""
    # Simple implementation - in practice would check for distributed training setup
    return True
