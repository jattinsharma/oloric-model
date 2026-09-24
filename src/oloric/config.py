"""
Configuration management for OLORIC.
"""
import os
from typing import Dict, Any
import yaml
from pathlib import Path


class Config:
    """Configuration manager for OLORIC."""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Any] = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all YAML configuration files."""
        config_files = [
            "model.yaml",
            "dataset.yaml", 
            "qlora.yaml",
            "evaluation.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_name = config_file.split('.')[0]
                    self._configs[config_name] = yaml.safe_load(f)
            else:
                # Create default config if file doesn't exist
                self._configs[config_file.split('.')[0]] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "model.base_model.name")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._configs.get("model", {})
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._configs.get("dataset", {})
    
    def get_qlora_config(self) -> Dict[str, Any]:
        """Get QLoRA configuration."""
        return self._configs.get("qlora", {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._configs.get("evaluation", {})


# Global config instance
config = Config()
