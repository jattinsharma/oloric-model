"""
Data handling for OLORIC.
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .schemas.dataset import TrainingExample
from .config import config


class OloricDataManager:
    """Manages data loading and processing for OLORIC."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data manager.
        
        Args:
            data_dir: Root directory for data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.generated_dir = self.data_dir / "generated"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of dictionaries
        """
        data = []
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line in {file_path}: {e}")
        return data
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """
        Save data to JSONL file.
        
        Args:
            data: List of dictionaries to save
            file_path: Path to save JSONL file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def load_training_examples(self, file_path: Optional[Path] = None) -> List[TrainingExample]:
        """
        Load training examples from JSONL file.
        
        Args:
            file_path: Path to training examples file (defaults to processed/train.jsonl)
            
        Returns:
            List of TrainingExample objects
        """
        if file_path is None:
            file_path = self.processed_dir / "train.jsonl"
            
        data = self.load_jsonl(file_path)
        examples = []
        for item in data:
            try:
                example = TrainingExample(**item)
                examples.append(example)
            except Exception as e:
                print(f"Warning: Failed to parse training example: {e}")
        return examples
    
    def save_training_examples(self, examples: List[TrainingExample], 
                             file_path: Optional[Path] = None) -> None:
        """
        Save training examples to JSONL file.
        
        Args:
            examples: List of TrainingExample objects
            file_path: Path to save file (defaults to processed/train.jsonl)
        """
        if file_path is None:
            file_path = self.processed_dir / "train.jsonl"
            
        data = [example.dict() for example in examples]
        self.save_jsonl(data, file_path)
    
    def get_dataset_paths(self) -> Dict[str, Path]:
        """
        Get paths for dataset files.
        
        Returns:
            Dictionary mapping dataset type to file path
        """
        dataset_config = config.get_dataset_config()
        paths = {
            "raw": self.raw_dir,
            "generated": self.generated_dir,
            "processed": self.processed_dir,
            "train": self.processed_dir / "train.jsonl",
            "validation": self.processed_dir / "validation.jsonl",
            "test": self.processed_dir / "test.jsonl"
        }
        
        # Override with config if specified
        if "paths" in dataset_config:
            path_config = dataset_config["paths"]
            for key, path_str in path_config.items():
                if key in ["train_file", "validation_file", "test_file"]:
                    # These are full paths
                    paths[key.replace("_file", "")] = Path(path_str)
                elif key in self.data_dir.parts or key in ["raw_data", "generated_data", "processed_data"]:
                    # These are directory names
                    dir_name = key.split("_")[0]  # raw, generated, processed
                    if dir_name in paths:
                        paths[dir_name] = Path(path_str)
                        
        return paths


# Global data manager instance
data_manager = OloricDataManager()
