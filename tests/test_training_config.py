"""
Pytest integration for training configuration validation.
"""
import os
import sys
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_src_dir = os.path.join(_repo_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import pytest
from scripts.test_training_config import test_training_configuration


def test_training_config():
    """Verify that training configuration resolves, passes type checks, and initializes optimizer."""
    test_training_configuration()
