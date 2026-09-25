#!/usr/bin/env python3
"""
Root-level wrapper for diagnostic single example inference.
Delegates to scripts.diagnostic_single_inference.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "src")
_scripts_dir = os.path.join(_script_dir, "scripts")

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from scripts.diagnostic_single_inference import main

if __name__ == "__main__":
    main()