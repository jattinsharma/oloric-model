"""
Unit tests for the diagnostic single-example inference script.
"""
import unittest
import os
import sys
import pathlib

# Ensure src and repo root are on sys.path
_repo_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "scripts"))
sys.path.insert(0, str(_repo_root))

from scripts.diagnostic_single_inference import run_diagnostic


class TestDiagnosticSingleInference(unittest.TestCase):
    def test_diagnostic_mock_mode_passes(self):
        """Mock diagnostic inference must succeed and return exit code 0."""
        code = run_diagnostic(
            model_path="./nonexistent_checkpoints",
            benchmark_file="./evaluation/benchmark.jsonl",
            mock_mode=True,
        )
        self.assertEqual(code, 0)

    def test_diagnostic_missing_model_fails_gracefully(self):
        """Missing model checkpoint must exit with 1 and not raise an unhandled exception."""
        code = run_diagnostic(
            model_path="./nonexistent_checkpoints_dir_12345",
            benchmark_file="./evaluation/benchmark.jsonl",
            mock_mode=False,
        )
        self.assertEqual(code, 1)

    def test_diagnostic_missing_benchmark_fails_gracefully(self):
        """Missing benchmark file must exit with 1 and not raise an unhandled exception."""
        code = run_diagnostic(
            model_path="./nonexistent",
            benchmark_file="./nonexistent_benchmark.jsonl",
            mock_mode=True,
        )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
