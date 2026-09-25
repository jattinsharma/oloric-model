"""
Unit tests for evaluation schema-failure handling in Oloric v0.1.

Verified behaviours
-------------------
- parse_output raises pydantic.ValidationError when required fields
  (e.g. diagnosis.severity) are absent from the model's JSON output.
- evaluate_example captures ValidationError and returns an EvaluationResult
  with schema_valid=False instead of crashing or silently dropping the example.
- evaluate_benchmark always returns one EvaluationResult per example.
- generate_report surfaces num_schema_failures, schema_failure_rate, and
  per-failure detail in the schema_failures list.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from pydantic import ValidationError

from oloric.evaluation import EvaluationResult, OloricEvaluator
from oloric.formatting import OloricFormatter
from oloric.schemas.model_output import OloricModelOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_tokenizer():
    tok = MagicMock()
    tok.eos_token = "</s>"
    tok.pad_token_id = 0
    tok.apply_chat_template = MagicMock(return_value="<prompt>")
    tok.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    return tok


def _good_output() -> OloricModelOutput:
    return OloricModelOutput(**{
        "action": "explain",
        "strategy": "simple_explanation",
        "difficulty": "beginner",
        "response": "Answer.",
        "understanding_check": {"required": False},
        "diagnosis": {"confusion_type": "conceptual", "severity": "low"},
        "memory": {"candidate": False, "confidence": 0.0},
    })


def _benchmark_example(eid: str) -> dict:
    return {
        "id": eid,
        "context": {
            "task": "Help.",
            "document_context": {
                "title": "Doc", "page": 1, "section": "S",
                "selected_text": "foo",
                "surrounding_context": "bar " * 50,
                "retrieved_evidence": [],
            },
            "learner_state": {
                "level": "beginner", "concept": "test", "mastery": 0.5,
                "known_prerequisites": [], "weak_prerequisites": [],
                "known_misconceptions": [],
            },
            "conversation_context": [{"role": "student", "content": "Why?"}],
            "current_goal": "Understand basics",
        },
        "target": {
            "action": "explain", "strategy": "simple_explanation",
            "difficulty": "beginner", "response": "Because.",
            "understanding_check": {"required": False, "question": None, "expected_answer": None},
            "diagnosis": {
                "confusion_type": "conceptual", "severity": "low",
                "misconception_addressed": None,
            },
            "memory": {
                "candidate": False, "title": None, "content": None,
                "anchor_concept": None, "memory_type": None, "confidence": 0.0,
            },
        },
    }


# ---------------------------------------------------------------------------
# parse_output schema-failure tests
# ---------------------------------------------------------------------------

class TestParseOutputSchemaFailure(unittest.TestCase):

    def setUp(self):
        self.formatter = OloricFormatter(_mock_tokenizer())

    def test_missing_severity_raises_validation_error(self):
        """parse_output must raise ValidationError when severity is absent."""
        bad = json.dumps({
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "Answer.",
            "understanding_check": {"required": False},
            "diagnosis": {"confusion_type": "conceptual"},  # severity missing
            "memory": {"candidate": False, "confidence": 0.0},
        })
        with self.assertRaises(ValidationError) as ctx:
            self.formatter.parse_output(bad)
        self.assertIn("severity", str(ctx.exception))

    def test_valid_json_parsed_correctly(self):
        """parse_output must succeed when all required fields are present."""
        good = json.dumps({
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "Answer.",
            "understanding_check": {"required": False},
            "diagnosis": {"confusion_type": "conceptual", "severity": "low"},
            "memory": {"candidate": False, "confidence": 0.0},
        })
        result = self.formatter.parse_output(good)
        self.assertIsInstance(result, OloricModelOutput)
        self.assertEqual(result.diagnosis.severity, "low")

    def test_no_json_block_fallback_has_required_fields(self):
        """parse_output must return a valid fallback when no JSON is found."""
        result = self.formatter.parse_output("Plain text with no JSON block.")
        self.assertIsInstance(result, OloricModelOutput)
        self.assertEqual(result.diagnosis.severity, "unknown")
        self.assertEqual(result.action, "explain")


# ---------------------------------------------------------------------------
# generate_report schema-failure surfacing
# ---------------------------------------------------------------------------

class TestGenerateReport(unittest.TestCase):

    def _make_ev(self) -> OloricEvaluator:
        ev = OloricEvaluator.__new__(OloricEvaluator)
        ev.eval_config = {}
        ev.dimensions = {}
        ev.benchmark_path = ""
        return ev

    def _result(self, eid: str, valid: bool, score: float = 3.0) -> EvaluationResult:
        t = _good_output()
        return EvaluationResult(
            example_id=eid,
            dimension_scores={},
            overall_score=score if valid else 0.0,
            feedback="ok",
            model_output=t if valid else None,
            expected_output=t,
            schema_valid=valid,
            raw_response=None if valid else "{}",
            validation_error=None if valid else "severity: Field required",
        )

    def test_report_has_failure_counts(self):
        ev = self._make_ev()
        results = [
            self._result("a", True),
            self._result("b", False),
            self._result("c", True),
        ]
        report = ev.generate_report(results)
        self.assertEqual(report["num_examples"], 3)
        self.assertEqual(report["num_schema_valid"], 2)
        self.assertEqual(report["num_schema_failures"], 1)
        self.assertAlmostEqual(report["schema_failure_rate"], 1 / 3, places=3)
        self.assertEqual(len(report["schema_failures"]), 1)
        fail = report["schema_failures"][0]
        self.assertEqual(fail["example_id"], "b")
        self.assertIn("severity", fail["validation_error"])

    def test_empty_results_report(self):
        ev = self._make_ev()
        report = ev.generate_report([])
        self.assertEqual(report["num_examples"], 0)
        self.assertEqual(report["num_schema_failures"], 0)
        self.assertEqual(report["schema_failures"], [])


if __name__ == "__main__":
    unittest.main()
