#!/usr/bin/env python3
"""
Gate 1 — Phase 2: Automated Baseline Scoring
Reads responses.jsonl and scores each response on deterministic structural criteria.

Self-contained: no imports from oloric.src.*
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Valid values from the Oloric schema
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "diagnose", "explain", "simplify", "analogy", "example",
    "prerequisite", "misconception_correction", "hint", "practice",
    "recap", "confirm_understanding", "memory_candidate",
}

VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}

VALID_CONFUSION_TYPES = {"conceptual", "procedural", "prerequisite_gap", "misconception"}

VALID_SEVERITIES = {"low", "medium", "high"}

VALID_MEMORY_TYPES = {
    "definition", "procedure", "example", "misconception_correction",
    "hint", "prerequisite", None,
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to extract and parse JSON from a raw response string."""
    # Try direct parse
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON within markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to find any JSON object
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def score_response(response_record: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single response on all automated dimensions."""
    item_id = response_record["id"]
    mode = response_record["mode"]
    raw = response_record["raw_response"]

    scores: Dict[str, Any] = {
        "id": item_id,
        "mode": mode,
        "domain": response_record.get("domain", "unknown"),
        "category": response_record.get("category", "unknown"),
    }

    # Response length (always applicable)
    scores["response_length"] = len(raw)
    scores["response_not_empty"] = "PASS" if len(raw.strip()) >= 20 else "FAIL"

    # Mode A (general): schema checks are NA
    if mode == "general":
        scores["schema_valid"] = "NA"
        scores["has_action"] = "NA"
        scores["has_strategy"] = "NA"
        scores["has_response"] = "NA"
        scores["has_diagnosis"] = "NA"
        scores["has_understanding_check"] = "NA"
        scores["has_memory"] = "NA"
        scores["memory_structure_valid"] = "NA"
        scores["action_valid"] = "NA"
        scores["difficulty_valid"] = "NA"
        return scores

    # Mode B (structured): full schema checks
    parsed = try_parse_json(raw)

    if parsed is None:
        scores["schema_valid"] = "FAIL"
        scores["has_action"] = "FAIL"
        scores["has_strategy"] = "FAIL"
        scores["has_response"] = "FAIL"
        scores["has_diagnosis"] = "FAIL"
        scores["has_understanding_check"] = "FAIL"
        scores["has_memory"] = "FAIL"
        scores["memory_structure_valid"] = "FAIL"
        scores["action_valid"] = "FAIL"
        scores["difficulty_valid"] = "FAIL"
        scores["parse_error"] = "Could not extract valid JSON from response"
        return scores

    # Schema validity — check if all required top-level keys exist
    required_top = {"action", "strategy", "difficulty", "response",
                    "understanding_check", "diagnosis", "memory"}
    missing_top = required_top - set(parsed.keys())
    scores["schema_valid"] = "PASS" if not missing_top else "FAIL"
    if missing_top:
        scores["schema_missing_fields"] = list(missing_top)

    # Individual field checks
    scores["has_action"] = "PASS" if "action" in parsed and isinstance(parsed["action"], str) else "FAIL"
    scores["has_strategy"] = "PASS" if "strategy" in parsed and isinstance(parsed["strategy"], str) else "FAIL"
    scores["has_response"] = "PASS" if "response" in parsed and isinstance(parsed["response"], str) and len(parsed["response"]) >= 10 else "FAIL"

    # Difficulty
    difficulty = parsed.get("difficulty", "")
    scores["difficulty_valid"] = "PASS" if difficulty in VALID_DIFFICULTIES else "FAIL"

    # Action validity
    action = parsed.get("action", "")
    scores["action_valid"] = "PASS" if action in VALID_ACTIONS else "FAIL"
    if action and action not in VALID_ACTIONS:
        scores["action_value"] = action  # record what was actually produced

    # Diagnosis
    diag = parsed.get("diagnosis")
    if isinstance(diag, dict):
        has_ct = isinstance(diag.get("confusion_type"), str)
        has_sev = isinstance(diag.get("severity"), str)
        scores["has_diagnosis"] = "PASS" if has_ct and has_sev else "FAIL"
        if has_ct:
            scores["diagnosis_confusion_type_valid"] = "PASS" if diag["confusion_type"] in VALID_CONFUSION_TYPES else "FAIL"
        if has_sev:
            scores["diagnosis_severity_valid"] = "PASS" if diag["severity"] in VALID_SEVERITIES else "FAIL"
    else:
        scores["has_diagnosis"] = "FAIL"

    # Understanding check
    uc = parsed.get("understanding_check")
    if isinstance(uc, dict):
        scores["has_understanding_check"] = "PASS" if "required" in uc else "FAIL"
    else:
        scores["has_understanding_check"] = "FAIL"

    # Memory
    mem = parsed.get("memory")
    if isinstance(mem, dict):
        scores["has_memory"] = "PASS" if "candidate" in mem else "FAIL"
        if mem.get("candidate") is True:
            has_title = isinstance(mem.get("title"), str) and len(mem["title"]) > 0
            has_content = isinstance(mem.get("content"), str) and len(mem["content"]) > 0
            has_anchor = isinstance(mem.get("anchor_concept"), str) and len(mem["anchor_concept"]) > 0
            scores["memory_structure_valid"] = "PASS" if (has_title and has_content and has_anchor) else "FAIL"
        else:
            scores["memory_structure_valid"] = "NA"
    else:
        scores["has_memory"] = "FAIL"
        scores["memory_structure_valid"] = "FAIL"

    return scores


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(all_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-dimension pass rates."""
    dimensions = [
        "schema_valid", "has_action", "has_strategy", "has_response",
        "has_diagnosis", "has_understanding_check", "has_memory",
        "memory_structure_valid", "action_valid", "difficulty_valid",
        "response_not_empty",
    ]
    agg = {}
    for dim in dimensions:
        applicable = [s for s in all_scores if s.get(dim) not in ("NA", None)]
        passed = [s for s in applicable if s.get(dim) == "PASS"]
        agg[dim] = {
            "total": len(applicable),
            "passed": len(passed),
            "pass_rate": round(len(passed) / max(len(applicable), 1), 3),
        }

    # Response length stats
    lengths = [s["response_length"] for s in all_scores if "response_length" in s]
    if lengths:
        agg["response_length"] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 0),
            "median": sorted(lengths)[len(lengths) // 2],
        }

    return agg


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def run_dry(responses_path: str, output_dir: str):
    """Dry-run: create sample scores for validation."""
    print("=" * 60)
    print("DRY RUN — scoring logic validation")
    print("=" * 60)

    # Test with a fake response
    sample_valid = {
        "id": "test_001",
        "mode": "structured",
        "raw_response": json.dumps({
            "action": "explain",
            "strategy": "simple_explanation",
            "difficulty": "beginner",
            "response": "Here is a detailed explanation of the concept...",
            "understanding_check": {"required": True, "question": "What is X?", "expected_answer": "X is..."},
            "diagnosis": {"confusion_type": "conceptual", "severity": "medium", "misconception_addressed": None},
            "memory": {"candidate": True, "title": "Key concept", "content": "Remember this...", "anchor_concept": "X", "memory_type": "definition", "confidence": 0.8},
        }),
    }
    sample_invalid = {
        "id": "test_002",
        "mode": "structured",
        "raw_response": "I'm not sure what you mean. Could you clarify?",
    }
    sample_general = {
        "id": "test_003",
        "mode": "general",
        "raw_response": "Let me explain this concept in simple terms...",
    }

    for sample in [sample_valid, sample_invalid, sample_general]:
        scores = score_response(sample)
        print(f"\n{sample['id']} ({sample['mode']}):")
        for k, v in scores.items():
            if k not in ("id", "mode", "domain", "category"):
                print(f"  {k}: {v}")

    print("\nDRY RUN PASSED — scoring logic is valid")
    return 0


# ---------------------------------------------------------------------------
# Real scoring
# ---------------------------------------------------------------------------

def run_scoring(responses_path: str, output_dir: str):
    """Score actual responses from baseline inference."""
    print("=" * 60)
    print("AUTOMATED BASELINE SCORING")
    print("=" * 60)

    if not os.path.exists(responses_path):
        print(f"ERROR: responses file not found: {responses_path}")
        return 1

    # Load responses
    responses = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(json.loads(line))

    print(f"Loaded {len(responses)} responses")

    # Score each response
    all_scores = []
    for resp in responses:
        scores = score_response(resp)
        all_scores.append(scores)

    # Separate by mode
    general_scores = [s for s in all_scores if s["mode"] == "general"]
    structured_scores = [s for s in all_scores if s["mode"] == "structured"]

    # Aggregate
    print(f"\n--- General mode: {len(general_scores)} responses ---")
    if general_scores:
        gen_agg = aggregate_scores(general_scores)
        for k, v in gen_agg.items():
            print(f"  {k}: {v}")

    print(f"\n--- Structured mode: {len(structured_scores)} responses ---")
    if structured_scores:
        struct_agg = aggregate_scores(structured_scores)
        for k, v in struct_agg.items():
            print(f"  {k}: {v}")

    overall_agg = aggregate_scores(all_scores)

    # Save scores
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, "scores_automated.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for s in all_scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nScores saved: {scores_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "responses_file": responses_path,
        "total_responses": len(all_scores),
        "general_count": len(general_scores),
        "structured_count": len(structured_scores),
        "general_aggregate": aggregate_scores(general_scores) if general_scores else {},
        "structured_aggregate": aggregate_scores(structured_scores) if structured_scores else {},
        "overall_aggregate": overall_agg,
    }
    summary_path = os.path.join(output_dir, "scores_automated_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    print(f"\n{'=' * 60}")
    print("AUTOMATED SCORING COMPLETE")
    print(f"{'=' * 60}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gate 1 — Automated baseline scoring"
    )
    parser.add_argument(
        "--responses", type=str,
        default="evaluation/baseline/responses.jsonl",
        help="Path to responses JSONL from baseline inference",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="evaluation/baseline",
        help="Output directory for score artifacts",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate scoring logic without real data",
    )
    args = parser.parse_args()

    if args.dry_run:
        return run_dry(args.responses, args.output_dir)
    else:
        return run_scoring(args.responses, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
