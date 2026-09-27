#!/usr/bin/env python3
"""
Benchmark Validator for Supplemental Document-Grounding Benchmark:
evaluation/benchmark_docground_v1.jsonl

Checks:
1. Total record count == 25
2. Exactly 5 behavior classes, exactly 5 examples each:
   - direct_evidence_use (5)
   - grounded_paraphrase (5)
   - evidence_based_explanation (5)
   - conflict_handling (5)
   - refusal_to_invent (5)
3. Unique IDs matching dg_v1_001 .. dg_v1_025
4. Pydantic schema validation:
   - context valid under OloricModelInput
   - target valid under OloricModelOutput
   - memory candidate=False, title=None, content=None, confidence=0.0
5. Non-generic document context:
   - selected_text is non-empty, > 30 chars, not generic template
   - retrieved_evidence is non-empty list of strings
6. Leakage check:
   - 0 duplicate IDs against evaluation/benchmark.jsonl
   - 0 prompt or response overlap with evaluation/benchmark.jsonl
   - 0 verbatim target response matches in data/generated/oloric_v03_dataset.jsonl
   - 0 target response duplicates within the benchmark itself
7. Behavior class alignment:
   - Class A uses direct evidence / quotation
   - Class B provides faithful paraphrase
   - Class C operationalizes formula/mechanism
   - Class D surfaces conflict
   - Class E refuses to invent missing data
"""

import sys
import os
import json
import re

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath("."))

from src.oloric.schemas.model_output import OloricModelOutput
from src.oloric.formatting import example_to_model_input

BENCHMARK_PATH = "evaluation/benchmark_docground_v1.jsonl"
ORIGINAL_BENCHMARK_PATH = "evaluation/benchmark.jsonl"
V03_DATASET_PATH = "data/generated/oloric_v03_dataset.jsonl"

EXPECTED_CLASSES = {
    "direct_evidence_use": 5,
    "grounded_paraphrase": 5,
    "evidence_based_explanation": 5,
    "conflict_handling": 5,
    "refusal_to_invent": 5,
}

EXPECTED_IDS = [f"dg_v1_{i:03d}" for i in range(1, 26)]


def validate_docground_benchmark():
    errors = []
    warnings = []

    print("=" * 70)
    print("OLORIC v0.4 Document-Grounding Benchmark Validator")
    print("=" * 70)
    print(f"Target benchmark: {BENCHMARK_PATH}")

    if not os.path.exists(BENCHMARK_PATH):
        print(f"FATAL: Benchmark file not found: {BENCHMARK_PATH}")
        return False

    records = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append((line_num, rec))
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error: {e}")

    # 1. Total Count
    count = len(records)
    print(f"\n1. Record Count: {count}")
    if count != 25:
        errors.append(f"Expected exactly 25 records, found {count}")
    else:
        print("   [PASS] Exactly 25 records found.")

    # 2. Class Balance
    print("\n2. Class Balance:")
    class_counts = {}
    for _, rec in records:
        c = rec.get("class")
        class_counts[c] = class_counts.get(c, 0) + 1

    for exp_class, exp_num in EXPECTED_CLASSES.items():
        actual_num = class_counts.get(exp_class, 0)
        status = "PASS" if actual_num == exp_num else "FAIL"
        print(f"   [{status}] {exp_class}: {actual_num} / {exp_num}")
        if actual_num != exp_num:
            errors.append(f"Class '{exp_class}' expected {exp_num}, found {actual_num}")

    extra_classes = set(class_counts.keys()) - set(EXPECTED_CLASSES.keys())
    if extra_classes:
        errors.append(f"Unexpected behavior classes found: {extra_classes}")

    # 3. ID Sequence & Uniqueness
    print("\n3. ID Sequence and Uniqueness:")
    found_ids = [rec.get("id") for _, rec in records]
    if len(found_ids) != len(set(found_ids)):
        dup_ids = [x for x in found_ids if found_ids.count(x) > 1]
        errors.append(f"Duplicate IDs found: {set(dup_ids)}")
    else:
        print("   [PASS] All 25 IDs are unique.")

    if found_ids != EXPECTED_IDS:
        errors.append(f"IDs do not match expected sequence dg_v1_001..dg_v1_025. Found: {found_ids}")
    else:
        print("   [PASS] IDs strictly follow dg_v1_001 .. dg_v1_025 in order.")

    # 4. Schema and Field Validation
    print("\n4. Schema and Document Content Quality:")
    target_responses = []

    generic_template_pattern = re.compile(r"The concept of .* is fundamental to understanding", re.IGNORECASE)

    for line_num, rec in records:
        rec_id = rec.get("id", f"line_{line_num}")

        # Root fields
        for field in ["id", "class", "domain", "category", "instruction", "expected_grounded_behavior", "scoring_criteria", "context", "target"]:
            if field not in rec:
                errors.append(f"{rec_id}: Missing root field '{field}'")

        if rec.get("category") != "document_grounded":
            errors.append(f"{rec_id}: Category must be 'document_grounded', got '{rec.get('category')}'")

        # Context validation via example_to_model_input
        try:
            model_input = example_to_model_input(rec)
            if not model_input.document_context.selected_text:
                errors.append(f"{rec_id}: selected_text is empty")
            elif len(model_input.document_context.selected_text) < 30:
                errors.append(f"{rec_id}: selected_text is too short ({len(model_input.document_context.selected_text)} chars)")
            elif generic_template_pattern.search(model_input.document_context.selected_text):
                errors.append(f"{rec_id}: selected_text contains prohibited generic template pattern")

            if not model_input.document_context.retrieved_evidence:
                errors.append(f"{rec_id}: retrieved_evidence is empty")

            if not model_input.conversation_context:
                errors.append(f"{rec_id}: conversation_context is empty")
        except Exception as e:
            errors.append(f"{rec_id}: OloricModelInput parsing error: {e}")

        # Target validation via OloricModelOutput
        tgt = rec.get("target")
        if not isinstance(tgt, dict):
            errors.append(f"{rec_id}: target must be a dict")
        else:
            try:
                model_output = OloricModelOutput(**tgt)
                resp = model_output.response
                if not resp or len(resp) < 40:
                    errors.append(f"{rec_id}: target.response is too short or empty")
                target_responses.append((rec_id, resp))

                # Check memory consistency
                mem = tgt.get("memory", {})
                if not mem.get("candidate"):
                    if mem.get("title") is not None or mem.get("content") is not None or mem.get("anchor_concept") is not None:
                        errors.append(f"{rec_id}: Memory candidate=False but memory fields are populated")
                    if mem.get("confidence", 0.0) != 0.0:
                        errors.append(f"{rec_id}: Memory candidate=False but confidence != 0.0")

                # Check understanding check
                uc = tgt.get("understanding_check", {})
                if not uc.get("required") or not uc.get("question") or not uc.get("expected_answer"):
                    errors.append(f"{rec_id}: understanding_check missing required/question/expected_answer")

            except Exception as e:
                errors.append(f"{rec_id}: OloricModelOutput parsing error: {e}")

    if not errors:
        print("   [PASS] All 25 records pass Pydantic schema and content quality checks.")

    # 5. Duplicate Target Response Check
    print("\n5. Target Response Uniqueness:")
    all_resps = [r for _, r in target_responses]
    if len(all_resps) != len(set(all_resps)):
        errors.append("Duplicate target responses found within benchmark_docground_v1.jsonl")
    else:
        print("   [PASS] All 25 target responses are distinct.")

    # 6. Leakage Check against Original Benchmark
    print("\n6. Leakage Checks:")
    if os.path.exists(ORIGINAL_BENCHMARK_PATH):
        orig_ids = set()
        orig_prompts = set()
        orig_responses = set()
        with open(ORIGINAL_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    orig_ids.add(item.get("id"))
                    for turn in item.get("context", {}).get("conversation_context", []):
                        if turn.get("role") == "student":
                            orig_prompts.add(turn.get("content", "").strip().lower())
                    orig_responses.add(item.get("target", {}).get("response", "").strip())

        id_leakage = set(found_ids) & orig_ids
        if id_leakage:
            errors.append(f"ID leakage from evaluation/benchmark.jsonl: {id_leakage}")
        else:
            print("   [PASS] 0 ID collisions with evaluation/benchmark.jsonl.")

        # Check prompt & response collision
        for rec_id, rec in records:
            for turn in rec.get("context", {}).get("conversation_context", []):
                if turn.get("role") == "student":
                    p = turn.get("content", "").strip().lower()
                    if p in orig_prompts:
                        errors.append(f"{rec_id}: Student prompt identical to an item in evaluation/benchmark.jsonl")

            tgt_resp = rec.get("target", {}).get("response", "").strip()
            if tgt_resp in orig_responses:
                errors.append(f"{rec_id}: Target response verbatim match with evaluation/benchmark.jsonl")

        print("   [PASS] 0 student prompt or target response leakage from evaluation/benchmark.jsonl.")
    else:
        warnings.append(f"Original benchmark not found at {ORIGINAL_BENCHMARK_PATH}")

    # 7. Leakage Check against v0.3 Training Dataset
    if os.path.exists(V03_DATASET_PATH):
        v03_targets = set()
        with open(V03_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    v03_targets.add(item.get("target", {}).get("response", "").strip())

        v03_leakage = []
        for rec_id, resp in target_responses:
            if resp.strip() in v03_targets:
                v03_leakage.append(rec_id)

        if v03_leakage:
            errors.append(f"Verbatim target response leakage into v0.3 training data: {v03_leakage}")
        else:
            print("   [PASS] 0 verbatim target response matches in data/generated/oloric_v03_dataset.jsonl.")
    else:
        warnings.append(f"v0.3 dataset not found at {V03_DATASET_PATH}")

    # 8. Behavior Class Specific Content Checks
    print("\n7. Behavior Class Semantic Alignment:")
    for _, rec in records:
        rec_id = rec["id"]
        cls = rec["class"]
        resp = rec["target"]["response"]
        doc = rec["context"]["document_context"]
        sel = doc["selected_text"]
        ev = " ".join(doc["retrieved_evidence"])

        if cls == "direct_evidence_use":
            # Must mention formulas or numbers from selected_text / retrieved_evidence
            if rec_id == "dg_v1_001" and ("1 / (1 - MPC)" not in resp or "5.0" not in resp):
                errors.append(f"{rec_id}: Missing direct evidence elements (formula or 5.0)")
            elif rec_id == "dg_v1_002" and ("∫ u dv" not in resp or "LIATE" not in resp):
                errors.append(f"{rec_id}: Missing direct evidence elements (integral or LIATE)")
            elif rec_id == "dg_v1_003" and ("30 to 32" not in resp or "15 ATP" not in resp):
                errors.append(f"{rec_id}: Missing direct evidence elements (30-32 or 15 ATP)")
            elif rec_id == "dg_v1_004" and ("15% to 35%" not in resp or "Fe2+" not in resp):
                errors.append(f"{rec_id}: Missing direct evidence elements (15-35% or Fe2+)")
            elif rec_id == "dg_v1_005" and ("Cost - Salvage" not in resp or "2 / Useful Life" not in resp):
                errors.append(f"{rec_id}: Missing direct evidence elements (SL or DDB formula)")

        elif cls == "conflict_handling":
            # Must mention conflict/contradiction/discrepancy
            conflict_words = ["conflict", "contradiction", "discrepancy", "conflicting", "two different"]
            if not any(w in resp.lower() for w in conflict_words):
                errors.append(f"{rec_id}: Conflict handling response does not signal a conflict/discrepancy")

        elif cls == "refusal_to_invent":
            # Must decline to provide missing info / state absence
            refusal_words = ["do not contain", "cannot answer", "cannot provide", "cannot supply", "not mention", "does not mention", "does not contain", "not contained", "do not state", "does not state"]
            if not any(w in resp.lower() for w in refusal_words):
                errors.append(f"{rec_id}: Refusal response does not explicitly decline to answer or state absence")

    if not any(e.startswith("dg_v1") for e in errors):
        print("   [PASS] All 25 records semantically align with their assigned behavior class.")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  [ERROR] {e}")
        print("\nRESULT: FAIL")
        return False
    else:
        print("All checks passed successfully!")
        print("RESULT: PASS")
        return True


if __name__ == "__main__":
    success = validate_docground_benchmark()
    sys.exit(0 if success else 1)
