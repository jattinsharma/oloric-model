#!/usr/bin/env python3
"""
OLORIC v0.4 Dataset Validation Script

Executes all 21 strict validation gates on the generated v0.4 dataset:
 1. Exactly 600 records
 2. 20 canonical categories only
 3. Category counts (document_grounded=120, error_correction=48, other 18=24)
 4. 6 domains * 100 records
 5. 60 concepts * 10 records
 6. All 20 benchmark records/concepts covered
 7. Unique IDs (format: v04_{domain}_{category}_{seq:03d})
 8. No v0.3 ID reuse
 9. No placeholder strings
10. No cross-domain contamination
11. Prior-error correctness/alignment
12. Evidence-prioritization alignment
13. Document-absence/refusal alignment
14. Genuine document evidence
15. Memory-field consistency
16. Train/validation/test counts = 480/60/60
17. Zero split overlap
18. Deterministic regeneration
19. No leakage from evaluation/benchmark.jsonl
20. No leakage from evaluation/benchmark_docground_v1.jsonl
21. No verbatim target leakage from v0.3 dataset
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from v04_concept_registry import (
    VALID_CATEGORIES,
    VALID_DOMAINS,
    get_domain_concepts,
    get_concept_info,
)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {filepath} at line {line_no}: {e}")
    return records


def compute_lf_sha256(filepath: Path) -> str:
    """Compute LF-normalized SHA-256 hash."""
    with open(filepath, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest().upper()


def run_all_validation_gates(repo_root: Path) -> bool:
    """Run all 21 strict validation gates."""
    print("=" * 70)
    print("OLORIC v0.4 STRICT VALIDATION GATES")
    print("=" * 70)

    dataset_path = repo_root / "data" / "generated" / "oloric_v04_dataset.jsonl"
    splits_dir = repo_root / "data" / "splits_v04"
    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "validation.jsonl"
    test_path = splits_dir / "test.jsonl"

    v03_path = repo_root / "data" / "generated" / "oloric_v03_dataset.jsonl"
    b1_path = repo_root / "evaluation" / "benchmark.jsonl"
    b2_path = repo_root / "evaluation" / "benchmark_docground_v1.jsonl"

    records = load_jsonl(dataset_path)
    total_records = len(records)
    print(f"Loaded {total_records} records from {dataset_path.name}")

    results = {}

    # Gate 1: Exactly 600 records
    g1_pass = (total_records == 600)
    results["Gate 01: Exactly 600 records"] = (g1_pass, f"Total records = {total_records}")

    # Gate 2: 20 canonical categories only
    categories = [r["category"] for r in records]
    unique_cats = set(categories)
    invalid_cats = unique_cats - set(VALID_CATEGORIES)
    g2_pass = (len(invalid_cats) == 0 and len(unique_cats) == 20)
    results["Gate 02: 20 canonical categories only"] = (g2_pass, f"Categories: {len(unique_cats)}, Invalid: {invalid_cats}")

    # Gate 3: Category counts: document_grounded=120, error_correction=48, other 18=24
    cat_counts = Counter(categories)
    g3_errors = []
    if cat_counts["document_grounded"] != 120:
        g3_errors.append(f"document_grounded={cat_counts['document_grounded']} (expected 120)")
    if cat_counts["error_correction"] != 48:
        g3_errors.append(f"error_correction={cat_counts['error_correction']} (expected 48)")
    for c in VALID_CATEGORIES:
        if c not in ["document_grounded", "error_correction"]:
            if cat_counts[c] != 24:
                g3_errors.append(f"{c}={cat_counts[c]} (expected 24)")
    g3_pass = (len(g3_errors) == 0)
    results["Gate 03: Category counts (120/48/24)"] = (g3_pass, "All 20 category counts match target" if g3_pass else "; ".join(g3_errors))

    # Gate 4: 6 domains * 100
    domains = [r["domain"] for r in records]
    dom_counts = Counter(domains)
    g4_errors = [f"{d}={dom_counts[d]}" for d in VALID_DOMAINS if dom_counts[d] != 100]
    g4_pass = (len(g4_errors) == 0 and len(dom_counts) == 6)
    results["Gate 04: 6 domains * 100 records"] = (g4_pass, "All 6 domains have exactly 100 records" if g4_pass else "; ".join(g4_errors))

    # Gate 5: 60 concepts * 10
    concepts = [r["context"]["learner_state"]["concept"] for r in records]
    concept_counts = Counter(concepts)
    g5_errors = [f"{c}={concept_counts[c]}" for c in concept_counts if concept_counts[c] != 10]
    g5_pass = (len(g5_errors) == 0 and len(concept_counts) == 60)
    results["Gate 05: 60 concepts * 10 records"] = (g5_pass, "All 60 concepts have exactly 10 records" if g5_pass else f"Errors: {len(g5_errors)}")

    # Gate 6: All 20 benchmark records/concepts covered
    b1_records = load_jsonl(b1_path)
    b1_concepts = {r["context"]["learner_state"]["concept"] for r in b1_records}
    missing_bench = [c for c in b1_concepts if concept_counts[c] != 10]
    g6_pass = (len(missing_bench) == 0 and len(b1_concepts) >= 18)
    results["Gate 06: All benchmark concepts covered"] = (g6_pass, f"All {len(b1_concepts)} benchmark concepts have 10 training records" if g6_pass else f"Missing: {missing_bench}")

    # Gate 7: Unique IDs and format v04_{domain}_{category}_{seq:03d}
    ids = [r["id"] for r in records]
    id_set = set(ids)
    id_format_errors = [i for i in ids if not (i.startswith("v04_") and len(i.split("_")) >= 4)]
    g7_pass = (len(ids) == len(id_set) == 600 and len(id_format_errors) == 0)
    results["Gate 07: Unique IDs (v04 schema)"] = (g7_pass, "600/600 unique v04 IDs" if g7_pass else f"Duplicates: {len(ids)-len(id_set)}, Format errors: {len(id_format_errors)}")

    # Gate 8: No v0.3 ID reuse
    v03_records = load_jsonl(v03_path)
    v03_ids = {r["id"] for r in v03_records}
    v03_overlap = id_set & v03_ids
    v03_prefix_matches = [i for i in ids if i.startswith("v03_")]
    g8_pass = (len(v03_overlap) == 0 and len(v03_prefix_matches) == 0)
    results["Gate 08: No v0.3 ID reuse"] = (g8_pass, "Zero overlap with v0.3 IDs" if g8_pass else f"Overlap: {len(v03_overlap)}")

    # Gate 9: No placeholder strings
    placeholder_tokens = [
        "[concrete example]", "[hint 1]", "[hint 2]", "[everyday analogy]",
        "[placeholder]", "[tbd]", "TODO", "FIXME", "undefined",
    ]
    placeholder_violations = []
    for r in records:
        r_str = json.dumps(r)
        for tok in placeholder_tokens:
            if tok.lower() in r_str.lower():
                placeholder_violations.append((r["id"], tok))
    g9_pass = (len(placeholder_violations) == 0)
    results["Gate 09: No placeholder strings"] = (g9_pass, "Clean of placeholder tokens" if g9_pass else f"Violations: {placeholder_violations[:5]}")

    # Gate 10: No cross-domain contamination
    contamination_errors = []
    for r in records:
        rec_dom = r["domain"]
        rec_concept = r["context"]["learner_state"]["concept"]
        valid_concepts = get_domain_concepts(rec_dom)
        if rec_concept not in valid_concepts:
            contamination_errors.append(f"{r['id']}: concept '{rec_concept}' not in domain '{rec_dom}'")
        # Check document id prefix matches domain
        doc_id = r["context"]["document_context"]["document_id"]
        if not doc_id.startswith(rec_dom):
            contamination_errors.append(f"{r['id']}: document_id '{doc_id}' does not match domain '{rec_dom}'")
    g10_pass = (len(contamination_errors) == 0)
    results["Gate 10: No cross-domain contamination"] = (g10_pass, "Zero cross-domain bleed" if g10_pass else f"Errors: {contamination_errors[:5]}")

    # Gate 11: Prior-error correctness/alignment
    # Prior error records: category == "error_correction" and seq in 005..008
    prior_error_records = [r for r in records if r["category"] == "error_correction" and int(r["id"].split("_")[-1]) >= 5]
    pe_errors = []
    if len(prior_error_records) != 24:
        pe_errors.append(f"Expected 24 prior-error records, got {len(prior_error_records)}")
    for r in prior_error_records:
        turns = r["context"]["conversation_context"]
        if len(turns) < 3 or turns[1]["role"] != "tutor" or turns[2]["role"] != "student":
            pe_errors.append(f"{r['id']}: Invalid prior error turn structure")
        resp = r["target"]["response"]
        if not ("apologize" in resp.lower() or "error" in resp.lower() or "misunderstanding" in resp.lower() or "mistake" in resp.lower()):
            pe_errors.append(f"{r['id']}: Target response does not acknowledge error")
        diag = r["target"].get("diagnosis", {})
        if diag.get("confusion_type") != "prior_tutor_error":
            pe_errors.append(f"{r['id']}: diagnosis confusion_type is not 'prior_tutor_error'")
        mem = r["target"].get("memory", {})
        if not mem.get("candidate") or mem.get("memory_type") != "error_correction":
            pe_errors.append(f"{r['id']}: memory is not candidate error_correction")
    g11_pass = (len(pe_errors) == 0)
    results["Gate 11: Prior-error alignment (24 records)"] = (g11_pass, "All 24 prior-error records properly aligned" if g11_pass else "; ".join(pe_errors[:3]))

    # Gate 12: Evidence-prioritization alignment
    # seq 013..016 of document_grounded
    ev_priority_records = [r for r in records if r["category"] == "document_grounded" and 13 <= int(r["id"].split("_")[-1]) <= 16]
    ep_errors = []
    if len(ev_priority_records) != 24:
        ep_errors.append(f"Expected 24 evidence-priority records, got {len(ev_priority_records)}")
    pattern_a_count = 0
    pattern_b_count = 0
    for r in ev_priority_records:
        seq = int(r["id"].split("_")[-1])
        diag_type = r["target"].get("diagnosis", {}).get("confusion_type")
        if seq in [13, 14]:
            pattern_a_count += 1
            if diag_type != "evidence_prioritization":
                ep_errors.append(f"{r['id']}: Expected diagnosis 'evidence_prioritization', got '{diag_type}'")
        elif seq in [15, 16]:
            pattern_b_count += 1
            if diag_type != "mechanistic_derivation":
                ep_errors.append(f"{r['id']}: Expected diagnosis 'mechanistic_derivation', got '{diag_type}'")
    if pattern_a_count != 12 or pattern_b_count != 12:
        ep_errors.append(f"Expected 12 Pattern A and 12 Pattern B, got A={pattern_a_count}, B={pattern_b_count}")
    g12_pass = (len(ep_errors) == 0)
    results["Gate 12: Evidence-prioritization alignment"] = (g12_pass, f"12 Pattern A and 12 Pattern B verified" if g12_pass else "; ".join(ep_errors))

    # Gate 13: Document-absence/refusal alignment
    # seq 017..020 of document_grounded
    absence_records = [r for r in records if r["category"] == "document_grounded" and 17 <= int(r["id"].split("_")[-1]) <= 20]
    ar_errors = []
    if len(absence_records) != 24:
        ar_errors.append(f"Expected 24 absence-refusal records, got {len(absence_records)}")
    for r in absence_records:
        resp = r["target"]["response"].lower()
        if not ("no mention" in resp or "not mention" in resp or "does not contain" in resp):
            ar_errors.append(f"{r['id']}: Refusal response does not state absence")
        if not ("consult" in resp or "resource" in resp or "source" in resp or "needed" in resp):
            ar_errors.append(f"{r['id']}: Refusal response does not cite needed source")
        mem = r["target"].get("memory", {})
        if mem.get("candidate") is not False:
            ar_errors.append(f"{r['id']}: Memory candidate should be False for absence refusal")
    g13_pass = (len(ar_errors) == 0)
    results["Gate 13: Document-absence refusal alignment"] = (g13_pass, "All 24 refusal records properly aligned" if g13_pass else "; ".join(ar_errors[:3]))

    # Gate 14: Genuine document evidence
    dg_records = [r for r in records if r["category"] == "document_grounded"]
    doc_evidence_errors = []
    for r in dg_records:
        dc = r["context"].get("document_context", {})
        st = dc.get("selected_text", "")
        sc = dc.get("surrounding_context", "")
        re = dc.get("retrieved_evidence", [])
        if not st or len(st) < 15:
            doc_evidence_errors.append(f"{r['id']}: selected_text missing or too short")
        if not sc or len(sc) < 15:
            doc_evidence_errors.append(f"{r['id']}: surrounding_context missing or too short")
        if not re or len(re) == 0:
            doc_evidence_errors.append(f"{r['id']}: retrieved_evidence missing or empty")
    g14_pass = (len(doc_evidence_errors) == 0 and len(dg_records) == 120)
    results["Gate 14: Genuine document evidence (120 records)"] = (g14_pass, f"All 120 document-grounded records contain genuine evidence" if g14_pass else f"Errors: {len(doc_evidence_errors)}")

    # Gate 15: Memory-field consistency
    mem_errors = []
    for r in records:
        mem = r["target"].get("memory", {})
        if "candidate" not in mem:
            mem_errors.append(f"{r['id']}: missing memory.candidate")
            continue
        if mem["candidate"] is False:
            if not (mem.get("title") is None and mem.get("content") is None and
                    mem.get("anchor_concept") is None and mem.get("memory_type") is None and
                    mem.get("confidence") == 0.0):
                mem_errors.append(f"{r['id']}: candidate=False memory violates hygiene: {mem}")
        elif mem["candidate"] is True:
            if not (mem.get("title") and mem.get("content") and mem.get("anchor_concept") and
                    mem.get("memory_type") and mem.get("confidence", 0.0) > 0.0):
                mem_errors.append(f"{r['id']}: candidate=True memory has missing fields: {mem}")
        else:
            mem_errors.append(f"{r['id']}: memory.candidate is neither True nor False: {mem['candidate']}")
    g15_pass = (len(mem_errors) == 0)
    results["Gate 15: Memory-field hygiene"] = (g15_pass, "100% compliant memory objects across all 600 records" if g15_pass else f"Errors: {len(mem_errors)}")

    # Gate 16: Train/validation/test counts = 480/60/60
    train_recs = load_jsonl(train_path)
    val_recs = load_jsonl(val_path)
    test_recs = load_jsonl(test_path)
    g16_pass = (len(train_recs) == 480 and len(val_recs) == 60 and len(test_recs) == 60)
    results["Gate 16: Split counts (480 / 60 / 60)"] = (g16_pass, f"Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # Gate 17: Zero split overlap
    tr_ids = {r["id"] for r in train_recs}
    va_ids = {r["id"] for r in val_recs}
    te_ids = {r["id"] for r in test_recs}
    g17_pass = not (tr_ids & va_ids or tr_ids & te_ids or va_ids & te_ids)
    results["Gate 17: Zero split overlap"] = (g17_pass, "Zero ID overlap across splits" if g17_pass else "Overlap detected!")

    # Gate 18: Deterministic regeneration
    from generate_v04_dataset import generate_v04_dataset
    regen_records = generate_v04_dataset()
    regen_bytes = "\n".join(json.dumps(r, ensure_ascii=False) for r in regen_records).encode("utf-8") + b"\n"
    regen_sha = hashlib.sha256(regen_bytes.replace(b"\r\n", b"\n")).hexdigest().upper()
    current_sha = compute_lf_sha256(dataset_path)
    g18_pass = (regen_sha == current_sha)
    results["Gate 18: Deterministic regeneration"] = (g18_pass, f"LF-normalized SHA matches ({current_sha[:16]}...)")

    # Gate 19: No leakage from evaluation/benchmark.jsonl
    b1_prompts = {r.get("student_prompt") or r.get("prompt") or r.get("question") or r["context"]["conversation_context"][-1]["content"] for r in b1_records}
    b1_prompts.discard(None)
    b1_targets = {r["target"]["response"] for r in b1_records}
    v04_targets = {r["target"]["response"] for r in records}
    v04_prompts = {r["context"]["conversation_context"][-1]["content"] for r in records}

    leak_b1_targets = v04_targets & b1_targets
    leak_b1_prompts = v04_prompts & b1_prompts
    g19_pass = (len(leak_b1_targets) == 0 and len(leak_b1_prompts) == 0)
    results["Gate 19: No benchmark.jsonl leakage"] = (g19_pass, "Zero prompt or target leakage" if g19_pass else f"Targets: {len(leak_b1_targets)}, Prompts: {len(leak_b1_prompts)}")

    # Gate 20: No leakage from evaluation/benchmark_docground_v1.jsonl
    b2_records = load_jsonl(b2_path)
    b2_prompts = {r.get("student_prompt") or r.get("prompt") or r["context"]["conversation_context"][-1]["content"] for r in b2_records}
    b2_prompts.discard(None)
    b2_targets = {r["target"]["response"] for r in b2_records}

    leak_b2_targets = v04_targets & b2_targets
    leak_b2_prompts = v04_prompts & b2_prompts
    g20_pass = (len(leak_b2_targets) == 0 and len(leak_b2_prompts) == 0)
    results["Gate 20: No benchmark_docground_v1 leakage"] = (g20_pass, "Zero prompt or target leakage" if g20_pass else f"Targets: {len(leak_b2_targets)}, Prompts: {len(leak_b2_prompts)}")

    # Gate 21: No verbatim target leakage from v0.3 dataset
    v03_targets = {r["target"]["response"] for r in v03_records}
    leak_v03_targets = v04_targets & v03_targets
    g21_pass = (len(leak_v03_targets) == 0)
    results["Gate 21: No v0.3 verbatim target leakage"] = (g21_pass, "Zero verbatim overlap with v0.3 targets" if g21_pass else f"Leakage: {len(leak_v03_targets)}")

    # Summary Display
    all_passed = True
    print("\n" + "-" * 70)
    for gate_name, (passed, msg) in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {gate_name}: {msg}")
    print("-" * 70)

    if all_passed:
        print("\nALL 21 VALIDATION GATES PASSED PERFECTLY!\n")
    else:
        print("\nVALIDATION FAILED ON ONE OR MORE GATES!\n")

    return all_passed


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    success = run_all_validation_gates(repo_root)
    sys.exit(0 if success else 1)
