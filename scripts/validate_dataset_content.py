#!/usr/bin/env python3
"""
Strict content-quality (placeholder) validator for Oloric datasets.

Checks EVERY string field of every record for unresolved template placeholders:

  ERROR (fails validation):
    - Square-bracket slots: [reason], [explanation], [hint 1], [clear, concise definition], ...
      with an allowlist for legitimate notation:
        * chemistry concentration/ions: [H+], [OH-], [e-], [Na+], [Ca2+] ...
        * numeric intervals / matrix elements: [0.4, 0.6], [1,2]
    - Angle-bracket placeholders: <explanation>, <TODO>, ...
    - Generic markers: "placeholder", TODO, TBD, FIXME, LOREM, "to be filled",
      "to be determined", "fill in the blank", "insert ... here"

  WARNING (human review, not fatal):
    - Literal (unbracketed) template phrasing such as "question 1", "hint 1",
      "clear, concise definition", "everyday analogy" — these can be legitimate
      English ("Here's a concrete example:"), so they are reported but not failed.
    - Memory-shape inconsistencies (candidate=True but missing title/content/anchor;
      candidate=False but filled content).

Exit code: 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import re
import sys
from collections import Counter

BRACKET = re.compile(r"\[[^\[\]\n]{1,80}\]")
ANGLE = re.compile(r"<[a-zA-Z_][a-zA-Z0-9_\-\s]{0,60}>")
GENERIC = re.compile(
    r"\bplaceholder\b|\bTODO\b|\bTBD\b|\bFIXME\b|\bLOREM\b"
    r"|to be filled|to be determined|fill in the blank|insert [^\n]{0,40} here",
    re.IGNORECASE,
)

ALLOWLIST = [
    re.compile(r"^\[H\u207a\]$"),            # [H+]
    re.compile(r"^\[OH\u207b\]$"),           # [OH-]
    re.compile(r"^\[e\u207b\]$"),            # [e-]
    re.compile(r"^\[H\u2083O\u207a\]$"),     # [H3O+]
    re.compile(r"^\[[A-Z][a-z]?[0-9]*[\u207a\u207b\u00b2\u00b3]{1,2}\]$"),  # [Na+], [Ca2+], ions
    re.compile(r"^\[\s*-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)*\s*\]$"),        # [0.4, 0.6], [1,2]
    # Legitimate mathematical notation:
    # 1. Composite function notation: [f(g(x))], [g(x)], [f(x)]
    re.compile(r"^\[[a-zA-Z]\([a-zA-Z0-9_\(\)]+\)\]$"),
    # 2. Definite integral evaluation brackets / single-variable algebraic terms: [3x²/2], [x²], [2x]
    re.compile(r"^\[[-+]?\d*[a-zA-Z](\u00b2|\u00b3|\u2074|\u2075|\u2076|\u2077|\u2078|\u2079|\u2070|\^\d+)?(\s*/\s*\d+)?\]$"),
]

LITERAL_SUSPECTS = [
    "clear, concise definition", "detailed explanation", "everyday analogy",
    "specific evidence", "key evidence", "key point about concept",
    "key point to remember", "question 1", "question 2", "question 3",
    "hint 1", "hint 2", "hint 3",
]


def walk_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def validate_file(path):
    """Returns (passed, errors, warnings, stats)."""
    errors = []
    warnings = []
    occ_by_field = Counter()
    occ_samples = Counter()
    affected_records = set()
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id", f"line{ln}")
            for spath, s in walk_strings(rec):
                top_field = spath.split(".")[0].split("[")[0]
                for m in BRACKET.finditer(s):
                    tok = m.group(0)
                    if any(p.match(tok) for p in ALLOWLIST):
                        continue
                    total += 1
                    occ_by_field[top_field] += 1
                    occ_samples[tok] += 1
                    affected_records.add(rid)
                    errors.append(
                        f"{rid} [{spath}]: unresolved slot {tok!r}"
                    )
                for m in ANGLE.finditer(s):
                    tok = m.group(0)
                    total += 1
                    occ_by_field[top_field] += 1
                    occ_samples[tok] += 1
                    affected_records.add(rid)
                    errors.append(f"{rid} [{spath}]: angle placeholder {tok!r}")
                gm = GENERIC.search(s)
                if gm:
                    total += 1
                    occ_by_field[top_field] += 1
                    occ_samples[gm.group(0)] += 1
                    affected_records.add(rid)
                    errors.append(f"{rid} [{spath}]: marker {gm.group(0)!r}")
                low = s.lower()
                for phr in LITERAL_SUSPECTS:
                    if phr.lower() in low and f"[{phr}" not in s:
                        warnings.append(f"{rid} [{spath}]: literal phrase {phr!r} (review)")

            # memory-shape hygiene warnings
            mem = rec.get("target", {}).get("memory", {})
            if isinstance(mem, dict):
                if mem.get("candidate"):
                    for k in ("title", "content", "anchor_concept"):
                        if not mem.get(k):
                            warnings.append(f"{rid}: memory.candidate=True but {k} missing")
                else:
                    filled = [k for k in ("title", "content", "anchor_concept", "memory_type")
                              if mem.get(k)]
                    if filled:
                        warnings.append(f"{rid}: memory.candidate=False but {filled} filled")
            uc = rec.get("target", {}).get("understanding_check", {})
            if isinstance(uc, dict) and uc.get("required") and not uc.get("question"):
                warnings.append(f"{rid}: understanding_check.required=True but question missing")

    stats = {
        "file": path,
        "placeholder_occurrences": total,
        "affected_records": len(affected_records),
        "by_top_field": dict(occ_by_field),
        "top_slot_strings": occ_samples.most_common(15),
    }
    return len(errors) == 0, errors, warnings, stats


def main():
    ap = argparse.ArgumentParser(description="Strict placeholder/content validator")
    ap.add_argument("--data-path", required=True, help="JSONL dataset to validate")
    args = ap.parse_args()

    passed, errors, warnings, stats = validate_file(args.data_path)
    print("=" * 64)
    print(f"CONTENT VALIDATION: {args.data_path}")
    print("=" * 64)
    print(f"placeholder occurrences : {stats['placeholder_occurrences']}")
    print(f"affected records        : {stats['affected_records']}")
    print(f"by top field            : {stats['by_top_field']}")
    print(f"top slot strings        : {stats['top_slot_strings']}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  X {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  ! {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    print()
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
