#!/usr/bin/env python3
"""
Scan model-generated output (raw text or JSONL) for unresolved template
placeholders. Intended as the post-smoke-run gate:
"no placeholder strings in generated output".

Accepts:
  - a plain text file (e.g. one generated response)
  - a JSONL file (every string field of every record is scanned, including
    evaluate.py's responses.jsonl raw_response / raw_generated_text fields)

Reuses the exact error patterns from scripts/validate_dataset_content.py
(bracket slots, angle placeholders, generic markers). Same allowlist rules
apply (e.g. [H+] ion notation is legitimate).

Exit code: 0 = PASS (no placeholders found), 1 = FAIL.

Usage (from repo root):
  PYTHONIOENCODING=utf-8 python scripts/scan_smoke_output.py --output-path <file>
"""
import argparse
import json
import re
import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo_root, "scripts"))

from validate_dataset_content import BRACKET, ANGLE, GENERIC, ALLOWLIST, walk_strings  # noqa: E402

# Extended ion allowlist for generated output: covers multi-element ions such as
# [HCO3-] (bicarbonate), [CH3COO-] (acetate) that the dataset validator's
# single-element pattern does not. Still requires a charge superscript, so
# word-like slots ([reason], [explanation], [hint 1]) remain errors.
EXTRA_ION_ALLOWLIST = [
    re.compile(r"^\[[A-Za-z][A-Za-z0-9]{0,5}[0-9]*[\u207a\u207b\u00b2\u00b3]{1,2}\]$"),
]


def scan_text(text: str, label: str) -> list:
    """Return list of placeholder hits in a single string."""
    hits = []
    for m in BRACKET.finditer(text):
        tok = m.group(0)
        if any(p.match(tok) for p in ALLOWLIST) or any(p.match(tok) for p in EXTRA_ION_ALLOWLIST):
            continue
        hits.append(f"{label}: unresolved slot {tok!r}")
    for m in ANGLE.finditer(text):
        hits.append(f"{label}: angle placeholder {m.group(0)!r}")
    gm = GENERIC.search(text)
    if gm:
        hits.append(f"{label}: marker {gm.group(0)!r}")
    return hits


def scan_file(path: str) -> list:
    hits = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if path.endswith(".jsonl"):
        for ln, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                hits.append(f"line{ln}: not valid JSON — treated as raw text")
                hits.extend(scan_text(line, f"line{ln}"))
                continue
            for spath, s in walk_strings(rec):
                hits.extend(scan_text(s, f"line{ln} [{spath}]"))
    else:
        hits.extend(scan_text(content, os.path.basename(path)))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Placeholder scan for generated model output")
    ap.add_argument("--output-path", required=True,
                    help="Generated output file (.txt for raw text, .jsonl for records)")
    args = ap.parse_args()

    print("=" * 64)
    print(f"SMOKE OUTPUT PLACEHOLDER SCAN: {args.output_path}")
    print("=" * 64)
    hits = scan_file(args.output_path)
    if hits:
        print(f"\nHITS ({len(hits)}):")
        for h in hits[:20]:
            print(f"  X {h}")
        if len(hits) > 20:
            print(f"  ... and {len(hits) - 20} more")
        print("\nRESULT: FAIL")
        return 1
    print("placeholder occurrences: 0")
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
