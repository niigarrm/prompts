#!/usr/bin/env python3
"""Checks that the holes-kept extraction is a strict superset of the default one and counts holes per class.

Usage:
    python3 verify_variants.py default.jsonl holes_kept.jsonl
"""

import json
import re
import sys
from collections import Counter

ACK_RE = re.compile(
    r"^(?:We|I|The\s+authors?)\b[^.]{0,60}?\b(?:are\s+grateful|thanks?|"
    r"would\s+like\s+to\s+thank|acknowledges?|gratefully)\b", re.IGNORECASE)
THEOREM_RE = re.compile(
    r"^(?:Lemma|Theorem|Corollary|Proposition|Definition|Claim|Remark|Fact|"
    r"Observation|Example|Assumption|Algorithm)\s+\d+", re.IGNORECASE)
BIBKEY_RE = re.compile(r"\b[A-Za-z][a-z\-]+(?:19|20)\d{2}[a-z]{3,}\b")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest(path):
    try:
        with open(path + ".manifest.txt", encoding="utf-8") as f:
            header = f.readline().strip()
            files = [l.strip() for l in f if l.strip()]
    except OSError:
        return None, []
    return header, files


def key(r):
    return (r["paper_id"], r["paragraph_index"], r["sentence_index"])


def main(default_path, holes_path):
    problems = []

    hd, fd = manifest(default_path)
    hh, fh = manifest(holes_path)
    print("--- Manifests ---")
    print(f"  default   : {hd}")
    print(f"  keep-holes: {hh}")
    if hd is None or hh is None:
        problems.append("manifest missing")
    else:
        strip = lambda h: re.sub(r"\s*keep_holes=\w+", "", h)
        if strip(hd) != strip(hh):
            problems.append("manifest headers differ beyond keep_holes")
        if "script_version" not in hd:
            problems.append("default manifest has no script_version "
                            "(pre-v4 file)")
        if fd != fh:
            problems.append("file lists differ")
        print(f"  file lists identical: {fd == fh} ({len(fd)} files)")

    d = load(default_path)
    h = load(holes_path)
    kd = {key(r): r for r in d}
    kh = {key(r): r for r in h}
    print("\n--- Sizes ---")
    print(f"  default    {len(d):>9,} rows, {len(kd):,} unique keys")
    print(f"  keep-holes {len(h):>9,} rows, {len(kh):,} unique keys")
    if len(kd) != len(d) or len(kh) != len(h):
        problems.append("duplicate keys inside a file")

    missing = [k for k in kd if k not in kh]
    text_diff = [k for k in kd if k in kh and
                 (kd[k]["text"] != kh[k]["text"] or
                  kd[k]["label"] != kh[k]["label"])]
    extra = [k for k in kh if k not in kd]
    print("\n--- Superset check ---")
    print(f"  default rows missing from keep-holes : {len(missing)}")
    print(f"  shared rows with different text/label: {len(text_diff)}")
    print(f"  rows only in keep-holes              : {len(extra)}")
    if missing:
        problems.append(f"{len(missing)} default rows not in keep-holes")
    if text_diff:
        problems.append(f"{len(text_diff)} shared rows differ")
    if not extra:
        problems.append("keep-holes adds nothing (not a STRICT superset)")

    extra_no_hole = [k for k in extra if not kh[k].get("hole")]
    default_with_hole = [k for k in kd if kd[k].get("hole")]
    print(f"  extra rows without a hole label      : {len(extra_no_hole)}")
    print(f"  default rows carrying a hole label   : {len(default_with_hole)}")
    if extra_no_hole:
        problems.append("extra rows without hole label")
    if default_with_hole:
        problems.append("default file contains hole rows")

    pos = [r for r in h if r["label"] == 1]
    neg = [r for r in h if r["label"] == 0]
    hp = Counter(r["hole"] for r in pos if r.get("hole"))
    hn = Counter(r["hole"] for r in neg if r.get("hole"))
    tp, tn = sum(hp.values()), sum(hn.values())
    print("\n--- Holes per class (keep-holes file = quality-passing population) ---")
    print(f"  positives {len(pos):,}, with hole {tp:,} "
          f"({100.0*tp/max(len(pos),1):.2f}%)")
    for k, v in hp.most_common():
        print(f"      {k:22s} {v:,}")
    print(f"  negatives {len(neg):,}, with hole {tn:,} "
          f"({100.0*tn/max(len(neg),1):.2f}%)")
    for k, v in hn.most_common():
        print(f"      {k:22s} {v:,}")
    if tp + tn:
        print(f"  share of all holes that sit in positives: "
              f"{100.0*tp/(tp+tn):.1f}%")
        print(f"  positive rate: default {100.0*sum(r['label'] for r in d)/max(len(d),1):.2f}%  "
              f"keep-holes {100.0*len(pos)/max(len(h),1):.2f}%")

    print("\n--- Residue checks on keep-holes file ---")
    checks = {
        "newline inside sentence": sum("\n" in r["text"] for r in h),
        "'FORMULA 's' / 'REF 's'": sum(bool(re.search(r"\b(FORMULA|REF) 's\b", r["text"])) for r in h),
        "bibtex key residue": sum(bool(BIBKEY_RE.search(r["text"])) for r in h),
        "acknowledgement": sum(bool(ACK_RE.match(r["text"])) for r in h),
        "theorem fragment": sum(bool(THEOREM_RE.match(r["text"])) for r in h),
        "double comma ',,'": sum(",," in r["text"] for r in h),
        "sentence starts 'E.g.' fragment": sum(r["text"].startswith("E.g.") and len(r["text"].split()) < 8 for r in h),
        "'et al' survived": sum("et al" in r["text"] for r in h),
    }
    for name, n in checks.items():
        flag = "   " if n == 0 else " <-"
        print(f"  {name:34s} {n:,}{flag}")

    print("\n--- Verdict ---")
    if problems:
        for p_ in problems:
            print(f"  FAIL  {p_}")
        return 1
    print("  OK    keep-holes output is a strict superset of the default output")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
