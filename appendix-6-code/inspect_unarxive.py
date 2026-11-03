#!/usr/bin/env python3
"""Prints the structure of the unarXive files and a first estimate of the Computer Science share."""

import glob
import gzip
import json
import os
from collections import Counter

INPUT_DIR = "/Users/nigarmamedova/Desktop/thesis/unarxive_data"

SAMPLE_SIZE = 2000


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def find_files(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.jsonl*"), recursive=True))


def show_first_paper(path):
    with open_maybe_gzip(path) as f:
        paper = json.loads(f.readline())

    print("Top-level keys :", list(paper.keys()))
    print("discipline     :", paper.get("discipline"))

    meta = paper.get("metadata") or {}
    print("metadata keys  :", list(meta.keys()))
    print("categories     :", meta.get("categories"))

    body = paper.get("body_text", [])
    print(f"\nbody_text has {len(body)} entries.")

    content_types = Counter(
        e.get("content_type") for e in body if isinstance(e, dict)
    )
    print("content_type counts in this paper:", dict(content_types))

    print("\nFirst paragraph-type entry:\n")
    for entry in body:
        if isinstance(entry, dict) and entry.get("content_type") == "paragraph":
            text = entry.get("text", "")
            print("  content_type:", entry.get("content_type"))
            print("  text (first 500 chars):")
            print("  " + repr(text[:500]))
            print("\n  contains '{{cite:' marker:", "{{cite:" in text)
            break
    else:
        print("  WARNING: no entry with content_type == 'paragraph' found.")


def estimate_disciplines(files, sample_size):
    counts = Counter()
    read = 0

    for path in files:
        if read >= sample_size:
            break
        with open_maybe_gzip(path) as f:
            for line in f:
                if read >= sample_size:
                    break
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                counts[paper.get("discipline")] += 1
                read += 1

    return counts, read


def main():
    files = find_files(INPUT_DIR)

    if not files:
        print(f"No .jsonl files found in {INPUT_DIR!r}.")
        print("Did you extract the archive and set INPUT_DIR correctly?")
        print("  mkdir unarxive_data")
        print("  tar -xJf unarxive_open.tar.xz -C unarxive_data")
        return

    print(f"Found {len(files)} file(s) under {INPUT_DIR!r}.")
    if len(files) == 1:
        print("WARNING: only one file found. You may have the sample, not the archive.")
    print(f"\nReading the first paper from: {files[0]}\n")

    show_first_paper(files[0])

    print(f"\n--- Discipline sample (up to {SAMPLE_SIZE} papers) ---\n")
    counts, read = estimate_disciplines(files, SAMPLE_SIZE)
    for discipline, n in counts.most_common():
        share = 100.0 * n / read if read else 0.0
        print(f"  {str(discipline):30s} {n:6d}  ({share:5.1f}%)")

    cs = counts.get("Computer Science", 0)
    if read:
        print(f"\nComputer Science share in sample: {100.0 * cs / read:.1f}%")
        print("The open subset holds ~165,000 papers, so estimated CS papers:")
        print(f"  ~{int(165000 * cs / read):,}")

    print("\n--- Checklist before moving to Stage 1 ---")
    print("  [ ] More than one file found")
    print("  [ ] 'discipline' prints a real value, and 'Computer Science' appears")
    print("  [ ] Paragraph text contains inline {{cite:...}} markers")
    print("\nIf any of these fail, send the output above before running the pilot.")


if __name__ == "__main__":
    main()
