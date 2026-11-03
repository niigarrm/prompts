#!/usr/bin/env python3
"""Random survey of the unarXive open subset: 300 files, at most 8 papers each, seed 42 (Table 2.1)."""

import argparse
import glob
import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import spacy

INPUT_DIR = "unarxive_data"

TOTAL_PAPERS_ESTIMATE = 165_000

YYMM_RE = re.compile(r"arXiv_src_(\d{2})(\d{2})_")
CITE_RE = re.compile(r"\{\{cite:[^}]*\}\}")

random.seed(42)


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def year_from_filename(path):
    m = YYMM_RE.search(os.path.basename(path))
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def build_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    return nlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument(
        "--files", type=int, default=300,
        help="How many files to sample at random across the archive.",
    )
    parser.add_argument(
        "--papers-per-file", type=int, default=8,
        help="How many papers to read from each sampled file.",
    )
    args = parser.parse_args()

    all_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**", "*.jsonl*"), recursive=True)
    )
    if not all_files:
        print(f"No .jsonl files found in {args.input_dir!r}.", file=sys.stderr)
        return 1

    print(f"Archive contains {len(all_files):,} files.")
    n = min(args.files, len(all_files))
    sample_files = random.sample(all_files, n)
    print(f"Sampling {n} of them at random, "
          f"up to {args.papers_per_file} papers each.\n")

    nlp = build_nlp()

    disciplines = Counter()
    cs_by_year = Counter()
    all_by_year = Counter()

    cs_papers = 0
    cs_sentences = 0
    cs_cite_sentences = 0
    cs_empty_body = 0
    cs_junk_first_para = 0
    sentences_per_year = defaultdict(list)

    papers_read = 0

    for path in sample_files:
        year = year_from_filename(path)
        try:
            with open_maybe_gzip(path) as f:
                for i, line in enumerate(f):
                    if i >= args.papers_per_file:
                        break
                    try:
                        paper = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    papers_read += 1
                    disc = paper.get("discipline")
                    disciplines[disc] += 1
                    all_by_year[year] += 1

                    if disc != "Computer Science":
                        continue

                    cs_papers += 1
                    cs_by_year[year] += 1

                    paras = [
                        e.get("text", "")
                        for e in paper.get("body_text", [])
                        if isinstance(e, dict)
                        and e.get("content_type") == "paragraph"
                        and e.get("text", "").strip()
                    ]
                    if not paras:
                        cs_empty_body += 1
                        continue

                    first = paras[0].strip()
                    if len(first) < 40 or not first[0].isalpha():
                        cs_junk_first_para += 1

                    n_sents = 0
                    for doc in nlp.pipe(paras, batch_size=50):
                        for sent in doc.sents:
                            text = sent.text.strip()
                            if len(text.split()) < 5:
                                continue
                            n_sents += 1
                            if CITE_RE.search(text):
                                cs_cite_sentences += 1

                    cs_sentences += n_sents
                    if year:
                        sentences_per_year[year].append(n_sents)
        except OSError:
            continue

    print(f"Read {papers_read:,} papers.\n")

    print("--- Discipline distribution (random sample) ---")
    for disc, cnt in disciplines.most_common():
        print(f"  {str(disc):26s} {cnt:6,}  ({100.0 * cnt / papers_read:5.1f}%)")

    cs_share = cs_papers / papers_read if papers_read else 0
    print(f"\n  Computer Science share: {100 * cs_share:.1f}%")

    print("\n--- CS papers by year ---")
    for year in sorted(all_by_year):
        total = all_by_year[year]
        cs = cs_by_year.get(year, 0)
        share = 100.0 * cs / total if total else 0.0
        med = ""
        if sentences_per_year.get(year):
            vals = sorted(sentences_per_year[year])
            med = f"  median {vals[len(vals) // 2]:4d} sents/paper"
        print(f"  {year}  papers {total:5,}  CS {cs:4,}  ({share:5.1f}%){med}")

    print("\n--- CS text quality ---")
    if cs_papers:
        print(f"  CS papers sampled          {cs_papers:,}")
        print(f"  with empty body_text       {cs_empty_body:,} "
              f"({100.0 * cs_empty_body / cs_papers:.1f}%)")
        print(f"  first para looks like junk {cs_junk_first_para:,} "
              f"({100.0 * cs_junk_first_para / cs_papers:.1f}%)")
        print(f"  mean sentences per paper   {cs_sentences / cs_papers:.0f}")

    if cs_sentences:
        pos_rate = 100.0 * cs_cite_sentences / cs_sentences
        print(f"\n  natural positive rate      {pos_rate:.1f}%")
        print("  (published work reports roughly 10-30%)")

    print("\n--- Projection for the full CS extraction ---")
    est_cs_papers = int(TOTAL_PAPERS_ESTIMATE * cs_share)
    per_paper = cs_sentences / cs_papers if cs_papers else 0
    est_sentences = int(est_cs_papers * per_paper)
    print(f"  estimated CS papers        ~{est_cs_papers:,}")
    print(f"  sentences per CS paper     ~{per_paper:.0f}")
    print(f"  estimated raw CS sentences ~{est_sentences:,}")

    if cs_sentences:
        est_pos = int(est_sentences * cs_cite_sentences / cs_sentences)
        print(f"  estimated positives        ~{est_pos:,}")
        print(f"  balanced set would be      ~{2 * est_pos:,}")

    print("\nNote: these are pre-filter counts. Quality filtering in")
    print("extract_and_label.py will remove some of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
