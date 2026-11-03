#!/usr/bin/env python3
"""Draws the blind review sheet (75 + 75 sentences, seed 7) and scores one annotator against the rule (Chapter 3)."""

import argparse
import json
import random
import re
import sys
from collections import Counter

INPUT = "sentences_labelled.jsonl"
SHEET = "review_sheet.txt"
KEY = "review_key.json"

N_POSITIVE = 75
N_NEGATIVE = 75
SEED = 7

ANSWER_RE = re.compile(r"^ANSWER:\s*([YyNn])?\s*$")
ID_RE = re.compile(r"^### (\d+)")
NOTES_RE = re.compile(r"^NOTES:\s*(.*)$")


def make_sheet(args):
    positives, negatives = [], []
    rng = random.Random(args.seed)

    n_pos_seen = n_neg_seen = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec["label"] == 1:
                n_pos_seen += 1
                if len(positives) < args.n_positive:
                    positives.append(rec)
                else:
                    j = rng.randrange(n_pos_seen)
                    if j < args.n_positive:
                        positives[j] = rec
            else:
                n_neg_seen += 1
                if len(negatives) < args.n_negative:
                    negatives.append(rec)
                else:
                    j = rng.randrange(n_neg_seen)
                    if j < args.n_negative:
                        negatives[j] = rec

    sample = positives + negatives
    rng.shuffle(sample)

    key = {}
    with open(args.sheet, "w", encoding="utf-8") as out:
        out.write("STAGE 2 REVIEW SHEET\n")
        out.write("=" * 70 + "\n\n")
        out.write("For each sentence, decide WITHOUT seeing the label:\n")
        out.write("  Would this sentence, in a published paper, need a\n")
        out.write("  citation to support it?\n\n")
        out.write("Write Y or N after ANSWER:. Use NOTES: for anything odd --\n")
        out.write("broken sentence, leftover bracket, bad split, over-scrubbed\n")
        out.write("text where a name was removed and the sentence no longer\n")
        out.write("makes sense.\n\n")
        out.write("Leave ANSWER blank if you genuinely cannot tell.\n")
        out.write("=" * 70 + "\n\n")

        for i, rec in enumerate(sample, start=1):
            sid = f"{i:03d}"
            key[sid] = {
                "label": rec["label"],
                "paper_id": rec.get("paper_id"),
                "year": rec.get("year"),
                "text": rec["text"],
            }
            out.write(f"### {sid}\n")
            out.write(f"TEXT:   {rec['text']}\n")
            out.write("ANSWER: \n")
            out.write("NOTES:  \n\n")

    with open(args.key, "w", encoding="utf-8") as kf:
        json.dump(key, kf, ensure_ascii=False, indent=1)

    print(f"Wrote {len(sample)} sentences to {args.sheet}")
    print(f"  {len(positives)} labelled positive, {len(negatives)} negative")
    print(f"  (order shuffled, labels hidden)")
    print(f"Key written to {args.key} -- do not read it before reviewing.")
    print(f"\nCorpus seen: {n_pos_seen:,} positive, {n_neg_seen:,} negative")
    print(f"\nNow open {args.sheet} and fill in the ANSWER lines.")
    print("Budget about two hours. Do it in two sittings if you need to.")


def score_sheet(args):
    with open(args.key, encoding="utf-8") as kf:
        key = json.load(kf)

    answers, notes = {}, {}
    current = None
    with open(args.sheet, encoding="utf-8") as f:
        for line in f:
            m = ID_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if current is None:
                continue
            m = ANSWER_RE.match(line.strip())
            if m:
                if m.group(1):
                    answers[current] = m.group(1).upper()
                continue
            m = NOTES_RE.match(line.strip())
            if m and m.group(1).strip():
                notes[current] = m.group(1).strip()

    if not answers:
        print("No answers found. Did you fill in the ANSWER: lines?")
        return 1

    counts = Counter()
    disagreements = []

    for sid, ans in answers.items():
        if sid not in key:
            continue
        human = 1 if ans == "Y" else 0
        machine = key[sid]["label"]
        if human == machine:
            counts["agree_positive" if machine else "agree_negative"] += 1
        else:
            counts["false_positive" if machine == 1 else "false_negative"] += 1
            disagreements.append((sid, machine, human, key[sid]["text"]))

    n = sum(counts.values())
    skipped = len(key) - len(answers)

    print("--- Label validation ---")
    print(f"  reviewed                    {n}")
    print(f"  left blank                  {skipped}")
    print(f"  agreement                   {100.0 * (counts['agree_positive'] + counts['agree_negative']) / n:.1f}%")
    print()
    print(f"  agree, both positive        {counts['agree_positive']}")
    print(f"  agree, both negative        {counts['agree_negative']}")
    print(f"  labelled 1, you said no     {counts['false_positive']}  (false positives)")
    print(f"  labelled 0, you said yes    {counts['false_negative']}  (false negatives)")

    n_machine_pos = counts["agree_positive"] + counts["false_positive"]
    n_machine_neg = counts["agree_negative"] + counts["false_negative"]

    if n_machine_pos:
        print(f"\n  positive precision          {100.0 * counts['agree_positive'] / n_machine_pos:.1f}%")
        print("  (of sentences the pipeline called citation-worthy,")
        print("   how many you agreed with)")
    if n_machine_neg:
        print(f"\n  negative contamination      {100.0 * counts['false_negative'] / n_machine_neg:.1f}%")
        print("  (of sentences the pipeline called not citation-worthy,")
        print("   how many actually needed a citation -- this is the")
        print("   irreducible noise floor of distant supervision)")

    if notes:
        print(f"\n--- Sentences you flagged ({len(notes)}) ---")
        for sid, note in sorted(notes.items()):
            print(f"  {sid}  [{note}]")
            print(f"       {key[sid]['text'][:110]}")

    if disagreements:
        print(f"\n--- Disagreements ({len(disagreements)}) ---")
        for sid, machine, human, text in sorted(disagreements)[:25]:
            print(f"  {sid}  pipeline={machine} you={human}")
            print(f"       {text[:110]}")
        if len(disagreements) > 25:
            print(f"  ... and {len(disagreements) - 25} more")

    print("\nRecord these numbers. They belong in your methodology chapter,")
    print("and no paper in your Table 2.1 reports an equivalent figure.")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--input", default=INPUT)
    p.add_argument("--sheet", default=SHEET)
    p.add_argument("--key", default=KEY)
    p.add_argument("--n-positive", type=int, default=N_POSITIVE)
    p.add_argument("--n-negative", type=int, default=N_NEGATIVE)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.make:
        return make_sheet(args) or 0
    if args.score:
        return score_sheet(args)

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
