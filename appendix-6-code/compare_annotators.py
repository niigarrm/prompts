#!/usr/bin/env python3
"""Compares two annotators with each other and with the rule's labels: agreement, Cohen's kappa,
positive precision and negative contamination (Chapter 3).
"""

import argparse
import json
import re
import sys
from collections import Counter

ID_RE = re.compile(r"^### (\d+)")
ANSWER_RE = re.compile(r"^ANSWER:\s*([YyNn])?\s*$")


def read_answers(path):
    answers = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = ID_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if current is None:
                continue
            m = ANSWER_RE.match(line.strip())
            if m and m.group(1):
                answers[current] = 1 if m.group(1).upper() == "Y" else 0
                current = None
    return answers


def cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return 0.0

    observed = sum(1 for a, b in pairs if a == b) / n

    a_pos = sum(1 for a, _ in pairs if a == 1) / n
    b_pos = sum(1 for _, b in pairs if b == 1) / n
    expected = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret(k):
    if k < 0.00:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def vs_pipeline(name, answers, key):
    shared = [s for s in answers if s in key]
    c = Counter()
    for sid in shared:
        human = answers[sid]
        machine = key[sid]["label"]
        if human == machine:
            c["agree_pos" if machine else "agree_neg"] += 1
        else:
            c["false_pos" if machine == 1 else "false_neg"] += 1

    n = len(shared)
    if not n:
        return

    agree = c["agree_pos"] + c["agree_neg"]
    print(f"\n--- {name} vs pipeline labels ---")
    print(f"  reviewed                  {n}")
    print(f"  agreement                 {100.0 * agree / n:.1f}%")
    print(f"  labelled 1, said no       {c['false_pos']}")
    print(f"  labelled 0, said yes      {c['false_neg']}")

    n_pos = c["agree_pos"] + c["false_pos"]
    n_neg = c["agree_neg"] + c["false_neg"]
    if n_pos:
        print(f"  positive precision        {100.0 * c['agree_pos'] / n_pos:.1f}%")
    if n_neg:
        print(f"  negative contamination    {100.0 * c['false_neg'] / n_neg:.1f}%")

    pairs = [(answers[s], key[s]["label"]) for s in shared]
    k = cohens_kappa(pairs)
    print(f"  Cohen's kappa             {k:.3f}  ({interpret(k)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="first annotator's sheet")
    p.add_argument("--b", required=True, help="second annotator's sheet")
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--key", default="review_key.json")
    args = p.parse_args()

    a = read_answers(args.a)
    b = read_answers(args.b)
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)

    if not a or not b:
        print("One of the sheets has no filled answers.", file=sys.stderr)
        return 1

    both = sorted(set(a) & set(b))
    print(f"{args.a_name}: {len(a)} answered.  "
          f"{args.b_name}: {len(b)} answered.  "
          f"Both: {len(both)}.")

    pairs = [(a[s], b[s]) for s in both]
    raw = 100.0 * sum(1 for x, y in pairs if x == y) / len(pairs)
    k = cohens_kappa(pairs)

    print(f"\n--- {args.a_name} vs {args.b_name} (inter-annotator) ---")
    print(f"  raw agreement             {raw:.1f}%")
    print(f"  Cohen's kappa             {k:.3f}  ({interpret(k)})")
    print()
    print("  This is the ceiling. If two careful readers agree only this")
    print("  often, a model scoring far above it against the automatic")
    print("  labels is fitting label noise, not learning the task.")

    both_yes = sum(1 for x, y in pairs if x == 1 and y == 1)
    both_no = sum(1 for x, y in pairs if x == 0 and y == 0)
    a_only = sum(1 for x, y in pairs if x == 1 and y == 0)
    b_only = sum(1 for x, y in pairs if x == 0 and y == 1)
    print(f"\n  both said yes             {both_yes}")
    print(f"  both said no              {both_no}")
    print(f"  only {args.a_name} said yes{' ' * max(1, 14 - len(args.a_name))}{a_only}")
    print(f"  only {args.b_name} said yes{' ' * max(1, 14 - len(args.b_name))}{b_only}")

    vs_pipeline(args.a_name, a, key)
    vs_pipeline(args.b_name, b, key)

    consensus = {s: a[s] for s in both if a[s] == b[s]}
    if consensus:
        print(f"\n--- Consensus ({len(consensus)} sentences both agreed on) "
              f"vs pipeline ---")
        agree = sum(1 for s in consensus if consensus[s] == key[s]["label"])
        print(f"  pipeline matches consensus {100.0 * agree / len(consensus):.1f}%")
        print("  This is your cleanest label-quality estimate: it drops the")
        print("  sentences where humans themselves could not agree.")

    hard = [s for s in both if a[s] != b[s]]
    if hard:
        print(f"\n--- Sentences you disagreed on ({len(hard)}) ---")
        print("  Good candidates to quote in the thesis as genuinely")
        print("  ambiguous cases.\n")
        for sid in hard[:20]:
            print(f"  {sid}  {args.a_name}={a[sid]} {args.b_name}={b[sid]} "
                  f"pipeline={key[sid]['label']}")
            print(f"       {key[sid]['text'][:110]}")
        if len(hard) > 20:
            print(f"  ... and {len(hard) - 20} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
