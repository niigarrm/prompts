#!/usr/bin/env python3
"""Diagnostic (E-012): runs extraction v3 and v4.0 on the same 20 files and compares their sentences
to explain the extra no_terminal_punct rejections of v4.0.
"""

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict

WS = re.compile(r"\s+")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3 = load("extract_and_label.py", "v3")
v4 = load("extract_and_label_v4.py", "v4")

files, total, eligible = v4.select_files("unarxive_data", 20, 2010, 42)
print(f"{len(files)} files (of {eligible} eligible)")


def run(mod):
    nlp = mod.build_nlp()
    recs = defaultdict(list)
    for path in files:
        year = mod.year_from_filename(path)
        with mod.open_maybe_gzip(path) as f:
            for line in f:
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if paper.get("discipline") != mod.DISCIPLINE:
                    continue
                pid = paper.get("paper_id") or paper.get("id")
                paras = list(mod.iter_paragraphs(paper))
                if not paras:
                    continue
                protected = [mod.protect_abbreviations(t) for _, t in paras]
                for (pi, _), doc in zip(paras, nlp.pipe(protected, batch_size=200)):
                    for si, sent in enumerate(doc.sents):
                        raw = mod.restore_abbreviations(sent.text).strip()
                        if not raw:
                            continue
                        label = 1 if mod.CITE_RE.search(raw) else 0
                        text = mod.clean_sentence(raw)
                        if not text:
                            verdict = "empty_after_cleaning"
                        else:
                            ok, why = mod.passes_quality(text)
                            verdict = "pass" if ok else why
                        recs[(pid, pi)].append({
                            "si": si, "raw": raw, "norm": WS.sub(" ", raw).strip(),
                            "text": text, "label": label, "verdict": verdict})
    return recs


print("running v3 ...", flush=True)
r3 = run(v3)
print("running v4 ...", flush=True)
r4 = run(v4)

n3 = sum(len(v) for v in r3.values())
n4 = sum(len(v) for v in r4.values())
print(f"segmented: v3 {n3:,}  v4 {n4:,}")
print("verdicts v3:", dict(Counter(r["verdict"] for v in r3.values() for r in v)))
print("verdicts v4:", dict(Counter(r["verdict"] for v in r4.values() for r in v)))

matched = 0
unmatched3 = 0
changed = Counter()
examples = defaultdict(list)
for key, lst3 in r3.items():
    lst4 = r4.get(key, [])
    idx4 = {r["norm"]: r for r in lst4}
    for a in lst3:
        b = idx4.get(a["norm"])
        if b is None:
            unmatched3 += 1
            continue
        matched += 1
        if a["verdict"] != b["verdict"]:
            changed[(a["verdict"], b["verdict"])] += 1
            if len(examples[(a["verdict"], b["verdict"])]) < 12:
                examples[(a["verdict"], b["verdict"])].append((a, b))

print(f"\nmatched on raw text: {matched:,}; v3 sentences with no raw-text match in v4: {unmatched3:,}")
print("\nverdict changes v3 -> v4 (count):")
for (va, vb), n in changed.most_common():
    print(f"  {va:24s} -> {vb:24s} {n:,}")

for (va, vb), ex in examples.items():
    if n := changed[(va, vb)]:
        print(f"\n=== {va} -> {vb}  ({n}) ===")
        for a, b in ex:
            print(f"  RAW  ...{a['raw'][-70:]!r}")
            print(f"  v3   ...{a['text'][-60:]!r}")
            print(f"  v4   ...{b['text'][-60:]!r}")
            print()

print("\n=== v3 sentences whose raw text does not exist in v4 (segmentation differs) ===")
shown = 0
for key, lst3 in r3.items():
    idx4 = {r["norm"] for r in r4.get(key, [])}
    for a in lst3:
        if a["norm"] not in idx4 and shown < 10:
            print(f"  v3 raw: {a['raw'][:160]!r}")
            near = [r["raw"][:160] for r in r4.get(key, []) if r["raw"][:25] == a["raw"][:25]]
            for nr in near[:2]:
                print(f"  v4 raw: {nr!r}")
            print()
            shown += 1
