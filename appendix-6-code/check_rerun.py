#!/usr/bin/env python3
"""Checks that the drop split files are unchanged from the first run and prints the Table 2.4 counts (v2).

    python3 check_rerun.py
"""

import json

OLD_DROP = {'val_bal': 496436, 'val_bal_pos': 248218, 'test_natural': 2002611, 'test_bal': 490326,
            'test_bal_pos': 245163, 'train_bal': 400018, 'train_bal_pos': 200009, 'llm_subset': 3000,
            'llm_subset_pos': 360, 'train_papers_used': 8139, 'test_positives': 245163,
            'natural_positive_rate_test_pct': 12.24}
OLD_PAPERS = {"train": 29968, "val": 9989, "test": 9990}
CS_SELECTED = 50790

s = json.load(open("data/splits/split_stats.json"))
d, k, P = s["drop"], s["keep"], s["papers"]

bad = {x: (v, d.get(x)) for x, v in OLD_DROP.items() if d.get(x) != v}
if P != OLD_PAPERS: bad["papers"] = (OLD_PAPERS, P)
if d.get("papers_outside_split", 0) != 0: bad["papers_outside_split"] = (0, d["papers_outside_split"])
print("CHECK drop:", "SAME as first run" if not bad else f"DIFFERS (old, new): {bad}")
print("keep raw:", k)

f = lambda x: f"{x:,}"
n = P["train"] + P["val"] + P["test"]
def bal(v, name, papers):
    return f"{f(v[name])} ({f(v[name + '_pos'])} pos.); from {f(papers)} papers"
def nat(v):
    return f"{f(v['test_natural'])} ({f(v['test_positives'])} pos., {v['natural_positive_rate_test_pct']} %); {f(P['test'])} papers"
def llm(v):
    return f"{f(v['llm_subset'])} ({v['llm_subset_pos']} pos., {100 * v['llm_subset_pos'] / v['llm_subset']:.1f} %)"
split_txt = f"{f(P['train'])} / {f(P['val'])} / {f(P['test'])}"
rows = [
    ("CS papers selected (2010 and later)", f(CS_SELECTED), f(CS_SELECTED)),
    ("Papers with at least one kept sentence", f(n), f(n + k["papers_outside_split"])),
    ("Papers left out of the split", "0",
     f"{f(k['papers_outside_split'])} ({f(k['sentences_outside_split'])} sentences, {f(k['positives_outside_split'])} pos.)"),
    ("Papers in the split, train / val / test", split_txt, split_txt),
    ("Natural-rate validation", f"{f(d['val_natural'])} ({f(d['val_natural_pos'])} pos., {d['val_natural_rate_pct']} %)",
     f"{f(k['val_natural'])} ({f(k['val_natural_pos'])} pos., {k['val_natural_rate_pct']} %)"),
    ("Balanced train", bal(d, "train_bal", d["train_papers_used"]), bal(k, "train_bal", k["train_papers_used"])),
    ("Balanced validation", bal(d, "val_bal", P["val"]), bal(k, "val_bal", P["val"])),
    ("Balanced test", bal(d, "test_bal", P["test"]), bal(k, "test_bal", P["test"])),
    ("Natural-rate test", nat(d), nat(k)),
    ("LLM prompting subset", llm(d), llm(k)),
]
print("\nTable 3.4 rows (Quantity | Holes dropped | Holes kept):")
for r in rows: print(" | ".join(r))
print("Rows 'Sentences after filtering' and 'Positives / natural rate' stay as they are.")
print("Balanced validation and balanced test stay in the table as built files; the runs no longer use them.")
