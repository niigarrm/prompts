#!/usr/bin/env python3
"""Context inputs (Section 5.4.1, v1): sentence with one neighbour on each side, and the whole paragraph,
with the target marked [TGT] ... [/TGT].
"""

import argparse, json
from collections import defaultdict

def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--files", nargs="+", default=["train_bal", "val_bal", "test_bal"])
    a = ap.parse_args()
    data = {f: read(f"{a.splits}/{f}.jsonl") for f in a.files}
    need = {(r["paper_id"], r["paragraph_index"]) for rows in data.values() for r in rows}
    para = defaultdict(dict)
    with open(a.full, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); k = (r["paper_id"], r["paragraph_index"])
            if k in need: para[k][r["sentence_index"]] = r["text"]
    for name, rows in data.items():
        for r in rows:
            sents = para[(r["paper_id"], r["paragraph_index"])]; si = r["sentence_index"]
            tgt = f"[TGT] {r['text']} [/TGT]"
            prev = sents.get(si - 1, ""); nxt = sents.get(si + 1, "")
            r["text_neighbour"] = " ".join(x for x in (prev, tgt, nxt) if x)
            r["text_paragraph"] = " ".join(tgt if j == si else t for j, t in sorted(sents.items()) if j == si or t)
        for sfx in ("neighbour", "paragraph"):
            with open(f"{a.splits}/{name}_{sfx}.jsonl", "w", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), "written")

if __name__ == "__main__":
    main()
