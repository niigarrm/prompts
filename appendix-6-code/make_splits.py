#!/usr/bin/env python3
"""Document-level split, balanced training set, natural-rate validation and test sets, LLM subset (Section 2.5, v3).

    python3 make_splits.py --input runs/full/sentences_v4.jsonl --holes runs/full/sentences_v4_holes.jsonl --out data/splits --seed 42 --train-cap 400000 --llm-subset 3000
"""

import argparse, json, os, random
from array import array
from collections import Counter

def stream(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True); p.add_argument("--holes", required=True)
    p.add_argument("--out", default="data/splits"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-cap", type=int, default=400_000, help="max balanced training sentences (0 = no cap)")
    p.add_argument("--llm-subset", type=int, default=3000)
    p.add_argument("--val-natural-size", type=int, default=500_000, help="sentences in the natural-rate validation set")
    a = p.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    papers, seen, ppos = [], set(), Counter()
    for _, r in stream(a.input):
        pid = r["paper_id"]
        if pid not in seen: seen.add(pid); papers.append(pid)
        if r["label"] == 1: ppos[pid] += 1
    rng.shuffle(papers)
    n = len(papers); n_tr = int(n * 0.6); n_va = int(n * 0.2)
    split = {pid: "train" for pid in papers[:n_tr]}
    split.update({pid: "val" for pid in papers[n_tr:n_tr + n_va]})
    split.update({pid: "test" for pid in papers[n_tr + n_va:]})
    json.dump(split, open(f"{a.out}/paper_split.json", "w"))
    print(f"papers {n:,}: train {n_tr:,}  val {n_va:,}  test {n - n_tr - n_va:,}")
    stats = {"seed": a.seed, "train_cap": a.train_cap,
             "papers": {"train": n_tr, "val": n_va, "test": n - n_tr - n_va}}
    train_papers = [pid for pid in papers if split[pid] == "train"]

    for variant, path in (("drop", a.input), ("keep", a.holes)):
        vrng = random.Random(a.seed + (0 if variant == "drop" else 1))
        od = f"{a.out}/{variant}"; os.makedirs(od, exist_ok=True)
        npos, vppos = Counter(), Counter()
        neg_idx = {s: array("q") for s in ("train", "val", "test")}
        nat_test, nat_val = 0, 0
        out_rows, out_pos, out_papers = 0, 0, set()
        for i, r in stream(path):
            s = split.get(r["paper_id"])
            if s is None:
                out_rows += 1; out_pos += r["label"]; out_papers.add(r["paper_id"]); continue
            if r["label"] == 1: npos[s] += 1; vppos[r["paper_id"]] += 1
            else: neg_idx[s].append(i)
            if s == "test": nat_test += 1
            if s == "val": nat_val += 1
        if a.train_cap and 2 * npos["train"] > a.train_cap:
            kept, acc = set(), 0
            for pid in train_papers:
                if acc >= a.train_cap // 2: break
                kept.add(pid); acc += vppos.get(pid, 0)
        else:
            kept, acc = set(train_papers), npos["train"]
        chosen = {}
        for s in ("val", "test"):
            chosen[s] = set(vrng.sample(list(neg_idx[s]), min(npos[s], len(neg_idx[s]))))
        tr = array("q")
        for i, r in stream(path):
            if r["label"] == 0 and r["paper_id"] in kept and split.get(r["paper_id"]) == "train":
                tr.append(i)
        chosen["train"] = set(vrng.sample(list(tr), min(acc, len(tr))))
        files = {k: open(f"{od}/{k}.jsonl", "w", encoding="utf-8")
                 for k in ("train_bal", "val_bal", "test_bal", "test_natural", "val_natural")}
        vnrng = random.Random(a.seed + 100 + (0 if variant == "drop" else 1))
        keep_p = min(1.0, a.val_natural_size / max(nat_val, 1))
        counts, res, seen_t = Counter(), [], 0
        for i, r in stream(path):
            s = split.get(r["paper_id"])
            if s is None: continue
            r["id"] = f"{r['paper_id']}|{r['paragraph_index']}|{r['sentence_index']}"
            line = json.dumps(r, ensure_ascii=False) + "\n"
            if s == "val" and vnrng.random() < keep_p:
                files["val_natural"].write(line); counts["val_natural"] += 1
                counts["val_natural_pos"] += r["label"]
            if s == "test":
                files["test_natural"].write(line); counts["test_natural"] += 1
                seen_t += 1
                if len(res) < a.llm_subset: res.append(r)
                else:
                    j = vrng.randrange(seen_t)
                    if j < a.llm_subset: res[j] = r
            take = (r["label"] == 1 and (s != "train" or r["paper_id"] in kept)) or \
                   (r["label"] == 0 and i in chosen[s])
            if take:
                files[f"{s}_bal"].write(line); counts[f"{s}_bal"] += 1
                counts[f"{s}_bal_pos"] += r["label"]
        for f in files.values(): f.close()
        with open(f"{od}/llm_subset.jsonl", "w", encoding="utf-8") as f:
            for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts["llm_subset"] = len(res); counts["llm_subset_pos"] = sum(r["label"] for r in res)
        counts["train_papers_used"] = len(kept); counts["test_positives"] = npos["test"]
        counts["val_natural_rate_pct"] = round(100 * counts["val_natural_pos"] / max(counts["val_natural"], 1), 2)
        counts["natural_positive_rate_test_pct"] = round(100 * npos["test"] / max(nat_test, 1), 2)
        counts["papers_outside_split"] = len(out_papers)
        counts["sentences_outside_split"] = out_rows
        counts["positives_outside_split"] = out_pos
        if out_papers:
            with open(f"{od}/papers_outside_split.txt", "w") as f:
                f.write("\n".join(sorted(out_papers)) + "\n")
        stats[variant] = dict(counts); print(variant, dict(counts))
    json.dump(stats, open(f"{a.out}/split_stats.json", "w"), indent=1)
    print("stats ->", f"{a.out}/split_stats.json  (Table 3.4)")

if __name__ == "__main__":
    main()
