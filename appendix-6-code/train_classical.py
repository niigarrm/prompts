#!/usr/bin/env python3
"""Majority, all-positive and random baselines; TF-IDF logistic regression, linear SVM and LR with features
(Section 4.2, v4).
"""

import argparse, json, os, random, re, time
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from evaluate import evaluate_run, metrics, best_threshold

KEEP = ("id", "label", "text", "paper_id", "paragraph_index", "sentence_index")
TRIVIAL = ("majority", "allpos", "random")

def slim(r): return {k: r[k] for k in KEEP if k in r}

def read(path):
    with open(path, encoding="utf-8") as f:
        return [slim(json.loads(l)) for l in f if l.strip()]

def chunks(path, n):
    buf = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                buf.append(slim(json.loads(l)))
                if len(buf) == n: yield buf; buf = []
    if buf: yield buf

def subsample(rows, fraction, seed):
    if fraction >= 1: return rows
    papers = sorted({r["paper_id"] for r in rows}); random.Random(seed).shuffle(papers)
    keep = set(papers[:max(1, int(len(papers) * fraction))])
    return [r for r in rows if r["paper_id"] in keep]

CAP = re.compile(r"\b[A-Z][a-z]+\b")
def engineered(rows):
    X = []
    for r in rows:
        t = r["text"]; toks = t.split()
        X.append([len(toks), len(CAP.findall(t[1:])) if t else 0, int(any(c.isdigit() for c in t)),
                  int(r["sentence_index"] == 0), int("FORMULA" in t), int("REF" in t),
                  float(np.log(r["paragraph_index"] + 1))])
    return sp.csr_matrix(np.array(X, dtype=np.float32))

def write_rows(f, rows, prob):
    for r, p in zip(rows, prob):
        f.write(json.dumps({"id": r["id"], "label": r["label"], "prob": float(p)}) + "\n")

def score_file(src, dst, scorer, chunk):
    with open(dst, "w", encoding="utf-8") as f:
        for rows in chunks(src, chunk): write_rows(f, rows, scorer(rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lr", "svm", "lr_feat", *TRIVIAL])
    ap.add_argument("--variant", default="drop"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0); ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--chunk", type=int, default=100_000, help="sentences scored at a time")
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0], help="C values to try")
    ap.add_argument("--cross", action="store_true", help="also score the other variant's test sets")
    a = ap.parse_args()
    d = f"{a.splits}/{a.variant}"
    src = {"val": f"{d}/val_natural.jsonl", "test_natural": f"{d}/test_natural.jsonl"}
    name = f"{a.model}_{a.variant}_f{a.fraction:g}_s{a.seed}"
    if a.cross:
        other = "keep" if a.variant == "drop" else "drop"
        src[f"test_natural_{other}"] = f"{a.splits}/{other}/test_natural.jsonl"
    os.makedirs("preds", exist_ok=True); paths = {k: f"preds/{name}_{k}.jsonl" for k in src}
    t0 = time.time()
    tr = subsample(read(f"{d}/train_bal.jsonl"), a.fraction, a.seed)
    ytr = np.array([r["label"] for r in tr]); bestC, fixed, val_by_C = None, None, {}

    if a.model in TRIVIAL:
        fixed = 0.5
        rng = np.random.default_rng(a.seed)
        const = {"majority": 0.0, "allpos": 1.0}.get(a.model)
        scorer = (lambda rows: np.full(len(rows), const)) if const is not None else (lambda rows: rng.random(len(rows)))
        for k in src: score_file(src[k], paths[k], scorer, a.chunk)
    else:
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, dtype=np.float32)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=200_000,
                             sublinear_tf=True, dtype=np.float32)
        def feats(rows, fit=False):
            tx = [r["text"] for r in rows]
            parts = [wv.fit_transform(tx) if fit else wv.transform(tx), cv.fit_transform(tx) if fit else cv.transform(tx)]
            if a.model == "lr_feat": parts.append(engineered(rows))
            return sp.hstack(parts, format="csr", dtype=np.float32)
        Xtr = feats(tr, fit=True)
        for v in (wv, cv): v.stop_words_ = None
        va = read(src["val"]); Xva = feats(va); yva = np.array([r["label"] for r in va]); best = -1
        for C in sorted(a.Cs):
            clf = (LinearSVC(C=C, random_state=a.seed) if a.model == "svm"
                   else LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=a.seed))
            clf.fit(Xtr, ytr)
            s = clf.decision_function(Xva); f1 = metrics(yva, s, best_threshold(yva, s))["f1_pos"]
            print(f"  C={C}: val F1={f1:.4f}", flush=True); val_by_C[f"{C:g}"] = round(float(f1), 4)
            if f1 > best: best, bestC, bestclf = f1, C, clf
        print(f"chosen C={bestC}", flush=True)
        if len(a.Cs) > 1 and bestC in (min(a.Cs), max(a.Cs)):
            print(f"NOTE: the best C ({bestC:g}) is at the edge of the grid; extend --Cs beyond it", flush=True)
        score = lambda X: 1 / (1 + np.exp(-bestclf.decision_function(X)))
        with open(paths["val"], "w", encoding="utf-8") as f: write_rows(f, va, score(Xva))
        del Xtr, Xva, va
        for k in src:
            if k != "val": score_file(src[k], paths[k], lambda rows: score(feats(rows)), a.chunk)

    extra = {"model": a.model, "variant": a.variant, "fraction": a.fraction, "seed": a.seed,
             "train_sentences": len(tr), "train_papers": len({r["paper_id"] for r in tr}),
             "elapsed_s": round(time.time() - t0, 1), "C": bestC, "C_grid_val_f1": val_by_C or None,
             "cross": bool(a.cross)}
    evaluate_run(paths["val"], {k: paths[k] for k in src if k != "val"}, name, extra=extra, fixed_threshold=fixed)
    with open("results/log.csv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{json.dumps(extra)}\n")

if __name__ == "__main__":
    main()
