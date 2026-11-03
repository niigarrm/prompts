#!/usr/bin/env python3
"""Evaluation (Section 4.1, v4): threshold from the natural-rate validation set, metrics on the natural test set
and the balanced view, bootstrap intervals, reports.
"""

import argparse, json, math, os, glob
from collections import defaultdict
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, average_precision_score,
                             accuracy_score, precision_recall_curve)

def load(path):
    ids, y, p = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); ids.append(r["id"]); y.append(int(r["label"])); p.append(float(r["prob"]))
    return ids, np.array(y), np.array(p)

def best_threshold(y, p, w_pos=1.0):
    if len(np.unique(p)) <= 2:
        return 0.5
    o = np.argsort(-p, kind="mergesort"); ys, ps = y[o], p[o]
    tp = np.cumsum(ys) * w_pos; fp = np.cumsum(1 - ys); fn = ys.sum() * w_pos - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    last = np.r_[ps[1:] != ps[:-1], True]
    return float(ps[int(np.argmax(np.where(last, f1, -1.0)))])

def metrics(y, p, thr, w_pos=1.0):
    yhat = (p >= thr).astype(int)
    sw = np.where(y == 1, float(w_pos), 1.0) if w_pos != 1.0 else None
    P, R, F, _ = precision_recall_fscore_support(y, yhat, average=None, labels=[0, 1], zero_division=0, sample_weight=sw)
    mp, mr, mf, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0, sample_weight=sw)
    wp, wr, wf, _ = precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0, sample_weight=sw)
    rate = float((w_pos * y.sum()) / (w_pos * y.sum() + (len(y) - y.sum())))
    maj = max(rate, 1 - rate)
    return {"threshold": thr, "n": int(len(y)), "weight_positive": float(w_pos), "positive_rate": rate,
            "precision_pos": float(P[1]), "recall_pos": float(R[1]), "f1_pos": float(F[1]),
            "macro_f1": float(mf), "weighted_f1": float(wf),
            "pr_auc": float(average_precision_score(y, p, sample_weight=sw)) if len(np.unique(p)) > 2 else None,
            "accuracy": float(accuracy_score(y, yhat, sample_weight=sw)), "majority_accuracy": float(maj),
            "tp": int(((yhat == 1) & (y == 1)).sum()), "fp": int(((yhat == 1) & (y == 0)).sum()),
            "fn": int(((yhat == 0) & (y == 1)).sum()), "tn": int(((yhat == 0) & (y == 0)).sum())}

def bootstrap_f1(y, p, thr, n=1000, seed=0, w_pos=1.0):
    rng = np.random.default_rng(seed); yhat = (p >= thr).astype(int); vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); yy, hh = y[idx], yhat[idx]
        tp = w_pos * ((hh == 1) & (yy == 1)).sum(); fp = ((hh == 1) & (yy == 0)).sum()
        fn = w_pos * ((hh == 0) & (yy == 1)).sum()
        vals.append(2 * tp / max(2 * tp + fp + fn, 1e-9))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def evaluate_run(val_path, test_paths, name, out_dir="results", extra=None, fixed_threshold=None,
                 balanced_view=True):
    os.makedirs(out_dir, exist_ok=True)
    _, yv, pv = load(val_path)
    thr = float(fixed_threshold) if fixed_threshold is not None else best_threshold(yv, pv)
    w_val = float((len(yv) - yv.sum()) / max(yv.sum(), 1))
    thr_b = thr if fixed_threshold is not None else best_threshold(yv, pv, w_pos=w_val)
    rule = f"fixed {thr:g}" if fixed_threshold is not None else "validation, max positive-F1"
    res = {"name": name, "threshold_from_val": thr, "threshold_balanced_view": thr_b,
           "threshold_rule": rule, "val": metrics(yv, pv, thr), "test": {}}
    for tag, path in test_paths.items():
        _, y, p = load(path); m = metrics(y, p, thr); m["f1_pos_ci95"] = bootstrap_f1(y, p, thr)
        res["test"][tag] = m
        if balanced_view and 0 < y.sum() < len(y):
            w = float((len(y) - y.sum()) / y.sum())
            b = metrics(y, p, thr_b, w_pos=w); b["f1_pos_ci95"] = bootstrap_f1(y, p, thr_b, w_pos=w)
            res["test"][f"{tag}_balanced_view"] = b
    if extra: res.update(extra)
    json.dump(res, open(f"{out_dir}/{name}.json", "w"), indent=1)
    for tag, m in res["test"].items():
        print(f"{name:40s} {tag:14s} P={m['precision_pos']:.3f} R={m['recall_pos']:.3f} "
              f"F1={m['f1_pos']:.3f} [{m['f1_pos_ci95'][0]:.3f},{m['f1_pos_ci95'][1]:.3f}] "
              f"macroF1={m['macro_f1']:.3f} PR-AUC={m['pr_auc'] if m['pr_auc'] is None else round(m['pr_auc'],3)} "
              f"acc={m['accuracy']:.3f} (maj {m['majority_accuracy']:.3f})")
    return res

def mcnemar(a_path, b_path, thr_a=None, thr_b=None):
    ia, ya, pa = load(a_path); ib, yb, pb = load(b_path)
    assert ia == ib, "prediction files must cover the same ids in the same order"
    ha = (pa >= (thr_a if thr_a is not None else 0.5)); hb = (pb >= (thr_b if thr_b is not None else 0.5))
    ca, cb = ha == ya, hb == yb
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    if n01 + n10 == 0:
        return {"a_correct_b_wrong": 0, "b_correct_a_wrong": 0, "chi2": 0.0, "p": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    pval = math.erfc(math.sqrt(stat / 2))
    return {"a_correct_b_wrong": n01, "b_correct_a_wrong": n10, "chi2": stat, "p": pval}

def report(files, out):
    groups = defaultdict(list)
    for f in files:
        r = json.load(open(f)); base = r["name"].rsplit("_s", 1)[0]; groups[base].append(r)
    lines = ["| Model | Test set | P (pos) | R (pos) | F1 (pos) | Macro-F1 | PR-AUC | Accuracy | Seeds |", "|---|---|---|---|---|---|---|---|---|"]
    for base, rs in sorted(groups.items()):
        tags = list(dict.fromkeys(t for r in rs for t in r["test"]))
        for tag in tags:
            have = [r for r in rs if tag in r["test"]]
            def ms(k):
                v = [r["test"][tag][k] for r in have if r["test"][tag].get(k) is not None]
                return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if v else "—")
            lines.append(f"| {base} | {tag} | {ms('precision_pos')} | {ms('recall_pos')} | {ms('f1_pos')} | {ms('macro_f1')} | {ms('pr_auc')} | {ms('accuracy')} | {len(have)} |")
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))

def subset_preds(pred_path, subset_path, out_path):
    want = set()
    with open(subset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): want.add(json.loads(line)["id"])
    n = 0
    with open(pred_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as g:
        for line in f:
            if line.strip() and json.loads(line)["id"] in want: g.write(line); n += 1
    if n != len(want):
        print(f"WARNING: {pred_path} covers {n} of the {len(want)} subset sentences")
    return out_path

def val_report(files, out):
    rows = []
    for f in files:
        r = json.load(open(f)); v = r["val"]
        rows.append((v["f1_pos"], r["name"], r.get("lr", "—"), r.get("batch", "—"), r.get("max_len", "—"),
                     v["precision_pos"], v["recall_pos"], v["pr_auc"]))
    rows.sort(reverse=True)
    lines = ["| Run | LR | Batch | Max len | P (val) | R (val) | F1 (val) | PR-AUC (val) |", "|---|---|---|---|---|---|---|---|"]
    for f1, name, lr, b, L, p, rc, auc in rows:
        lines.append(f"| {name} | {lr} | {b} | {L} | {p:.3f} | {rc:.3f} | {f1:.3f} | {'—' if auc is None else f'{auc:.3f}'} |")
    open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    if rows: print(f"best: {rows[0][1]}  (val F1 {rows[0][0]:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val"); ap.add_argument("--test", nargs="*"); ap.add_argument("--name")
    ap.add_argument("--report", nargs="*"); ap.add_argument("--out", default="results/table.md")
    ap.add_argument("--mcnemar", nargs=2); ap.add_argument("--val-report", nargs="*")
    ap.add_argument("--subset", help="ids file, e.g. data/splits/drop/llm_subset.jsonl")
    a = ap.parse_args()
    if a.mcnemar:
        print(json.dumps(mcnemar(*a.mcnemar), indent=1))
    elif a.report:
        report(sum([glob.glob(g) for g in a.report], []), a.out)
    elif a.val_report:
        val_report(sum([glob.glob(g) for g in a.val_report], []), a.out if a.out != "results/table.md" else "results/grid.md")
    elif a.subset:
        tag = os.path.basename(a.subset).split(".")[0]
        sub = subset_preds(a.test[0], a.subset, f"preds/{a.name}_{tag}.jsonl")
        evaluate_run(a.val, {tag: sub}, f"{a.name}_{tag}")
    else:
        evaluate_run(a.val, {os.path.basename(t).split(".")[0]: t for t in a.test}, a.name)
