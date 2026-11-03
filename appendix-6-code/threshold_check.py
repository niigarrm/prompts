#!/usr/bin/env python3
"""Threshold diagnostic on saved predictions (E-017, E-020, E-023); trains nothing (v1)."""

import json, sys
import numpy as np

PRIOR = 0.1230

def load(path):
    y, p = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); y.append(int(r["label"])); p.append(float(r["prob"]))
    return np.array(y), np.array(p)

def best_threshold(y, p, w_neg=1.0):
    o = np.argsort(-p, kind="mergesort"); ys, ps = y[o], p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys) * w_neg; fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    last = np.r_[ps[1:] != ps[:-1], True]
    i = int(np.argmax(np.where(last, f1, -1.0)))
    return float(ps[i])

def prf(y, p, t):
    h = p >= t; tp = int((h & (y == 1)).sum()); fp = int((h & (y == 0)).sum()); fn = int((~h & (y == 1)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-12)

def main():
    name = sys.argv[1]
    global PRIOR
    if len(sys.argv) > 2:
        v = float(sys.argv[2]); PRIOR = v / 100 if v > 1 else v
    yv, pv = load(f"preds/{name}_val.jsonl"); yt, pt = load(f"preds/{name}_test_natural.jsonl")
    if len(np.unique(pv)) <= 2: sys.exit("constant or hard-label scores: nothing to tune (trivial baseline or LLM prompting)")
    val_rate = yv.mean()
    w = (1 - PRIOR) / PRIOR * val_rate / (1 - val_rate)
    ta, tb, tc = best_threshold(yv, pv), best_threshold(yv, pv, w), best_threshold(yt, pt)
    print(f"{name}: validation {len(yv):,} ({100*val_rate:.1f} % pos), natural test {len(yt):,} ({100*yt.mean():.2f} % pos), prior {100*PRIOR:.2f} %, negative weight {w:.2f}")
    for tag, t in (("A balanced-val threshold (current)", ta), ("B prior-corrected val threshold  ", tb),
                   ("C best on test - DIAGNOSTIC ONLY  ", tc)):
        P, R, F = prf(yt, pt, t)
        print(f"  {tag}  t={t:.4f}  natural test P={P:.3f} R={R:.3f} F1={F:.3f}")

if __name__ == "__main__":
    main()
