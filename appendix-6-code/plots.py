#!/usr/bin/env python3
"""Figures 2.2, 5.1, 5.2, 5.3 and 6.1 (v5)."""

import argparse, glob, json, os, random, re
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from sklearn.metrics import precision_recall_curve
from evaluate import load
os.makedirs("figures", exist_ok=True)
STY = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
CAP = 100

def _stats(xs):
    a = np.asarray(xs)
    q = lambda v: round(float(v), 1)
    return {"n": int(len(a)), "mean": q(a.mean()), "median": q(np.median(a)), "p90": q(np.percentile(a, 90)),
            "p95": q(np.percentile(a, 95)), "p99": q(np.percentile(a, 99)), "max": int(a.max())}

def lengths(path, tokenizer=None, sample=20000, seed=42):
    L, texts = {0: [], 1: []}, {0: [], 1: []}
    with open(path, encoding="utf-8") as f:
        for l in f:
            if not l.strip(): continue
            r = json.loads(l); L[r["label"]].append(len(r["text"].split()))
            if tokenizer: texts[r["label"]].append(r["text"])
    stats = {"file": path, "unit": "words (text split on whitespace)", "plot_cap_words": CAP}
    for y in (0, 1):
        s = _stats(L[y]); s["over_cap_pct"] = round(100 * sum(x > CAP for x in L[y]) / len(L[y]), 2)
        stats[f"label_{y}"] = s
    plt.figure(figsize=(6, 3.5)); bins = list(range(0, CAP + 5, 4))
    plt.hist(np.minimum(L[0], CAP), bins, alpha=.5, label="without citation (label 0)", color="grey")
    plt.hist(np.minimum(L[1], CAP), bins, label="with citation (label 1)", color="black", histtype="step", lw=1.5)
    ticks = [0, 20, 40, 60, 80, CAP + 2]
    plt.xticks(ticks, [str(t) for t in ticks[:-1]] + [f"{CAP}+"])
    plt.xlabel("words per sentence"); plt.ylabel("sentences"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/fig_3_2.png", dpi=300)

    if tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained(tokenizer); rng = random.Random(seed)
        ts = {"tokenizer": tokenizer, "sample_per_class": sample, "seed": seed,
              "special_tokens": "counted, as the tokenizer adds them (these count toward max_len)"}
        for y in (0, 1):
            xs = texts[y] if len(texts[y]) <= sample else rng.sample(texts[y], sample)
            n = [len(e.ids) for e in tok.encode_batch(xs)]
            s = _stats(n)
            for m in (128, 256, 512): s[f"truncated_at_{m}_pct"] = round(100 * sum(x > m for x in n) / len(n), 2)
            s["tokens_per_word"] = round(sum(n) / max(sum(len(t.split()) for t in xs), 1), 2)
            ts[f"label_{y}"] = s
        stats["tokens"] = ts

    json.dump(stats, open("figures/fig_3_2_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

def pr(files):
    plt.figure(figsize=(6, 4.5))
    for i, f in enumerate(files):
        _, y, p = load(f); P, R, _ = precision_recall_curve(y, p)
        plt.plot(R, P, ls=STY[i % len(STY)], color="black", label=os.path.basename(f).split("_test")[0])
        base = y.mean()
    plt.axhline(base, color="grey", ls=":", label=f"positive rate {base:.2f}")
    plt.xlabel("recall (positive class)"); plt.ylabel("precision"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/fig_5_1.png", dpi=300)

NAMES = {"lr_drop": "Logistic regression (TF-IDF)", "svm_drop": "Linear SVM (TF-IDF)",
         "lr_feat_drop": "LR + engineered features", "mmBERT-base_drop_sentence": "mmBERT-base",
         "ModernBERT-base_drop_sentence": "ModernBERT-base", "scibert_scivocab_uncased_drop_sentence": "SciBERT"}
MARKS = ["o", "s", "^", "D", "v"]

def curve(patterns):
    fig, ax = plt.subplots(figsize=(5.7, 3.9))
    sizes, lo, hi = {}, 1.0, 0.0
    for i, pat in enumerate(patterns):
        pts = {}
        for f in glob.glob(pat):
            r = json.load(open(f))
            if "test_natural" not in r.get("test", {}):
                print(f"skipped (no natural-test block): {f}"); continue
            x = r["fraction"]
            pts.setdefault(x, []).append(r["test"]["test_natural"]["f1_pos"])
            if r.get("train_sentences"): sizes.setdefault(x, []).append(r["train_sentences"])
        if not pts:
            print(f"no results match {pat}"); continue
        xs = sorted(pts); m = [float(np.mean(pts[x])) for x in xs]; sd = [float(np.std(pts[x])) for x in xs]
        base = re.sub(r"_f\*.*", "", os.path.basename(pat))
        ax.errorbar(xs, m, yerr=sd, ls=STY[i % len(STY)], marker=MARKS[i % len(MARKS)], color="black",
                    capsize=3, ms=5, lw=1.4, label=NAMES.get(base, base), zorder=3)
        for x, y, e in zip(xs, m, sd):
            ax.annotate(f"{y:.3f}", (x, y + e if i % 2 == 0 else y - e), textcoords="offset points",
                        xytext=(0, 7 if i % 2 == 0 else -15), ha="center", fontsize=8, zorder=4)
        lo, hi = min(lo, min(m) - max(sd)), max(hi, max(m) + max(sd))
    xs = sorted(sizes) if sizes else sorted({x for x in sizes})
    ax.set_xscale("log"); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{100 * x:g} %\n{np.mean(sizes[x]) / 1000:.0f} k" if np.mean(sizes[x]) >= 10000
                        else f"{100 * x:g} %\n{np.mean(sizes[x]) / 1000:.1f} k" for x in xs], fontsize=9)
    ax.set_xlim(xs[0] / 1.35, xs[-1] * 1.35)
    ax.set_ylim(np.floor((lo - 0.025) * 50) / 50, np.ceil((hi + 0.02) * 50) / 50)
    ax.yaxis.set_major_locator(MultipleLocator(0.02)); ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(which="major", color="0.82", lw=0.7); ax.grid(which="minor", axis="y", color="0.9", lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of the training set (training sentences)", fontsize=10)
    ax.set_ylabel("F1 for \u2018needs a citation\u2019\n(natural test set)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right", frameon=True)
    plt.tight_layout(); plt.savefig("figures/fig_5_2.png", dpi=300)

def forest(files):
    rows = []
    for f in files:
        r = json.load(open(f)); t = r["test"].get("test_natural") or list(r["test"].values())[0]
        rows.append((r["name"], t["f1_pos"], t["f1_pos_ci95"]))
    rows.sort(key=lambda x: x[1]); plt.figure(figsize=(6, 0.4 * len(rows) + 1))
    for i, (n, f1, ci) in enumerate(rows):
        plt.plot(ci, [i, i], color="black"); plt.plot(f1, i, "ko")
    plt.yticks(range(len(rows)), [r[0] for r in rows], fontsize=7); plt.xlabel("F1 (positive class) with 95 % bootstrap CI")
    plt.tight_layout(); plt.savefig("figures/fig_6_3.png", dpi=300)

def confusion(files):
    n = len(files); fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    for ax, f in zip(np.atleast_1d(axes), files):
        _, y, p = load(f); r = json.load(open(f"results/{os.path.basename(f).split('_test')[0]}.json")); thr = r["threshold_from_val"]
        h = (p >= thr).astype(int); M = np.array([[((y == 0) & (h == 0)).sum(), ((y == 0) & (h == 1)).sum()], [((y == 1) & (h == 0)).sum(), ((y == 1) & (h == 1)).sum()]])
        ax.imshow(M, cmap="Greys"); ax.set_title(os.path.basename(f).split("_test")[0], fontsize=7)
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=7, color="red")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout(); plt.savefig("figures/fig_7_1.png", dpi=300)

SIG_PR = [("lr", "-", "0.0"), ("lr_feat", "--", "0.0"), ("modernbert", "-", "0.45"), ("lora", ":", "0.0"),
          ("mb_neighbour", "-.", "0.0"), ("mb_paragraph", "-", "0.0")]
SHORT = {"lr": "LR, TF-IDF", "svm": "SVM, TF-IDF", "lr_feat": "LR, engineered features", "scibert": "SciBERT",
         "modernbert": "ModernBERT-base", "mmbert": "mmBERT-base", "lora": "Gemma 4 E4B + LoRA (r = 32)",
         "mb_neighbour": "ModernBERT-base, ± 1 neighbour", "mb_paragraph": "ModernBERT-base, paragraph"}

def from_significance(path):
    S = json.load(open(path)); M = S["models"]; base = S["positive_rate"]
    fig, ax = plt.subplots(figsize=(5.9, 4.4))
    for i, (k, ls, col) in enumerate(SIG_PR):
        m = M[k]; R, P = m["pr_curve"]["recall"], m["pr_curve"]["precision"]
        ax.plot(R, P, ls=ls, color=col, lw=1.6 if k != "mb_paragraph" else 2.2,
                label=f"{SHORT[k]} (PR-AUC {m['pr_auc']:.3f})")
        ax.plot([m["recall"]], [m["precision"]], marker=MARKS[i % len(MARKS)], color=col, ms=6, ls="none", zorder=4)
    ax.axhline(base, color="0.6", ls=":", lw=1, label=f"positive rate {base:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(color="0.88", lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Recall (needs a citation)", fontsize=10); ax.set_ylabel("Precision (needs a citation)", fontsize=10)
    ax.tick_params(labelsize=9); ax.legend(fontsize=7.5, loc="lower left", frameon=True)
    plt.tight_layout(); plt.savefig("figures/fig_5_1.png", dpi=300); plt.close()
    keys = sorted(M, key=lambda k: M[k]["f1"])
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    for yv, k in enumerate(keys):
        m = M[k]; lo, hi = m["f1_ci95"]
        ax.errorbar(m["f1"], yv, xerr=[[m["f1"] - lo], [hi - m["f1"]]], fmt="o", color="black", ms=4, capsize=3, lw=1.2)
        ax.annotate(f"{m['f1']:.3f} [{lo:.3f}, {hi:.3f}]", (hi, yv), textcoords="offset points", xytext=(6, -3), fontsize=7.5)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([SHORT[k] for k in keys], fontsize=8.5)
    ax.set_xlim(0.50, 0.77); ax.xaxis.set_major_locator(MultipleLocator(0.05)); ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis="x", which="major", color="0.85", lw=0.7); ax.grid(axis="x", which="minor", color="0.93", lw=0.5)
    ax.set_axisbelow(True); ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Positive-class F1, 95% bootstrap interval", fontsize=9)
    plt.tight_layout(); plt.savefig("figures/fig_5_3.png", dpi=300); plt.close()
    order = ["lr", "svm", "lr_feat", "scibert", "modernbert", "mmbert", "lora", "mb_neighbour", "mb_paragraph"]
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.6))
    for ax, k in zip(axes.flat, order):
        c = M[k]["cm"]; A = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]]); share = A / A.sum(1, keepdims=True)
        ax.imshow(share, cmap="Greys", vmin=0, vmax=1.6)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{A[i, j]:,}\n{100 * share[i, j]:.1f}%", ha="center", va="center", fontsize=7)
        ax.set_title(SHORT[k], fontsize=7.5); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["no", "yes"], fontsize=7); ax.set_yticklabels(["no", "yes"], fontsize=7)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7)
    plt.tight_layout(); plt.savefig("figures/fig_6_1.png", dpi=300); plt.close()
    print("figures/fig_5_1.png, fig_5_3.png, fig_6_1.png written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths"); ap.add_argument("--tokenizer", help="e.g. jhu-clsp/mmBERT-base (with --lengths)")
    ap.add_argument("--sample", type=int, default=20000, help="sentences per class for token lengths")
    ap.add_argument("--pr", nargs="*"); ap.add_argument("--curve", nargs="*"); ap.add_argument("--significance")
    ap.add_argument("--forest", nargs="*"); ap.add_argument("--confusion", nargs="*")
    a = ap.parse_args()
    if a.lengths: lengths(a.lengths, a.tokenizer, a.sample)
    if a.pr: pr(a.pr)
    if a.curve: curve(a.curve)
    if a.significance: from_significance(a.significance)
    if a.forest: forest(a.forest)
    if a.confusion: confusion(a.confusion)
    print("figures/ updated")
