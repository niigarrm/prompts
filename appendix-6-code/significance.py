#!/usr/bin/env python3
"""Bootstrap intervals, paired F1 differences and Holm-corrected McNemar tests on the saved test predictions
(Section 5.5, Table 5.7; inputs of Figs 5.1, 5.3 and 6.1; v1).
"""

import json, os, sys, time
import numpy as np
from scipy.stats import chi2, binomtest
from sklearn.metrics import precision_recall_curve, average_precision_score

ENC = "drop_sentence_f1_lr2e-05_b128_L128_s42"
MODELS = [
    ("lr", "LR, TF-IDF", "lr_drop_f1_s42"),
    ("svm", "SVM, TF-IDF", "svm_drop_f1_s42"),
    ("lr_feat", "LR, engineered features", "lr_feat_drop_f1_s42"),
    ("scibert", "SciBERT", f"scibert_scivocab_uncased_{ENC}"),
    ("modernbert", "ModernBERT-base", f"ModernBERT-base_{ENC}"),
    ("mmbert", "mmBERT-base", f"mmBERT-base_{ENC}"),
    ("lora", "Gemma 4 E4B + LoRA, r = 32", "lora_gemma-4-E4B-it_r32_a64_n100000_4bit_s42_rescored"),
    ("mb_neighbour", "ModernBERT-base, sentence ± 1 neighbour", "ModernBERT-base_drop_neighbour_f1_lr2e-05_b128_L256_s42"),
    ("mb_paragraph", "ModernBERT-base, full paragraph", "ModernBERT-base_drop_paragraph_f1_lr2e-05_b128_L512_s42"),
]
PAIRS = [
    ("modernbert", "scibert", "5.1"), ("modernbert", "mmbert", "5.1"), ("scibert", "mmbert", "5.1"),
    ("modernbert", "lr", "5.1"), ("lr_feat", "lr", "5.1"), ("lr", "svm", "5.1"),
    ("lora", "modernbert", "5.1, 5.3.2"), ("lora", "lr", "5.1, 5.3.2"),
    ("mb_neighbour", "modernbert", "5.4.1"), ("mb_paragraph", "modernbert", "5.4.1"), ("mb_paragraph", "mb_neighbour", "5.4.1"),
]
N_BOOT, SEED, MIN_DIFF, ALPHA = 1000, 0, 0.01, 0.05

def load(path):
    ids, y, p = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); ids.append(r["id"]); y.append(int(r["label"])); p.append(float(r["prob"]))
    return ids, np.array(y, dtype=np.int8), np.array(p)

def f1_from_codes(counts):
    tp, fp, fn = counts[..., 3], counts[..., 2], counts[..., 1]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)

def mcnemar(b, c):
    if b + c == 0: return 0.0, 1.0, "none"
    if b + c < 25: return float("nan"), float(binomtest(min(b, c), b + c, 0.5).pvalue), "exact binomial"
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(chi2.sf(stat, 1)), "chi-square, continuity corrected"

def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj

def fmt_p(p):
    return "< 1e-300" if p == 0 else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")

def main():
    t0 = time.time()
    missing = [f for _, _, n in MODELS for f in (f"preds/{n}_test_natural.jsonl", f"results/{n}.json") if not os.path.exists(f)]
    if missing: raise SystemExit("missing files:\n  " + "\n  ".join(missing))
    ref_ids = y = None; M = {}
    for key, label, name in MODELS:
        ids, yy, p = load(f"preds/{name}_test_natural.jsonl")
        res = json.load(open(f"results/{name}.json")); thr = float(res["threshold_from_val"])
        if ref_ids is None:
            ref_ids, y = ids, yy; pos = {i: k for k, i in enumerate(ref_ids)}
        elif ids != ref_ids:
            assert len(ids) == len(ref_ids) and set(ids) == pos.keys(), f"{name}: not the same test sentences"
            o = np.empty(len(ids), dtype=np.int64); o[[pos[i] for i in ids]] = np.arange(len(ids))
            yy, p = yy[o], p[o]
        assert np.array_equal(yy, y), f"{name}: labels differ from the first model"
        h = (p >= thr).astype(np.int8); codes = 2 * h + y
        cm = np.bincount(codes, minlength=4); f1 = float(f1_from_codes(cm))
        f1_file = res["test"]["test_natural"]["f1_pos"]
        assert abs(f1 - f1_file) < 1e-6, f"{name}: F1 {f1:.6f} here but {f1_file:.6f} in the results file"
        P, R, _ = precision_recall_curve(y, p); ap = float(average_precision_score(y, p))
        o = np.argsort(R); Rs, Ps = R[o], P[o]
        grid = np.linspace(0, 1, 400); idx = np.unique(np.clip(np.searchsorted(Rs, grid), 0, len(Rs) - 1))
        M[key] = dict(label=label, name=name, thr=thr, codes=codes, correct=(h == y), f1=f1,
                      cm=dict(TN=int(cm[0]), FN=int(cm[1]), FP=int(cm[2]), TP=int(cm[3])),
                      precision=float(cm[3] / max(cm[3] + cm[2], 1)), recall=float(cm[3] / max(cm[3] + cm[1], 1)),
                      pr_auc=ap, pr_curve=dict(recall=[round(float(v), 5) for v in Rs[idx]], precision=[round(float(v), 5) for v in Ps[idx]]))
        print(f"{key:13s} thr {thr:.3f}  F1 {f1:.4f} (file {f1_file:.4f})  PR-AUC {ap:.4f}  {time.time()-t0:.0f}s", flush=True)
    n = len(y); keys = [k for k, _, _ in MODELS]
    rng = np.random.default_rng(SEED); boot = {k: np.empty(N_BOOT) for k in keys}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for k in keys: boot[k][b] = f1_from_codes(np.bincount(M[k]["codes"][idx], minlength=4))
        if b % 100 == 99: print(f"bootstrap {b+1}/{N_BOOT}  {time.time()-t0:.0f}s", flush=True)
    for k in keys: M[k]["f1_ci95"] = [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]
    rows = []
    for a, b_, sec in PAIRS:
        ca, cb = M[a]["correct"], M[b_]["correct"]
        n_a_only = int((ca & ~cb).sum()); n_b_only = int((~ca & cb).sum()); both_wrong = int((~ca & ~cb).sum())
        stat, p, test = mcnemar(n_a_only, n_b_only); d = boot[a] - boot[b_]
        rows.append(dict(A=a, B=b_, section=sec, A_label=M[a]["label"], B_label=M[b_]["label"],
                         dF1=M[a]["f1"] - M[b_]["f1"], dF1_ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         A_right_B_wrong=n_a_only, A_wrong_B_right=n_b_only, discordant=n_a_only + n_b_only,
                         both_wrong=both_wrong, test=test, statistic=stat, p=p))
    adj = holm(np.array([r["p"] for r in rows]))
    for r, pa in zip(rows, adj):
        r["p_holm"] = float(pa); r["significant"] = bool(pa < ALPHA); r["ci_excludes_0"] = bool(r["dF1_ci95"][0] > 0 or r["dF1_ci95"][1] < 0)
        r["meaningful_proposed_rule"] = bool(r["significant"] and abs(r["dF1"]) >= MIN_DIFF)
    out = dict(n_test=n, positive_rate=float(y.mean()), n_boot=N_BOOT, seed=SEED, min_diff=MIN_DIFF, alpha=ALPHA,
               models={k: {kk: vv for kk, vv in M[k].items() if kk not in ("codes", "correct")} for k in keys}, pairs=rows,
               seconds=round(time.time() - t0))
    json.dump(out, open("results/significance.json", "w"), indent=1)
    with open("results/significance.md", "w") as f:
        f.write(f"Natural test set: {n:,} sentences, {100*y.mean():.2f}% positive. Each model at its own validation threshold.\n\n")
        f.write("| Model | Threshold | P | R | F1 [95% CI] | PR-AUC | TN | FP | FN | TP |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for k in keys:
            m = M[k]; c = m["cm"]
            f.write(f"| {m['label']} | {m['thr']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                    f"[{m['f1_ci95'][0]:.3f}, {m['f1_ci95'][1]:.3f}] | {m['pr_auc']:.3f} | {c['TN']:,} | {c['FP']:,} | {c['FN']:,} | {c['TP']:,} |\n")
        f.write("\n| A | B | dF1 (A - B) [95% CI] | A right, B wrong | A wrong, B right | McNemar p | Holm p | Meaningful (Holm p < 0.05 and abs(dF1) >= 0.01) |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['A_label']} | {r['B_label']} | {r['dF1']:+.3f} [{r['dF1_ci95'][0]:+.3f}, {r['dF1_ci95'][1]:+.3f}] | "
                    f"{r['A_right_B_wrong']:,} | {r['A_wrong_B_right']:,} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} | "
                    f"{'yes' if r['meaningful_proposed_rule'] else 'no'} |\n")
    print(open("results/significance.md").read()); print(f"done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
