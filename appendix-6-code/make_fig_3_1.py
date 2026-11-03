#!/usr/bin/env python3
"""Draws the dataset construction pipeline (Fig. 2.1); the numbers are copied from the run logs."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "svg.fonttype": "none", "pdf.fonttype": 42})

W, H = 5.75, 6.55
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK, GREY, FILL = "black", "#555555", "#ececec"
LW = 0.7

def rbox(x, y, w, h, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                lw=LW, ec=INK, fc=fill))

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=LW, color=INK, shrinkA=0, shrinkB=0))

L, R = 0.05, W - 0.05
C1, C2 = 1.62, 4.18
RH, GAP = 0.50, 0.24
rows = [
    ("Source corpus",          ["unarXive open subset, files from 2010 on", "(5,437 of 5,599 files)"],
                               ["163,323 papers"]),
    ("Discipline filter",      ["Computer Science only"],
                               ["50,790 papers"]),
    ("Sentence segmentation",  ["spaCy sentencizer; abbreviations,", "decimals and initials protected"],
                               ["12,717,320 sentences"]),
    ("Labelling and cleaning", ["label 1 if a citation marker was present;", "markers removed or replaced by FORMULA / REF"],
                               ["label 0 / 1 per sentence"]),
    ("Quality filter",         ["12 rules: fragments, non-prose,", "unresolved citation keys (Table 3.2)"],
                               ["10,187,815 sentences"]),
    ("Hole detection",         ["hole sentences dropped (Table 3.3); a copy with", "holes kept is used for the ablation (6.4.2)"],
                               ["10,020,763 sentences", "12.30% positive"]),
    ("Document-level split",   ["papers divided 60 / 20 / 20"],
                               ["29,968 / 9,989 / 9,990", "papers"]),
]
removed = {1: "112,533 papers of other disciplines", 4: "2,529,505 sentences rejected",
           5: "167,052 sentences with a hole"}

top = H - 0.30
for x, t in ((L + (C1 - L) / 2, "Step"), (C1 + (C2 - C1) / 2, "Operation"), (C2 + (R - C2) / 2, "Result")):
    ax.text(x, top + 0.13, t, ha="center", va="center", fontsize=8.5, style="italic", color=GREY)

ys = []
for i, (step, what, out) in enumerate(rows):
    y = top - (i + 1) * RH - i * GAP
    ys.append(y)
    rbox(L, y, R - L, RH)
    for xd in (C1, C2):
        ax.plot([xd, xd], [y, y + RH], lw=LW * 0.8, color=INK)
    ax.text(L + 0.08, y + RH / 2, f"{i + 1}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(L + 0.26, y + RH / 2, step, ha="left", va="center", fontsize=9, fontweight="bold")
    ax.text(C1 + 0.08, y + RH / 2, "\n".join(what), ha="left", va="center", fontsize=8.3, linespacing=1.25)
    ax.text(C2 + (R - C2) / 2, y + RH / 2, "\n".join(out), ha="center", va="center", fontsize=8.5,
            linespacing=1.2)

xa = C1 + 0.25
for i in range(len(rows) - 1):
    arrow(xa, ys[i], xa, ys[i + 1] + RH)
    if i in removed:
        ax.text(xa + 0.10, ys[i] - GAP / 2, "− " + removed[i], ha="left", va="center",
                fontsize=7.8, style="italic", color=GREY)

sets = [("Training set", ["balanced (undersampled)", "400,018", "50% positive"]),
        ("Validation set", ["natural rate", "499,199", "12.43% positive"]),
        ("Test set", ["natural rate", "2,002,611", "12.24% positive"]),
        ("LLM subset", ["from the test set", "3,000", "12.0% positive"])]
bw, bh = 1.30, 0.72
g = (R - L - 4 * bw) / 3
by = 0.30
ybus = ys[-1] - 0.20
ax.plot([xa, xa], [ys[-1], ybus], lw=LW, color=INK)
centres = [L + bw / 2 + k * (bw + g) for k in range(4)]
ax.plot([centres[0], centres[-1]], [ybus, ybus], lw=LW, color=INK)
for (t, l), cxk in zip(sets, centres):
    rbox(cxk - bw / 2, by, bw, bh, fill=FILL)
    arrow(cxk, ybus, cxk, by + bh)
    ax.text(cxk, by + bh - 0.10, t, ha="center", va="top", fontsize=9, fontweight="bold")
    ax.text(cxk, by + bh - 0.29, "\n".join(l), ha="center", va="top", fontsize=8.1, linespacing=1.15)

os.makedirs("figures", exist_ok=True)
for ext, kw in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
    fig.savefig(f"figures/fig_3_1.{ext}", **kw)
print("wrote figures/fig_3_1.png / .pdf / .svg")
