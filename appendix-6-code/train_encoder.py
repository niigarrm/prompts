#!/usr/bin/env python3
"""Fine-tunes one encoder with one setting and one seed (Section 4.3, v2)."""

import argparse, dataclasses, json, os, random, shutil, time
import numpy as np
from evaluate import evaluate_run, best_threshold, metrics, subset_preds

FIELDS = {"sentence": "text", "neighbour": "text_neighbour", "paragraph": "text_paragraph"}

def read(path, col="text", limit=0):
    if not os.path.exists(path):
        raise SystemExit(f"missing file: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            rows.append({"id": r["id"], "label": int(r["label"]), "paper_id": r["paper_id"], "text": r[col]})
            if limit and len(rows) >= limit: break
    return rows

def subsample(rows, fraction, seed):
    if fraction >= 1: return rows
    papers = sorted({r["paper_id"] for r in rows}); random.Random(seed).shuffle(papers)
    keep = set(papers[:max(1, int(len(papers) * fraction))])
    return [r for r in rows if r["paper_id"] in keep]

def training_args(**kw):
    from transformers import TrainingArguments
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "warmup_ratio" in kw and "warmup_ratio" not in known:
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    if "eval_strategy" in kw and "eval_strategy" not in known and "evaluation_strategy" in known:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    for k in [k for k in kw if k not in known]:
        print(f"NOTE: TrainingArguments has no '{k}' in this transformers version; ignored"); kw.pop(k)
    return TrainingArguments(**kw)

def softmax_pos(logits):
    z = np.asarray(logits, dtype=np.float64); z = z - z.max(-1, keepdims=True)
    e = np.exp(z); return e[:, 1] / e.sum(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--variant", default="drop", choices=["drop", "keep"])
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--context", default="sentence", choices=list(FIELDS))
    ap.add_argument("--splits", default="data/splits"); ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="dry run: only the first N rows of every file")
    ap.add_argument("--cross", action="store_true", help="also score the other variant's natural test set")
    ap.add_argument("--no-test", action="store_true", help="validation only (grid runs, dry runs)")
    ap.add_argument("--predict", nargs="*", default=[], help="extra files to score (tag = file name)")
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--keep-model", action="store_true")
    a = ap.parse_args()
    if a.cross and a.context != "sentence":
        raise SystemExit("--cross is defined for sentence input only")

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                              DataCollatorWithPadding, EarlyStoppingCallback)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    d = f"{a.splits}/{a.variant}"; sfx = "" if a.context == "sentence" else f"_{a.context}"; col = FIELDS[a.context]
    tr = subsample(read(f"{d}/train_bal{sfx}.jsonl", col, a.limit), a.fraction, a.seed)
    va = read(f"{d}/val_natural{sfx}.jsonl", col, a.limit)
    tests = {}
    if not a.no_test:
        tests["test_natural"] = f"{d}/test_natural{sfx}.jsonl"
        if a.cross:
            other = "keep" if a.variant == "drop" else "drop"
            tests[f"test_natural_{other}"] = f"{a.splits}/{other}/test_natural.jsonl"
        for p in a.predict: tests[os.path.basename(p).split(".")[0]] = p
    print(f"train {len(tr):,} sentences ({sum(r['label'] for r in tr):,} positive) from "
          f"{len({r['paper_id'] for r in tr}):,} papers; validation {len(va):,} ({sum(r['label'] for r in va):,} positive)", flush=True)

    short = a.model.rstrip("/").split("/")[-1]
    name = f"{short}_{a.variant}_{a.context}_f{a.fraction:g}_lr{a.lr:g}_b{a.batch}_L{a.max_len}{('_'+a.tag) if a.tag else ''}_s{a.seed}"
    out = f"runs_gpu/{name}"; os.makedirs(out, exist_ok=True); os.makedirs("preds", exist_ok=True); os.makedirs("results", exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    def enc(rows):
        ds = Dataset.from_list([{"text": r["text"], "label": r["label"]} for r in rows])
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len), batched=True, remove_columns=["text"])
    dtr, dva = enc(tr), enc(va)

    def compute(ev):
        logits, y = ev[0], ev[1]
        p = softmax_pos(logits); y = np.asarray(y).astype(int)
        return {"f1_pos": metrics(y, p, best_threshold(y, p))["f1_pos"]}

    args = training_args(output_dir=out, learning_rate=a.lr, per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=256, num_train_epochs=a.epochs, warmup_ratio=0.06, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1_pos",
        greater_is_better=True, save_total_limit=1, seed=a.seed, fp16=a.fp16, bf16=a.bf16, logging_steps=200,
        report_to=[], dataloader_num_workers=a.workers)
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute, callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    t0 = time.time(); trainer.train(); train_s = time.time() - t0

    def predict(rows, path):
        prob = softmax_pos(trainer.predict(enc(rows)).predictions)
        with open(path, "w", encoding="utf-8") as f:
            for r, p in zip(rows, prob): f.write(json.dumps({"id": r["id"], "label": r["label"], "prob": float(p)}) + "\n")
        return path
    t1 = time.time()
    vp = predict(va, f"preds/{name}_val.jsonl"); del dva
    tp = {}
    for tag, path in tests.items():
        tp[tag] = predict(read(path, col if tag == "test_natural" else "text", a.limit), f"preds/{name}_{tag}.jsonl")
    sub = f"{d}/llm_subset.jsonl"
    if "test_natural" in tp and os.path.exists(sub) and not a.limit:
        tp["llm_subset"] = subset_preds(tp["test_natural"], sub, f"preds/{name}_llm_subset.jsonl")
    score_s = time.time() - t1

    extra = {"model": a.model, "variant": a.variant, "context": a.context, "fraction": a.fraction, "lr": a.lr,
             "batch": a.batch, "epochs_max": a.epochs, "epochs_run": round(float(trainer.state.epoch or 0), 2),
             "max_len": a.max_len, "seed": a.seed, "cross": a.cross, "train_sentences": len(tr),
             "train_papers": len({r["paper_id"] for r in tr}), "val_sentences": len(va), "train_seconds": round(train_s),
             "score_seconds": round(score_s), "best_val_f1_pos": trainer.state.best_metric,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "limit": a.limit}
    evaluate_run(vp, tp, name, extra=extra)
    with open("results/log.csv", "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{json.dumps(extra)}\n")
    if a.keep_model:
        trainer.save_model(f"models/{name}"); tok.save_pretrained(f"models/{name}")
    shutil.rmtree(out, ignore_errors=True)

if __name__ == "__main__":
    main()
