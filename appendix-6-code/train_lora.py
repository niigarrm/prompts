#!/usr/bin/env python3
"""QLoRA fine-tuning of the chosen language model and scoring with the Yes/No answer probability
(Section 4.4.2, v3.1).
"""

import argparse, contextlib, dataclasses, json, os, random, time
import numpy as np
from evaluate import evaluate_run

VAL_SEED = 42
SKIP_PARTS = ("vision", "visual", "audio", "multi_modal_projector", "mm_projector", "embed_vision", "embed_audio")

def read(path, limit=0):
    if not os.path.exists(path): raise SystemExit(f"missing file: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); rows.append({"id": r["id"], "label": int(r["label"]), "text": r["text"]})
                if limit and len(rows) >= limit: break
    return rows

def load_data(a):
    d = f"{a.splits}/{a.variant}"
    tr = read(f"{d}/train_bal.jsonl"); random.Random(a.seed).shuffle(tr); tr = tr[:a.train_size]
    va = read(f"{d}/val_natural.jsonl"); random.Random(VAL_SEED).shuffle(va); va = va[:a.val_size]
    tests = {"llm_subset": f"{d}/llm_subset.jsonl"}
    if a.full_test: tests["test_natural"] = f"{d}/test_natural.jsonl"
    return tr, va, tests

def lora_targets(model, wanted):
    names = []
    for n, m in model.named_modules():
        last = n.split(".")[-1]
        if last in wanted and not any(p in n for p in SKIP_PARTS) and hasattr(m, "weight"):
            names.append(n)
    if not names: raise SystemExit(f"no layers named {wanted} found; check --targets for this model")
    return names

def answer_ids(tok):
    ids = {w: tok.encode(w, add_special_tokens=False) for w in ("Yes", "No")}
    for w, v in ids.items():
        if len(v) != 1: print(f"NOTE: '{w}' is {len(v)} tokens for this tokenizer; the first one is used")
    yes, no = ids["Yes"][0], ids["No"][0]
    if yes == no: raise SystemExit("'Yes' and 'No' start with the same token; cannot score")
    return yes, no

def training_args(**kw):
    from transformers import TrainingArguments
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "warmup_ratio" in kw and "warmup_ratio" not in known:
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    for k in [k for k in kw if k not in known]:
        print(f"NOTE: TrainingArguments has no '{k}' in this transformers version; ignored"); kw.pop(k)
    return TrainingArguments(**kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--variant", default="drop", choices=["drop", "keep"])
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train-size", type=int, default=100_000); ap.add_argument("--val-size", type=int, default=5_000)
    ap.add_argument("--load-4bit", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="data/splits"); ap.add_argument("--limit", type=int, default=0,
                    help="dry run: only the first N sentences of each test file")
    ap.add_argument("--full-test", action="store_true"); ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--adapter", default="", help="score this trained adapter instead of training a new one")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import Trainer, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from prompt_llm import PROMPT, load_llm, chat_text
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)

    tr, va, tests = load_data(a)
    short = a.model.rstrip("/").split("/")[-1]
    name = f"lora_{short}_r{a.rank}_a{a.alpha}_n{len(tr)}{'_4bit' if a.load_4bit else ''}_s{a.seed}"
    if a.adapter:
        if not os.path.isdir(a.adapter): raise SystemExit(f"missing adapter folder: {a.adapter}")
        name += "_rescored"
    model, tok, proc = load_llm(a.model, a.load_4bit)
    if a.load_4bit: model = prepare_model_for_kbit_training(model)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter); targets = []
    else:
        targets = lora_targets(model, set(a.targets.split(",")))
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    yes_id, no_id = answer_ids(tok)

    def encode(r):
        p = tok(chat_text(tok, proc, PROMPT.format(sentence=r["text"])), add_special_tokens=False)["input_ids"]
        ans = [yes_id if r["label"] == 1 else no_id]
        return {"input_ids": p + ans, "attention_mask": [1] * (len(p) + 1), "labels": [-100] * len(p) + ans}
    train_s = 0.0
    if not a.adapter:
        tok.padding_side = "right"
        dtr = Dataset.from_list([encode(r) for r in tr])
        args = training_args(output_dir=f"runs_gpu/{name}", learning_rate=a.lr, per_device_train_batch_size=a.batch,
            gradient_accumulation_steps=max(1, 32 // a.batch), num_train_epochs=a.epochs, lr_scheduler_type="cosine",
            warmup_ratio=0.03, logging_steps=50, save_strategy="no", bf16=True, seed=a.seed, report_to=[],
            remove_unused_columns=False)
        trainer = Trainer(model=model, args=args, train_dataset=dtr,
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
        t0 = time.time(); trainer.train(); train_s = time.time() - t0

    model.eval(); tok.padding_side = "left"
    @torch.no_grad()
    def predict(rows, path):
        os.makedirs("preds", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(0, len(rows), a.eval_batch):
                batch = rows[i:i + a.eval_batch]
                enc = tok([chat_text(tok, proc, PROMPT.format(sentence=r["text"])) for r in batch], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(model.device)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if (a.adapter and torch.cuda.is_available()) \
                    else contextlib.nullcontext()
                with amp:
                    logits = model(**enc).logits[:, -1, :]
                pr = torch.softmax(torch.stack([logits[:, no_id], logits[:, yes_id]], -1).float(), -1)[:, 1]
                for r, p in zip(batch, pr.tolist()): f.write(json.dumps({"id": r["id"], "label": r["label"], "prob": p}) + "\n")
        return path
    t1 = time.time()
    vp = predict(va, f"preds/{name}_val.jsonl")
    tp = {tag: predict(read(path, a.limit), f"preds/{name}_{tag}.jsonl") for tag, path in tests.items()}
    extra = {"model": a.model, "rank": a.rank, "alpha": a.alpha, "dropout": a.dropout, "targets": a.targets,
             "lora_layers": len(targets) if targets else "from adapter", "adapter": a.adapter or None, "quantisation": "nf4 4-bit (QLoRA)" if a.load_4bit else "bf16",
             "epochs": a.epochs, "lr": a.lr, "schedule": "cosine, warmup 3%", "batch_effective": a.batch * max(1, 32 // a.batch),
             "train_size": len(tr), "val_size": len(va), "val_seed": VAL_SEED, "seed": a.seed,
             "train_seconds": round(train_s), "score_seconds": round(time.time() - t1), "limit": a.limit,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    evaluate_run(vp, tp, name, extra=extra)
    with open("results/log.csv", "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{json.dumps(extra)}\n")
    if not a.adapter:
        model.save_pretrained(f"runs_gpu/{name}/adapter")

if __name__ == "__main__":
    main()
