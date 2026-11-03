#!/usr/bin/env python3
"""Zero-shot and few-shot prompting of the language models on the 3,000-sentence subset
(Section 4.4.1, Table 5.4; v2.1).
"""

import argparse, json, os, re, time, random
from evaluate import evaluate_run

PROMPT = ("You are checking a sentence from a computer-science research paper. "
          "Decide whether this sentence, as written in a published paper, needs a citation "
          "to support it — a claim, result, method, dataset or system that comes from prior work. "
          "Answer with exactly one word: Yes or No.\n\nSentence: {sentence}\nAnswer:")

SHOTS_FILE = "prompts/shots.json"

def get_shots(train_path="data/splits/drop/train_bal.jsonl", seed=42, k_each=3):
    if os.path.exists(SHOTS_FILE):
        return [tuple(x) for x in json.load(open(SHOTS_FILE, encoding="utf-8"))["shots"]]
    rng = random.Random(seed); res = {0: [], 1: []}; seen = {0: 0, 1: 0}
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line); n = len(r["text"].split()); y = int(r["label"])
            if not 8 <= n <= 40: continue
            seen[y] += 1
            if len(res[y]) < k_each: res[y].append(r)
            else:
                j = rng.randrange(seen[y])
                if j < k_each: res[y][j] = r
    shots = []
    for i in range(k_each):
        shots.append((res[1][i]["text"], "Yes")); shots.append((res[0][i]["text"], "No"))
    os.makedirs(os.path.dirname(SHOTS_FILE), exist_ok=True)
    json.dump({"source": train_path, "seed": seed, "rule": "3 positive + 3 negative, 8-40 words, reservoir sample",
               "ids": [r["id"] for i in range(k_each) for r in (res[1][i], res[0][i])], "shots": shots},
              open(SHOTS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"few-shot examples drawn and saved to {SHOTS_FILE}")
    return shots

def build(sentence, mode, shots=None):
    if mode == "zero":
        return PROMPT.format(sentence=sentence)
    demo = "\n\n".join(f"Sentence: {s}\nAnswer: {a}" for s, a in shots)
    return PROMPT.replace("Sentence: {sentence}\nAnswer:", demo + f"\n\nSentence: {sentence}\nAnswer:")

def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().lower()
    m = re.match(r"\W*(yes|no)\b", t)
    return (1 if m.group(1) == "yes" else 0) if m else None


def dtype_kw(torch):
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch.bfloat16} if (major, minor) >= (4, 56) else {"torch_dtype": torch.bfloat16}

def load_llm(model_id, load_4bit=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    kw = {"device_map": "auto", **dtype_kw(torch)}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw); kind = "causal-lm"
    except (ValueError, KeyError) as e:
        print(f"NOTE: AutoModelForCausalLM failed ({str(e)[:120]}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw); kind = "image-text-to-text"
    tok = AutoTokenizer.from_pretrained(model_id)
    proc = None
    if not getattr(tok, "chat_template", None):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    print(f"loaded {model_id} as {kind} ({type(model).__name__}); chat template from {'processor' if proc else 'tokenizer'}")
    return model, tok, proc

THINK_PREFILL = [False]

def chat_text(tok, proc, user):
    text = _chat_text(tok, proc, user)
    if THINK_PREFILL[0] and not text.rstrip().endswith("</think>"):
        text = text + ("\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n")
    return text

def _chat_text(tok, proc, user):
    if proc is not None:
        msgs, owner = [{"role": "user", "content": [{"type": "text", "text": user}]}], proc
    else:
        msgs, owner = [{"role": "user", "content": user}], tok
    try:    return owner.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError: return owner.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

class HF:
    def __init__(self, model, load_4bit, max_new):
        import torch
        self.torch, self.max_new = torch, max_new
        self.model, self.tok, self.proc = load_llm(model, load_4bit); self.model.eval()
    def batch(self, users):
        texts = [chat_text(self.tok, self.proc, u) for u in users]
        self.tok.padding_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

class MLX:
    def __init__(self, model, max_new):
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok = load(model); self.gen = generate; self.sampler = make_sampler(temp=0.0); self.max_new = max_new
    def batch(self, users):
        res = []
        for user in users:
            p = chat_text(self.tok, None, user)
            res.append(self.gen(self.model, self.tok, prompt=p, max_tokens=self.max_new, sampler=self.sampler, verbose=False))
        return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "mlx"], default="hf"); ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["zero", "few"], default="zero"); ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--input", default="data/splits/drop/llm_subset.jsonl"); ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="dry run: only the first N sentences")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--think-prefill", action="store_true", help="empty <think></think> block before the answer")
    a = ap.parse_args(); THINK_PREFILL[0] = a.think_prefill
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.limit: rows = rows[:a.limit]
    shots = get_shots() if a.mode == "few" else None
    tag = a.tag or os.path.basename(a.input).split(".")[0]
    name = f"llm_{a.model.rstrip('/').split('/')[-1]}_{a.mode}_{tag}" + (f"_limit{a.limit}" if a.limit else "")
    llm = MLX(a.model, a.max_new_tokens) if a.backend == "mlx" else HF(a.model, a.load_4bit, a.max_new_tokens)
    os.makedirs("preds", exist_ok=True); os.makedirs("results", exist_ok=True)
    t0 = time.time(); fails = 0
    with open(f"preds/{name}.jsonl", "w", encoding="utf-8") as out, \
         open(f"results/{name}_raw.jsonl", "w", encoding="utf-8") as raw_log:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i:i + a.batch]
            for r, text in zip(chunk, llm.batch([build(r["text"], a.mode, shots) for r in chunk])):
                y = parse(text)
                if y is None: fails += 1; y = 1 - int(r["label"])
                out.write(json.dumps({"id": r["id"], "label": r["label"], "prob": float(y)}) + "\n")
                raw_log.write(json.dumps({"id": r["id"], "raw": text}, ensure_ascii=False) + "\n")
            print(f"{min(i + a.batch, len(rows))}/{len(rows)}  fails={fails}  {time.time()-t0:.0f}s", flush=True)
    gpu = "cpu"
    try:
        import torch
        if torch.cuda.is_available(): gpu = torch.cuda.get_device_name(0)
    except ImportError: pass
    evaluate_run(f"preds/{name}.jsonl", {tag: f"preds/{name}.jsonl"}, name, fixed_threshold=0.5,
                 extra={"backend": a.backend, "model": a.model, "mode": a.mode, "parse_failures": fails,
                        "parse_failure_rate": fails / len(rows), "n": len(rows), "elapsed_s": round(time.time() - t0),
                        "batch": a.batch, "max_new_tokens": a.max_new_tokens, "load_4bit": a.load_4bit,
                        "shots_file": SHOTS_FILE if shots else None, "gpu": gpu, "think_prefill": a.think_prefill})
    with open("results/log.csv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{name},{json.dumps({'model': a.model, 'mode': a.mode, 'parse_failures': fails, 'n': len(rows)})}\n")
    print(f"parse failures: {fails}/{len(rows)}")

if __name__ == "__main__":
    main()
