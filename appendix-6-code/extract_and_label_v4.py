#!/usr/bin/env python3
"""Extraction v4.1 used for the full dataset (E-010, Chapter 2): sentence segmentation, distant-supervision labels,
marker cleaning, quality filter and hole detection.
"""

SCRIPT_VERSION = "v4.1"

import argparse
import glob
import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter

import spacy


INPUT_DIR = "unarxive_data"
OUTPUT_FILE = "sentences_labelled.jsonl"
DISCIPLINE = "Computer Science"
MIN_YEAR = 2010

MIN_TOKENS = 5
MAX_TOKENS = 100
MIN_ALPHA_RATIO = 0.70
MIN_REAL_WORDS = 4

BATCH_SIZE = 200
SEED = 42

FORMULA_TOKEN = "FORMULA"
REF_TOKEN = "REF"


ABBREVIATIONS = [
    "et al.", "e.g.", "i.e.", "cf.", "vs.", "etc.", "approx.", "resp.",
    "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.", "tab.", "tabs.",
    "ref.", "refs.", "alg.", "app.", "ch.", "def.", "thm.", "prop.",
    "lem.", "cor.", "no.", "vol.", "pp.", "dr.", "prof.", "mr.", "ms.",
    "mrs.", "st.", "inc.", "ltd.", "a.k.a.", "w.r.t.", "s.t.", "i.i.d.",
]

DOT = "\uE000"
DECIMAL_RE = re.compile(r"(\d)\.(\d)")
INITIAL_RE = re.compile(r"\b([A-Z])\.\s")
YYMM_RE = re.compile(r"arXiv_src_(\d{2})(\d{2})_")

_ABBR_RE = re.compile(
    "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True)),
    re.IGNORECASE,
)


def protect_abbreviations(text):
    text = _ABBR_RE.sub(lambda m: m.group(0).replace(".", DOT), text)
    text = DECIMAL_RE.sub(r"\1" + DOT + r"\2", text)
    text = INITIAL_RE.sub(r"\1" + DOT + " ", text)
    return text


def restore_abbreviations(text):
    return text.replace(DOT, ".")


def year_from_filename(path):
    m = YYMM_RE.search(os.path.basename(path))
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def build_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    return nlp


CITE_RE = re.compile(r"\{\{cite:[^}]*\}\}")
FORMULA_RE = re.compile(r"\{\{formula:[^}]*\}\}")
OTHER_MARKER_RE = re.compile(r"\{\{[^}]*\}\}")

REPEATED_PUNCT_RE = re.compile(r"([,;:])(?:\s*[,;:])+")
PUNCT_BEFORE_TERMINAL_RE = re.compile(r"[,;:]+\s*([.!?])")
EMPTY_BRACKETS_RE = re.compile(r"\(\s*[,;:]*\s*\)|\[\s*[,;:]*\s*\]")
DANGLING_OPEN_RE = re.compile(r"[\(\[]\s*(?=[.,;:])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
MULTISPACE_RE = re.compile(r"\s{2,}")
SPACE_IN_PAREN_RE = re.compile(r"\(\s+|\s+\)")

EMPTY_XREF_PAREN_RE = re.compile(
    r"\(\s*(?:Section|Sec\.|Chapter|Figure|Fig\.|Table|Tab\.|Equation|Eq\.|"
    r"Algorithm|Alg\.|Appendix|App\.)\s*\)", re.IGNORECASE)
BARE_XREF_RE = re.compile(
    r"\b(Sections?|Chapters?|Figures?|Fig\.|Tables?|Tab\.|Equations?|Eq\.|"
    r"Algorithms?|Alg\.|Appendix|App\.)\s+(?=[.,;:)])", re.IGNORECASE)

LATEX_LABEL_RE = re.compile(
    r"\b(?:eq|eqn|sec|subsec|fig|tab|thm|lem|prop|cor|def|defn|app|alg|ch|"
    r"rem|clm|ex|obs|assum)[:_][A-Za-z0-9_:\-]+", re.IGNORECASE)

NAME_TELL_RE = re.compile(
    r"\b[A-Z][a-zA-Z\-']+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z\-']+)?\s+et\s+al\.?")
YEAR_PAREN_RE = re.compile(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)")


def clean_sentence(text, remove_name_tells=True):
    text = CITE_RE.sub(" ", text)
    text = FORMULA_RE.sub(FORMULA_TOKEN, text)
    text = OTHER_MARKER_RE.sub(REF_TOKEN, text)

    if remove_name_tells:
        text = NAME_TELL_RE.sub(" ", text)
        text = YEAR_PAREN_RE.sub(" ", text)

    text = LATEX_LABEL_RE.sub(REF_TOKEN, text)
    text = EMPTY_XREF_PAREN_RE.sub("(" + REF_TOKEN + ")", text)
    text = BARE_XREF_RE.sub(r"\1 " + REF_TOKEN, text)

    text = EMPTY_BRACKETS_RE.sub(" ", text)
    text = DANGLING_OPEN_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = PUNCT_BEFORE_TERMINAL_RE.sub(r"\1", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = SPACE_IN_PAREN_RE.sub(lambda m: m.group(0).strip(), text)
    text = TOKEN_POSSESSIVE_RE.sub(r"\1's", text)
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


TOKEN_POSSESSIVE_RE = re.compile(r"\b(FORMULA|REF)\s+'s\b")
WHITESPACE_RE = re.compile(r"\s+")
BLOCK_MARKER_AFTER_END_RE = re.compile(
    r"([.!?])[ \t\u00a0]*(\{\{(?:figure|table):)")

BIBKEY_RE = re.compile(r"\b[A-Za-z][a-z\-]+(?:19|20)\d{2}[a-z]{3,}\b")


HOLE_PATTERNS = [
    (re.compile(
        r"\b(?:of|in|by|from|to|with|for|as|than|via|see|following|like|"
        r"upon|over|under|through|between|among|into|onto)\s*[.!?]\s*$",
        re.IGNORECASE), "trailing_preposition"),
    (re.compile(r"\s+'s\b"), "orphan_possessive"),
    (re.compile(
        r"^(?:In|By|From|Unlike|Following|Like|Similar\s+to|Compared\s+to|"
        r"According\s+to|Based\s+on)\s*,", re.IGNORECASE), "initial_gap"),
    (re.compile(
        r"\b(?:of|in|by|from|to|with|for|as|than|via|following|like)\s+,",
        re.IGNORECASE), "midsentence_gap"),
    (re.compile(
        r"(?:\bas|,)\s+(?:points?\s+out|shows?|showed|notes?|noted|"
        r"argues?|argued|proposed?|proposes|suggests?|demonstrates?|"
        r"introduced?|introduces)\b"), "missing_subject"),
    (re.compile(
        r"^(?:proposed|showed|introduced|presented|demonstrated|argued|"
        r"suggested|developed)\b"), "leading_verb"),
]


def find_hole(text):
    for rx, name in HOLE_PATTERNS:
        if rx.search(text):
            return name
    return None


URL_RE = re.compile(r"https?://|www\.")
REAL_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

ACK_RE = re.compile(
    r"^(?:We|I|The\s+authors?)\b[^.]{0,60}?\b(?:are\s+grateful|thanks?|"
    r"would\s+like\s+to\s+thank|acknowledges?|gratefully)\b", re.IGNORECASE)

THEOREM_FRAGMENT_RE = re.compile(
    r"^(?:Lemma|Theorem|Corollary|Proposition|Definition|Claim|Remark|Fact|"
    r"Observation|Example|Assumption|Algorithm)\s+\d+", re.IGNORECASE)


def passes_quality(text):
    tokens = text.split()

    if len(tokens) < MIN_TOKENS:
        return False, "too_short"
    if len(tokens) > MAX_TOKENS:
        return False, "too_long"
    if not text[0].isupper():
        return False, "no_capital_start"
    if text[-1] not in ".!?":
        return False, "no_terminal_punct"

    if len(REAL_WORD_RE.findall(text)) < MIN_REAL_WORDS:
        return False, "too_few_real_words"
    if THEOREM_FRAGMENT_RE.match(text):
        return False, "theorem_fragment"
    if ACK_RE.match(text):
        return False, "acknowledgement"

    alpha = sum(1 for c in text if c.isalpha() or c.isspace())
    if alpha / len(text) < MIN_ALPHA_RATIO:
        return False, "low_alpha_ratio"

    if URL_RE.search(text):
        return False, "contains_url"
    if "{{" in text or "}}" in text:
        return False, "leftover_marker"
    if BIBKEY_RE.search(text):
        return False, "unresolved_citation_key"

    return True, None


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def select_files(input_dir, n_files, min_year, seed):
    all_files = sorted(
        glob.glob(os.path.join(input_dir, "**", "*.jsonl*"), recursive=True))
    if not all_files:
        return [], 0, 0
    eligible = [p for p in all_files if (year_from_filename(p) or 0) >= min_year]
    if n_files is None or n_files >= len(eligible):
        return eligible, len(all_files), len(eligible)
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, n_files)), len(all_files), len(eligible)


def iter_paragraphs(paper):
    for idx, entry in enumerate(paper.get("body_text", [])):
        if not isinstance(entry, dict):
            continue
        if entry.get("content_type") != "paragraph":
            continue
        t = entry.get("text", "")
        if t and t.strip():
            yield idx, BLOCK_MARKER_AFTER_END_RE.sub(r"\1\n\2", t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--files", type=int, default=None)
    p.add_argument("--min-year", type=int, default=MIN_YEAR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--keep-holes", action="store_true",
                   help="Keep sentences with a syntactic hole (for measuring "
                        "how much the confound is worth).")
    p.add_argument("--keep-name-tells", action="store_true")
    args = p.parse_args()

    files, total, eligible = select_files(
        args.input_dir, args.files, args.min_year, args.seed)
    if not files:
        print(f"No usable files in {args.input_dir!r}.", file=sys.stderr)
        return 1

    print(f"Archive: {total:,} files, {eligible:,} from {args.min_year} onward.")
    print(f"Processing {len(files):,} file(s), seed {args.seed} -> {args.output}")
    print(f"Syntactic holes: {'KEPT' if args.keep_holes else 'DROPPED'}\n")

    with open(args.output + ".manifest.txt", "w", encoding="utf-8") as mf:
        mf.write(f"# script_version={SCRIPT_VERSION} seed={args.seed} "
                 f"min_year={args.min_year} n_files={len(files)} "
                 f"keep_holes={args.keep_holes} "
                 f"keep_name_tells={args.keep_name_tells}\n")
        mf.writelines(p_ + "\n" for p_ in files)

    nlp = build_nlp()
    stats, rejected, years = Counter(), Counter(), Counter()
    holes_pos, holes_neg = Counter(), Counter()
    start = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for path in files:
            year = year_from_filename(path)
            try:
                fh = open_maybe_gzip(path)
            except OSError:
                continue
            with fh as f:
                for line in f:
                    try:
                        paper = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stats["papers_seen"] += 1
                    if DISCIPLINE and paper.get("discipline") != DISCIPLINE:
                        continue
                    stats["papers_kept"] += 1
                    years[year] += 1
                    pid = paper.get("paper_id") or paper.get("id")

                    paras = list(iter_paragraphs(paper))
                    if not paras:
                        continue
                    protected = [protect_abbreviations(t) for _, t in paras]

                    for (pi, _), doc in zip(
                            paras, nlp.pipe(protected, batch_size=BATCH_SIZE)):
                        for si, sent in enumerate(doc.sents):
                            raw = restore_abbreviations(sent.text).strip()
                            if not raw:
                                continue
                            stats["sentences_segmented"] += 1

                            label = 1 if CITE_RE.search(raw) else 0
                            text = clean_sentence(
                                raw, remove_name_tells=not args.keep_name_tells)
                            if not text:
                                rejected["empty_after_cleaning"] += 1
                                continue

                            ok, why = passes_quality(text)
                            if not ok:
                                rejected[why] += 1
                                continue

                            hole = find_hole(text)
                            if hole:
                                (holes_pos if label else holes_neg)[hole] += 1
                                stats["holes_in_positive" if label
                                      else "holes_in_negative"] += 1
                                if not args.keep_holes:
                                    rejected["syntactic_hole"] += 1
                                    continue

                            stats["sentences_kept"] += 1
                            stats["positive" if label else "negative"] += 1
                            out.write(json.dumps({
                                "paper_id": pid, "year": year,
                                "paragraph_index": pi, "sentence_index": si,
                                "text": text, "label": label,
                                "hole": hole,
                            }, ensure_ascii=False) + "\n")

                    if stats["papers_kept"] % 2000 == 0:
                        print(f"  {stats['papers_kept']:,} papers, "
                              f"{stats['sentences_kept']:,} sentences, "
                              f"{time.time()-start:.0f}s", flush=True)

    el = time.time() - start
    print("\n--- Extraction summary ---")
    for k in ["papers_seen", "papers_kept", "sentences_segmented",
              "sentences_kept", "positive", "negative"]:
        print(f"  {k:26s} {stats[k]:,}")

    kept = stats["sentences_kept"]
    if kept:
        print(f"\n  positive rate              "
              f"{100.0*stats['positive']/kept:.2f}%")

    print("\n--- Syntactic holes found ---")
    tp, tn = sum(holes_pos.values()), sum(holes_neg.values())
    print(f"  in POSITIVE sentences      {tp:,}")
    for k, v in holes_pos.most_common():
        print(f"      {k:24s} {v:,}")
    print(f"  in NEGATIVE sentences      {tn:,}")
    for k, v in holes_neg.most_common():
        print(f"      {k:24s} {v:,}")
    if tp + tn:
        print(f"\n  {100.0*tp/(tp+tn):.1f}% of holes are in positives.")
        print("  A strong asymmetry is the confound: report this figure.")
    pos_pop = stats["positive"] + (0 if args.keep_holes else tp)
    neg_pop = stats["negative"] + (0 if args.keep_holes else tn)
    if pos_pop and neg_pop:
        print(f"  {100.0*tp/pos_pop:.2f}% of quality-passing positives carry "
              f"a hole; {100.0*tn/neg_pop:.2f}% of negatives do.")

    summary = {
        "script_version": SCRIPT_VERSION, "seed": args.seed,
        "min_year": args.min_year, "n_files": len(files),
        "keep_holes": args.keep_holes, "keep_name_tells": args.keep_name_tells,
        "stats": dict(stats), "rejected": dict(rejected),
        "holes_positive": dict(holes_pos), "holes_negative": dict(holes_neg),
        "papers_by_year": {str(k): v for k, v in sorted(years.items())},
        "elapsed_s": round(el, 1),
    }
    with open(args.output + ".stats.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=1)

    if rejected:
        print("\n--- Rejected by reason ---")
        seg = max(stats["sentences_segmented"], 1)
        for r, n in rejected.most_common():
            print(f"  {r:26s} {n:,}  ({100.0*n/seg:.1f}%)")

    print(f"\n--- Timing ---\n  elapsed {el:.0f}s, "
          f"{el/max(len(files),1):.2f}s/file, "
          f"full corpus ~{el/max(len(files),1)*eligible/60:.0f} min")
    print(f"\nWrote {kept:,} sentences to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
