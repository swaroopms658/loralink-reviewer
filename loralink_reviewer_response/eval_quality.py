"""Held-out perplexity + BLEU/ROUGE for a LoRA adapter (or bare base model).

Standard HuggingFace eval script used by notebook 02 (`02_task_quality.ipynb`).
Not a reimplementation of any LoraLink component.

`peft` and `evaluate` are imported lazily inside the functions that need them so
this module imports cleanly in an environment where only torch / transformers /
datasets are installed (the offline test relies on that).
"""
from __future__ import annotations

import csv
import math
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

_COLS = ["arm", "seed", "dataset", "base_model", "perplexity", "bleu",
         "rougeL", "n_eval", "adapter_dir", "slice_bounds"]


def _load(base_model, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
    if adapter_dir:
        from peft import PeftModel  # lazy: only needed for adapter eval
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok


def _load_e2e(limit):
    """load GEM/e2e_nlg, tolerating datasets v4 dropping trust_remote_code."""
    try:
        ds = load_dataset("GEM/e2e_nlg", split="validation",
                          cache_dir="./dataset", trust_remote_code=True)
        print("[eval_quality] loaded GEM/e2e_nlg with trust_remote_code=True")
    except TypeError:
        ds = load_dataset("GEM/e2e_nlg", split="validation", cache_dir="./dataset")
        print("[eval_quality] loaded GEM/e2e_nlg without trust_remote_code (datasets v4)")
    return ds.select(range(min(limit, len(ds))))


@torch.no_grad()
def _perplexity(model, tok, texts, max_length=256):
    dev = next(model.parameters()).device
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(dev)
        if ids.numel() < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.numel() - 1
        nll += out.loss.item() * n
        ntok += n
    return math.exp(nll / max(ntok, 1))


@torch.no_grad()
def _gen_bleu_rouge(model, tok, mrs, refs, max_new_tokens):
    import evaluate as _ev  # lazy: only needed for the e2e path
    dev = next(model.parameters()).device
    preds = []
    for mr in mrs:
        prompt = f"Data: {mr}\nText:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        preds.append(tok.decode(gen[0][enc.input_ids.shape[1]:],
                                skip_special_tokens=True).strip())
    bleu = _ev.load("sacrebleu").compute(
        predictions=preds, references=[[r] for r in refs])["score"]
    rouge = _ev.load("rouge").compute(
        predictions=preds, references=refs)["rougeL"]
    return bleu, rouge * 100.0


def evaluate_adapter(base_model, adapter_dir, dataset, *, eval_holdout=200,
                     max_new_tokens=48, limit=100, arm="", seed=0,
                     out_csv="results_quality.csv"):
    model, tok = _load(base_model, adapter_dir)
    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          cache_dir="./dataset")
        texts = [x for x in ds["text"] if x.strip()][:limit]
        ppl = _perplexity(model, tok, texts)
        bleu = rougeL = ""
        n_eval = len(texts)
        bounds = f"test[0:{len(texts)}]"
    else:  # e2e (anything not "wikitext")
        ds = _load_e2e(limit)
        mrs = ds["meaning_representation"]
        refs = ds["target"]
        texts = [f"Data: {m}\nText: {t}" for m, t in zip(mrs, refs)]
        ppl = _perplexity(model, tok, texts)
        bleu, rougeL = _gen_bleu_rouge(model, tok, mrs, refs, max_new_tokens)
        n_eval = len(ds)
        bounds = f"validation[0:{len(ds)}]"

    row = {"arm": arm, "seed": seed, "dataset": dataset,
           "base_model": base_model, "perplexity": round(ppl, 4),
           "bleu": bleu if bleu == "" else round(bleu, 3),
           "rougeL": rougeL if rougeL == "" else round(rougeL, 3),
           "n_eval": n_eval, "adapter_dir": adapter_dir or "(base)",
           "slice_bounds": bounds}
    p = pathlib.Path(out_csv)
    need = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        if need:
            w.writeheader()
        w.writerow(row)
    return row
