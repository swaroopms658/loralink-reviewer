import torch
from datasets import load_dataset
from typing import Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ── Dataset Registry ──────────────────────────────────────────────────────────
# Each entry maps a user-friendly name to its HuggingFace loading args and a
# function that converts raw examples into a flat list of text strings.

def _format_wikitext(examples: Dict[str, Any]) -> Dict[str, Any]:
    """WikiText: already has a 'text' column."""
    return {"text": examples["text"]}


def _format_dolly(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Dolly-15k: combine instruction + context + response into a single prompt."""
    texts = []
    for instr, ctx, resp in zip(
        examples["instruction"], examples["context"], examples["response"]
    ):
        parts = [f"### Instruction:\n{instr}"]
        if ctx and ctx.strip():
            parts.append(f"### Context:\n{ctx}")
        parts.append(f"### Response:\n{resp}")
        texts.append("\n\n".join(parts))
    return {"text": texts}


def _format_e2e(examples: Dict[str, Any]) -> Dict[str, Any]:
    """E2E NLG: combine meaning representation and target."""
    texts = []
    for mr, tgt in zip(examples["meaning_representation"], examples["target"]):
        texts.append(f"Data: {mr}\nText: {tgt}")
    return {"text": texts}


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wikitext": {
        "path": "wikitext",
        "config": "wikitext-2-raw-v1",
        "formatter": _format_wikitext,
        "remove_columns": ["text"],
        "description": "WikiText-2 (raw)",
    },
    "dolly": {
        "path": "databricks/databricks-dolly-15k",
        "config": None,
        "formatter": _format_dolly,
        "remove_columns": ["instruction", "context", "response", "category"],
        "description": "Databricks Dolly 15k",
    },
    "e2e": {
        "path": "GEM/e2e_nlg",
        "config": None,
        "formatter": _format_e2e,
        "remove_columns": ["meaning_representation", "target", "references"],
        "description": "E2E NLG Challenge",
        "trust_remote_code": True,
    },
}


# ── Public API ────────────────────────────────────────────────────────────────

def get_data_loader(
    tokenizer: AutoTokenizer,
    num_samples: int = 5,
    dataset_name: str = "wikitext",
    *,
    split: str = "train",
    eval_holdout: int = 0,
) -> torch.utils.data.DataLoader:
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert isinstance(num_samples, int) and num_samples > 0
    assert dataset_name in DATASET_REGISTRY, (
        f"Unknown dataset '{dataset_name}'. "
        f"Available: {list(DATASET_REGISTRY.keys())}"
    )

    ds_info = DATASET_REGISTRY[dataset_name]
    print(f"📊 Loading dataset: {ds_info['description']} ({ds_info['path']})")

    # Load from HuggingFace (uses local cache if already downloaded)
    load_args = {"path": ds_info["path"], "split": "train", "cache_dir": "./dataset"}
    if ds_info["config"]:
        load_args["name"] = ds_info["config"]
    if ds_info.get("trust_remote_code"):
        load_args["trust_remote_code"] = True
    dataset = load_dataset(**load_args)

    # Format into a unified 'text' column
    dataset = dataset.map(ds_info["formatter"], batched=True)

    # Filter out empty texts (wikitext has many blank lines)
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

    # Determine columns to drop (keep only tokenizer outputs)
    remove_cols = [c for c in dataset.column_names if c != "input_ids" and c != "attention_mask"]
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=remove_cols)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    total = len(tokenized_dataset)
    if split == "eval":
        val = None
        for cand in ("validation", "test"):
            try:
                v = load_dataset(**{**load_args, "split": cand})
                v = v.map(ds_info["formatter"], batched=True)
                v = v.filter(lambda x: len(x["text"].strip()) > 0)
                v = v.map(tokenize_function, batched=True, remove_columns=remove_cols)
                v.set_format(type="torch", columns=["input_ids", "attention_mask"])
                val = v
                break
            except Exception:
                continue
        if val is not None:
            end = min(num_samples, len(val))
            subset_dataset = val.select(range(end))
            bounds = (0, end, cand)
        else:
            start = max(0, total - max(num_samples, eval_holdout))
            end = min(total, start + num_samples)
            subset_dataset = tokenized_dataset.select(range(start, end))
            bounds = (start, end, "train-tail")
    else:
        # F8: honour num_samples even when eval_holdout is set.
        end = min(num_samples, total - eval_holdout) if eval_holdout else min(num_samples, total)
        subset_dataset = tokenized_dataset.select(range(0, end))
        bounds = (0, end, "train")
    print(f"📐 slice bounds: {bounds}")

    dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=1, shuffle=False)
    dataloader.slice_bounds = bounds
    print(f"✅ DataLoader ready: {len(subset_dataset)} samples")

    return dataloader
