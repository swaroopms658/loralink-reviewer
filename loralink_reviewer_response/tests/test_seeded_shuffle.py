"""Seeds must vary the training run, not just LoRA's initialization.

Reviewer concern 1 (R3.4 / R3-Q7) asks for "repeated runs, standard deviations,
confidence intervals". The loader used `shuffle=False`, so all five seeds trained
on the same samples in the same order; the only seed-dependent quantity was
LoRA's `A` init, and since `B` starts at zero and 60 steps at lr=1e-4 barely move
it, the runs were near-identical (cross-seed std 0.0016 on a loss of 4.77). That
reports determinism, not robustness.

Training now shuffles under a generator seeded from `--seed`. Evaluation stays
ordered so held-out metrics remain comparable across runs. Batch size stays 1,
matching the paper's hyperparameter table.
"""
import torch

import data_loader


class _FakeDS(list):
    column_names = ["input_ids", "attention_mask"]

    def select(self, rng):
        return _FakeDS(self[i] for i in rng)

    def set_format(self, *a, **k):
        pass

    def map(self, *a, **k):
        return self

    def filter(self, *a, **k):
        return self

    def __len__(self):
        return list.__len__(self)


def _rows(n=25):
    return [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])}
            for i in range(n)]


def _order(monkeypatch, tok, *, seed, split="train"):
    monkeypatch.setattr(data_loader, "load_dataset", lambda **k: _FakeDS(_rows()))
    loader = data_loader.get_data_loader(
        tok, num_samples=20, dataset_name="wikitext", split=split, seed=seed)
    return [int(b["input_ids"][0]) for b in loader]


def test_same_seed_reproduces_order(tokenizer, monkeypatch):
    a = _order(monkeypatch, tokenizer, seed=0)
    b = _order(monkeypatch, tokenizer, seed=0)
    assert a == b, "a seed must reproduce its own data order"


def test_different_seeds_give_different_orders(tokenizer, monkeypatch):
    a = _order(monkeypatch, tokenizer, seed=0)
    b = _order(monkeypatch, tokenizer, seed=1)
    assert a != b, "seeds must vary the data order, or CI measures nothing"
    assert sorted(a) == sorted(b), "same samples, only the order should differ"


def test_eval_split_stays_ordered(tokenizer, monkeypatch):
    a = _order(monkeypatch, tokenizer, seed=0, split="eval")
    b = _order(monkeypatch, tokenizer, seed=7, split="eval")
    assert a == b, "evaluation order must not depend on the seed"


def test_seedless_call_is_still_deterministic(tokenizer, monkeypatch):
    """Omitting seed keeps the old, unshuffled behaviour."""
    a = _order(monkeypatch, tokenizer, seed=None)
    b = _order(monkeypatch, tokenizer, seed=None)
    assert a == b == sorted(a)


def test_batch_size_stays_one(tokenizer, monkeypatch):
    """The paper's hyperparameter table specifies batch size 1."""
    monkeypatch.setattr(data_loader, "load_dataset", lambda **k: _FakeDS(_rows()))
    loader = data_loader.get_data_loader(
        tokenizer, num_samples=20, dataset_name="wikitext", seed=0)
    assert loader.batch_size == 1
    assert next(iter(loader))["input_ids"].shape[0] == 1


def test_main_passes_the_seed_through():
    """A shuffle the coordinator never seeds would silently reintroduce the bug."""
    import inspect

    import main

    src = inspect.getsource(main.run_coordinator)
    call = src[src.index("get_data_loader("):]
    assert "seed=" in call[:400], "run_coordinator must pass seed= to get_data_loader"
