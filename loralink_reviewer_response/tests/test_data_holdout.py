import torch
from transformers import AutoTokenizer
import data_loader

class _FakeDS(list):
    column_names = ["input_ids", "attention_mask"]
    def select(self, rng): return _FakeDS(self[i] for i in rng)
    def set_format(self, *a, **k): pass
    def map(self, *a, **k): return self
    def filter(self, *a, **k): return self
    def __len__(self): return list.__len__(self)

def test_train_and_eval_slices_are_disjoint(monkeypatch):
    rows = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])}
            for i in range(20)]
    def mock_load_dataset(**k):
        # Only return data for train split; validation/test don't exist
        if k.get("split") == "train":
            return _FakeDS(rows)
        else:
            # No validation/test split - will cause exception and fall back to train-tail
            raise ValueError(f"No split '{k.get('split')}'")
    monkeypatch.setattr(data_loader, "load_dataset", mock_load_dataset)
    tok = AutoTokenizer.from_pretrained("gpt2")           # cached in CI image
    tr = data_loader.get_data_loader(tok, num_samples=15, dataset_name="wikitext",
                                     split="train", eval_holdout=5)
    ev = data_loader.get_data_loader(tok, num_samples=5, dataset_name="wikitext",
                                     split="eval", eval_holdout=5)
    tr_ids = {int(b["input_ids"][0]) for b in tr}
    ev_ids = {int(b["input_ids"][0]) for b in ev}
    assert tr_ids.isdisjoint(ev_ids)
    assert ev.slice_bounds[0] >= 15


def test_eval_uses_validation_split_when_present(monkeypatch):
    # F7: the PRIMARY path — dataset HAS a validation split (wikitext does).
    # The prior implementer's rewrite only covered the train-tail fallback.
    train_rows = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])}
                  for i in range(20)]
    val_rows = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])}
                for i in range(100, 120)]
    def mock_load_dataset(**k):
        split = k.get("split")
        if split == "validation":
            return _FakeDS(val_rows)
        if split == "train":
            return _FakeDS(train_rows)
        raise ValueError(f"No split '{split}'")
    monkeypatch.setattr(data_loader, "load_dataset", mock_load_dataset)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tr = data_loader.get_data_loader(tok, num_samples=15, dataset_name="wikitext",
                                     split="train", eval_holdout=5)
    ev = data_loader.get_data_loader(tok, num_samples=5, dataset_name="wikitext",
                                     split="eval", eval_holdout=5)
    tr_ids = {int(b["input_ids"][0]) for b in tr}
    ev_ids = {int(b["input_ids"][0]) for b in ev}
    assert ev.slice_bounds[2] == "validation"
    assert tr_ids.isdisjoint(ev_ids)
    assert ev_ids == {100, 101, 102, 103, 104}
