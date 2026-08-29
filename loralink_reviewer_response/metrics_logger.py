"""Append-only CSV writer for LoraLink reviewer-response experiments."""
from __future__ import annotations
import csv
import os
from pathlib import Path

RUN_COLUMNS = [
    "run_tag", "seed", "strategy", "compression", "dataset", "model",
    "n_workers", "epoch", "global_batch", "loss", "step_latency_s",
    "comp_ratio", "bytes_sent", "bytes_saved", "sim", "timestamp",
]
SUMMARY_COLUMNS = [
    "run_tag", "seed", "strategy", "compression", "dataset", "model",
    "n_workers", "sim", "n_batches", "mean_loss", "last_loss",
    "mean_step_latency_s", "total_bytes_sent", "total_bytes_saved",
    "overall_comp_ratio", "wall_time_s", "partition_map",
    "partition_balance_std",
]

def append_rows(csv_path, rows, columns):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if need_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
