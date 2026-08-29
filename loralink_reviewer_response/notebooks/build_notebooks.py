"""Generate the 7 sharded Colab experiment notebooks from ``_template.ipynb``.

Each notebook is the 5-cell template with ``MODEL_PLACEHOLDER`` / ``SHARD_PLACEHOLDER``
substituted and cell 4 (index 3) replaced by that notebook's body. Re-running is
idempotent -- it overwrites the 7 ``.ipynb`` files in place.
"""
from __future__ import annotations

import pathlib

import nbformat

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "_template.ipynb"
BODY_CELL = 3  # cells: [setup, model, params, body, download]

_IMPORT = "from loralink_reviewer_response.cluster_launch import run_cluster"

BODIES = {
    "00_setup_smoke": ("EleutherAI/gpt-neo-125M", "", f'''\
# --- Cell 4/5: body -- one tiny end-to-end run, prints SMOKE PASS/FAIL --------
import os, pandas as pd
{_IMPORT}

PER_RUN_ESTIMATE = 120
PLANNED, DONE = 1, 0
csv = f"results_smoke_{{ACCOUNT_TAG}}.csv"

if budget_left() < PER_RUN_ESTIMATE:
    print("budget exhausted, stopping")
else:
    run_cluster(n_workers=2, dataset="wikitext", seed=0, model=MODEL,
                num_samples=6, epochs=1, tag="smoke", results_csv=csv)
    DONE += 1

if os.path.exists(csv):
    df = pd.read_csv(csv)
    n_loss = int(df["loss"].notna().sum()) if "loss" in df.columns else 0
    print("SMOKE PASS" if n_loss >= 3 else "SMOKE FAIL", f"({{n_loss}} loss rows)")
else:
    print("SMOKE FAIL (no csv)")
'''),

    "01_stat_validation": ("EleutherAI/gpt-neo-125M", "wikitext", f'''\
# --- Cell 4/5: body -- 5 seeds x 1 dataset shard ----------------------------
{_IMPORT}

PER_RUN_ESTIMATE = 200
SEEDS = list(range(5))
PLANNED, DONE = len(SEEDS), 0
csv = f"results_stat_{{ACCOUNT_TAG}}.csv"

for seed in SEEDS:
    if budget_left() < PER_RUN_ESTIMATE:
        print("budget exhausted, stopping"); break
    run_cluster(2, SHARD, seed, model=MODEL, num_samples=60, epochs=1,
                compression=True, tag=f"stat-{{SHARD}}-s{{seed}}", results_csv=csv)
    DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),

    "02_task_quality": ("microsoft/phi-1_5", "e2e:0", f'''\
# --- Cell 4/5: body -- compression ON/OFF vs centralized reference ----------
{_IMPORT}
from loralink_reviewer_response.eval_quality import evaluate_adapter

PER_RUN_ESTIMATE = 420
parts = SHARD.split(":")
ds = parts[0]
seed = int(parts[1]) if len(parts) > 1 else 0
print(f"shard: dataset={{ds}} seed={{seed}}")
ARMS = ["ON", "OFF", "reference"]
PLANNED, DONE = len(ARMS), 0
sys_csv = f"results_qsys_{{ACCOUNT_TAG}}.csv"
q_csv = f"results_quality_{{ACCOUNT_TAG}}.csv"

for arm in ARMS:
    if budget_left() < PER_RUN_ESTIMATE:
        print("budget exhausted, stopping"); break
    nw = 1 if arm == "reference" else 2
    adir = f"adapters/{{ds}}_s{{seed}}_{{arm}}"
    run_cluster(nw, ds, seed, model=MODEL, num_samples=50, epochs=1,
                compression=(arm != "OFF"), eval_holdout=200, strategy="smart",
                tag=f"q-{{ds}}-s{{seed}}-{{arm}}", results_csv=sys_csv,
                save_adapters_to=adir)
    evaluate_adapter(MODEL, adir, ds, arm=arm, seed=seed, limit=200, out_csv=q_csv)
    DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),

    "02b_convergence": ("microsoft/phi-1_5", "e2e", f'''\
# --- Cell 4/5: body -- single 3-epoch run to trace the loss curve ----------
{_IMPORT}

PER_RUN_ESTIMATE = 900
PLANNED, DONE = 1, 0
csv = f"results_converge_{{ACCOUNT_TAG}}.csv"

if budget_left() < PER_RUN_ESTIMATE:
    print("budget exhausted, stopping")
else:
    run_cluster(2, "e2e", 0, model=MODEL, num_samples=50, epochs=3,
                compression=True, tag="conv-e2e", results_csv=csv)
    DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),

    "03_alt_scheduling": ("EleutherAI/gpt-neo-125M", "", f'''\
# --- Cell 4/5: body -- 4 partition strategies x 3 seeds -------------------
{_IMPORT}
from loralink_reviewer_response.metrics_logger import SUMMARY_COLUMNS, append_rows

PER_RUN_ESTIMATE = 120
STRATS = ["smart", "round_robin", "proportional", "random"]
SEEDS = list(range(3))
PLANNED, DONE = len(STRATS) * len(SEEDS), 0
csv = f"results_sched_{{ACCOUNT_TAG}}.csv"
summary_csv = f"results_sched_{{ACCOUNT_TAG}}.summary.csv"

stop = False
for strat in STRATS:
    if stop:
        break
    for seed in SEEDS:
        if budget_left() < PER_RUN_ESTIMATE:
            print("budget exhausted, stopping"); stop = True; break
        try:
            run_cluster(4, "wikitext", seed, model=MODEL, strategy=strat,
                        num_samples=30, tag=f"sched-{{strat}}-s{{seed}}",
                        results_csv=csv)
        except RuntimeError as e:  # device_manager.PartitionInfeasible subclasses this
            print(f"  {{strat}} s{{seed}}: infeasible ({{e}})")
            # SUMMARY_COLUMNS only -- append_rows' extrasaction="ignore" drops the
            # extra "note" key so the file keeps main.py's 18-col schema (a wider
            # header makes pandas.read_csv raise ParserError in aggregate._t4).
            append_rows(summary_csv, [{{"run_tag": f"sched-{{strat}}-s{{seed}}",
                                       "seed": seed, "strategy": strat,
                                       "n_batches": 0, "note": "infeasible"}}],
                        SUMMARY_COLUMNS)
        DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),

    "04_scalability_sim": ("EleutherAI/gpt-neo-125M", "", f'''\
# --- Cell 4/5: body -- worker-count sweep with the fake benchmark shim -----
import os
os.environ["LORALINK_FAKE_BENCHMARK"] = "1"
{_IMPORT}

PER_RUN_ESTIMATE = 180
# SHARD may hold a worker-count subset like "6,8" (split across accounts);
# empty = the full [2, 4, 6, 8] sweep. 4 counts x 2 reps ~= 24 min, reaches n=8.
NS = [int(x) for x in SHARD.split(",")] if SHARD.strip() else [2, 4, 6, 8]
REPS = list(range(2))
PLANNED, DONE = len(NS) * len(REPS), 0
csv = f"results_scale_{{ACCOUNT_TAG}}.csv"

stop = False
for n in NS:
    if stop:
        break
    for rep in REPS:
        if budget_left() < PER_RUN_ESTIMATE:
            print("budget exhausted, stopping"); stop = True; break
        run_cluster(n, "wikitext", 0, model=MODEL, num_samples=30,
                    tag=f"scale-n{{n}}-r{{rep}}", results_csv=csv)
        DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),

    "05_network_netem": ("EleutherAI/gpt-neo-125M", "", f'''\
# --- Cell 4/5: body -- delay x loss netem grid ---------------------------
{_IMPORT}

PER_RUN_ESTIMATE = 100
# SHARD may hold a delay subset like "0,25"; empty = all delays.
DELAYS = [int(x) for x in SHARD.split(",")] if SHARD.strip() else [0, 25, 50, 100]
LOSSES = [0, 1, 3]
PLANNED, DONE = len(DELAYS) * len(LOSSES), 0
csv = f"results_net_{{ACCOUNT_TAG}}.csv"

stop = False
for delay in DELAYS:
    if stop:
        break
    for loss in LOSSES:
        if budget_left() < PER_RUN_ESTIMATE:
            print("budget exhausted, stopping"); stop = True; break
        run_cluster(2, "wikitext", 0, model=MODEL, num_samples=20,
                    netem={{"delay_ms": delay, "loss_pct": loss}},
                    tag=f"net-d{{delay}}-l{{loss}}", results_csv=csv)
        DONE += 1
print(f"done {{DONE}}/{{PLANNED}}")
'''),
}


def build_one(name: str, model: str, shard: str, body: str) -> pathlib.Path:
    nb = nbformat.read(TEMPLATE, as_version=4)
    nb.cells[BODY_CELL].source = body
    for cell in nb.cells:
        cell.source = (cell.source
                       .replace("MODEL_PLACEHOLDER", model)
                       .replace("SHARD_PLACEHOLDER", shard))
    out = HERE / f"{name}.ipynb"
    nbformat.write(nb, out)
    return out


def main() -> None:
    for name, (model, shard, body) in BODIES.items():
        print("wrote", build_one(name, model, shard, body).name)


if __name__ == "__main__":
    main()
