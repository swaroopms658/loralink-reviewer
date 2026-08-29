"""Merge experiment CSVs + published baselines into reviewer-response tables/plots.

`main.py` writes two files per run: per-batch ``RUN_COLUMNS`` rows to
``<metrics_csv>`` (e.g. ``results_stat_<tag>.csv``) and the single
``SUMMARY_COLUMNS`` row to a sibling ``<stem>.summary.csv``.  So summary-level
tables read ``suffix=".summary.csv"`` and per-batch/loss-curve data reads the
plain ``.csv`` (with ``.summary.csv`` filtered out).  ``results_quality_*`` from
``eval_quality.py`` is a single self-contained schema and keeps its plain name.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pathlib
import re

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loralink_reviewer_response.statlib import mean_std_ci  # noqa: E402

PROV = ("[ours] = we ran it on Colab Free T4 - [published] = transcribed, "
        "see baselines/SOURCES.md")
LOOPBACK = "single-box loopback simulation - not WAN"


# --------------------------------------------------------------------------- io

def _read_many(results_dir, prefix, suffix=".csv"):
    """Concat every ``*{prefix}*{suffix}`` CSV under *results_dir*.

    ``suffix=".summary.csv"`` matches only the summary siblings; ``suffix=".csv"``
    matches the per-batch files and EXCLUDES anything ending ``.summary.csv``.
    Returns ``None`` when nothing matches / nothing is readable.
    """
    if suffix == ".summary.csv":
        files = sorted(glob.glob(os.path.join(results_dir, f"*{prefix}*.summary.csv")))
    else:
        files = [f for f in sorted(glob.glob(os.path.join(results_dir, f"*{prefix}*.csv")))
                 if not f.endswith(".summary.csv")]
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except (pd.errors.EmptyDataError, OSError):
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _write_csv(df, path, note):
    """DataFrame to CSV with a leading ``# <provenance>`` comment line."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {note}\n")
        df.to_csv(fh, index=False, float_format="%.4g")


def _round(obj):
    """Recursively round floats to 4dp and turn non-finite floats into None,
    so summary.json carries no 17-digit noise and no invalid ``NaN`` token."""
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(float(obj), 4)
    return obj


def _save_fig(f, ax, path_stem, caption):
    f.text(0.01, 0.01, caption, fontsize=6, wrap=True)
    f.tight_layout(rect=(0, 0.06, 1, 1))
    f.savefig(f"{path_stem}.png", dpi=150)
    f.savefig(f"{path_stem}.pdf")
    plt.close(f)


# --------------------------------------------------------------------------- T1

def _t1(results_dir, fig, summary):
    df = _read_many(results_dir, "results_stat_", ".summary.csv")
    if df is None:
        print("WARN: no data for T1")
        return
    caption = f"T1 stat-validation (mean +/- 95% CI over seeds) [ours]. {PROV}"
    rows, summ = [], {}
    for ds, g in df.groupby("dataset"):
        for col in ("mean_loss", "mean_step_latency_s", "overall_comp_ratio"):
            if col not in g:
                continue
            vals = _num(g[col]).dropna().tolist()
            if not vals:
                continue
            st = mean_std_ci(vals)
            st["tag"] = "[ours]"
            rows.append({"dataset": ds, "metric": col, **st})
            summ.setdefault(ds, {})[col] = st
    if rows:
        _write_csv(pd.DataFrame(rows), fig / "T1_stat_validation.csv", caption)
    summary["stat_validation"] = summ

    conv = _read_many(results_dir, "results_converge_", ".csv")
    f, ax = plt.subplots(figsize=(6, 4))
    if conv is not None and "global_batch" in conv and "loss" in conv:
        for ds, g in conv.groupby("dataset"):
            g = g.assign(_x=_num(g["global_batch"]), _y=_num(g["loss"])).dropna(
                subset=["_x", "_y"]).sort_values("_x")
            if not g.empty:
                ax.plot(g["_x"], g["_y"], label=f"{ds} [ours]")
        ax.set_xlabel("mini-batch")
        ax.set_ylabel("cross-entropy loss")
    else:
        for ds, g in df.groupby("dataset"):
            g = g.assign(_x=_num(g.get("seed")), _y=_num(g.get("mean_loss"))).dropna(
                subset=["_x", "_y"])
            if not g.empty:
                ax.scatter(g["_x"], g["_y"], label=f"{ds} [ours]")
        ax.set_xlabel("seed")
        ax.set_ylabel("mean loss")
    ax.set_title("Convergence / loss stability [ours]")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _save_fig(f, ax, str(fig / "T1_loss_curve"), caption)


# --------------------------------------------------------------------------- T2

def _t2(results_dir, fig, summary):
    df = _read_many(results_dir, "results_quality_", ".csv")
    if df is None:
        print("WARN: no data for T2")
        return
    caption = ("T2 task quality: compression ON vs OFF vs reference [ours] "
               f"(PPL lower better; BLEU/ROUGE higher better). {PROV}")
    out, summ = [], {}
    for ds, g in df.groupby("dataset"):
        agg = {}
        for arm, ga in g.groupby("arm"):
            for m in ("perplexity", "bleu", "rougeL"):
                if m not in ga:
                    continue
                vals = _num(ga[m]).dropna().tolist()
                if not vals:
                    continue
                st = mean_std_ci(vals)
                agg[(arm, m)] = st
                out.append({"dataset": ds, "arm": arm, "metric": m, "tag": "[ours]", **st})
        for m, key in (("perplexity", "delta_ppl_on_minus_off"),
                       ("bleu", "delta_bleu_on_minus_off"),
                       ("rougeL", "delta_rougeL_on_minus_off")):
            if ("ON", m) in agg and ("OFF", m) in agg:
                summ.setdefault(ds, {})[key] = agg[("ON", m)]["mean"] - agg[("OFF", m)]["mean"]
        if ds in summ:
            summ[ds]["tag"] = "[ours]"
            summ[ds]["n"] = int(df[df["dataset"] == ds].get("seed", pd.Series(dtype=float)).nunique() or len(g))
    if not out:
        print("WARN: no data for T2")
        return
    odf = pd.DataFrame(out)
    _write_csv(odf, fig / "T2_quality_vs_compression.csv", caption)
    summary["quality"] = summ

    piv = odf.pivot_table(index=["dataset", "metric"], columns="arm", values="mean")
    f, ax = plt.subplots(figsize=(7, 4))
    piv.plot.bar(ax=ax)
    ax.set_title("Task quality: compression ON vs OFF vs reference [ours]")
    ax.set_ylabel("value")
    _save_fig(f, ax, str(fig / "T2_quality_bars"), caption)


# --------------------------------------------------------------------------- T3

def _t3(results_dir, baselines_csv, fig, summary):
    try:
        bl = pd.read_csv(baselines_csv)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        bl = None
    if bl is None or bl.empty:
        print("WARN: no data for T3")
        return
    caption = ("T3 LoraLink vs published baselines. Metrics/units differ across "
               "sources - this is primarily a TABLE; the plot is a categorical "
               f"strip of raw values (log y) and is NOT a like-for-like comparison. {PROV}")
    rows = []

    # ours: quality perplexity per arm/dataset + stat compression ratio
    q = _read_many(results_dir, "results_quality_", ".csv")
    if q is not None:
        for (ds, arm), g in q.groupby(["dataset", "arm"]):
            v = _num(g.get("perplexity")).dropna()
            if not v.empty:
                rows.append({"method": f"LoraLink ({arm})", "metric": "perplexity",
                             "dataset": ds, "value": float(v.mean()),
                             "comparable": "direct", "tag": "[ours]"})
    s = _read_many(results_dir, "results_stat_", ".summary.csv")
    if s is not None and "overall_comp_ratio" in s:
        for ds, g in s.groupby("dataset"):
            v = _num(g["overall_comp_ratio"]).dropna()
            if not v.empty:
                rows.append({"method": "LoraLink", "metric": "overall_comp_ratio",
                             "dataset": ds, "value": float(v.mean()),
                             "comparable": "direct", "tag": "[ours]"})

    # published
    for _, r in bl.iterrows():
        val = pd.to_numeric(pd.Series([r.get("value")]), errors="coerce").iloc[0]
        rows.append({
            "method": str(r.get("method", "")),
            "metric": f"{r.get('metric', '')} [{r.get('unit', '')}]",
            "dataset": str(r.get("dataset", "")),
            "value": None if pd.isna(val) else float(val),
            "comparable": str(r.get("comparable", "")),
            "tag": f"[published, {r.get('source_ref', '')}]",
        })

    tdf = pd.DataFrame(rows)
    _write_csv(tdf, fig / "T3_ours_vs_published.csv", caption)
    summary["ours_vs_published"] = {
        "rows": rows,
        "n_ours": int((tdf["tag"] == "[ours]").sum()),
        "n_published": int(tdf["tag"].str.startswith("[published").sum()),
    }

    f, ax = plt.subplots(figsize=(8, 4.5))
    plotted = tdf[tdf["value"].notna() & (tdf["value"] > 0)].reset_index(drop=True)
    if not plotted.empty:
        for is_ours, marker in ((True, "o"), (False, "^")):
            sub = plotted[plotted["tag"].str.startswith("[ours]") == is_ours]
            if not sub.empty:
                ax.scatter(sub.index, sub["value"], marker=marker,
                           label="ours" if is_ours else "published", s=60)
        ax.set_yscale("log")
        ax.set_xticks(range(len(plotted)))
        ax.set_xticklabels(plotted["method"] + "\n" + plotted["metric"],
                           rotation=90, fontsize=6)
        ax.legend()
    else:
        ax.bar(tdf["comparable"].value_counts().index, tdf["comparable"].value_counts().values)
    ax.set_ylabel("raw value (units differ - see table)")
    ax.set_title("LoraLink vs published (NOT like-for-like) [ours] vs [published]")
    _save_fig(f, ax, str(fig / "T3_scatter"), caption)


# --------------------------------------------------------------------------- T4

def _t4(results_dir, fig, summary):
    df = _read_many(results_dir, "results_sched_", ".summary.csv")
    if df is None:
        print("WARN: no data for T4")
        return
    caption = ("T4 scheduling: partition balance std + mean step latency by "
               f"strategy [ours]. Infeasible strategies shown as a gap. {PROV}")
    rows, summ = [], {}
    note_col = df["note"].astype(str).str.lower() if "note" in df else None
    for strat, g in df.groupby("strategy"):
        idx = g.index
        nb = _num(g["n_batches"]).fillna(0) if "n_batches" in g else pd.Series(1, index=idx)
        infeasible_mask = nb <= 0
        if note_col is not None:
            infeasible_mask = infeasible_mask | note_col.loc[idx].str.contains("infeasible")
        feasible = g[~infeasible_mask]
        if feasible.empty:
            row = {"strategy": strat, "partition_balance_std": None,
                   "mean_step_latency_s": None, "status": "infeasible", "n": 0,
                   "tag": "[ours]"}
        else:
            bstd = _num(feasible["partition_balance_std"]).dropna()
            lat = _num(feasible["mean_step_latency_s"]).dropna()
            row = {
                "strategy": strat,
                "partition_balance_std": float(bstd.mean()) if not bstd.empty else None,
                "mean_step_latency_s": float(lat.mean()) if not lat.empty else None,
                "status": "ok",
                "n": int(len(feasible)),
                "tag": "[ours]",
            }
        rows.append(row)
        summ[strat] = {k: row[k] for k in ("partition_balance_std",
                                           "mean_step_latency_s", "status", "n", "tag")}
    rdf = pd.DataFrame(rows)
    _write_csv(rdf, fig / "T4_scheduling.csv", caption)
    summary["scheduling"] = summ

    f, ax = plt.subplots(figsize=(7, 4))
    x = range(len(rdf))
    w = 0.35
    ax.bar([i - w / 2 for i in x], rdf["partition_balance_std"].fillna(0), w,
           label="partition_balance_std")
    ax.bar([i + w / 2 for i in x], rdf["mean_step_latency_s"].fillna(0), w,
           label="mean_step_latency_s")
    for i, st in enumerate(rdf["status"]):
        if st == "infeasible":
            ax.annotate("infeasible", (i, 0), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7, rotation=90)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rdf["strategy"], rotation=30, ha="right")
    ax.set_title("Scheduling: balance vs latency by strategy [ours]")
    ax.legend()
    _save_fig(f, ax, str(fig / "T4_bars"), caption)


# --------------------------------------------------------------------------- T5

def _t5(results_dir, fig, summary):
    df = _read_many(results_dir, "results_scale_", ".summary.csv")
    if df is None:
        print("WARN: no data for T5")
        return
    caption = (f"T5 scalability: latency + throughput vs worker count [ours]. "
               f"{LOOPBACK}. {PROV}")
    df = df.assign(_nw=_num(df.get("n_workers")), _lat=_num(df.get("mean_step_latency_s")))
    df = df.dropna(subset=["_nw", "_lat"])
    rows = []
    for nw, g in df.groupby("_nw"):
        lat = g["_lat"].mean()
        thr = (nw / lat) if lat and lat > 0 else None
        rows.append({"n_workers": int(nw), "mean_step_latency_s": float(lat),
                     "throughput_workers_per_s": None if thr is None else float(thr),
                     "tag": "[ours]"})
    rdf = pd.DataFrame(rows).sort_values("n_workers")
    if rdf.empty:
        print("WARN: no data for T5")
        return
    _write_csv(rdf, fig / "T5_scalability_sim.csv", caption)
    summary["scalability"] = {
        "by_workers": rdf.to_dict(orient="records"),
        "tag": "[ours]", "n": int(len(rdf)), "note": LOOPBACK,
    }

    f, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rdf["n_workers"], rdf["mean_step_latency_s"], "o-", label="mean step latency (s)")
    ax2 = ax.twinx()
    ax2.plot(rdf["n_workers"], rdf["throughput_workers_per_s"], "s--",
             color="tab:orange", label="throughput (workers/s)")
    ax.set_xlabel("n_workers")
    ax.set_ylabel("mean step latency (s)")
    ax2.set_ylabel("throughput (workers/s)")
    ax.set_title(f"Scalability ({LOOPBACK}) [ours]")
    _save_fig(f, ax, str(fig / "T5_lines"), caption)


# --------------------------------------------------------------------------- T6

_NET_RE = re.compile(r"net-d(?P<delay>[0-9.]+)-l(?P<loss>[0-9.]+)")


def _t6(results_dir, fig, summary):
    df = _read_many(results_dir, "results_net_", ".summary.csv")
    if df is None:
        print("WARN: no data for T6")
        return
    caption = (f"T6 network: mean step latency over (delay x loss) [ours]. "
               f"loopback + tc/netem emulation, {LOOPBACK}. {PROV}")

    def _parse(tag, col):
        m = _NET_RE.search(str(tag))
        return float(m.group(col)) if m else float("nan")

    if "delay" in df and "loss" in df:
        df = df.assign(_delay=_num(df["delay"]), _loss=_num(df["loss"]))
    else:
        df = df.assign(_delay=df["run_tag"].map(lambda t: _parse(t, "delay")),
                       _loss=df["run_tag"].map(lambda t: _parse(t, "loss")))
    df = df.assign(_lat=_num(df.get("mean_step_latency_s"))).dropna(
        subset=["_delay", "_loss", "_lat"])
    if df.empty:
        print("WARN: no data for T6")
        return
    grid = df.groupby(["_delay", "_loss"])["_lat"].mean().reset_index()
    grid.columns = ["delay", "loss", "mean_step_latency_s"]
    grid["tag"] = "[ours]"
    _write_csv(grid, fig / "T6_network.csv", caption)
    summary["network"] = {
        "cells": grid.to_dict(orient="records"),
        "tag": "[ours]", "n": int(len(grid)), "note": f"loopback + tc/netem, {LOOPBACK}",
    }

    piv = grid.pivot_table(index="delay", columns="loss", values="mean_step_latency_s")
    f, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("packet loss (%)")
    ax.set_ylabel("added delay (ms)")
    ax.set_title(f"Network sensitivity ({LOOPBACK}) [ours]")
    f.colorbar(im, ax=ax, label="mean step latency (s)")
    _save_fig(f, ax, str(fig / "T6_heatmap"), caption)


# ------------------------------------------------------------------------- api

def build_all(results_dir, baselines_csv, out_dir):
    fig = pathlib.Path(out_dir)
    fig.mkdir(parents=True, exist_ok=True)
    summary = {"provenance": PROV, "loopback_disclaimer": LOOPBACK}
    _t1(results_dir, fig, summary)
    _t2(results_dir, fig, summary)
    _t3(results_dir, baselines_csv, fig, summary)
    _t4(results_dir, fig, summary)
    _t5(results_dir, fig, summary)
    _t6(results_dir, fig, summary)
    (fig / "summary.json").write_text(
        json.dumps(_round(summary), indent=2, allow_nan=False, default=float),
        encoding="utf-8")
    return summary


def render_response(summary_json, template_md, out_md):
    data = json.loads(pathlib.Path(summary_json).read_text(encoding="utf-8"))
    flat = {}

    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            flat[prefix] = obj

    _walk("", data)
    txt = pathlib.Path(template_md).read_text(encoding="utf-8")
    for k, v in flat.items():
        rep = format(v, ".4g") if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        txt = txt.replace("{{" + k + "}}", rep)
    leftover = sorted(set(re.findall(r"\{\{[^}]+\}\}", txt)))
    if leftover:
        print("WARN: unfilled placeholder(s): " + ", ".join(leftover))
    pathlib.Path(out_md).write_text(txt, encoding="utf-8")
    return out_md
