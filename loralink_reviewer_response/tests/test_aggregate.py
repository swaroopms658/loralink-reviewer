"""Tests for aggregate.py (Task 11) - synthetic CSVs, no GPU/network."""
import json

import pandas as pd

from loralink_reviewer_response.aggregate import _read_many, build_all, render_response


def _write(p, rows):
    pd.DataFrame(rows).to_csv(p, index=False)


# --- summary column sets (mirror metrics_logger.SUMMARY_COLUMNS) --------------
_SUMMARY_COLS = [
    "run_tag", "seed", "strategy", "compression", "dataset", "model",
    "n_workers", "sim", "n_batches", "mean_loss", "last_loss",
    "mean_step_latency_s", "total_bytes_sent", "total_bytes_saved",
    "overall_comp_ratio", "wall_time_s", "partition_map", "partition_balance_std",
]


def _summary_row(**over):
    base = dict(run_tag="r", seed=0, strategy="smart", compression="on",
                dataset="wikitext", model="gpt-neo-125M", n_workers=2, sim="loopback",
                n_batches=10, mean_loss=3.0, last_loss=2.9, mean_step_latency_s=0.5,
                total_bytes_sent=1000, total_bytes_saved=900, overall_comp_ratio=10.0,
                wall_time_s=5.0, partition_map="a:1;b:1", partition_balance_std=0.0)
    base.update(over)
    return {k: base[k] for k in _SUMMARY_COLS}


def test_stat_and_quality_tables(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    fig = tmp_path / "figures"
    _write(rdir / "results_stat_a.summary.csv", [
        _summary_row(seed=s, mean_loss=3.0 + 0.1 * s, mean_step_latency_s=0.5,
                     overall_comp_ratio=10.0) for s in range(5)])
    _write(rdir / "results_quality_b.csv", [
        {"arm": a, "seed": 0, "dataset": "e2e", "base_model": "phi-1_5",
         "perplexity": p, "bleu": bl, "rougeL": rl, "n_eval": 100}
        for a, p, bl, rl in [("ON", 22.0, 4.1, 18.0), ("OFF", 21.6, 4.3, 18.4),
                             ("reference", 21.4, 4.5, 18.7)]])
    bl = tmp_path / "baselines.csv"
    _write(bl, [{"method": "SplitLoRA", "source_ref": "x", "model": "LLaMA",
                 "params": "7B", "dataset": "E2E NLG",
                 "metric": "communication_per_round", "value": "120", "unit": "MB",
                 "hardware": "2x3090", "comparable": "direct", "notes": ""}])
    out = build_all(str(rdir), str(bl), str(fig))
    assert (fig / "T1_stat_validation.csv").exists()
    assert (fig / "T1_loss_curve.png").exists()
    assert (fig / "T2_quality_vs_compression.csv").exists()
    assert (fig / "T2_quality_bars.png").exists()
    s = json.loads((fig / "summary.json").read_text())
    assert s["stat_validation"]["wikitext"]["mean_loss"]["n"] == 5
    assert s["stat_validation"]["wikitext"]["mean_loss"]["tag"] == "[ours]"
    assert "delta_ppl_on_minus_off" in s["quality"]["e2e"]
    # [ours] tag present in the T2 table
    t2 = pd.read_csv(fig / "T2_quality_vs_compression.csv", comment="#")
    assert (t2["tag"] == "[ours]").all()
    assert out is not None


def test_missing_inputs_are_skipped(tmp_path):
    (tmp_path / "results").mkdir()
    bl = tmp_path / "b.csv"
    pd.DataFrame([{"method": "m", "source_ref": "r", "model": "x", "params": "1B",
                   "dataset": "d", "metric": "z", "value": "1", "unit": "u",
                   "hardware": "h", "comparable": "trend", "notes": ""}]).to_csv(bl, index=False)
    out = build_all(str(tmp_path / "results"), str(bl), str(tmp_path / "figures"))
    assert out is not None
    # every table skipped, but build_all still returns the summary dict
    assert out["provenance"]
    assert not (tmp_path / "figures" / "T1_stat_validation.csv").exists()


def test_t3(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    fig = tmp_path / "figures"
    _write(rdir / "results_quality_x.csv", [
        {"arm": a, "seed": 0, "dataset": "e2e", "base_model": "phi-1_5",
         "perplexity": p, "bleu": "", "rougeL": "", "n_eval": 50}
        for a, p in [("ON", 22.0), ("OFF", 21.5)]])
    _write(rdir / "results_stat_x.summary.csv", [_summary_row(overall_comp_ratio=9.5)])
    bl = tmp_path / "baselines.csv"
    _write(bl, [
        {"method": "SplitLoRA", "source_ref": "lin2024", "model": "GPT2-M",
         "params": "355M", "dataset": "E2E", "metric": "ppl_gap", "value": "0.04",
         "unit": "PPL", "hardware": "2x3090", "comparable": "direct", "notes": ""},
        {"method": "Petals", "source_ref": "borzunov2023", "model": "BLOOM",
         "params": "176B", "dataset": "inference", "metric": "throughput",
         "value": "1", "unit": "steps/s", "hardware": "consumer", "comparable": "direct",
         "notes": ""}])
    build_all(str(rdir), str(bl), str(fig))
    assert (fig / "T3_ours_vs_published.csv").exists()
    assert (fig / "T3_scatter.png").exists()
    t3 = pd.read_csv(fig / "T3_ours_vs_published.csv", comment="#")
    assert (t3["tag"] == "[ours]").any()
    assert t3["tag"].str.startswith("[published").any()


def test_t4_scheduling(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    fig = tmp_path / "figures"
    rows = [
        _summary_row(strategy="smart", partition_balance_std=0.2, mean_step_latency_s=0.4),
        _summary_row(strategy="round_robin", partition_balance_std=0.5, mean_step_latency_s=0.6),
        _summary_row(strategy="proportional", n_batches=0, mean_step_latency_s=0.0),
    ]
    df = pd.DataFrame([{**r} for r in rows])
    df["note"] = ["", "", "infeasible"]
    df.to_csv(rdir / "results_sched_x.summary.csv", index=False)
    out = build_all(str(rdir), str(tmp_path / "bl.csv"), str(fig))
    assert (fig / "T4_scheduling.csv").exists()
    assert (fig / "T4_bars.png").exists()
    assert "scheduling" in out
    assert out["scheduling"]["proportional"]["status"] == "infeasible"
    assert out["scheduling"]["proportional"]["mean_step_latency_s"] is None
    assert out["scheduling"]["smart"]["tag"] == "[ours]"
    t4 = pd.read_csv(fig / "T4_scheduling.csv", comment="#")
    assert (t4["tag"] == "[ours]").all()


def test_t5_scalability(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    fig = tmp_path / "figures"
    _write(rdir / "results_scale_x.summary.csv",
           [_summary_row(n_workers=nw, mean_step_latency_s=0.3 + 0.1 * nw)
            for nw in (2, 4, 8)])
    out = build_all(str(rdir), str(tmp_path / "bl.csv"), str(fig))
    assert (fig / "T5_scalability_sim.csv").exists()
    assert (fig / "T5_lines.png").exists()
    assert "scalability" in out
    assert out["scalability"]["n"] == 3
    assert "loopback" in out["scalability"]["note"]


def test_t6_network(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    fig = tmp_path / "figures"
    _write(rdir / "results_net_x.summary.csv", [
        _summary_row(run_tag=f"net-d{d}-l{l}", mean_step_latency_s=0.3 + d / 1000 + l / 10)
        for d in (10, 50) for l in (0, 1)])
    out = build_all(str(rdir), str(tmp_path / "bl.csv"), str(fig))
    assert (fig / "T6_network.csv").exists()
    assert (fig / "T6_heatmap.png").exists()
    assert "network" in out
    assert out["network"]["n"] == 4


def test_render_response_substitutes(tmp_path):
    sj = tmp_path / "summary.json"
    sj.write_text(json.dumps({"a": {"b": {"c": 42.5}}, "top": "hi"}))
    tpl = tmp_path / "tpl.md"
    tpl.write_text("value is {{a.b.c}} and {{top}}")
    out = tmp_path / "out.md"
    render_response(str(sj), str(tpl), str(out))
    txt = out.read_text()
    assert "value is 42.5 and hi" == txt


def test_render_response_warns_on_unfilled(tmp_path, capsys):
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({"x": 1}))
    tpl = tmp_path / "t.md"
    tpl.write_text("{{x}} but {{missing.key}}")
    out = tmp_path / "o.md"
    render_response(str(sj), str(tpl), str(out))
    assert "{{missing.key}}" in out.read_text()
    assert "unfilled placeholder" in capsys.readouterr().out


def test_read_many_suffix_excludes_summary(tmp_path):
    rd = tmp_path
    _write(rd / "results_converge_x.csv", [
        {"run_tag": "c", "global_batch": b, "loss": 3.0 - 0.01 * b} for b in range(6)])
    _write(rd / "results_converge_x.summary.csv", [_summary_row(n_batches=6)])
    per_batch = _read_many(str(rd), "results_converge_", ".csv")
    assert len(per_batch) == 6
    assert "global_batch" in per_batch.columns
    assert "n_batches" not in per_batch.columns
    summ = _read_many(str(rd), "results_converge_", ".summary.csv")
    assert len(summ) == 1
    assert "n_batches" in summ.columns
