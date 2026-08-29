"""Spawn a LoraLink coordinator + N workers on loopback for one experiment run."""
from __future__ import annotations
import os, sys, socket, time, shutil, subprocess, contextlib, pathlib, tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PORT = 29500


def _ip_plan(n_workers):
    """Coordinator on 127.0.0.1, workers on 127.0.0.2 .. 127.0.0.(n+1)."""
    return "127.0.0.1", [f"127.0.0.{i}" for i in range(2, 2 + n_workers)]


def _wait_port(ip, port, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        with contextlib.suppress(OSError):
            with socket.create_connection((ip, port), timeout=1):
                return True
        time.sleep(0.5)
    raise TimeoutError(f"{ip}:{port} never came up")


def _tc_available():
    # os.geteuid is POSIX-only; on Windows this yields 1 -> False -> in-process shim.
    return shutil.which("tc") is not None and getattr(os, "geteuid", lambda: 1)() == 0


def _apply_netem(netem):
    parts = ["tc", "qdisc", "add", "dev", "lo", "root", "netem"]
    if netem.get("delay_ms"):
        parts += ["delay", f'{netem["delay_ms"]}ms']
    if netem.get("loss_pct"):
        parts += ["loss", f'{netem["loss_pct"]}%']
    if netem.get("rate"):
        parts += ["rate", netem["rate"]]
    subprocess.run(parts, check=True)


def _clear_netem():
    subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"],
                   check=False, capture_output=True)


_MAIN = str(REPO_ROOT / "main.py")  # absolute: children run with cwd=workdir


def _worker_cmd(ip, model, seed):
    return [sys.executable, _MAIN, "--role", "worker",
            "--host-ip", ip, "--base-model", model, "--seed", str(seed)]


def _coord_cmd(coord_ip, worker_ips, *, model, dataset, seed, num_samples,
               epochs, eval_holdout, strategy, tag, csv_path):
    return [sys.executable, _MAIN, "--role", "coordinator",
            "--host-ip", coord_ip, "--workers", ",".join(worker_ips),
            "--base-model", model, "--dataset", dataset, "--seed", str(seed),
            "--num-samples", str(num_samples), "--epochs", str(epochs),
            "--eval-holdout", str(eval_holdout),
            "--partition-strategy", strategy, "--run-tag", tag,
            "--metrics-csv", str(csv_path)]


def run_cluster(n_workers, dataset, seed, *, model="EleutherAI/gpt-neo-125M",
                strategy="smart", compression=True, num_samples=60, epochs=1,
                eval_holdout=0, netem=None, tag="", run_timeout_s=900,
                results_csv="results.csv", save_adapters_to=None, workdir="."):
    workdir = pathlib.Path(workdir).resolve()
    csv_path = pathlib.Path(results_csv)
    if not csv_path.is_absolute():
        csv_path = workdir / csv_path
    coord_ip, worker_ips = _ip_plan(n_workers)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["LORALINK_LOSSY_COMPRESSION"] = "1" if compression else "0"
    env.setdefault("LORALINK_FAKE_BENCHMARK", "0")

    netem_mode = "none"
    procs = []
    errf = None
    try:
        if netem:
            if _tc_available():
                netem_mode = "tc-netem"  # set first so finally clears a partial install
                _apply_netem(netem)
            else:
                netem_mode = "in-process-shim"
                env["LORALINK_NET_SHIM"] = f'{netem.get("delay_ms", 0)},{netem.get("loss_pct", 0)}'
        csv_path.with_name(csv_path.name + ".netem").write_text(netem_mode)

        for ip in worker_ips:
            procs.append(subprocess.Popen(
                _worker_cmd(ip, model, seed), cwd=workdir, env=env))
        for ip in worker_ips:
            _wait_port(ip, PORT)

        errf = tempfile.TemporaryFile(mode="w+")  # a file, not a PIPE: no fill-up deadlock
        coord = subprocess.Popen(
            _coord_cmd(coord_ip, worker_ips, model=model, dataset=dataset,
                       seed=seed, num_samples=num_samples, epochs=epochs,
                       eval_holdout=eval_holdout, strategy=strategy, tag=tag,
                       csv_path=csv_path),
            cwd=workdir, env=env, stderr=errf)
        procs.append(coord)
        try:
            coord.wait(timeout=run_timeout_s)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"run '{tag}' exceeded {run_timeout_s}s")
        if coord.returncode != 0:
            _errlines = []
            with contextlib.suppress(Exception):
                errf.seek(0)
                _errlines = [ln.strip() for ln in errf.read().splitlines() if ln.strip()]
            _tail = _errlines[-1] if _errlines else "(no stderr)"
            raise RuntimeError(
                f"coordinator exited {coord.returncode} for run '{tag}': {_tail}")

        if save_adapters_to:
            src = workdir / "lora_adapters"
            if src.exists():
                shutil.copytree(src, save_adapters_to, dirs_exist_ok=True)
        return str(csv_path)
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate()
        time.sleep(2)
        for p in procs:
            with contextlib.suppress(Exception):
                p.kill()
        for p in procs:  # reap so no zombies and ports 29500 are released before return
            with contextlib.suppress(Exception):
                p.wait(timeout=5)
        if netem_mode == "tc-netem":
            _clear_netem()
        if errf is not None:
            with contextlib.suppress(Exception):
                errf.close()
