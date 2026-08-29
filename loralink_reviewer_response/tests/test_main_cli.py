import subprocess, sys

def test_help_lists_new_flags(repo_root):
    out = subprocess.run([sys.executable, "main.py", "--help"],
                         cwd=repo_root, capture_output=True, text=True).stdout
    for flag in ["--seed", "--num-samples", "--epochs", "--partition-strategy",
                 "--run-tag", "--metrics-csv", "--base-model", "--eval-holdout"]:
        assert flag in out, flag

def test_bad_strategy_rejected(repo_root):
    r = subprocess.run([sys.executable, "main.py", "--role", "worker",
                        "--partition-strategy", "nope"],
                       cwd=repo_root, capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr
