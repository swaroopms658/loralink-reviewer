"""Record / verify SHA-256 of every LoraLink source file that determines results.

The first five carry the additive reviewer-response instrumentation. The last two
carry the forward-pass correctness fix (restoring the final norm and GPT-Neo's
learned position embedding) -- see patch/README.md.
"""
import hashlib, json, sys, pathlib
FILES = ["main.py", "device_manager.py", "compression_engine.py",
         "benchmarking.py", "data_loader.py",
         "pipeline_engine.py", "model_registry.py"]

def _sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    sums = {f: _sha(root / f) for f in FILES}
    out = pathlib.Path(__file__).with_name("SHA256SUMS")
    if "--update" in sys.argv:
        out.write_text(json.dumps(sums, indent=2)); print("wrote", out); return
    if "--verify" in sys.argv:
        want = json.loads(out.read_text())
        bad = [f for f in FILES if want.get(f) != sums[f]]
        if bad:
            print("CHECKSUM MISMATCH:", bad); sys.exit(1)
        print("checksums OK")

if __name__ == "__main__":
    main()
