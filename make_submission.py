"""Package team_name.zip with the layout the organisers require:

team_name.zip
├── src/      (requirements.txt, main.py, README.md, echogesture/, scripts/, models/)
└── results/  (predictions.json, predictions.jsonl, results.csv)

    python make_submission.py --team echogesture_team
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "tflite", "backup_modelA", "backup_modelD", "backup_modelE", "old_modelD_E", "robust_E", "robust_F"}
SKIP_FILES = {"echogesture.pt", "echo_F60.pt", "benchmark_busy.json", "benchmark_prev.json", "_calib.npy", "_pre.onnx"}  # training checkpoint not needed at inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team_name")
    a = ap.parse_args()

    for req in ["src/main.py", "src/requirements.txt", "src/README.md", "src/models/labels.txt",
                "src/models/echogesture_int8.onnx", "results/predictions.json", "results/results.csv"]:
        if not (ROOT / req).exists():
            print(f"WARNING: missing {req}")

    out = ROOT / f"{a.team}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for top in ("src", "results"):
            for p in sorted((ROOT / top).rglob("*")):
                if p.is_dir() or any(s in p.parts for s in SKIP_DIRS) or p.name in SKIP_FILES:
                    continue
                if p.suffix == ".pyc":
                    continue
                z.write(p, p.relative_to(ROOT))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
