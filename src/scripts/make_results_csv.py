"""Fill results/results.csv (metric, target, measured, unit, device, runtime, pass)
from results/metrics.json + results/benchmark.json, plus values you measure on
the phone and pass on the command line (these override the laptop numbers).

    python scripts/make_results_csv.py --device "Galaxy S24, Snapdragon 8 Gen 3" \
        --runtime "ONNX Runtime INT8, CPU 2 threads" --avg-power-mw 420 \
        --latency-p95-ms 31.2 --thermal-drift-pct 9.5
"""
import argparse
import csv
import json
import operator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402

TARGETS = [  # metric, op, target, unit
    ("top1_accuracy", ">=", 70, "%"),
    ("latency_p95", "<=", 50, "ms"),
    ("model_size", "<=", 25, "MB"),
    ("accuracy_drop_robustness", "<=", 5, "pt"),
    ("avg_power", "<=", 500, "mW"),
    ("thermal_drift_10min", "<=", 20, "%"),
]
OPS = {">=": operator.ge, "<=": operator.le}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="")
    ap.add_argument("--runtime", default="ONNX Runtime INT8 (CPU)")
    ap.add_argument("--top1", type=float)
    ap.add_argument("--latency-p95-ms", type=float)
    ap.add_argument("--model-size-mb", type=float)
    ap.add_argument("--robustness-drop-pt", type=float)
    ap.add_argument("--avg-power-mw", type=float)
    ap.add_argument("--thermal-drift-pct", type=float)
    a = ap.parse_args()

    m, b = load(C.RESULTS_DIR / "metrics.json"), load(C.RESULTS_DIR / "benchmark.json")
    measured = {
        "top1_accuracy": a.top1 if a.top1 is not None else m.get("clean", {}).get("top1"),
        "latency_p95": a.latency_p95_ms if a.latency_p95_ms is not None else b.get("latency_e2e_ms_p95"),
        # organisers: 25 MB "for the entire model" -> classifier + MediaPipe hand/pose bundles
        "model_size": a.model_size_mb if a.model_size_mb is not None else b.get("total_pipeline_mb"),
        "accuracy_drop_robustness": a.robustness_drop_pt if a.robustness_drop_pt is not None
        else m.get("accuracy_drop_robustness_pt"),
        "avg_power": a.avg_power_mw,
        "thermal_drift_10min": a.thermal_drift_pct if a.thermal_drift_pct is not None
        else b.get("thermal_drift_pct"),
    }
    device = a.device or b.get("host", "")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        for name, op, tgt, unit in TARGETS:
            v = measured[name]
            ok = "" if v is None else ("pass" if OPS[op](v, tgt) else "fail")
            w.writerow([name, f"{op}{tgt}", "" if v is None else round(float(v), 2), unit, device, a.runtime, ok])
        # supporting numbers (no official target) so judges can see the full picture
        extra = [
            ("top5_accuracy", m.get("clean", {}).get("top5"), "%"),
            ("accuracy_drop_worst_condition", m.get("accuracy_drop_worst_pt"), "pt"),
            ("accuracy_drop_stress_occlusion_lowlight", m.get("accuracy_drop_stress_pt"), "pt"),
            ("latency_p50", b.get("latency_e2e_ms_p50") if a.latency_p95_ms is None else None, "ms"),
            ("classifier_size_int8", b.get("classifier_int8_mb"), "MB"),
            ("classifier_latency_p95", b.get("classifier_ms_p95"), "ms"),
        ]
        for name, v, unit in extra:
            if v is not None:
                w.writerow([name, "", round(float(v), 2), unit, device, a.runtime, "info"])
    print(out.read_text())
    if m.get("worst_condition"):
        print(f"worst robustness condition: {m['worst_condition']}")


if __name__ == "__main__":
    main()
