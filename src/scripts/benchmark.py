"""Latency / size / thermal-drift benchmark -> results/benchmark.json

    python scripts/benchmark.py --clips 20
    python scripts/benchmark.py --thermal-minutes 10      # sustained-load drift test

latency_p95 = p95 over clips of (per-frame landmark time + features + classifier),
i.e. the time between the last frame of a sign and its caption appearing.
thermal_drift = % change of median end-to-end latency, last minute vs first minute.
Power must be measured on the phone (see README); pass it to make_results_csv.py.
"""
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from echogesture import config as C  # noqa: E402
from echogesture.features import compute_features  # noqa: E402
from echogesture.runtime import Classifier, EchoGesturePipeline  # noqa: E402


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def pick_clips(n):
    idx = C.DATA_DIR / "wlasl_index.csv"
    if not idx.exists():
        return []
    with open(idx, encoding="utf-8") as f:
        rows = [r["path"] for r in csv.DictReader(f) if r["split"] == "test"]
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--thermal-minutes", type=float, default=0)
    a = ap.parse_args()

    out = {"host": f"{platform.processor() or platform.machine()} / {platform.system()}",
           "threads": a.threads}

    # --- model size
    out["classifier_int8_mb"] = round(mb(C.ONNX_INT8), 3)
    out["classifier_fp32_mb"] = round(mb(C.ONNX_FP32), 3)
    out["mediapipe_tasks_mb"] = round(mb(C.HAND_TASK) + mb(C.POSE_TASK), 3)
    out["total_pipeline_mb"] = round(out["classifier_int8_mb"] + out["mediapipe_tasks_mb"], 3)

    # --- classifier-only latency
    clf = Classifier(threads=a.threads)
    x = np.random.randn(C.SEQ_LEN, C.FEAT_DIM).astype(np.float32)
    for _ in range(20):
        clf.logits(x)
    ts = []
    for _ in range(500):
        t0 = time.perf_counter()
        clf.logits(x)
        ts.append((time.perf_counter() - t0) * 1000)
    out["classifier_ms_p50"] = float(np.percentile(ts, 50))
    out["classifier_ms_p95"] = float(np.percentile(ts, 95))

    # --- end-to-end on real clips
    clips = a.videos or pick_clips(a.clips)
    pipe = EchoGesturePipeline(threads=a.threads)
    if clips:
        pipe.predict_video(clips[0])            # warm-up (graph init, caches)
    e2e, per_frame = [], []
    for p in clips:
        r = pipe.predict_video(p)
        e2e.append(r["latency_ms"])
        per_frame.append(r["_timing"]["landmark_ms_per_frame"])
    if e2e:
        out["landmark_ms_per_frame_p50"] = float(np.percentile(per_frame, 50))
        out["latency_e2e_ms_p50"] = float(np.percentile(e2e, 50))
        out["latency_e2e_ms_p95"] = float(np.percentile(e2e, 95))
        out["n_clips"] = len(e2e)

    # --- sustained load (thermal throttling shows up as rising latency)
    if a.thermal_minutes > 0 and clips:
        print(f"thermal run: {a.thermal_minutes} min ...")
        seq, _ = pipe.extractor.process_video(clips[0])
        frames_lat, t_start, i = [], time.time(), 0
        from echogesture.landmarks import iter_video_frames
        frames = list(iter_video_frames(clips[0]))
        while time.time() - t_start < a.thermal_minutes * 60:
            f = frames[i % len(frames)]
            t0 = time.perf_counter()
            pipe.extractor.process_frame(f)
            if i % 8 == 0:                      # classify every 8 frames like the live demo
                clf.logits(compute_features(seq))
            frames_lat.append((time.time() - t_start, (time.perf_counter() - t0) * 1000))
            i += 1
        arr = np.array(frames_lat)
        first = np.median(arr[arr[:, 0] < 60, 1])
        last = np.median(arr[arr[:, 0] > arr[-1, 0] - 60, 1])
        out["thermal_first_min_ms"] = float(first)
        out["thermal_last_min_ms"] = float(last)
        out["thermal_drift_pct"] = float((last - first) / first * 100)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bp = C.RESULTS_DIR / "benchmark.json"
    old = json.loads(bp.read_text()) if bp.exists() else {}
    old.update(out)
    bp.write_text(json.dumps(old, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
