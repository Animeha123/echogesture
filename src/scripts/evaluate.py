"""Run predict() over a set of clips and write results/predictions.jsonl.

Sources (pick one):
    --videos DIR             every video in DIR (clip_id = file stem)
    --manifest FILE.csv      columns: clip_id or path [, label/gloss]   (paths relative to the csv)
    --wlasl-split test       our held-out WLASL split from data/wlasl_index.csv

    python scripts/evaluate.py --wlasl-split test --robustness
    python scripts/evaluate.py --videos /path/to/organiser_test_clips

--robustness re-runs every clip under 7 realistic recording conditions (dim room,
over-exposure, white-balance shift, blur, heavy compression, signer further away,
camera at an angle) and reports the mean top-1 drop -> accuracy_drop_robustness_pt.
--stress adds two deliberately harsh tests (grey box over the signing area, very
dark + noise), reported separately as accuracy_drop_stress_pt.
Each perturbation is seeded by clip_id, so runs are repeatable.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from echogesture import config as C  # noqa: E402
from main import predict  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ------------------------------------------------------------------ perturbations
def make_occlusion(clip_id: str):
    r = np.random.default_rng(int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16))
    fx, fy = r.uniform(0.15, 0.6), r.uniform(0.35, 0.7)
    fw, fh = r.uniform(0.18, 0.28), r.uniform(0.18, 0.28)

    def f(img):
        h, w = img.shape[:2]
        img = img.copy()
        x0, y0 = int(fx * w), int(fy * h)
        img[y0:y0 + int(fh * h), x0:x0 + int(fw * w)] = (40, 40, 40)
        return img
    return f


def make_lowlight(clip_id: str):
    r = np.random.default_rng(int(hashlib.md5(clip_id.encode()).hexdigest()[8:16], 16))
    gain = r.uniform(0.25, 0.4)

    def f(img):
        x = img.astype(np.float32) * gain
        x += r.normal(0, 4, x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)
    return f


def _rng(clip_id: str, salt: str):
    return np.random.default_rng(int(hashlib.md5((salt + clip_id).encode()).hexdigest()[:8], 16))


# ---- realistic conditions: what differs between ordinary sign-language datasets /
# real phone recordings (organisers: "not too far from commonly available datasets")
def make_dim(clip_id: str):
    """Dim indoor room: ~half exposure, mild sensor noise."""
    r = _rng(clip_id, "dim")
    gain, sd = r.uniform(0.45, 0.65), r.uniform(2, 4)

    def f(img):
        x = img.astype(np.float32) * gain + r.normal(0, sd, img.shape)
        return np.clip(x, 0, 255).astype(np.uint8)
    return f


def make_bright(clip_id: str):
    """Over-exposed / backlit: brighter, washed-out contrast."""
    r = _rng(clip_id, "bright")
    gain, lift = r.uniform(1.25, 1.5), r.uniform(20, 45)

    def f(img):
        return np.clip(img.astype(np.float32) * gain + lift, 0, 255).astype(np.uint8)
    return f


def make_colour(clip_id: str):
    """Different white balance (warm tungsten / cool LED) + saturation."""
    r = _rng(clip_id, "colour")
    cast = r.uniform(0.8, 1.2, 3).astype(np.float32)
    sat = r.uniform(0.6, 1.3)

    def f(img):
        x = img.astype(np.float32) * cast
        grey = x.mean(axis=2, keepdims=True)
        return np.clip(grey + (x - grey) * sat, 0, 255).astype(np.uint8)
    return f


def make_blur(clip_id: str):
    """Out-of-focus / cheap webcam softness."""
    r = _rng(clip_id, "blur")
    k = int(r.choice([5, 7]))
    return lambda img: cv2.GaussianBlur(img, (k, k), 0)


def make_compress(clip_id: str):
    """Heavy video compression (messaging apps, streaming)."""
    q = int(_rng(clip_id, "jpeg").integers(15, 30))
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), q]

    def f(img):
        ok, buf = cv2.imencode(".jpg", img, enc)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    return f


def make_distance(clip_id: str):
    """Signer further from the camera: content shrunk to 60-75 %, off-centre."""
    r = _rng(clip_id, "dist")
    s = r.uniform(0.6, 0.75)
    ox, oy = r.uniform(0, 1), r.uniform(0, 1)

    def f(img):
        h, w = img.shape[:2]
        sw, sh = int(w * s), int(h * s)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).mean(0)
        bg = np.empty_like(img)
        bg[:] = edge.astype(np.uint8)  # wall colour taken from the frame's own edges
        x0, y0 = int((w - sw) * ox), int((h - sh) * oy)
        bg[y0:y0 + sh, x0:x0 + sw] = small
        return bg
    return f


def make_angle(clip_id: str):
    """Camera not square-on: small roll + perspective (side/low placement)."""
    r = _rng(clip_id, "angle")
    roll = r.uniform(-12, 12)
    k = r.uniform(0.06, 0.12) * r.choice([-1, 1])

    def f(img):
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dy = k * h
        dst = np.float32([[0, max(0, dy)], [w, max(0, -dy)], [w, h - max(0, -dy)], [0, h - max(0, dy)]])
        M = cv2.getPerspectiveTransform(src, dst)
        R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0), [0, 0, 1]])
        return cv2.warpPerspective(img, R @ M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return f


REALISTIC = {"dim": make_dim, "bright": make_bright, "colour": make_colour, "blur": make_blur,
             "compress": make_compress, "distance": make_distance, "angle": make_angle}
# ---- stress tests: deliberately harsher than real recordings (reported separately)
STRESS = {"occlusion": make_occlusion, "lowlight": make_lowlight}
PERTURB = {**REALISTIC, **STRESS}


# ------------------------------------------------------------------ inputs
def gather(a):
    items = []  # (clip_id, path, label or None)
    if a.videos:
        for p in sorted(Path(a.videos).rglob("*")):
            if p.suffix.lower() in VIDEO_EXT:
                items.append((p.stem, str(p), None))
    elif a.manifest:
        base = Path(a.manifest).parent
        with open(a.manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                path = r.get("path") or r.get("video_path") or ""
                cid = r.get("clip_id") or Path(path).stem
                if not path:
                    cand = [base / f"{cid}{e}" for e in VIDEO_EXT]
                    path = str(next((c for c in cand if c.exists()), base / f"{cid}.mp4"))
                elif not Path(path).is_absolute():
                    path = str(base / path)
                lab = r.get("label") or r.get("gloss") or r.get("prediction")
                items.append((cid, path, lab.upper() if lab else None))
    else:
        with open(C.DATA_DIR / "wlasl_index.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["split"] == a.wlasl_split:
                    items.append((r["video_id"], r["path"], r["gloss"]))
    if a.limit:
        # seeded random sample: the list is ordered by gloss frequency, so a plain
        # prefix would only test the most common words
        if a.limit < len(items):
            pick = sorted(np.random.default_rng(0).choice(len(items), a.limit, replace=False))
            items = [items[i] for i in pick]
    return items


def run(items, perturb: str | None, out_jsonl: Path | None):
    preds, n, c1, c5, lat = [], 0, 0, 0, []
    t0 = time.time()
    fh = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None
    for i, (cid, path, lab) in enumerate(items):
        tf = PERTURB[perturb](cid) if perturb else None
        p = predict(path, frame_transform=tf)
        p["clip_id"] = cid
        preds.append(p)
        lat.append(p["latency_ms"])
        if fh:
            fh.write(json.dumps(p) + "\n")
        if lab:
            n += 1
            c1 += p["prediction"] == lab
            c5 += lab in p["top5"]
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            acc = f" top1 {100 * c1 / n:.1f}% top5 {100 * c5 / n:.1f}%" if n else ""
            print(f"  [{perturb or 'clean'}] {i + 1}/{len(items)}{acc}  ({time.time() - t0:.0f}s)", flush=True)
    if fh:
        fh.close()
    res = {"n": len(items), "n_labelled": n,
           "latency_p50_ms": float(np.percentile(lat, 50)) if lat else None,
           "latency_p95_ms": float(np.percentile(lat, 95)) if lat else None}
    if n:
        res.update(top1=100 * c1 / n, top5=100 * c5 / n)
    return res


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--videos")
    g.add_argument("--manifest")
    g.add_argument("--wlasl-split", default="test")
    ap.add_argument("--robustness", action="store_true",
                    help="realistic conditions: dim, bright, colour cast, blur, compression, distance, angle")
    ap.add_argument("--stress", action="store_true",
                    help="also the harsh stress tests (grey box over the signing area, very dark)")
    ap.add_argument("--only", default="",
                    help="comma list to run a subset of conditions, e.g. --only dim,angle (quick checks)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()

    items = gather(a)
    print(f"{len(items)} clips")
    if not items:
        src = a.videos or a.manifest or f"split '{a.wlasl_split}'"
        sys.exit(f"ERROR: no clips found in {src}. Nothing written - existing results kept.\n"
                 "Check the folder path (it must contain .mp4/.mov/... files).")
    missing = [p for _, p, _ in items if not Path(p).exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} video paths do not exist, e.g. {missing[0]}. Nothing written.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    jl = a.out_dir / "predictions.jsonl"
    metrics = {"clean": run(items, None, jl)}
    # the bundle diagram names it predictions.json -> ship both (same JSON-lines content)
    (a.out_dir / "predictions.json").write_text(jl.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"predictions -> {jl}")

    clean = metrics["clean"].get("top1")
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    bad = only - set(PERTURB)
    if bad:
        sys.exit(f"unknown condition(s) {sorted(bad)}; choose from {sorted(PERTURB)}")
    real = [k for k in REALISTIC if not only or k in only]
    stress = [k for k in STRESS if not only or k in only]
    if a.robustness and clean is not None and real:
        for k in real:
            metrics[k] = run(items, k, None)
        accs = {k: metrics[k]["top1"] for k in real}
        metrics["accuracy_drop_robustness_pt"] = clean - float(np.mean(list(accs.values())))
        metrics["accuracy_drop_worst_pt"] = clean - min(accs.values())
        metrics["worst_condition"] = min(accs, key=accs.get)
        metrics["robustness_conditions"] = real
    if a.stress and clean is not None and stress:
        for k in stress:
            metrics[k] = run(items, k, None)
        metrics["accuracy_drop_stress_pt"] = clean - float(np.mean([metrics[k]["top1"] for k in stress]))

    mp = a.out_dir / "metrics.json"
    # full runs replace the file (no stale conditions from older runs); --only merges
    old = json.loads(mp.read_text()) if mp.exists() and only else {}
    old.update(metrics)
    mp.write_text(json.dumps(old, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
