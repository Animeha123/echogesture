"""Run MediaPipe over every clip in the index once and cache landmarks as .npy.

Training then runs on these small arrays (seconds per epoch) instead of video.

    python scripts/extract_landmarks.py --workers 6
"""
import argparse
import csv
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402

_EXT = None
_USE_RANGE = False


def _init(use_range):
    global _EXT, _USE_RANGE
    os.environ.setdefault("GLOG_minloglevel", "2")
    from echogesture.landmarks import LandmarkExtractor
    _EXT = LandmarkExtractor()
    _USE_RANGE = use_range


def _vid_seed(vid: str) -> int:
    """Numeric WLASL ids keep their old seeds; other ids (aslc_...) get a stable hash."""
    if vid.isdigit():
        return 7919 * int(vid)
    return int(zlib.crc32(vid.encode())) * 7919


def _work(job):
    vid, path, out, fs, fe, pseed = job
    if Path(out).exists():
        return vid, "skip", 0
    try:
        rng = (int(fs), int(fe)) if _USE_RANGE else None
        tf = None
        if pseed is not None:
            from echogesture.pixel_aug import make_train_perturb
            tf = make_train_perturb(pseed)
        seq, info = _EXT.process_video(path, tf, frame_range=rng)
        np.save(out, seq.astype(np.float16))
        return vid, "ok", info["n_frames"]
    except Exception as e:
        return vid, f"err {e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=C.DATA_DIR / "wlasl_index.csv")
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "landmarks")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--use-frame-range", action="store_true",
                    help="apply WLASL frame_start/frame_end (only for raw, untrimmed downloads)")
    ap.add_argument("--overwrite", action="store_true", help="re-extract clips that are already cached")
    ap.add_argument("--pix-aug", type=int, default=0, metavar="N",
                    help="instead of clean landmarks, write N occluded/low-light variants of every "
                         "train/val clip to data/landmarks_pix (test clips are never used)")
    ap.add_argument("--pix-limit", type=int, default=0, metavar="K",
                    help="with --pix-aug: only a seeded random subset of K train/val clips")
    a = ap.parse_args()
    if a.pix_aug and a.out == C.DATA_DIR / "landmarks":
        a.out = C.DATA_DIR / "landmarks_pix"
    a.out.mkdir(parents=True, exist_ok=True)
    if a.overwrite:
        for p in a.out.glob("*.npy"):
            p.unlink()
    print(f"processing resolution: long side {C.PROC_LONG_SIDE}px")

    with open(a.index, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.pix_aug:
        rows = [r for r in rows if r["split"] in ("train", "val")]
        if a.pix_limit and a.pix_limit < len(rows):
            pick = np.sort(np.random.default_rng(0).choice(len(rows), a.pix_limit, replace=False))
            rows = [rows[i] for i in pick]
        jobs = [(r["video_id"], r["path"], str(a.out / f"{r['video_id']}__p{k}.npy"),
                 r["frame_start"], r["frame_end"], _vid_seed(r["video_id"]) + 104729 * k + 13)
                for r in rows for k in range(a.pix_aug)]
    else:
        jobs = [(r["video_id"], r["path"], str(a.out / f"{r['video_id']}.npy"),
                 r["frame_start"], r["frame_end"], None) for r in rows]
    print(f"{len(jobs)} clips, {a.workers} workers -> {a.out}")
    t0, done, errs = time.time(), 0, 0
    with Pool(a.workers, initializer=_init, initargs=(a.use_frame_range,)) as pool:
        for vid, status, n in pool.imap_unordered(_work, jobs, chunksize=4):
            done += 1
            if status.startswith("err"):
                errs += 1
                print(f"  {vid}: {status}")
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el / 60:.1f} min  eta {(len(jobs) - done) * el / done / 60:.1f} min  errors={errs}")


if __name__ == "__main__":
    main()
