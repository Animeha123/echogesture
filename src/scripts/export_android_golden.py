"""Write golden test vectors so the Android (Kotlin) feature port can be checked
against the Python reference: raw landmark sequences in, features + top-5 out.

    python scripts/export_android_golden.py

Files (float32 little-endian) go to android/app/src/test/resources/golden/:
    index.txt          one line per case: name n_frames
    <name>.lm          n_frames * 53 * 3   raw landmarks (NaN = missing)
    <name>.feat        48 * 352            expected compute_features() output
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402
from echogesture.features import compute_features  # noqa: E402

OUT = C.SRC_DIR.parent / "android" / "app" / "src" / "test" / "resources" / "golden"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(C.DATA_DIR / "wlasl_index.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng = np.random.default_rng(1)
    cases = []
    for r in [rows[i] for i in sorted(rng.choice(len(rows), 6, replace=False))]:
        p = C.DATA_DIR / "landmarks" / f"{r['video_id']}.npy"
        if p.exists():
            cases.append((f"wlasl_{r['video_id']}", np.load(p).astype(np.float32)))
    base = cases[0][1]
    # edge cases: no shoulders (fallback scale/centre), long hand gaps, one frame
    nopose = base.copy(); nopose[:, :C.N_POSE] = np.nan
    gaps = base.copy(); gaps[5:9, C.LH0:C.LH0 + C.N_HAND] = np.nan; gaps[10:30, C.RH0:] = np.nan
    cases += [("edge_nopose", nopose), ("edge_gaps", gaps), ("edge_single", base[:1].copy())]

    with open(OUT / "index.txt", "w") as idx:
        for name, seq in cases:
            seq.astype("<f4").tofile(OUT / f"{name}.lm")
            compute_features(seq).astype("<f4").tofile(OUT / f"{name}.feat")
            idx.write(f"{name} {len(seq)}\n")
    print(f"{len(cases)} golden cases -> {OUT}")


if __name__ == "__main__":
    main()
