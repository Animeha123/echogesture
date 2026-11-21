"""Build the training index from WLASL.

Expected input (e.g. the Kaggle "WLASL processed" dataset):
    <root>/WLASL_v0.3.json
    <root>/videos/<video_id>.mp4

Usage:
    python scripts/prepare_wlasl.py --root ../data/wlasl --subset 300

Writes:
    data/wlasl_index.csv   video_id,path,gloss,label,split,frame_start,frame_end
    src/models/labels.txt  one UPPERCASE gloss per line (class order)

--subset N keeps the first N glosses of the json (= official WLASL100/300/1000/2000).
--vocab FILE restricts to the organisers' fixed vocabulary list instead.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--vocab", type=Path, default=None, help="txt file, one gloss per line")
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "wlasl_index.csv")
    a = ap.parse_args()

    js = a.json or a.root / "WLASL_v0.3.json"
    entries = json.loads(js.read_text(encoding="utf-8"))
    vids_dir = a.root / "videos"

    if a.vocab:
        wanted = [ln.strip().upper() for ln in a.vocab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        by_gloss = {e["gloss"].upper(): e for e in entries}
        entries = [by_gloss[g] for g in wanted if g in by_gloss]
        missing = [g for g in wanted if g not in by_gloss]
        if missing:
            print(f"WARNING: {len(missing)} vocab words not in WLASL: {missing[:20]}")
    else:
        entries = entries[:a.subset]

    labels, rows, n_missing = [], [], 0
    for e in entries:
        gloss = e["gloss"].upper()
        lab = len(labels)
        labels.append(gloss)
        for inst in e["instances"]:
            p = vids_dir / f"{inst['video_id']}.mp4"
            if not p.exists():
                n_missing += 1
                continue
            rows.append(dict(video_id=inst["video_id"], path=str(p.resolve()), gloss=gloss, label=lab,
                             split=inst["split"], frame_start=inst.get("frame_start", 1),
                             frame_end=inst.get("frame_end", -1)))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    C.LABELS_TXT.write_text("\n".join(labels) + "\n", encoding="utf-8")

    from collections import Counter
    sp = Counter(r["split"] for r in rows)
    print(f"{len(labels)} classes, {len(rows)} clips {dict(sp)}; {n_missing} listed videos missing on disk")
    print(f"index  -> {a.out}\nlabels -> {C.LABELS_TXT}")


if __name__ == "__main__":
    main()
