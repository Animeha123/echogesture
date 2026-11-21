"""Download only the ASL Citizen clips whose gloss matches our WLASL vocabulary.

ASL Citizen (Microsoft Research, 2023): ~83k webcam clips, 2,731 signs, 52 signers.
Licence: non-commercial research use only; do not redistribute the videos.
Instead of pulling the whole 46 GB zip, this reads the zip's central directory with
HTTP range requests and fetches each selected member with one range request.

    python scripts/fetch_asl_citizen.py --per-word 12 --workers 8

Writes the videos to --video-dir and data/aslc_index.csv (same columns as
wlasl_index.csv, split=train, label = WLASL class id) for extract_landmarks.py.
"""
import argparse
import csv
import io
import json
import re
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402

URL = ("https://download.microsoft.com/download/b/8/8/"
       "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip")


def _get(start: int, end: int, tries: int = 6) -> bytes:
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of the remote zip (only used to parse the directory)."""

    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, o, w=0):
        self.pos = {0: o, 1: self.pos + o, 2: self.size + o}[w]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        d = _get(self.pos, min(self.pos + n, self.size) - 1)
        self.pos += len(d)
        return d


def load_directory(cache: Path):
    """{member name: (header_offset, compress_size, compress_type)} + split csv rows."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    print("reading remote zip directory ...", flush=True)
    z = zipfile.ZipFile(_RangeFile())
    members = {i.filename: (i.header_offset, i.compress_size, i.compress_type) for i in z.infolist()}
    rows = []
    for s in ("train", "val", "test"):
        txt = z.read(f"ASL_Citizen/splits/{s}.csv").decode("utf-8")
        rows += list(csv.DictReader(io.StringIO(txt)))
    data = {"members": members, "rows": rows}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_member(info, dst: Path) -> int:
    off, csize, ctype = info
    head = _get(off, off + 29)
    sig, *_ = struct.unpack("<I", head[:4])
    if sig != 0x04034B50:
        raise ValueError("bad local header")
    n_len, x_len = struct.unpack("<HH", head[26:30])
    start = off + 30 + n_len + x_len
    data = _get(start, start + csize - 1)
    if ctype == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif ctype != zipfile.ZIP_STORED:
        raise ValueError(f"unsupported compression {ctype}")
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return len(data)


def norm(g: str) -> str:
    """APPLE / apple1 / THANK YOU / THANKYOU -> comparable key."""
    g = re.sub(r"\d+$", "", g.upper().strip())
    return re.sub(r"[^A-Z]", "", g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-word", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--video-dir", type=Path, default=Path.home() / "datasets" / "asl_citizen" / "videos")
    ap.add_argument("--out", type=Path, default=C.DATA_DIR / "aslc_index.csv")
    ap.add_argument("--wlasl-index", type=Path, default=C.DATA_DIR / "wlasl_index.csv")
    a = ap.parse_args()

    d = load_directory(a.video_dir.parent / "aslc_directory.json")
    members, rows = d["members"], d["rows"]

    wl = {}
    with open(a.wlasl_index, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wl[int(r["label"])] = r["gloss"].upper()
    key2lab = {}
    for lab, g in wl.items():
        key2lab.setdefault(norm(g), lab)

    by_lab = defaultdict(list)
    for r in rows:
        lab = key2lab.get(norm(r["Gloss"]))
        if lab is not None:
            by_lab[lab].append(r)
    print(f"{len(by_lab)} of {len(wl)} WLASL words found in ASL Citizen "
          f"({sum(map(len, by_lab.values()))} clips available)")

    rng = np.random.default_rng(0)
    pick = []
    for lab in sorted(by_lab):
        # round-robin over signers so each word gets as many different people as possible
        per_p = defaultdict(list)
        for r in by_lab[lab]:
            per_p[r["Participant ID"]].append(r)
        ps = list(per_p)
        rng.shuffle(ps)
        for p in ps:
            rng.shuffle(per_p[p])
        chosen = []
        while len(chosen) < a.per_word and any(per_p[p] for p in ps):
            for p in ps:
                if per_p[p] and len(chosen) < a.per_word:
                    chosen.append(per_p[p].pop())
        pick += [(lab, r) for r in chosen]

    a.video_dir.mkdir(parents=True, exist_ok=True)
    out_rows, todo = [], []
    for lab, r in pick:
        name = f"ASL_Citizen/videos/{r['Video file']}"
        if name not in members:
            continue
        vid = "aslc_" + Path(r["Video file"]).stem.split("-")[0]
        dst = a.video_dir / f"{vid}.mp4"
        out_rows.append(dict(video_id=vid, path=str(dst.resolve()), gloss=wl[lab], label=lab,
                             split="train", frame_start=1, frame_end=-1))
        if not dst.exists():
            todo.append((members[name], dst))
    print(f"selected {len(out_rows)} clips ({a.per_word}/word max); {len(todo)} to download")

    t0, done, nbytes, errs = time.time(), 0, 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch_member, info, dst) for info, dst in todo]
        for fu in as_completed(futs):
            try:
                nbytes += fu.result()
            except Exception as e:  # keep going; a re-run picks up whatever failed
                errs += 1
                if errs <= 5:
                    print("  error:", e, flush=True)
            done += 1
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {nbytes / 1e9:.2f} GB  {nbytes / 1e6 / max(el, 1):.1f} MB/s  "
                      f"eta {(len(todo) - done) * el / max(done, 1) / 60:.0f} min  errors={errs}", flush=True)

    out_rows = [r for r in out_rows if Path(r["path"]).exists()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} clips on disk -> {a.out}")


if __name__ == "__main__":
    main()
