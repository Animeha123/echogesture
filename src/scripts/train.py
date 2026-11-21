"""Train the EchoGesture classifier on cached landmarks.

    python scripts/train.py --epochs 120
    python scripts/train.py --epochs 120 --final     # train on train+val for the submission model

Saves the best (val top-1) checkpoint to src/models/echogesture.pt.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echogesture import config as C  # noqa: E402
from echogesture.augment import augment  # noqa: E402
from echogesture.features import compute_features  # noqa: E402
from echogesture.model import build_model  # noqa: E402


def load_split(index: Path, lm_dir: Path, splits, pix_dir: Path | None = None):
    """Clean landmarks for `splits`; with pix_dir, also the pixel-perturbed variants
    (<vid>__p*.npy) of those same clips - only ever passed for training splits."""
    with open(index, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    pix = {}
    if pix_dir is not None:  # list the folder once instead of one glob per clip
        for q in sorted(pix_dir.glob("*__p*.npy")):
            pix.setdefault(q.name.rsplit("__p", 1)[0], []).append(q)
    X, y = [], []
    for r in rows:
        p = lm_dir / f"{r['video_id']}.npy"
        if p.exists():
            X.append(np.load(p).astype(np.float32))
            y.append(int(r["label"]))
        for q in pix.get(r["video_id"], []):
            X.append(np.load(q).astype(np.float32))
            y.append(int(r["label"]))
    return X, np.array(y, dtype=np.int64)


class LandmarkDS(Dataset):
    def __init__(self, X, y, train: bool, seed: int = 0):
        self.X, self.y, self.train = X, y, train
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = self.X[i]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            seq = augment(seq, rng)
        return torch.from_numpy(compute_features(seq)), self.y[i]


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    n = c1 = c5 = 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        top = model(xb).topk(5, dim=1).indices
        c1 += (top[:, 0] == yb).sum().item()
        c5 += (top == yb[:, None]).any(1).sum().item()
        n += len(yb)
    return c1 / max(n, 1), c5 / max(n, 1)


def mixup(x, y, n_cls, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    t = nn.functional.one_hot(y, n_cls).float()
    return lam * x + (1 - lam) * x[idx], lam * t + (1 - lam) * t[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=C.DATA_DIR / "wlasl_index.csv")
    ap.add_argument("--landmarks", type=Path, default=C.DATA_DIR / "landmarks")
    ap.add_argument("--pix-landmarks", type=Path, default=C.DATA_DIR / "landmarks_pix",
                    help="occluded/low-light variants from extract_landmarks.py --pix-aug (used if present)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--final", action="store_true", help="train on train+val, keep last epoch")
    ap.add_argument("--out", type=Path, default=C.CHECKPOINT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-index", type=Path, default=C.DATA_DIR / "aslc_index.csv",
                    help="extra TRAIN-only clips (e.g. ASL Citizen); used if the file exists")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # one gloss per line - glosses can contain spaces ("THANK YOU"), so never .split()
    labels = [ln.strip() for ln in C.LABELS_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_cls = len(labels)

    tr_splits = {"train", "val"} if a.final else {"train"}
    pix = a.pix_landmarks if a.pix_landmarks.is_dir() and any(a.pix_landmarks.glob("*.npy")) else None
    Xtr, ytr = load_split(a.index, a.landmarks, tr_splits, pix)
    if a.extra_index and a.extra_index.exists():
        Xe, ye = load_split(a.extra_index, a.landmarks, {"train"}, pix)
        Xtr, ytr = Xtr + Xe, np.concatenate([ytr, ye])
        print(f"extra training clips from {a.extra_index.name}: {len(Xe)}")
    print(f"pixel-perturbed variants: {'using ' + str(pix) if pix else 'none'}")
    Xva, yva = load_split(a.index, a.landmarks, {"test"} if a.final else {"val"})
    Xte, yte = load_split(a.index, a.landmarks, {"test"})
    print(f"{n_cls} classes | train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | device {dev}")

    tr_ds = LandmarkDS(Xtr, ytr, train=True, seed=a.seed)
    tr_dl = DataLoader(tr_ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                       persistent_workers=False, pin_memory=dev == "cuda")
    va_dl = DataLoader(LandmarkDS(Xva, yva, False), 128, num_workers=a.workers)
    te_dl = DataLoader(LandmarkDS(Xte, yte, False), 128, num_workers=a.workers)

    cfg = dict(in_dim=C.FEAT_DIM, n_classes=n_cls, d=a.dim, seq_len=C.SEQ_LEN, drop=a.drop)
    model = build_model(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "pos" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps = a.epochs * len(tr_dl)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")

    best = -1.0
    for ep in range(a.epochs):
        tr_ds.epoch = ep
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                if np.random.rand() < 0.5:
                    xm, tm = mixup(xb, yb, n_cls)
                    loss = torch.sum(-tm * torch.log_softmax(model(xm), 1), 1).mean()
                else:
                    loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        v1, v5 = evaluate(model, va_dl, dev)
        tag = ""
        if a.final or v1 > best:
            best = max(best, v1)
            torch.save({"cfg": cfg, "state_dict": model.state_dict(), "labels": labels,
                        "epoch": ep, "val_top1": v1}, a.out)
            tag = " *saved"
        print(f"ep {ep + 1:3d}/{a.epochs} loss {tot / n:.3f} val top1 {v1 * 100:5.1f} top5 {v5 * 100:5.1f} "
              f"({time.time() - t0:.0f}s){tag}", flush=True)

    ck = torch.load(a.out, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    t1, t5 = evaluate(model, te_dl, dev)
    print(f"\nTEST (landmark cache, clean): top1 {t1 * 100:.2f}%  top5 {t5 * 100:.2f}%  "
          f"(checkpoint epoch {ck['epoch'] + 1})")
    # The hidden test uses an unknown subset of the 2000 glosses; WLASL-N subsets are the
    # first N glosses, so report each one (model still chooses among all n_cls classes).
    if n_cls > 100:
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in te_dl:
                preds.append(model(xb.to(dev)).topk(5, 1).indices.cpu())
        preds, yt = torch.cat(preds), torch.as_tensor(np.asarray(yte))
        for k in (100, 300, 1000, 2000):
            m = yt < k
            if k <= n_cls and m.any():
                p1 = (preds[m, 0] == yt[m]).float().mean().item() * 100
                p5 = (preds[m] == yt[m, None]).any(1).float().mean().item() * 100
                print(f"  WLASL-{k} test clips ({int(m.sum())}): top1 {p1:.2f}%  top5 {p5:.2f}%")


if __name__ == "__main__":
    main()
