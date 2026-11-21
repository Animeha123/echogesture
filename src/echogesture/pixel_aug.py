"""Pixel-level training perturbations (occluders, low light, blur, compression,
over-exposure, colour casts).

Landmark-level augmentation only simulates *missing* detections. Real occlusion and
darkness also make MediaPipe return *wrong* landmarks, so we re-run MediaPipe on
perturbed copies of the TRAIN/VAL videos and train on those landmarks too.

These are deliberately broader than (and seeded differently from) the fixed
perturbations in scripts/evaluate.py, so the robustness test is not memorised:
moving or static occluders of many shapes/colours/sizes, a range of gains,
noise, blur and colour casts. Test-split clips are never perturbed for training.
"""
import cv2
import numpy as np


def _occluder(r):
    n = 1 if r.random() < 0.75 else 2
    boxes = []
    for _ in range(n):
        w, h = r.uniform(0.10, 0.32), r.uniform(0.10, 0.32)
        x0, y0 = r.uniform(0.05, 0.85 - w), r.uniform(0.15, 0.90 - h)
        moving = r.random() < 0.4
        dx, dy = (r.uniform(-0.25, 0.25), r.uniform(-0.15, 0.15)) if moving else (0.0, 0.0)
        colour = tuple(int(c) for c in (r.integers(0, 256, 3) if r.random() < 0.5
                                        else np.full(3, r.integers(0, 90))))
        ellipse = r.random() < 0.35
        boxes.append((x0, y0, w, h, dx, dy, colour, ellipse))
    return boxes


def make_train_perturb(seed: int, n_frames_hint: int = 64):
    """Return a frame -> frame callable with one randomly drawn perturbation."""
    r = np.random.default_rng(seed)
    kind = r.choice(["occlusion", "lowlight", "both", "compress", "bright", "colour"],
                    p=[0.15, 0.15, 0.05, 0.25, 0.20, 0.20])
    boxes = _occluder(r) if kind in ("occlusion", "both") else []
    gain = r.uniform(0.15, 0.55) if kind in ("lowlight", "both") else 1.0
    noise = r.uniform(2, 9) if gain < 1 else 0.0
    cast = r.uniform(0.85, 1.15, 3) if gain < 1 and r.random() < 0.5 else None
    blur = int(r.choice([0, 3, 5])) if r.random() < 0.3 else 0
    # compression: low JPEG quality, sometimes after a down/up-scale (low-bitrate stream)
    jpeg_q = int(r.integers(6, 36)) if kind == "compress" else 0
    down = float(r.uniform(0.3, 0.7)) if kind == "compress" and r.random() < 0.5 else 1.0
    # over-exposure: gain > 1, lifted blacks, reduced contrast
    bgain, boff, bcon = ((r.uniform(1.2, 2.0), r.uniform(10, 70), r.uniform(0.5, 0.9))
                         if kind == "bright" else (1.0, 0.0, 1.0))
    # colour: strong per-channel white-balance shift and saturation change
    wb = r.uniform(0.6, 1.4, 3) if kind == "colour" else None
    sat = r.uniform(0.2, 1.8) if kind == "colour" else 1.0
    state = {"i": 0}
    nr = np.random.default_rng(seed + 1)

    def f(img):
        t = min(state["i"] / max(1, n_frames_hint - 1), 1.0)
        state["i"] += 1
        h, w = img.shape[:2]
        img = img.copy()
        for x0, y0, bw, bh, dx, dy, col, ell in boxes:
            x, y = int((x0 + dx * t) * w), int((y0 + dy * t) * h)
            pw, ph = int(bw * w), int(bh * h)
            if ell:
                cv2.ellipse(img, (x + pw // 2, y + ph // 2), (pw // 2, ph // 2), 0, 0, 360, col, -1)
            else:
                cv2.rectangle(img, (x, y), (x + pw, y + ph), col, -1)
        if blur:
            img = cv2.GaussianBlur(img, (blur, blur), 0)
        if gain < 1:
            x = img.astype(np.float32) * gain
            if cast is not None:
                x *= cast
            x += nr.normal(0, noise, x.shape)
            img = np.clip(x, 0, 255).astype(np.uint8)
        if bgain != 1.0:
            x = img.astype(np.float32)
            x = (x - 128.0) * bcon + 128.0
            img = np.clip(x * bgain + boff, 0, 255).astype(np.uint8)
        if wb is not None:
            x = img.astype(np.float32) * wb
            g = x.mean(axis=2, keepdims=True)
            img = np.clip(g + (x - g) * sat, 0, 255).astype(np.uint8)
        if jpeg_q:
            if down < 1.0:
                small = cv2.resize(img, (max(16, int(w * down)), max(16, int(h * down))),
                                   interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                img = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (w, h),
                                 interpolation=cv2.INTER_LINEAR)
            else:
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    return f
