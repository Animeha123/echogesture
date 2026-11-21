"""Landmark-level augmentations that simulate the exocentric conditions in the
brief: angled signers, moving heads/cameras, partial occlusion, missed
detections in low light, left-handed signers and varying signing speed."""
from __future__ import annotations

import numpy as np

from . import config as C
from .features import mirror


def _rand(rng, lo, hi):
    return rng.uniform(lo, hi)


def augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = seq.copy()
    T = len(seq)

    # left-handed signers
    if rng.random() < 0.5:
        seq = mirror(seq)

    valid = np.isfinite(seq[..., 0])
    if valid.any():
        center = np.nanmean(seq[..., :3].reshape(-1, 3), axis=0)
        center = np.nan_to_num(center, nan=0.5)
    else:
        center = np.array([0.5, 0.5, 0.0], np.float32)
    p = seq - center

    # 3D yaw (signer turned away from camera) + small pitch
    if rng.random() < 0.7:
        yaw = np.deg2rad(_rand(rng, -30, 30))
        pitch = np.deg2rad(_rand(rng, -10, 10))
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        p = p @ (Rx @ Ry).T

    # in-plane rotation, anisotropic scale, shear (camera tilt / lens)
    if rng.random() < 0.8:
        a = np.deg2rad(_rand(rng, -15, 15))
        sx, sy_ = _rand(rng, 0.85, 1.15), _rand(rng, 0.85, 1.15)
        sh = _rand(rng, -0.15, 0.15)
        c, s = np.cos(a), np.sin(a)
        A = np.array([[c, -s], [s, c]], np.float32) @ np.array([[sx, sh], [0, sy_]], np.float32)
        p[..., :2] = p[..., :2] @ A.T

    # moving camera / head: slow drift
    if rng.random() < 0.3:
        drift = np.linspace(0, 1, T)[:, None, None] * rng.normal(0, 0.03, (1, 1, 3))
        drift[..., 2] = 0
        p = p + drift.astype(np.float32)

    seq = p + center

    # temporal crop (keep 75-100%)
    if T > 8 and rng.random() < 0.5:
        keep = int(T * _rand(rng, 0.75, 1.0))
        s = rng.integers(0, T - keep + 1)
        seq = seq[s:s + keep]
        T = len(seq)

    # speed change via resampling the raw frames
    if T > 4 and rng.random() < 0.5:
        newT = max(4, int(T * _rand(rng, 0.7, 1.4)))
        idx = np.linspace(0, T - 1, newT).round().astype(int)
        seq = seq[idx]
        T = len(seq)

    # missed hand detections (low light / motion blur), scattered and in short bursts
    if rng.random() < 0.7:
        for o in (C.LH0, C.RH0):
            m = rng.random(T) < _rand(rng, 0.0, 0.3)
            seq[m, o:o + C.N_HAND] = np.nan
    if rng.random() < 0.3:
        for _ in range(rng.integers(1, 4)):
            o = C.LH0 if rng.random() < 0.5 else C.RH0
            L = rng.integers(2, 8)
            s = rng.integers(0, max(T - L, 1))
            seq[s:s + L, o:o + C.N_HAND] = np.nan

    # contiguous occlusion of one hand (object / other person passing), up to half the sign
    if rng.random() < 0.5:
        for _ in range(1 if rng.random() < 0.7 else 2):
            o = C.LH0 if rng.random() < 0.5 else C.RH0
            L = int(T * _rand(rng, 0.1, 0.5))
            s = rng.integers(0, max(T - L, 1))
            seq[s:s + L, o:o + C.N_HAND] = np.nan

    # one hand hidden for the whole clip (held object, out of frame)
    if rng.random() < 0.1:
        o = C.LH0 if rng.random() < 0.5 else C.RH0
        seq[:, o:o + C.N_HAND] = np.nan

    # noisier landmarks, as produced by MediaPipe on dark/grainy frames
    if rng.random() < 0.3:
        hands = seq[:, C.LH0:]
        seq[:, C.LH0:] = hands + rng.normal(0, _rand(rng, 0.005, 0.015), hands.shape).astype(np.float32)

    # dropped pose points (face occluded, arm out of frame)
    if rng.random() < 0.3:
        k = rng.integers(1, 4)
        pts = rng.choice(C.N_POSE, size=k, replace=False)
        seq[:, pts] = np.nan

    # landmark jitter
    seq = seq + rng.normal(0, 0.004, seq.shape).astype(np.float32)
    return seq.astype(np.float32)
