"""Landmarks -> pose-invariant feature tensor (SEQ_LEN, FEAT_DIM).

Feature groups per frame (352 dims):
  106  body-normalised 2D positions of all 53 points (centre = mid-shoulder,
       scale = shoulder width, in-plane rotation removed)
  126  canonical 3D hand shape: each hand expressed in its own palm frame
       (wrist origin, wrist->middle-MCP axis, palm normal) -> invariant to hand
       position, size and 3D orientation, i.e. to camera angle
   12  hand orientation (palm direction + normal) in the body frame
    2  hand presence flags
  106  velocities of the body-normalised positions
"""
from __future__ import annotations

import numpy as np

from . import config as C

# L/R swap pairs within the 11-point pose subset (for mirroring)
POSE_SWAP = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def hand_present(seq: np.ndarray) -> np.ndarray:
    """(T,) bool: at least one hand detected."""
    lh = ~np.isnan(seq[:, C.LH0, 0])
    rh = ~np.isnan(seq[:, C.RH0, 0])
    return lh | rh


def trim_idle(seq: np.ndarray, margin: int = 2) -> np.ndarray:
    """Drop leading/trailing frames with no hands (signer at rest)."""
    pres = np.where(hand_present(seq))[0]
    if len(pres) < 2:
        return seq
    s = max(pres[0] - margin, 0)
    e = min(pres[-1] + margin + 1, len(seq))
    return seq[s:e]


def fill_hand_gaps(seq: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Linearly interpolate short runs of missed hand detections (blur, occlusion,
    low light). Gaps longer than `max_gap` frames stay missing."""
    seq = seq.copy()
    for o in (C.LH0, C.RH0):
        pres = np.where(~np.isnan(seq[:, o, 0]))[0]
        if len(pres) < 2:
            continue
        for a, b in zip(pres[:-1], pres[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                w = (np.arange(1, gap + 1) / (gap + 1))[:, None, None].astype(np.float32)
                ha, hb = seq[a, o:o + C.N_HAND], seq[b, o:o + C.N_HAND]
                seq[a + 1:b, o:o + C.N_HAND] = (1 - w) * ha + w * hb
    return seq


def resample(seq: np.ndarray, T: int = C.SEQ_LEN) -> np.ndarray:
    """Linear temporal resampling that tolerates NaNs (falls back to nearest frame)."""
    n = len(seq)
    if n == 1:
        return np.repeat(seq, T, axis=0)
    pos = np.linspace(0, n - 1, T)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0)[:, None, None].astype(np.float32)
    a, b = seq[i0], seq[i1]
    lin = (1 - w) * a + w * b
    near = seq[np.round(pos).astype(int)]
    return np.where(np.isnan(lin), near, lin).astype(np.float32)


def _body_frame(seq: np.ndarray):
    """Per-clip centre, scale and in-plane angle from the shoulders (robust medians)."""
    ls, rs = seq[:, C.P_LSHO, :2], seq[:, C.P_RSHO, :2]
    ok = ~(np.isnan(ls).any(1) | np.isnan(rs).any(1))
    if ok.sum() >= 1:
        mid = (ls[ok] + rs[ok]) / 2
        v = ls[ok] - rs[ok]
        center = np.median(mid, axis=0)
        scale = float(np.median(np.linalg.norm(v, axis=1)))
        ang = float(np.median(np.arctan2(v[:, 1], v[:, 0])))
        # a signer facing the camera has their left shoulder on image-right
        if abs(ang) > np.pi / 2:
            ang = ang - np.sign(ang) * np.pi
    else:
        hands = seq[:, C.LH0:, :2].reshape(-1, 2)
        hands = hands[~np.isnan(hands).any(1)]
        center = hands.mean(0) if len(hands) else np.array([0.5, 0.5], np.float32)
        scale, ang = 0.0, 0.0
    if not np.isfinite(scale) or scale < 0.05:
        scale = _hand_scale_fallback(seq)
    return center.astype(np.float32), np.float32(scale), np.float32(ang)


def _hand_scale_fallback(seq: np.ndarray) -> float:
    sizes = []
    for o in (C.LH0, C.RH0):
        d = np.linalg.norm(seq[:, o + 9, :2] - seq[:, o, :2], axis=1)
        sizes.append(d[~np.isnan(d)])
    sizes = np.concatenate(sizes)
    # shoulder width is ~2.5x the wrist->middle-MCP length
    return float(np.median(sizes) * 2.5) if len(sizes) else 0.25


def _rot2(ang: float) -> np.ndarray:
    c, s = np.cos(-ang), np.sin(-ang)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _canonical_hand(h: np.ndarray, R2: np.ndarray):
    """h: (T,21,3). Returns canonical shape (T,21,3) and orientation (T,6)."""
    rel = h - h[:, :1]
    size = np.linalg.norm(rel[:, 9], axis=1, keepdims=True)[:, :, None]
    rel = rel / np.maximum(size, 1e-6)
    u = rel[:, 9]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    side = rel[:, 5] - rel[:, 17]
    n = np.cross(u, side)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    v = np.cross(n, u)
    R = np.stack([u, v, n], axis=1)                 # (T,3,3) rows = axes
    canon = np.einsum("tij,tkj->tki", R, rel)       # rel @ R^T
    # orientation in the body frame (rotate xy only)
    u_b = u.copy(); n_b = n.copy()
    u_b[:, :2] = u[:, :2] @ R2.T
    n_b[:, :2] = n[:, :2] @ R2.T
    return canon, np.concatenate([u_b, n_b], axis=1)


def compute_features(seq: np.ndarray, T: int = C.SEQ_LEN, trim: bool = True) -> np.ndarray:
    """seq: (n, 53, 3) raw landmarks with NaN for missing -> (T, FEAT_DIM) float32."""
    seq = np.asarray(seq, dtype=np.float32)
    if trim:
        seq = trim_idle(seq)
    seq = fill_hand_gaps(seq)
    seq = resample(seq, T)
    center, scale, ang = _body_frame(seq)
    R2 = _rot2(ang)

    xy = (seq[..., :2] - center) / scale @ R2.T          # (T,53,2)
    xy = np.clip(xy, -6, 6)

    canon_l, ori_l = _canonical_hand(seq[:, C.LH0:C.LH0 + C.N_HAND], R2)
    canon_r, ori_r = _canonical_hand(seq[:, C.RH0:C.RH0 + C.N_HAND], R2)
    pres = np.stack([~np.isnan(seq[:, C.LH0, 0]), ~np.isnan(seq[:, C.RH0, 0])], 1)

    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    vel = np.nan_to_num(vel, nan=0.0) * 4.0

    feat = np.concatenate([
        xy.reshape(T, -1),
        np.clip(canon_l, -4, 4).reshape(T, -1), np.clip(canon_r, -4, 4).reshape(T, -1),
        ori_l, ori_r,
        pres.astype(np.float32),
        np.clip(vel, -6, 6).reshape(T, -1),
    ], axis=1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert feat.shape[1] == C.FEAT_DIM, feat.shape
    return feat


def mirror(seq: np.ndarray) -> np.ndarray:
    """Left-right mirror at the landmark level (swap hands + pose sides, flip x)."""
    out = seq.copy()
    cx = np.nanmean(seq[..., 0]) if np.isfinite(seq[..., 0]).any() else 0.5
    out[..., 0] = 2 * cx - out[..., 0]
    l, r = out[:, C.LH0:C.LH0 + C.N_HAND].copy(), out[:, C.RH0:C.RH0 + C.N_HAND].copy()
    out[:, C.LH0:C.LH0 + C.N_HAND], out[:, C.RH0:C.RH0 + C.N_HAND] = r, l
    for a, b in POSE_SWAP:
        out[:, [a, b]] = out[:, [b, a]]
    return out
