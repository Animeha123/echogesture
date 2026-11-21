"""On-device landmark extraction with MediaPipe Tasks (hands + upper-body pose).

Everything here runs locally: the .task model bundles are loaded from disk and
no frame ever leaves the process.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from . import config as C

FrameTransform = Optional[Callable[[np.ndarray], np.ndarray]]


def _require_models() -> None:
    missing = [p for p in (C.HAND_TASK, C.POSE_TASK) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing MediaPipe model(s): {[str(m) for m in missing]}. "
            "Run: python scripts/download_mediapipe_models.py")


def enhance_low_light(bgr: np.ndarray, thresh: float = 85.0) -> np.ndarray:
    """CLAHE on luminance when the frame is dark -> helps hand detection in low light."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mean = float(ycc[..., 0].mean())
    if mean >= thresh:
        return bgr
    if mean < 60:  # very dark: lift brightness with gamma before local contrast
        gamma = 0.45 if mean < 35 else 0.6
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        ycc[..., 0] = cv2.LUT(ycc[..., 0], lut)
        ycc[..., 0] = cv2.medianBlur(ycc[..., 0], 3)  # suppress amplified sensor noise
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ycc[..., 0] = clahe.apply(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def resize_long_side(bgr: np.ndarray, long_side: int = C.PROC_LONG_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return bgr
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class LandmarkExtractor:
    """Wraps MediaPipe HandLandmarker + PoseLandmarker in VIDEO (tracking) mode.

    Output per frame: float32 array (N_POINTS, 3), NaN where a part is missing.
    Coordinates are image-normalised with x and z multiplied by the aspect ratio
    so that x, y, z share the same unit.
    """

    def __init__(self, num_threads_hint: int = 0):
        os.environ.setdefault("GLOG_minloglevel", "2")  # hide MediaPipe's harmless warning spam
        _require_models()
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._vision = vision
        self._BaseOptions = BaseOptions
        self._hand = None
        self._pose = None
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="pose")
        self.reset()

    def reset(self) -> None:
        """Fresh trackers (call between independent clips so tracking state does not leak)."""
        self.close()
        v, B = self._vision, self._BaseOptions
        self._hand = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=B(model_asset_path=str(C.HAND_TASK)),
            running_mode=v.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35))
        self._pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=B(model_asset_path=str(C.POSE_TASK)),
            running_mode=v.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.4, min_tracking_confidence=0.4))
        self._ts = 0
        self._frame_i = 0
        self._last_pose = None

    def close(self) -> None:
        for m in (self._hand, self._pose):
            if m is not None:
                m.close()
        self._hand = self._pose = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ frames
    def process_frame(self, bgr: np.ndarray, ts_ms: Optional[int] = None) -> np.ndarray:
        bgr = enhance_low_light(resize_long_side(bgr))
        h, w = bgr.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if ts_ms is None:
            self._ts += 33
        else:
            self._ts = max(self._ts + 1, int(ts_ms))
        # Pose and hands are independent graphs and MediaPipe releases the GIL, so run
        # pose on a worker thread while hands run here (~12 ms/frame saved on 4+ cores).
        run_pose = self._frame_i % C.POSE_EVERY == 0 or self._last_pose is None
        fut = self._pool.submit(self._pose.detect_for_video, img, self._ts) if run_pose else None
        hand_res = self._hand.detect_for_video(img, self._ts)

        out = np.full((C.N_POINTS, 3), np.nan, dtype=np.float32)
        if fut is not None:
            pose_res = fut.result()
            pose = np.full((C.N_POSE, 3), np.nan, dtype=np.float32)
            if pose_res.pose_landmarks:
                pl = pose_res.pose_landmarks[0]
                for j, idx in enumerate(C.POSE_IDX):
                    p = pl[idx]
                    vis = getattr(p, "visibility", None)
                    if vis is None or vis > 0.3:
                        pose[j] = (p.x * aspect, p.y, p.z * aspect)
            self._last_pose = pose
        out[:C.N_POSE] = self._last_pose
        self._frame_i += 1

        hands = []
        for i, hl in enumerate(hand_res.hand_landmarks or []):
            arr = np.array([(p.x * aspect, p.y, p.z * aspect) for p in hl], dtype=np.float32)
            label = hand_res.handedness[i][0].category_name if hand_res.handedness else "Left"
            hands.append((arr, label))
        for arr, side in self._assign_hands(hands, out):
            o = C.LH0 if side == "L" else C.RH0
            out[o:o + C.N_HAND] = arr
        return out

    @staticmethod
    def _assign_hands(hands, pose_pts):
        """Assign detected hands to the signer's left/right hand.

        Prefer proximity to the pose wrists (robust for a third-person camera);
        fall back to MediaPipe handedness, which assumes a *mirrored* image, so
        on a normal rear camera its "Left" is the signer's right hand.
        """
        if not hands:
            return []
        lw, rw = pose_pts[C.P_LWRI, :2], pose_pts[C.P_RWRI, :2]
        if not (np.isnan(lw).any() or np.isnan(rw).any()):
            if len(hands) == 1:
                d_l = np.linalg.norm(hands[0][0][0, :2] - lw)
                d_r = np.linalg.norm(hands[0][0][0, :2] - rw)
                return [(hands[0][0], "L" if d_l < d_r else "R")]
            a, b = hands[0][0], hands[1][0]
            cost_ab = np.linalg.norm(a[0, :2] - lw) + np.linalg.norm(b[0, :2] - rw)
            cost_ba = np.linalg.norm(a[0, :2] - rw) + np.linalg.norm(b[0, :2] - lw)
            return [(a, "L"), (b, "R")] if cost_ab <= cost_ba else [(a, "R"), (b, "L")]
        out, used = [], set()
        for arr, label in hands[:2]:
            side = "R" if label == "Left" else "L"
            if side in used:
                side = "L" if side == "R" else "R"
            used.add(side)
            out.append((arr, side))
        return out

    # ------------------------------------------------------------------ videos
    def process_video(self, video_path: str, frame_transform: FrameTransform = None,
                      max_frames: int = C.MAX_VIDEO_FRAMES,
                      frame_range: tuple[int, int] | None = None) -> tuple[np.ndarray, dict]:
        """Returns (landmarks[T, N_POINTS, 3], timing info)."""
        self.reset()
        frames = list(iter_video_frames(video_path, max_frames, frame_range))
        if not frames:
            raise ValueError(f"Could not read any frames from {video_path}")
        seq = np.empty((len(frames), C.N_POINTS, 3), dtype=np.float32)
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            if frame_transform is not None:
                f = frame_transform(f)
            seq[i] = self.process_frame(f)
        dt = (time.perf_counter() - t0) * 1000
        return seq, {"n_frames": len(frames), "landmark_ms_total": dt,
                     "landmark_ms_per_frame": dt / len(frames)}


def iter_video_frames(video_path: str, max_frames: int = C.MAX_VIDEO_FRAMES,
                      frame_range: tuple[int, int] | None = None) -> Iterator[np.ndarray]:
    """Decode a clip, uniformly sub-sampling to at most `max_frames` frames.

    frame_range: optional 1-based inclusive (start, end) as in WLASL json (end=-1 -> last).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    frames = []
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(resize_long_side(f))
    finally:
        cap.release()
    if frame_range is not None:
        s, e = frame_range
        s = max(int(s) - 1, 0)
        e = len(frames) if e is None or int(e) < 0 else int(e)
        if e - s >= 4:
            frames = frames[s:e]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    yield from frames
