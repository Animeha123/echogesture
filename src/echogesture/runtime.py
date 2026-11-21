"""Inference runtime: MediaPipe landmarks -> features -> INT8 ONNX classifier.

No network access anywhere in this module.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as C
from .features import compute_features
from .landmarks import FrameTransform, LandmarkExtractor


def load_labels(path: Path = C.LABELS_TXT) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip()]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class Classifier:
    """ONNX Runtime session over the quantised classifier."""

    def __init__(self, model_path: Path | None = None, threads: int = 2):
        import onnxruntime as ort
        if model_path is None:
            model_path = C.ONNX_INT8 if C.ONNX_INT8.exists() else C.ONNX_FP32
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.model_path = Path(model_path)
        self.labels = load_labels()

    def logits(self, feats: np.ndarray) -> np.ndarray:
        x = feats[None].astype(np.float32) if feats.ndim == 2 else feats.astype(np.float32)
        return self.sess.run(None, {self.input_name: x})[0]

    def classify(self, feats: np.ndarray, k: int = 5) -> dict:
        t0 = time.perf_counter()
        # a model may have spare output units beyond the real classes - ignore them
        probs = softmax(self.logits(feats)[..., :len(self.labels)])[0]
        dt = (time.perf_counter() - t0) * 1000
        top = np.argsort(-probs)[:k]
        return {"prediction": self.labels[top[0]],
                "top5": [self.labels[i] for i in top],
                "confidence": float(probs[top[0]]),
                "classifier_ms": dt}


class EchoGesturePipeline:
    def __init__(self, model_path: Optional[Path] = None, threads: int = 2):
        self.extractor = LandmarkExtractor()
        self.clf = Classifier(model_path, threads)

    def predict_video(self, video_path: str, frame_transform: FrameTransform = None) -> dict:
        seq, timing = self.extractor.process_video(video_path, frame_transform)
        t0 = time.perf_counter()
        feats = compute_features(seq)
        feat_ms = (time.perf_counter() - t0) * 1000
        out = self.clf.classify(feats)
        # Streaming latency = per-frame landmarking + features + classifier: the delay
        # between the last frame of a sign and its caption on the HUD.
        out["latency_ms"] = round(timing["landmark_ms_per_frame"] + feat_ms + out["classifier_ms"], 2)
        out["_timing"] = {**timing, "feature_ms": feat_ms, "classifier_ms": out.pop("classifier_ms")}
        return out
