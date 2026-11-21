"""EchoGesture - on-device isolated sign recognition.

Entry point required by the evaluator:

    from main import predict
    predict("path/to/test_0412.mp4")
    # -> {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...],
    #     "confidence": 0.81, "latency_ms": 23.4}

Everything runs locally (MediaPipe Tasks + ONNX Runtime CPU); no network calls.

CLI:
    python main.py clip.mp4 [clip2.mp4 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echogesture.runtime import EchoGesturePipeline  # noqa: E402

_PIPELINE: EchoGesturePipeline | None = None


def _pipeline() -> EchoGesturePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = EchoGesturePipeline()
    return _PIPELINE


def predict(video_path: str, frame_transform=None) -> dict:
    """Return the predicted gloss/word for one video clip. Must run fully on-device."""
    clip_id = Path(video_path).stem
    try:
        out = _pipeline().predict_video(str(video_path), frame_transform)
    except Exception as e:  # never crash the evaluator on one bad clip
        print(f"[predict] {clip_id}: {e}", file=sys.stderr)
        labels = _pipeline().clf.labels
        return {"clip_id": clip_id, "prediction": labels[0], "top5": labels[:5],
                "confidence": 0.0, "latency_ms": 0.0}
    return {"clip_id": clip_id,
            "prediction": out["prediction"],
            "top5": out["top5"],
            "confidence": round(out["confidence"], 4),
            "latency_ms": out["latency_ms"]}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        print(json.dumps(predict(p)))
