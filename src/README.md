# EchoGesture — On-Device Sign Language Recognition (Track 02)

Live camera video → word captions (and optional speech), **fully on-device**:
no frame, landmark or feature ever leaves the process.

```
video frame ──► MediaPipe Hands + Pose (TFLite, on-device) ──► 53 3D landmarks/frame
            ──► pose-invariant features (352-d) ──► INT8 ONNX classifier (4.1 MB, 2,000 words)
            ──► top-5 glosses + confidence ──► HUD captions / offline TTS
```

| Stage | Cost (laptop i5-1334U, power-saving, CPU only) |
|---|---|
| MediaPipe landmarks (hands every frame, pose every 2nd) | ~39 ms / frame |
| Feature extraction (numpy) | ~0.6 ms / clip |
| INT8 classifier (ONNX Runtime, 2 threads) | ~3 ms / clip |
| **Model size**: classifier INT8 | 4.1 MB (2,000 classes) |
| Model size incl. MediaPipe bundles | 17.7 MB total (< 25 MB) |

## Why it is robust (exocentric view)
* **Pose-invariant 3D hand features**: each hand is expressed in its own palm
  coordinate frame (wrist origin, wrist→middle-MCP axis, palm normal), so hand
  *shape* is invariant to camera angle, distance and hand position. Hand
  orientation and body-relative position are separate features.
* **Body normalisation**: centred on mid-shoulder, scaled by shoulder width,
  in-plane camera roll removed.
* **Hand assignment by pose wrists**, not MediaPipe handedness (which assumes a
  mirrored selfie image and breaks on a rear/third-person camera).
* **Augmentations** for angled signers (3D yaw ±30°), camera tilt/shear, moving
  camera drift, left-handed signers (mirroring), speed changes, missed hand
  detections (low light), contiguous hand occlusion, dropped pose points.
* **Low-light pre-processing**: CLAHE on luminance for dark frames.

## Setup (Python 3.11)
```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/mac: source .venv/bin/activate
pip install -r src/requirements.txt                   # inference only
pip install -r src/requirements-train.txt             # + training / conversion
python src/scripts/download_mediapipe_models.py       # one-time; only network step
```

## Inference (what the evaluator runs)
```python
from main import predict          # run from src/
predict("test_0412.mp4")
# {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}
```
`latency_ms` = per-frame landmark time + features + classifier, i.e. the delay
between the last frame of a sign and its caption.

Batch over the test set → `results/predictions.jsonl` (+ identical `predictions.json`):
```bash
cd src
python scripts/evaluate.py --videos PATH/TO/TEST_CLIPS
python scripts/evaluate.py --manifest test_manifest.csv      # clip_id[,path][,label]
```

## Reproduce training
```bash
cd src
# 1. WLASL (e.g. Kaggle "WLASL processed": WLASL_v0.3.json + videos/<id>.mp4) under ../data/wlasl
python scripts/prepare_wlasl.py --root ../data/wlasl --subset 2000     # or --vocab organiser_vocab.txt
# 1b. optional extra signers: ASL Citizen clips for words shared with WLASL (range-downloads only those clips)
python scripts/fetch_asl_citizen.py --per-word 30          # -> ../data/aslc_index.csv (TRAIN only)
# 2. cache landmarks once (the slow step, parallel)
python scripts/extract_landmarks.py --workers 6
python scripts/extract_landmarks.py --index ../data/aslc_index.csv --workers 6
# 2b. pixel-degraded copies (compression, over-exposure, colour cast, occlusion, low light)
python scripts/extract_landmarks.py --index ../data/aslc_index.csv --pix-aug 1 --pix-limit 6000
# 3. train (minutes per run on landmarks; GPU optional)
python scripts/train.py --epochs 100 --final --dim 256 --lr 3e-3 --bs 128   # picks up ASL Citizen + pixel copies if present
# 4. conversion script: ONNX FP32 -> static INT8 (QDQ, per-channel), prints FP32 vs INT8 accuracy
python scripts/export_quantize.py            # add --tflite for an extra TFLite build via onnx2tf
# 5. accuracy + robustness (synthetic occlusion & low light) on held-out split
python scripts/evaluate.py --wlasl-split test --robustness --stress --limit 150
# 6. latency / size / sustained-load thermal drift
python scripts/benchmark.py --clips 50 --thermal-minutes 10
# 7. results.csv with pass/fail (add the phone measurements)
python scripts/make_results_csv.py --device "Galaxy S24, Snapdragon 8 Gen 3" --avg-power-mw 410
```

## Live demo (HUD)
```bash
python scripts/live_demo.py --speak                     # webcam, captions + offline speech
python scripts/live_demo.py --source clip.mp4 --record demo.mp4
```
Shows hand landmarks, live top-3 guesses with confidence bars, committed
captions, FPS/latency and an "ON-DEVICE | OFFLINE" badge.

## Files
```
src/main.py                    predict(video_path) -> dict
src/echogesture/landmarks.py   MediaPipe Tasks extraction, low-light enhancement
src/echogesture/features.py    pose-invariant features (shared by train & inference)
src/echogesture/augment.py     exocentric augmentations
src/echogesture/model.py       temporal CNN + attention classifier
src/echogesture/runtime.py     ONNX Runtime inference pipeline
src/models/                    echogesture_int8.onnx, labels.txt, MediaPipe .task bundles
src/scripts/                   prepare / extract / train / export_quantize / evaluate / benchmark / live_demo
```
