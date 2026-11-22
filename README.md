# EchoGesture: on-device sign-language recognition

**XR Hackathon 2026 · Track 02 · Team `gods_plan` · VIT Vellore**
Bhavya Singla · Vasabdatwa Patra · Yug Arora · Animeha Goutam

EchoGesture recognises isolated American Sign Language signs from video. It covers the full **2,000-word WLASL vocabulary** and runs **entirely on-device** with no network access, both on a laptop (Python) and on an Android phone (Kotlin app).

```
camera / video ─► MediaPipe Hands (every frame) + Pose-lite (every 2nd frame, parallel)
               ─► 53 landmarks / frame ─► body-normalised features (48 frames × 352)
               ─► Conv1d + self-attention classifier, INT8 ONNX (4.1 MB)
               ─► word + top-5 + confidence ─► sentence builder (app)
```

## Results

| Metric | Target | Ours | |
|---|---|---|---|
| Top-1 accuracy, WLASL-2000 test (1,414 clips) | ≥ 70 % | **61.0 %** (top-5 91.0 %) | ✗ |
| p95 latency (last frame → caption) | ≤ 50 ms | **91.5 ms** on Galaxy S23 FE, Exynos 2200 (p50 52 ms, 18 FPS) | ✗ |
| Model size, whole pipeline (classifier + MediaPipe) | ≤ 25 MB | **17.7 MB** | ✓ |
| Robustness drop (7 realistic conditions) | ≤ 5 pt | **1.0 pt** | ✓ |
| Average power | ≤ 500 mW | **≈ 2,575 mW** for ML (4,930 mW with ML on for 3 min − 2,355 mW with ML paused for 2 min, same session; camera + screen on in both) | ✗ |
| Thermal drift (10 min sustained) | ≤ 20 % | **−22.5 %** on phone over 11.9 min (got faster, no throttling) | ✓ |

Accuracy on vocabulary subsets (same model, choosing among all 2,000 words):

| Subset | Top-1 | Top-5 |
|---|---|---|
| WLASL-100 | 67.0 % | 91.0 % |
| WLASL-300 | 68.1 % | 94.0 % |
| WLASL-1000 | 64.8 % | 92.8 % |
| WLASL-2000 | 61.0 % | 91.0 % |

INT8 quantization costs **0.3 pt** top-1 compared with FP32 (61.3 % → 61.0 %, 15.1 MB → 4.1 MB).

Robustness is measured on 150 held-out test clips under these conditions: dim, over-exposed, colour cast, blur, heavy compression, signer far away, and camera angle. Stress tests add occlusion and very low light. Per-condition numbers are in `results/metrics.json`.

## Repository layout

```
src/                    everything the evaluator needs (Python 3.11)
  main.py               predict(video_path) -> {clip_id, prediction, top5, confidence, latency_ms}
  echogesture/          landmarks, features, augmentation, model, ONNX runtime
  scripts/              data prep, landmark extraction, training, export_quantize (conversion),
                        evaluate (accuracy + robustness), benchmark (latency, size, thermal)
  models/               echogesture_int8.onnx (deployed), echogesture_fp32.onnx, echogesture.pt
                        (training checkpoint), labels.txt, MediaPipe .task bundles
results/                predictions.jsonl / .json, results.csv, metrics.json, benchmark.json
android/                Android Studio project (CameraX + MediaPipe + ONNX Runtime), see android/README.md
training/               Colab notebook + bundling script used for GPU training
docs/                   submission deck (.pptx) and demo video
gods_plan.zip           packaged submission: src/ + results/
```

## Quick start (Python 3.11, CPU only)

```bash
pip install -r src/requirements.txt
cd src
python main.py path/to/clip.mp4                          # prints the JSON for one clip
python scripts/evaluate.py --videos path/to/clips/       # -> results/predictions.jsonl
python scripts/evaluate.py --wlasl-split test --robustness --stress --limit 150
python scripts/benchmark.py --clips 50 --thermal-minutes 10
python scripts/make_results_csv.py --avg-power-mw <phone value>
```

```python
from main import predict      # run from src/
predict("test_0412.mp4")
# {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}
```

Full training and reproduction steps are in [`src/README.md`](src/README.md).

## Android app

Open `android/` in Android Studio and press **Run**. Every model is bundled in `assets/`, so the app needs no network. It shows the live word with top-5 and confidence, builds a sentence (tap an alternative to correct a word), and segments signs automatically when your hands rise and fall. A stats overlay shows latency p50/p95, FPS, power (ML on vs off), battery temperature and thermal drift. See [`android/README.md`](android/README.md).

## How we got here

| Step | WLASL-2000 top-1 | Robustness drop |
|---|---|---|
| Landmark model on WLASL only (dim 256) | 45.1 % | 6.1 pt |
| Bigger model (dim 384) | 40.0 % (overfits, rejected) | – |
| + 16,620 ASL Citizen clips for the 1,385 words it shares with WLASL | 56.2 % | 4.3 pt |
| + 7,826 pixel-augmented copies (compression, exposure, colour, occlusion) | 57.0 % FP32 / 56.5 % INT8 | 3.6 pt |
| **+ up to 30 ASL Citizen clips per word (41,464 clips in total)** | **61.3 %** FP32 / **61.0 %** INT8 | **1.0 pt** |

## Known limitations and next steps

- **Latency and power are dominated by MediaPipe landmark detection, not the classifier.** On the Galaxy S23 FE a frame takes ~52 ms (p50), of which features + classifier are only ~5-20 ms. Hand landmarks run on every frame and pose on every second frame, all on the CPU at ~18 FPS, which keeps the big cores busy (~2.6 W for ML).
- **Next steps:** run MediaPipe hands/pose on the GPU delegate (typically 2-4x less power and lower latency), cap the camera at 15 FPS / 640x480, skip pose while no hands are visible, and try NNAPI/GPU for the classifier. These were not tested in time because the app was built without an Android SDK on the training laptop.
- **Accuracy (61.0 % vs 70 % target):** the model sees only 53 landmarks per frame; the hardest confusions are between signs with near-identical hand shapes and motion. More signer data and a larger temporal model are the main levers.

## Data and credits

Training videos are **not** included in this repository; both datasets' licences forbid redistribution.

- **WLASL**: Li et al., *Word-level Deep Sign Language Recognition from Video*, WACV 2020. Train/val/test splits as published; all reported accuracies are on the official **test** split, which was never used for training.
- **ASL Citizen**: Desai et al., *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, NeurIPS 2023 Datasets & Benchmarks (Microsoft Research). Used as **additional training data only**, under its research-only licence.
- **MediaPipe** Hand Landmarker and Pose Landmarker (lite) by Google, Apache 2.0.

`src/scripts/prepare_wlasl.py` and `src/scripts/fetch_asl_citizen.py` rebuild the data indexes from the original sources.
