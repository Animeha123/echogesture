# EchoGesture: sign language to text, entirely on your phone

**XR Hackathon 2026 · Track 02 · Team `gods_plan` · VIT Vellore**
Bhavya Singla · Vasabdatwa Patra · Yug Arora · Animeha Goutam

Point your phone at someone signing in American Sign Language and EchoGesture turns their signs into words, then sentences, then speech. Nothing leaves the phone: no server, no internet, no video upload.

We built it to be something a person could actually install and use today, not just a model in a notebook.

🎬 **[Watch the demo video](docs/demo_video.mp4)**


https://github.com/user-attachments/assets/1cd6ab78-074c-4ffa-a776-58d44ed02bb2



## Why EchoGesture stands out

- **📱 A real, installable Android app.** [**Download the APK from Google Drive**](https://drive.google.com/file/d/1x4vo_CWDhcoI93I15XHD4scHbvaA99iE/view?usp=drive_link), install it, and start signing. No Android Studio and no setup. The full source is in `android/` if you want to build it yourself.
- **🔤 The complete 2,000-word WLASL vocabulary.** Many sign-recognition demos shrink the problem to 100 or 300 words. We kept all 2,000, and every number below is our model choosing among all of them.
- **✅ Tested on every one of the 1,414 official test videos** through the same `predict()` function the evaluator runs: **61.0 % top-1, 91.1 % top-5**. The test split was never used for training.
- **💪 Barely notices bad conditions.** Across dim light, glare, colour casts, blur, heavy compression, distance and camera angle, accuracy drops by only **1.0 pt** on average (target ≤ 5 pt).
- **🪶 Small.** The whole pipeline, hand and pose trackers included, is **17.7 MB**. Our own classifier is 4.1 MB in INT8 and loses only 0.3 pt to quantization.
- **🔒 Private by design.** Hand landmarks are computed on the device and the raw video is never stored or sent anywhere.
- **🗣️ Built for real conversations.** The app does more than show a label:
  - It **builds whole sentences**, one sign after another.
  - It **reads them aloud** with text-to-speech.
  - **Tap one of the top-5 suggestions** to fix a misread word.
  - Signs are **detected automatically** when your hands rise and fall, or you can use **press-and-hold** mode instead.
  - A **searchable list of all 2,000 words** shows what it knows.
- **📊 Measured on a real phone, honestly.** The app has a built-in stats overlay (latency, FPS, ML power on vs off, battery temperature, drift). We report what it showed on a Galaxy S23 FE, including where we fell short.
- **📚 Extra data.** Besides WLASL we trained on **41,464 clips from ASL Citizen**: different signers, cameras and homes, which is a big part of why it holds up in messy real-world video.

```
camera ─► MediaPipe Hands (every frame) + Pose-lite (every 2nd frame)
       ─► 53 landmarks per frame ─► body-normalised features (48 frames × 352)
       ─► Conv1d + self-attention classifier, INT8 ONNX (4.1 MB)
       ─► word + top-5 + confidence ─► sentence ─► speech
```

## Results

| Metric | Target | Ours | |
|---|---|---|---|
| Top-1 accuracy, all 1,414 WLASL-2000 test videos | ≥ 70 % | **61.0 %** (top-5 91.1 %) | ✗ |
| p95 latency per frame | ≤ 50 ms | **91.5 ms** on Galaxy S23 FE, Exynos 2200 (p50 52 ms, 18 FPS) | ✗ |
| Model size, whole pipeline (classifier + MediaPipe) | ≤ 25 MB | **17.7 MB** | ✓ |
| Robustness drop (7 realistic conditions) | ≤ 5 pt | **1.0 pt** | ✓ |
| Average power for ML | ≤ 500 mW | **≈ 2,575 mW** (4,930 mW ML on − 2,355 mW ML paused, same session) | ✗ |
| Thermal drift (10+ min sustained) | ≤ 20 % | **−22.5 %** over 11.9 min (it got faster; no throttling) | ✓ |

We didn't hit every target, and we'd rather say so than hide it. The section on known limitations below explains why, and what we would do next.

Accuracy by vocabulary size (same model, still choosing among all 2,000 words):

| Subset | Top-1 | Top-5 |
|---|---|---|
| WLASL-100 | 67.0 % | 91.0 % |
| WLASL-300 | 68.1 % | 94.0 % |
| WLASL-1000 | 64.8 % | 92.8 % |
| WLASL-2000 | 61.0 % | 91.1 % |

The robustness test uses 150 held-out test clips. It covers seven realistic conditions: dim light, over-exposure, colour cast, blur, heavy compression, signer far away, and a different camera angle. Two extra stress tests add a partly hidden hand and very low light. Per-condition numbers are in `results/metrics.json`.

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

**Easiest:** download the [APK from Google Drive](https://drive.google.com/file/d/1x4vo_CWDhcoI93I15XHD4scHbvaA99iE/view?usp=drive_link) on an Android phone (Android 8.0 or newer), tap it, and allow installing from this source. **From source:** open `android/` in Android Studio and press **Run**. Every model is bundled in `assets/`, so the app needs no network. It shows the live word with top-5 and confidence, builds a sentence (tap an alternative to correct a word), and segments signs automatically when your hands rise and fall. A stats overlay shows latency p50/p95, FPS, power (ML on vs off), battery temperature and thermal drift. See [`android/README.md`](android/README.md).

## How we got here

We got here one step at a time, and kept only what actually helped:

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

## AI usage disclosure

We used an AI coding assistant (Claude, by Anthropic) during development, mainly for implementation support and debugging.

**What the AI did**
- Assisted with portions of the code implementation and debugging: the Python pipeline scripts (landmark extraction, features, training, quantization, evaluation), the Android app code and UI, and the packaging scripts.
- Generated initial drafts of some documentation tables and text, which the team reviewed, rewrote, and finalised.

**What the team did**
- Came up with the idea and designed the pipeline architecture: what to extract, how the stages fit together, and how it runs fully on-device.
- Chose the approach and datasets, and decided which models to keep or drop.
- Wrote the final README text and deck content ourselves.
- Ran GPU training on Colab and Kaggle.
- Built, installed, and tested the Android app on a real phone (Samsung Galaxy S23 FE).
- Took all on-device measurements (latency, power, thermal drift).
- Recorded the demo video, reviewed everything, and submitted the work.

All reported numbers come from our own runs and are reproducible with the scripts in `src/scripts/`. No AI model is part of the shipped product: EchoGesture runs only our own trained classifier and Google's MediaPipe, fully on-device.

## Data and credits

Training videos are **not** included in this repository; both datasets' licences forbid redistribution.

- **WLASL**: Li et al., *Word-level Deep Sign Language Recognition from Video*, WACV 2020. Train/val/test splits as published; all reported accuracies are on the official **test** split, which was never used for training.
- **ASL Citizen**: Desai et al., *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition*, NeurIPS 2023 Datasets & Benchmarks (Microsoft Research). Used as **additional training data only**, under its research-only licence.
- **MediaPipe** Hand Landmarker and Pose Landmarker (lite) by Google, Apache 2.0.

`src/scripts/prepare_wlasl.py` and `src/scripts/fetch_asl_citizen.py` rebuild the data indexes from the original sources.
