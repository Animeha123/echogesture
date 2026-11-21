# EchoGesture: Android app

This app does on-device sign recognition: CameraX → MediaPipe hands + pose → features (a Kotlin port of `features.py`) → INT8 ONNX classifier. No frame leaves the phone.

## Build

1. Install **Android Studio** (it includes JDK 17 and the Android SDK).
2. **File → Open** this `android/` folder and wait for the Gradle sync to finish. If it asks, accept "Use Gradle wrapper" or "Download Gradle".
3. **Check the feature port:** open `app/src/test/java/com/echogesture/app/FeaturesTest.kt`, right-click it and choose **Run 'FeaturesTest'**. It must pass; it compares Kotlin against the Python reference outputs.
4. Phone setup: enable Developer options and **USB debugging**, plug the phone in, then press **Run ▶**.

## Using the app

- **Auto mode:** raise your hands into the signing space and sign. Lowering your hands ends the sign, and the word appears.
- **Hold to sign:** hold the button while signing and release it to classify. This is more reliable for demos.
- Top-1 words above 35% confidence are added to the sentence automatically. **Tap an alternative** in the top-5 row to correct the word added for that sign. **Undo** and **Clear** edit the sentence.
- **Stats overlay:** per-frame latency p50/p95, FPS, caption latency (last frame + features + classifier), and classifier time.
  - Power: shows the whole phone's power **when unplugged**, split by ML on vs ML off (**Pause ML**).
  - Battery temperature, and thermal drift: the median latency of the last minute vs the first, available after 2 min.

### Measuring for results.csv

1. Unplug the phone, then tap **Reset stats**.
2. Sign or point the camera at a person for **10+ minutes**.
3. Read off: p95 latency, power with ML on, and drift %.
4. **Pause ML** for 1–2 min to get the power with ML off. The model's share of the power is roughly ML on minus ML off.

## Updating the model

Copy these files from `src/models/` into `app/src/main/assets/`:
- `echogesture_int8.onnx`
- `labels.txt`
- `hand_landmarker.task`
- `pose_landmarker_lite.task`

After a feature change, re-run `python src/scripts/export_android_golden.py` and then `FeaturesTest`.

## Known differences from the Python pipeline

- No CLAHE low-light enhancement on the phone, which lowers recall slightly in dim light.
- Frames are resized bilinearly rather than with INTER_AREA.
- Timestamps are real camera times, not a fixed 33 ms step.
