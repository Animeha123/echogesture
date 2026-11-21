# EchoGesture Android app: start here

**What it is:** an on-device sign-language recognizer (2,000 ASL words, WLASL vocabulary).

`Camera → MediaPipe hand + pose landmarks → feature vector → INT8 ONNX classifier → word + top-5 → sentence`

Nothing goes to a server. The full app is written but has **never been built or run on a phone yet**, so your first job is to get it running.

## 1. First run (≈20 min)

1. Unzip and choose **File → Open** in Android Studio, selecting the `android` folder (the one containing `settings.gradle.kts`).
2. Let the Gradle sync finish. If it asks about the Gradle wrapper or JDK, accept the defaults. It needs **JDK 17**: Settings → Build Tools → Gradle → Gradle JDK = 17 or the embedded JBR.
3. **Run the correctness test first:** open `app/src/test/java/com/echogesture/app/FeaturesTest.kt`, right-click it and choose **Run 'FeaturesTest'**. It checks that the Kotlin feature code produces exactly the same numbers as our Python training code. **It must pass.** If it fails, send the output to the team; don't "fix" the numbers.
4. Plug in a phone with **Developer options → USB debugging** turned on, then press **Run ▶**. Allow the camera permission.
5. Report back:
   - Does it build? Paste any red errors.
   - Does the skeleton overlay track your hands?
   - The stats numbers (p95 latency, FPS).
   - The phone model.

## 2. How the code is organised (`app/src/main/java/com/echogesture/app/`)

| File | What it does | Can you change it? |
|---|---|---|
| `MainActivity.kt` | Camera setup, UI wiring, sentence building, buttons, stats ticker | **Yes**, the UI parts |
| `OverlayView.kt` | Draws the hand skeleton over the camera | **Yes** |
| `res/layout/activity_main.xml`, `res/values/*` | Layout, colours, theme, strings | **Yes**, freely |
| `Segmenter.kt` | Decides when a sign starts and ends in Auto mode (hands raised = signing) | Tuning thresholds is OK; tell us |
| `Stats.kt` | Latency p50/p95, FPS, power, thermal drift | Display yes, maths no |
| `Classifier.kt` | Runs the ONNX model, softmax, top-5 | **No** |
| `Features.kt` | Exact port of the Python feature code (`Cfg` constants, normalisation, resampling to 48 frames × 352 values) | **No**: the model breaks silently if this changes |
| `LandmarkEngine.kt` | MediaPipe hands every frame, pose every 2nd frame, left/right hand assignment | **No** (except performance settings, after checking with us) |
| `assets/` | `echogesture_int8.onnx` (model), `labels.txt` (2,000 words), MediaPipe `.task` files | Only replaced by us when the model improves |

**The rule:** anything that changes the numbers going into the model (`Features.kt`, `LandmarkEngine.kt`, frame rotation or scaling in `MainActivity.analyze`) must still pass `FeaturesTest` and needs a heads-up. Everything visual is yours.

## 3. What the app already does

- Live camera view with the hand skeleton overlay. The front camera is mirrored.
- **Auto mode:** raise your hands to sign and lower them to finish. Alternatively, use **Hold to sign**.
- The word with its confidence, plus a **top-5 row**. Tap an alternative to replace the word just added.
- Builds a **sentence**, with **Undo** and **Clear**.
- **Stats overlay:** latency p50/p95, FPS, caption latency, classifier time, power (only when unplugged), battery temperature and thermal drift.
- **Pause ML** and **Reset stats** for measurements, plus **Flip camera**.

## 4. What would make it a winning demo (pick in this order)

1. **Get it running and stable** on the phone (section 1). This matters more than everything below.
2. **Text-to-speech:** a "Speak" button that reads the sentence aloud (`android.speech.tts.TextToSpeech`, a few lines). It's very effective in a demo.
3. **Polish the UI:** an app icon, a clean splash or onboarding screen explaining "raise hands → sign → lower hands", larger readable word text, and a nicer top-5 chip style. Keep the stats overlay but let it be toggled (tap to hide).
4. **Signing indicator:** clearly show when the app thinks you're signing (for example a red recording dot or a border glow). `Segmenter` already knows this.
5. **Word history or list:** a searchable list of the supported words from `labels.txt`, so judges can pick a word to try.
6. Optional: share or copy the sentence, landscape layout, dark/light theme.

## 5. Measurements we need from the phone (for the submission)

On a real phone, unplugged:
1. Tap **Reset stats**, then sign or point the camera at someone for **10+ minutes**.
2. Note **p95 latency**, **FPS**, **power with ML on**, and **thermal drift %**.
3. Tap **Pause ML** for 1–2 minutes and note **power with ML off**.
4. Send a screenshot of the stats overlay together with the phone model.

Targets: p95 ≤ 50 ms per frame, power ≤ 500 mW for the model's share, drift ≤ 20%.

## 6. Working together

- **Model updates:** we'll send a new `echogesture_int8.onnx` (and maybe `labels.txt`). Drop them into `app/src/main/assets/`, replacing the old ones. No code changes needed.
- **Sending changes back:** zip the `android` folder **without** the `build/`, `app/build/` and `.gradle/` folders, or tell us which files you changed.
- The demo video is recorded with this app, so a smooth, clear UI matters.

Tech: Kotlin, CameraX 1.3.4, MediaPipe tasks-vision 0.10.20, ONNX Runtime Android 1.19.2, AGP 8.5.2, Gradle 8.7, minSdk 26, compileSdk 34.
