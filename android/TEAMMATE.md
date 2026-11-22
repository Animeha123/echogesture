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
| `MainActivity.kt` | Camera setup, UI wiring, sentence building, speech, word list, stats ticker | UI parts only; `analyze()` is pipeline code, don't change it |
| `OverlayView.kt` | Draws the hand skeleton over the camera | **Yes** |
| `res/layout/*.xml`, `res/values/*`, `res/drawable/*` | Layout, word-list sheet, colours, theme, icons | **Yes** |
| `Segmenter.kt` | Decides when a sign starts and ends in Auto mode (hands raised = signing) | Tuning thresholds is OK; tell us |
| `Stats.kt` | Latency p50/p95, FPS, power, thermal drift | Display yes, maths no |
| `Classifier.kt` | Runs the ONNX model, softmax, top-5 | **No** |
| `Features.kt` | Exact port of the Python feature code (`Cfg` constants, normalisation, resampling to 48 frames × 352 values) | **No**: the model breaks silently if this changes |
| `LandmarkEngine.kt` | MediaPipe hands every frame, pose every 2nd frame, left/right hand assignment | **No** (except performance settings, after checking with us) |
| `assets/` | `echogesture_int8.onnx` (model), `labels.txt` (2,000 words), MediaPipe `.task` files | Only replaced by us when the model improves |

**The rule:** anything that changes the numbers going into the model (`Features.kt`, `LandmarkEngine.kt`, frame rotation or scaling in `MainActivity.analyze`) must still pass `FeaturesTest` and needs a heads-up. Everything visual is yours.

## 3. What the app does (UI finished on 21 Nov, needs testing only)

Dark, minimal design: a rounded camera card on top, the recognised word below it, then the sentence.

- **First launch** shows a 3-step guide ("prop the phone up, raise hands and sign, tap to fix and speak"). The **?** icon reopens it.
- **Status chip** (top-left of the camera): a green dot means ready. While a sign is being recorded the dot pulses coral, the camera border turns coral and the hand skeleton turns coral.
- **Recognised word** is large and animates in with a light vibration. A confidence pill is green (60% or more), amber (35–60%, still added) or grey (below 35%, not added: tap the right word).
- **Suggestions:** five chips (top-5). Tapping one replaces the word just added.
- **Sentence card** with **Undo**, **Clear**, **Copy** and a large **Speak** button (phone text-to-speech, US English).
- **Auto / Hold** toggle. Auto: raise hands to sign, lower to finish. Hold: a large "Press and hold while signing" button appears.
- **Words** button: a searchable bottom sheet listing all 2,000 words, so judges can pick one to try.
- **Stats icon** (top-right) shows the performance panel with **Pause ML** and **Reset stats**, used for the measurements in section 5. Hidden by default so the demo stays clean.
- **Flip camera** icon (top-right). App icon: a mint hand on a dark background.

## 4. Testing checklist (tomorrow)

1. Build and run. Paste any red errors to the team; the UI code was written without a compiler, so a small fix may be needed.
2. Run `FeaturesTest` (section 1). It must pass.
3. First launch: guide appears; "Start signing" closes it; it does not come back on restart; **?** reopens it.
4. Skeleton lines up with your hands for both the front and back camera (the preview now fills the card).
5. Auto mode: raise hands, the border turns coral; lower them, a word appears with the pill and chips.
6. Tap a chip: the last word in the sentence changes. Undo, Clear, Copy work.
7. Speak reads the sentence aloud (turn the volume up).
8. Hold mode: the big button appears; holding it records; releasing gives a word.
9. Words sheet opens, search filters ("thank").
10. Stats icon shows the panel; then do section 5.

Small visual tweaks (colours in `res/values/colors.xml`, sizes in `activity_main.xml`) are fine. Don't restructure the screen the night before the deadline.

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
