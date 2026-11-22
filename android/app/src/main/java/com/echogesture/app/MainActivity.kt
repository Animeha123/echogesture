package com.echogesture.app

import android.Manifest
import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.util.Size
import android.view.HapticFeedbackConstants
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.widget.ArrayAdapter
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.widget.doAfterTextChanged
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "EchoGesture"
        private const val AUTO_ADD_CONF = 0.35f   // auto-append top-1 to the sentence above this
        private const val SURE_CONF = 0.60f       // confidence pill turns green above this
        private const val PREFS = "echogesture"
        private const val KEY_INTRO = "intro_seen"
    }

    // ---- views
    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var camCard: MaterialCardView
    private lateinit var statusView: TextView
    private lateinit var statusDot: View
    private lateinit var statsPanel: View
    private lateinit var statsView: TextView
    private lateinit var wordView: TextView
    private lateinit var confView: TextView
    private lateinit var top5View: LinearLayout
    private lateinit var sentenceView: TextView
    private lateinit var modeGroup: MaterialButtonToggleGroup
    private lateinit var modeAuto: MaterialButton
    private lateinit var modeHold: MaterialButton
    private lateinit var holdBtn: MaterialButton
    private lateinit var pauseBtn: MaterialButton
    private lateinit var onboarding: View

    // ---- colours
    private var cAccent = 0; private var cAccentDim = 0; private var cRec = 0; private var cAmber = 0
    private var cText = 0; private var cMuted = 0; private var cOutline = 0; private var cSurface2 = 0

    // ---- ML pipeline (unchanged behaviour)
    private val analysisExec = Executors.newSingleThreadExecutor { r -> Thread(r, "analysis") }
    private var engine: LandmarkEngine? = null
    private var classifier: Classifier? = null
    private val segmenter = Segmenter()
    private val stats = Stats()
    private lateinit var power: PowerMeter
    private val ui = Handler(Looper.getMainLooper())

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    @Volatile private var mlOn = true
    @Volatile private var autoMode = true
    @Volatile private var holding = false
    private val manualBuf = ArrayList<FloatArray>()
    private var lastRec = false   // analysis thread only: last recording state sent to the UI

    // classifier timing (analysis thread writes, UI reads)
    @Volatile private var lastClfMs = Double.NaN
    @Volatile private var lastFeatMs = Double.NaN
    @Volatile private var lastCaptionMs = Double.NaN

    // sentence state (UI thread only)
    private val words = ArrayList<String>()
    private var segCounter = 0
    private var lastAddedSeg = -1
    private var modelsReady = false

    // ---- text to speech
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var pulse: ValueAnimator? = null

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else setStatus("Camera permission is needed", cRec)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        cAccent = color(R.color.accent); cAccentDim = color(R.color.accent_dim); cRec = color(R.color.rec)
        cAmber = color(R.color.amber); cText = color(R.color.text); cMuted = color(R.color.muted)
        cOutline = color(R.color.outline); cSurface2 = color(R.color.surface2)

        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        camCard = findViewById(R.id.camCard)
        statusView = findViewById(R.id.status)
        statusDot = findViewById(R.id.statusDot)
        statsPanel = findViewById(R.id.statsPanel)
        statsView = findViewById(R.id.stats)
        wordView = findViewById(R.id.word)
        confView = findViewById(R.id.conf)
        top5View = findViewById(R.id.top5)
        sentenceView = findViewById(R.id.sentence)
        modeGroup = findViewById(R.id.modeGroup)
        modeAuto = findViewById(R.id.modeAuto)
        modeHold = findViewById(R.id.modeHold)
        holdBtn = findViewById(R.id.hold)
        pauseBtn = findViewById(R.id.pause)
        onboarding = findViewById(R.id.onboarding)

        // FILL_CENTER fills the rounded card; COMPATIBLE (TextureView) lets the card clip its corners.
        // OverlayView uses the same fill mapping.
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        sentenceView.movementMethod = ScrollingMovementMethod()
        power = PowerMeter(this)

        // ---- mode: Auto (hands up = signing) or Hold (press the button while signing)
        styleModeButtons()
        modeGroup.addOnButtonCheckedListener { _, id, checked ->
            if (!checked) return@addOnButtonCheckedListener
            autoMode = id == R.id.modeAuto
            holding = false
            holdBtn.visibility = if (autoMode) View.GONE else View.VISIBLE
            analysisExec.execute { segmenter.reset() }
            styleModeButtons()
            setRecording(false)
        }
        holdBtn.setOnTouchListener { v, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> {
                    holding = true
                    v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                    setRecording(true)
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> holding = false
            }
            false
        }

        // ---- sentence actions
        findViewById<ImageButton>(R.id.undo).setOnClickListener {
            if (words.isNotEmpty()) words.removeAt(words.size - 1)
            lastAddedSeg = -1; renderSentence()
        }
        findViewById<ImageButton>(R.id.clear).setOnClickListener {
            words.clear(); lastAddedSeg = -1; renderSentence()
            top5View.removeAllViews(); showIdleWord()
        }
        findViewById<ImageButton>(R.id.copy).setOnClickListener {
            val text = sentenceText()
            if (text.isEmpty()) { toast("Nothing to copy yet"); return@setOnClickListener }
            val cm = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("EchoGesture", text))
            toast("Copied")
        }
        findViewById<ImageButton>(R.id.speak).setOnClickListener { speak(sentenceText()) }

        // ---- camera-card buttons
        findViewById<ImageButton>(R.id.flip).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            analysisExec.execute { segmenter.reset() }
            overlay.update(emptyList(), 1, 1, false, false)
            startCamera()
        }
        findViewById<ImageButton>(R.id.statsBtn).setOnClickListener {
            statsPanel.visibility = if (statsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        findViewById<ImageButton>(R.id.helpBtn).setOnClickListener { onboarding.visibility = View.VISIBLE }
        findViewById<MaterialButton>(R.id.introDone).setOnClickListener {
            onboarding.visibility = View.GONE
            getSharedPreferences(PREFS, MODE_PRIVATE).edit().putBoolean(KEY_INTRO, true).apply()
        }
        findViewById<MaterialButton>(R.id.wordsBtn).setOnClickListener { showWordList() }

        // ---- measurement tools (inside the stats panel)
        pauseBtn.setOnClickListener {
            mlOn = !mlOn
            pauseBtn.text = if (mlOn) "Pause ML" else "Resume ML"
            if (!mlOn) {
                overlay.update(emptyList(), 1, 1, false, false)
                setStatus("Recognition paused", cAmber)
            } else setRecording(false)
        }
        findViewById<MaterialButton>(R.id.resetStats).setOnClickListener {
            stats.reset(); power.reset()
            toast("Stats reset")
        }

        if (!getSharedPreferences(PREFS, MODE_PRIVATE).getBoolean(KEY_INTRO, false))
            onboarding.visibility = View.VISIBLE
        showIdleWord()
        setStatus("Loading models…", cMuted)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.setLanguage(Locale.US)
                tts?.setSpeechRate(0.95f)
                ttsReady = true
            }
        }

        // load models off the UI thread (~1 s)
        analysisExec.execute {
            try {
                engine = LandmarkEngine(this)
                classifier = Classifier(this)
                runOnUiThread {
                    modelsReady = true
                    setRecording(false)
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) startCamera()
                    else askCamera.launch(Manifest.permission.CAMERA)
                }
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread { setStatus("Model load failed: ${e.message}", cRec) }
            }
        }
        ui.post(statsTicker)
    }

    // ================= stats panel =================

    private val statsTicker = object : Runnable {
        override fun run() {
            power.tick(mlOn)
            val drift = stats.thermalDriftPct()
            val on = power.avg(true); val off = power.avg(false)
            statsView.text = buildString {
                append("frame  p50 %5.1f  p95 %5.1f ms\n".format(stats.p50(), stats.p95()))
                append("FPS %4.1f   caption %5.1f ms\n".format(stats.fps(), lastCaptionMs))
                append("features %4.1f  classifier %4.1f ms\n".format(lastFeatMs, lastClfMs))
                if (power.pluggedIn()) append("power: unplug to measure\n")
                else append("power ML on %s | off %s mW\n".format(on?.let { "%.0f".format(it) } ?: "-",
                    off?.let { "%.0f".format(it) } ?: "-"))
                append("battery %s C | drift %s | %.1f min".format(
                    power.tempC()?.let { "%.1f".format(it) } ?: "-",
                    drift?.let { "%+.1f%%".format(it) } ?: "(2 min)", stats.minutes()))
            }
            ui.postDelayed(this, 1000)
        }
    }

    // ================= camera + analysis (pipeline unchanged) =================

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector43 = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(640, 480),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(selector43).build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector43)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExec, ::analyze)
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this,
                    CameraSelector.Builder().requireLensFacing(lensFacing).build(), preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera bind failed", e)
                setStatus("Camera error: ${e.message}", cRec)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Analysis thread: frame -> landmarks -> segment -> features -> classifier. */
    private fun analyze(img: ImageProxy) {
        try {
            val eng = engine ?: return
            val clf = classifier ?: return
            if (!mlOn) return
            val t0 = SystemClock.elapsedRealtimeNanos()
            val src = img.toBitmap()
            val rot = img.imageInfo.rotationDegrees
            val rw = if (rot % 180 == 0) src.width else src.height
            val rh = if (rot % 180 == 0) src.height else src.width
            val s = min(1f, Cfg.PROC_LONG_SIDE.toFloat() / max(rw, rh))
            val m = Matrix().apply { postRotate(rot.toFloat()); postScale(s, s) }
            val frame: Bitmap = Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
            val nowMs = SystemClock.uptimeMillis()
            val res = eng.process(frame, nowMs)
            val frameMs = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            stats.addFrame(frameMs, nowMs)

            // segmentation: hold-to-sign has priority over auto mode
            var seg: List<FloatArray>? = null
            if (holding) {
                manualBuf.add(res.frame)
            } else if (manualBuf.isNotEmpty()) {
                if (manualBuf.size >= 5) seg = ArrayList(manualBuf)
                manualBuf.clear(); segmenter.reset()
            } else if (autoMode) {
                seg = segmenter.push(res.frame, nowMs)
            }
            val rec = holding || (autoMode && segmenter.recording)
            overlay.update(res.handsXY, frame.width, frame.height,
                lensFacing == CameraSelector.LENS_FACING_FRONT, rec)

            // UI state change first, then the prediction (both post to the main thread in order)
            if (rec != lastRec) {
                lastRec = rec
                runOnUiThread { setRecording(rec) }
            }
            if (seg != null) {
                val f0 = SystemClock.elapsedRealtimeNanos()
                val feats = Features.compute(Features.capFrames(seg))
                lastFeatMs = (SystemClock.elapsedRealtimeNanos() - f0) / 1e6
                val pred = clf.classify(feats)
                lastClfMs = pred.ms
                // caption latency: last frame's processing + features + classifier
                lastCaptionMs = frameMs + lastFeatMs + pred.ms
                runOnUiThread { showPrediction(pred) }
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            img.close()
        }
    }

    // ================= UI state =================

    private fun setStatus(text: String, dotColor: Int) {
        statusView.text = text
        statusDot.backgroundTintList = ColorStateList.valueOf(dotColor)
    }

    /** Signing indicator: coral card border + pulsing dot while a sign is being recorded. */
    private fun setRecording(on: Boolean) {
        if (!modelsReady || !mlOn) return
        camCard.strokeColor = if (on) cRec else cOutline
        camCard.strokeWidth = dp(if (on) 3f else 1f).toInt()
        pulse?.cancel(); statusDot.alpha = 1f
        if (on) {
            setStatus(if (autoMode) "Listening… lower your hands to finish" else "Listening… release when done", cRec)
            pulse = ValueAnimator.ofFloat(1f, 0.25f).apply {
                duration = 600; repeatMode = ValueAnimator.REVERSE; repeatCount = ValueAnimator.INFINITE
                addUpdateListener { statusDot.alpha = it.animatedValue as Float }
                start()
            }
        } else {
            setStatus(if (autoMode) "Raise your hands to sign" else "Press and hold the button to sign", cAccent)
        }
    }

    private fun showIdleWord() {
        wordView.text = "Ready when you are"
        wordView.setTextColor(cMuted)
        wordView.textSize = 22f
        confView.visibility = View.INVISIBLE
    }

    private fun showPrediction(p: Classifier.Prediction) {
        val seg = ++segCounter
        val conf = p.probs[0]
        val sure = conf >= AUTO_ADD_CONF

        // the word: large, animated in
        wordView.text = pretty(p.top[0])
        wordView.setTextColor(cText)
        wordView.textSize = 40f
        wordView.alpha = 0f; wordView.scaleX = 0.92f; wordView.scaleY = 0.92f
        wordView.animate().alpha(1f).scaleX(1f).scaleY(1f).setDuration(180).start()
        wordView.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)

        // confidence pill: green = sure, amber = likely, grey = unsure
        val (fg, bg) = when {
            conf >= SURE_CONF -> cAccent to cAccentDim
            sure -> cAmber to 0x33F5B841
            else -> cMuted to cSurface2
        }
        confView.text = "%.0f%%".format(conf * 100)
        confView.setTextColor(fg)
        confView.backgroundTintList = ColorStateList.valueOf(bg)
        confView.visibility = View.VISIBLE

        // suggestions: tapping one corrects the word added for this sign
        top5View.removeAllViews()
        for ((i, w) in p.top.withIndex()) {
            val chip = Chip(this).apply {
                text = pretty(w)
                isCheckable = false
                setTextColor(if (i == 0) cAccent else cText)
                chipBackgroundColor = ColorStateList.valueOf(if (i == 0) cAccentDim else cSurface2)
                chipStrokeColor = ColorStateList.valueOf(if (i == 0) cAccentDim else cOutline)
                chipStrokeWidth = dp(1f)
                textSize = 15f
                setOnClickListener {
                    if (lastAddedSeg == seg && words.isNotEmpty()) words[words.size - 1] = w
                    else words.add(w)
                    lastAddedSeg = seg
                    wordView.text = pretty(w)
                    renderSentence()
                    it.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                }
            }
            val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(8f).toInt() }
            top5View.addView(chip, lp)
        }
        (top5View.parent as? View)?.scrollTo(0, 0)

        if (sure) {
            words.add(p.top[0]); lastAddedSeg = seg
            setStatus("Added. Tap a suggestion to fix it", cAccent)
        } else {
            setStatus("Not sure. Tap the right word below", cAmber)
        }
        renderSentence()
    }

    private fun renderSentence() {
        sentenceView.text = sentenceText()
    }

    private fun sentenceText(): String {
        val s = words.joinToString(" ") { it.lowercase(Locale.US) }
        return s.replaceFirstChar { it.titlecase(Locale.US) }
    }

    /** Labels are upper-case ("THANK YOU"); show them as "Thank you". */
    private fun pretty(label: String): String =
        label.lowercase(Locale.US).replaceFirstChar { it.titlecase(Locale.US) }

    private fun speak(text: String) {
        if (text.isEmpty()) { toast("Sign a few words first"); return }
        val t = tts
        if (t == null || !ttsReady) { toast("Speech is not available on this phone"); return }
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "sentence")
    }

    private fun showWordList() {
        val labels = classifier?.labels
        if (labels == null) { toast("Still loading"); return }
        val sheet = BottomSheetDialog(this)
        val v = LayoutInflater.from(this).inflate(R.layout.sheet_words, null)
        val items = labels.map { pretty(it) }.sorted()
        val adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, items)
        v.findViewById<TextView>(R.id.wordsTitle).text = "%,d words EchoGesture knows".format(items.size)
        v.findViewById<ListView>(R.id.wordsList).adapter = adapter
        v.findViewById<EditText>(R.id.wordsSearch).doAfterTextChanged { adapter.filter.filter(it) }
        sheet.setContentView(v)
        sheet.show()
    }

    // ================= helpers =================

    private fun styleModeButtons() {
        for (b in listOf(modeAuto, modeHold)) {
            val on = b.id == modeGroup.checkedButtonId
            b.backgroundTintList = ColorStateList.valueOf(if (on) cAccentDim else 0x00000000)
            b.setTextColor(if (on) cAccent else cMuted)
            b.strokeColor = ColorStateList.valueOf(cOutline)
        }
    }

    private fun color(id: Int) = ContextCompat.getColor(this, id)
    private fun dp(v: Float) = v * resources.displayMetrics.density
    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacks(statsTicker)
        pulse?.cancel()
        tts?.shutdown()
        analysisExec.execute { engine?.close(); classifier?.close() }
        analysisExec.shutdown()
    }
}
