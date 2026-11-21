package com.echogesture.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.MotionEvent
import android.widget.LinearLayout
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
import com.google.android.material.button.MaterialButton
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "EchoGesture"
        private const val AUTO_ADD_CONF = 0.35f   // auto-append top-1 to the sentence above this
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var statsView: TextView
    private lateinit var statusView: TextView
    private lateinit var wordView: TextView
    private lateinit var confView: TextView
    private lateinit var top5View: LinearLayout
    private lateinit var sentenceView: TextView
    private lateinit var autoBtn: MaterialButton
    private lateinit var pauseBtn: MaterialButton

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

    // classifier timing (analysis thread writes, UI reads)
    @Volatile private var lastClfMs = Double.NaN
    @Volatile private var lastFeatMs = Double.NaN
    @Volatile private var lastCaptionMs = Double.NaN

    // sentence state (UI thread only)
    private val words = ArrayList<String>()
    private var segCounter = 0
    private var lastAddedSeg = -1

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else statusView.text = "Camera permission is needed"
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        statsView = findViewById(R.id.stats)
        statusView = findViewById(R.id.status)
        wordView = findViewById(R.id.word)
        confView = findViewById(R.id.conf)
        top5View = findViewById(R.id.top5)
        sentenceView = findViewById(R.id.sentence)
        autoBtn = findViewById(R.id.auto)
        pauseBtn = findViewById(R.id.pause)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        power = PowerMeter(this)

        findViewById<MaterialButton>(R.id.hold).setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> { holding = true; statusView.text = "Signing... release when done" }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> holding = false
            }
            false
        }
        autoBtn.setOnClickListener {
            autoMode = !autoMode
            autoBtn.text = if (autoMode) "Auto: on" else "Auto: off"
            analysisExec.execute { segmenter.reset() }
            statusView.text = if (autoMode) "Raise your hands and sign" else "Hold the button while signing"
        }
        findViewById<MaterialButton>(R.id.undo).setOnClickListener {
            if (words.isNotEmpty()) words.removeAt(words.size - 1)
            lastAddedSeg = -1; renderSentence()
        }
        findViewById<MaterialButton>(R.id.clear).setOnClickListener {
            words.clear(); lastAddedSeg = -1; renderSentence()
        }
        findViewById<MaterialButton>(R.id.flip).setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT)
                CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT
            analysisExec.execute { segmenter.reset() }
            startCamera()
        }
        pauseBtn.setOnClickListener {
            mlOn = !mlOn
            pauseBtn.text = if (mlOn) "Pause ML" else "Resume ML"
            if (!mlOn) overlay.update(emptyList(), 1, 1, false, false)
        }
        findViewById<MaterialButton>(R.id.resetStats).setOnClickListener {
            stats.reset(); power.reset()
            Toast.makeText(this, "Stats reset", Toast.LENGTH_SHORT).show()
        }

        // load models off the UI thread (~1 s)
        analysisExec.execute {
            try {
                engine = LandmarkEngine(this)
                classifier = Classifier(this)
                runOnUiThread {
                    statusView.text = "Raise your hands and sign (${classifier!!.labels.size} words)"
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        == PackageManager.PERMISSION_GRANTED) startCamera()
                    else askCamera.launch(Manifest.permission.CAMERA)
                }
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread { statusView.text = "Model load failed: ${e.message}" }
            }
        }
        ui.post(statsTicker)
    }

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
                statusView.text = "Camera error: ${e.message}"
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

            if (seg != null) {
                val f0 = SystemClock.elapsedRealtimeNanos()
                val feats = Features.compute(Features.capFrames(seg))
                lastFeatMs = (SystemClock.elapsedRealtimeNanos() - f0) / 1e6
                val pred = clf.classify(feats)
                lastClfMs = pred.ms
                // caption latency: last frame's processing + features + classifier
                lastCaptionMs = frameMs + lastFeatMs + pred.ms
                runOnUiThread { showPrediction(pred) }
            } else if (rec) {
                runOnUiThread { if (!holding) statusView.text = "Signing..." }
            } else if (autoMode) {
                runOnUiThread { statusView.text = "Raise your hands and sign" }
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            img.close()
        }
    }

    private fun showPrediction(p: Classifier.Prediction) {
        val seg = ++segCounter
        wordView.text = p.top[0]
        confView.text = "%.0f%%".format(p.probs[0] * 100)
        top5View.removeAllViews()
        for ((i, w) in p.top.withIndex()) {
            val b = MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle)
            b.text = "%s %.0f%%".format(w, p.probs[i] * 100)
            b.isAllCaps = false
            b.setOnClickListener {
                // tapping an alternative corrects the word added for this sign
                if (lastAddedSeg == seg && words.isNotEmpty()) words[words.size - 1] = w
                else words.add(w)
                lastAddedSeg = seg
                renderSentence()
            }
            val lp = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginEnd = 8 }
            top5View.addView(b, lp)
        }
        if (p.probs[0] >= AUTO_ADD_CONF) {
            words.add(p.top[0]); lastAddedSeg = seg
            statusView.text = "Added ${p.top[0]} - tap an alternative to correct"
        } else {
            statusView.text = "Not sure - tap the right word"
        }
        renderSentence()
    }

    private fun renderSentence() {
        sentenceView.text = words.joinToString(" ")
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacks(statsTicker)
        analysisExec.execute { engine?.close(); classifier?.close() }
        analysisExec.shutdown()
    }
}
