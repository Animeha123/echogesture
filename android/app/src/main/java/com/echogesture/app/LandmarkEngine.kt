package com.echogesture.app

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.hypot

/**
 * Port of src/echogesture/landmarks.py: MediaPipe hands (every frame) + pose
 * (every 2nd frame, on a parallel thread) -> one 53x3 landmark frame.
 * All inference is on-device; no frame leaves the phone.
 */
class LandmarkEngine(ctx: Context) : AutoCloseable {

    class Result(
        val frame: FloatArray,                  // 53*3, model coordinates (x,z scaled by aspect)
        val handsXY: List<FloatArray>,          // raw normalised (x,y)*21 per hand, for drawing
    )

    private val hand: HandLandmarker
    private val pose: PoseLandmarker
    private val poseExec = Executors.newSingleThreadExecutor { r -> Thread(r, "pose") }
    private var lastPose: FloatArray? = null
    private var frameI = 0
    private var lastTs = -1L

    init {
        hand = HandLandmarker.createFromOptions(ctx, HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task")
                .setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.35f)
            .setMinHandPresenceConfidence(0.35f)
            .setMinTrackingConfidence(0.35f)
            .build())
        pose = PoseLandmarker.createFromOptions(ctx, PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task")
                .setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.4f)
            .setMinTrackingConfidence(0.4f)
            .build())
    }

    /** bitmap: upright ARGB_8888 frame, long side <= PROC_LONG_SIDE. */
    fun process(bitmap: Bitmap, tsMs: Long): Result {
        val ts = if (tsMs <= lastTs) lastTs + 1 else tsMs   // VIDEO mode needs increasing timestamps
        lastTs = ts
        val aspect = bitmap.width.toFloat() / bitmap.height
        val img = BitmapImageBuilder(bitmap).build()

        val runPose = frameI % Cfg.POSE_EVERY == 0 || lastPose == null
        val fut: Future<PoseLandmarkerResult>? =
            if (runPose) poseExec.submit<PoseLandmarkerResult> { pose.detectForVideo(img, ts) } else null
        val hr = hand.detectForVideo(img, ts)

        val out = FloatArray(Cfg.FRAME) { Float.NaN }
        if (fut != null) {
            val pr = fut.get()
            val p = FloatArray(Cfg.N_POSE * 3) { Float.NaN }
            if (pr.landmarks().isNotEmpty()) {
                val pl = pr.landmarks()[0]
                for ((j, idx) in Cfg.POSE_IDX.withIndex()) {
                    val lm = pl[idx]
                    val vis = if (lm.visibility().isPresent) lm.visibility().get() else null
                    if (vis == null || vis > 0.3f) {
                        p[j * 3] = lm.x() * aspect; p[j * 3 + 1] = lm.y(); p[j * 3 + 2] = lm.z() * aspect
                    }
                }
            }
            lastPose = p
        }
        System.arraycopy(lastPose!!, 0, out, 0, Cfg.N_POSE * 3)
        frameI++

        val hands = ArrayList<Pair<FloatArray, String>>()
        val raw = ArrayList<FloatArray>()
        val lms = hr.landmarks()
        for (i in lms.indices) {
            val hl: List<NormalizedLandmark> = lms[i]
            val arr = FloatArray(Cfg.N_HAND * 3)
            val xy = FloatArray(Cfg.N_HAND * 2)
            for (k in 0 until Cfg.N_HAND) {
                arr[k * 3] = hl[k].x() * aspect; arr[k * 3 + 1] = hl[k].y(); arr[k * 3 + 2] = hl[k].z() * aspect
                xy[k * 2] = hl[k].x(); xy[k * 2 + 1] = hl[k].y()
            }
            val hd = hr.handedness()
            val label = if (i < hd.size && hd[i].isNotEmpty()) hd[i][0].categoryName() else "Left"
            hands += arr to label
            raw += xy
        }
        for ((arr, side) in assignHands(hands, out)) {
            val o = if (side == 'L') Cfg.LH0 else Cfg.RH0
            System.arraycopy(arr, 0, out, o * 3, Cfg.N_HAND * 3)
        }
        return Result(out, raw)
    }

    /** Signer's left/right hand: nearest pose wrist, else MediaPipe handedness (mirrored). */
    private fun assignHands(hands: List<Pair<FloatArray, String>>, pose: FloatArray): List<Pair<FloatArray, Char>> {
        if (hands.isEmpty()) return emptyList()
        val lwx = pose[Cfg.P_LWRI * 3]; val lwy = pose[Cfg.P_LWRI * 3 + 1]
        val rwx = pose[Cfg.P_RWRI * 3]; val rwy = pose[Cfg.P_RWRI * 3 + 1]
        fun d(h: FloatArray, x: Float, y: Float) = hypot(h[0] - x, h[1] - y)
        if (!(lwx.isNaN() || lwy.isNaN() || rwx.isNaN() || rwy.isNaN())) {
            if (hands.size == 1) {
                val h = hands[0].first
                return listOf(h to if (d(h, lwx, lwy) < d(h, rwx, rwy)) 'L' else 'R')
            }
            val a = hands[0].first; val b = hands[1].first
            val ab = d(a, lwx, lwy) + d(b, rwx, rwy)
            val ba = d(a, rwx, rwy) + d(b, lwx, lwy)
            return if (ab <= ba) listOf(a to 'L', b to 'R') else listOf(a to 'R', b to 'L')
        }
        val out = ArrayList<Pair<FloatArray, Char>>()
        val used = HashSet<Char>()
        for ((arr, label) in hands.take(2)) {
            var side = if (label == "Left") 'R' else 'L'
            if (side in used) side = if (side == 'R') 'L' else 'R'
            used += side
            out += arr to side
        }
        return out
    }

    override fun close() {
        hand.close(); pose.close(); poseExec.shutdown()
    }
}
