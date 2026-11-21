package com.echogesture.app

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

/** Mirrors src/echogesture/config.py - must stay identical to training. */
object Cfg {
    val POSE_IDX = intArrayOf(0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16)
    const val N_POSE = 11
    const val N_HAND = 21
    const val LH0 = N_POSE
    const val RH0 = N_POSE + N_HAND
    const val N_POINTS = N_POSE + 2 * N_HAND      // 53
    const val FRAME = N_POINTS * 3                 // floats per frame
    const val P_LSHO = 5
    const val P_RSHO = 6
    const val P_LWRI = 9
    const val P_RWRI = 10
    const val SEQ_LEN = 48
    const val FEAT_DIM = 352
    const val MAX_VIDEO_FRAMES = 160
    const val PROC_LONG_SIDE = 384
    const val POSE_EVERY = 2
}

/**
 * Exact port of src/echogesture/features.py (compute_features).
 * A frame is FloatArray(53*3) laid out [point][x,y,z], NaN where missing.
 * Verified against Python golden vectors by FeaturesTest.
 */
object Features {

    fun handPresent(f: FloatArray): Boolean =
        !f[Cfg.LH0 * 3].isNaN() || !f[Cfg.RH0 * 3].isNaN()

    /** Uniform sub-sampling to at most MAX_VIDEO_FRAMES (as iter_video_frames). */
    fun capFrames(seq: List<FloatArray>, maxFrames: Int = Cfg.MAX_VIDEO_FRAMES): List<FloatArray> {
        if (seq.size <= maxFrames) return seq
        val n = seq.size
        return List(maxFrames) { i -> seq[Math.rint(i * ((n - 1).toDouble() / (maxFrames - 1))).toInt()] }
    }

    fun trimIdle(seq: List<FloatArray>, margin: Int = 2): List<FloatArray> {
        val pres = seq.indices.filter { handPresent(seq[it]) }
        if (pres.size < 2) return seq
        val s = max(pres.first() - margin, 0)
        val e = min(pres.last() + margin + 1, seq.size)
        return seq.subList(s, e)
    }

    fun fillHandGaps(seq0: List<FloatArray>, maxGap: Int = 10): List<FloatArray> {
        val seq = seq0.map { it.copyOf() }
        for (o in intArrayOf(Cfg.LH0, Cfg.RH0)) {
            val pres = seq0.indices.filter { !seq0[it][o * 3].isNaN() }
            if (pres.size < 2) continue
            for (k in 0 until pres.size - 1) {
                val a = pres[k]
                val b = pres[k + 1]
                val gap = b - a - 1
                if (gap in 1..maxGap) {
                    for (g in 1..gap) {
                        val w = (g.toDouble() / (gap + 1)).toFloat()
                        for (j in o * 3 until (o + Cfg.N_HAND) * 3) {
                            seq[a + g][j] = (1 - w) * seq0[a][j] + w * seq0[b][j]
                        }
                    }
                }
            }
        }
        return seq
    }

    /** Linear temporal resampling; NaN results fall back to the nearest frame. */
    fun resample(seq: List<FloatArray>, T: Int = Cfg.SEQ_LEN): Array<FloatArray> {
        val n = seq.size
        if (n == 1) return Array(T) { seq[0].copyOf() }
        val step = (n - 1).toDouble() / (T - 1)            // as np.linspace
        return Array(T) { t ->
            val pos = if (t == T - 1) (n - 1).toDouble() else t * step
            val i0 = floor(pos).toInt()
            val i1 = min(i0 + 1, n - 1)
            val w = (pos - i0).toFloat()
            val near = seq[Math.rint(pos).toInt()]             // np.round = half-to-even
            val a = seq[i0]
            val b = seq[i1]
            FloatArray(a.size) { j ->
                val v = (1 - w) * a[j] + w * b[j]
                if (v.isNaN()) near[j] else v
            }
        }
    }

    private fun median(xs: List<Double>): Double {
        val s = xs.sorted()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2.0
    }

    class Body(val cx: Float, val cy: Float, val scale: Float, val ang: Float)

    fun bodyFrame(seq: Array<FloatArray>): Body {
        val mx = ArrayList<Double>(); val my = ArrayList<Double>()
        val norms = ArrayList<Double>(); val angs = ArrayList<Double>()
        val ls = Cfg.P_LSHO * 3; val rs = Cfg.P_RSHO * 3
        for (f in seq) {
            val lx = f[ls]; val ly = f[ls + 1]; val rx = f[rs]; val ry = f[rs + 1]
            if (lx.isNaN() || ly.isNaN() || rx.isNaN() || ry.isNaN()) continue
            mx += ((lx + rx) / 2f).toDouble(); my += ((ly + ry) / 2f).toDouble()
            val vx = (lx - rx).toDouble(); val vy = (ly - ry).toDouble()
            norms += sqrt(vx * vx + vy * vy)
            angs += atan2(vy, vx)
        }
        var cx: Double; var cy: Double; var scale: Double; var ang: Double
        if (mx.isNotEmpty()) {
            cx = median(mx); cy = median(my); scale = median(norms); ang = median(angs)
            // a signer facing the camera has their left shoulder on image-right
            if (abs(ang) > PI / 2) ang -= sign(ang) * PI
        } else {
            var sx = 0.0; var sy = 0.0; var c = 0
            for (f in seq) for (p in Cfg.LH0 until Cfg.N_POINTS) {
                val x = f[p * 3]; val y = f[p * 3 + 1]
                if (!x.isNaN() && !y.isNaN()) { sx += x; sy += y; c++ }
            }
            if (c > 0) { cx = sx / c; cy = sy / c } else { cx = 0.5; cy = 0.5 }
            scale = 0.0; ang = 0.0
        }
        if (!scale.isFinite() || scale < 0.05) scale = handScaleFallback(seq)
        return Body(cx.toFloat(), cy.toFloat(), scale.toFloat(), ang.toFloat())
    }

    private fun handScaleFallback(seq: Array<FloatArray>): Double {
        val sizes = ArrayList<Double>()
        for (o in intArrayOf(Cfg.LH0, Cfg.RH0)) for (f in seq) {
            val dx = (f[(o + 9) * 3] - f[o * 3]).toDouble()
            val dy = (f[(o + 9) * 3 + 1] - f[o * 3 + 1]).toDouble()
            val d = sqrt(dx * dx + dy * dy)
            if (!d.isNaN()) sizes += d
        }
        // shoulder width is ~2.5x the wrist->middle-MCP length
        return if (sizes.isNotEmpty()) median(sizes) * 2.5 else 0.25
    }

    private fun clip(v: Float, lo: Float, hi: Float): Float =
        if (v.isNaN()) v else v.coerceIn(lo, hi)

    /** NaN-propagating max, like np.maximum. */
    private fun nmax(a: Float, b: Float): Float = if (a.isNaN() || b.isNaN()) Float.NaN else max(a, b)

    /**
     * Canonical 3D hand shape in its own palm frame (63) + orientation in the body
     * frame (6). c, s = cos(-ang), sin(-ang).
     */
    private fun canonicalHand(f: FloatArray, o: Int, c: Float, s: Float,
                              canon: FloatArray, cOff: Int, ori: FloatArray, oOff: Int) {
        val rel = FloatArray(Cfg.N_HAND * 3)
        val wx = f[o * 3]; val wy = f[o * 3 + 1]; val wz = f[o * 3 + 2]
        for (k in 0 until Cfg.N_HAND) {
            rel[k * 3] = f[(o + k) * 3] - wx
            rel[k * 3 + 1] = f[(o + k) * 3 + 1] - wy
            rel[k * 3 + 2] = f[(o + k) * 3 + 2] - wz
        }
        val size = norm(rel[27], rel[28], rel[29])
        val d = nmax(size, 1e-6f)
        for (i in rel.indices) rel[i] = rel[i] / d
        val un = nmax(norm(rel[27], rel[28], rel[29]), 1e-6f)
        val ux = rel[27] / un; val uy = rel[28] / un; val uz = rel[29] / un
        val sx = rel[15] - rel[51]; val sy = rel[16] - rel[52]; val sz = rel[17] - rel[53]
        var nx = uy * sz - uz * sy; var ny = uz * sx - ux * sz; var nz = ux * sy - uy * sx
        val nn = nmax(norm(nx, ny, nz), 1e-6f)
        nx /= nn; ny /= nn; nz /= nn
        val vx = ny * uz - nz * uy; val vy = nz * ux - nx * uz; val vz = nx * uy - ny * ux
        for (k in 0 until Cfg.N_HAND) {
            val x = rel[k * 3]; val y = rel[k * 3 + 1]; val z = rel[k * 3 + 2]
            canon[cOff + k * 3] = clip(ux * x + uy * y + uz * z, -4f, 4f)
            canon[cOff + k * 3 + 1] = clip(vx * x + vy * y + vz * z, -4f, 4f)
            canon[cOff + k * 3 + 2] = clip(nx * x + ny * y + nz * z, -4f, 4f)
        }
        ori[oOff] = c * ux - s * uy; ori[oOff + 1] = s * ux + c * uy; ori[oOff + 2] = uz
        ori[oOff + 3] = c * nx - s * ny; ori[oOff + 4] = s * nx + c * ny; ori[oOff + 5] = nz
    }

    private fun norm(x: Float, y: Float, z: Float) = sqrt(x * x + y * y + z * z)

    /** seq: raw landmark frames -> (SEQ_LEN * FEAT_DIM) float32, row-major. */
    fun compute(input: List<FloatArray>, trim: Boolean = true): FloatArray {
        val T = Cfg.SEQ_LEN
        var seq = input
        if (trim) seq = trimIdle(seq)
        seq = fillHandGaps(seq)
        val rs = resample(seq, T)
        val body = bodyFrame(rs)
        val c = cos(-body.ang); val s = sin(-body.ang)

        val xy = Array(T) { FloatArray(Cfg.N_POINTS * 2) }
        for (t in 0 until T) for (p in 0 until Cfg.N_POINTS) {
            val dx = (rs[t][p * 3] - body.cx) / body.scale
            val dy = (rs[t][p * 3 + 1] - body.cy) / body.scale
            xy[t][p * 2] = clip(c * dx - s * dy, -6f, 6f)
            xy[t][p * 2 + 1] = clip(s * dx + c * dy, -6f, 6f)
        }

        val out = FloatArray(T * Cfg.FEAT_DIM)
        val canon = FloatArray(126)
        val ori = FloatArray(12)
        for (t in 0 until T) {
            val b = t * Cfg.FEAT_DIM
            System.arraycopy(xy[t], 0, out, b, 106)
            canonicalHand(rs[t], Cfg.LH0, c, s, canon, 0, ori, 0)
            canonicalHand(rs[t], Cfg.RH0, c, s, canon, 63, ori, 6)
            System.arraycopy(canon, 0, out, b + 106, 126)
            System.arraycopy(ori, 0, out, b + 232, 12)
            out[b + 244] = if (rs[t][Cfg.LH0 * 3].isNaN()) 0f else 1f
            out[b + 245] = if (rs[t][Cfg.RH0 * 3].isNaN()) 0f else 1f
            for (j in 0 until 106) {
                var v = if (t == 0) 0f else xy[t][j] - xy[t - 1][j]
                if (v.isNaN()) v = 0f
                out[b + 246 + j] = clip(v * 4f, -6f, 6f)
            }
        }
        for (i in out.indices) if (!out[i].isFinite()) out[i] = 0f
        return out
    }
}
