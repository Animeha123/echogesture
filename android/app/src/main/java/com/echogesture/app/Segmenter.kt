package com.echogesture.app

import kotlin.math.hypot

/**
 * Cuts the live landmark stream into single signs.
 * A sign starts when a hand is raised into the signing space (above roughly
 * mid-torso) and ends when hands drop / leave the frame for END_GAP_MS.
 */
class Segmenter {
    companion object {
        const val START_FRAMES = 2       // consecutive active frames to start
        const val END_GAP_MS = 450L      // inactivity that ends a sign
        const val MIN_ACTIVE = 5         // ignore twitches
        const val MAX_MS = 5000L         // force a cut on very long segments
        const val PRE_ROLL = 4           // frames kept before the start (trim margin)
        const val TAIL = 2
    }

    private val pre = ArrayDeque<FloatArray>()
    private var rec: ArrayList<FloatArray>? = null
    private var recStartMs = 0L
    private var lastActiveIdx = -1
    private var lastActiveMs = 0L
    private var activeCount = 0
    private var streak = 0

    val recording get() = rec != null

    fun reset() { pre.clear(); rec = null; streak = 0 }

    /** Hand present and raised above ~mid-torso (resting hands in the lap do not count). */
    fun active(f: FloatArray): Boolean {
        if (!Features.handPresent(f)) return false
        val ls = Cfg.P_LSHO * 3; val rs = Cfg.P_RSHO * 3
        if (f[ls].isNaN() || f[rs].isNaN()) return true
        val sy = (f[ls + 1] + f[rs + 1]) / 2
        val sw = hypot(f[ls] - f[rs], f[ls + 1] - f[rs + 1])
        val limit = sy + 1.7f * sw
        for (o in intArrayOf(Cfg.LH0, Cfg.RH0)) {
            val y = f[o * 3 + 1]
            if (!y.isNaN() && y < limit) return true
        }
        return false
    }

    /** Feed one frame; returns a finished sign (list of frames) or null. */
    fun push(f: FloatArray, tMs: Long): List<FloatArray>? {
        val act = active(f)
        val r = rec
        if (r == null) {
            pre.addLast(f); if (pre.size > PRE_ROLL) pre.removeFirst()
            streak = if (act) streak + 1 else 0
            if (streak >= START_FRAMES) {
                rec = ArrayList(pre); recStartMs = tMs; lastActiveMs = tMs
                lastActiveIdx = rec!!.size - 1; activeCount = streak; pre.clear()
            }
            return null
        }
        r.add(f)
        if (act) { lastActiveIdx = r.size - 1; lastActiveMs = tMs; activeCount++ }
        val ended = tMs - lastActiveMs >= END_GAP_MS
        val tooLong = tMs - recStartMs >= MAX_MS
        if (!ended && !tooLong) return null
        val seg = r.subList(0, minOf(r.size, lastActiveIdx + 1 + TAIL)).toList()
        val ok = activeCount >= MIN_ACTIVE
        rec = null; streak = 0; pre.clear()
        return if (ok) seg else null
    }
}
