package com.echogesture.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlin.math.abs

/** Rolling latency/FPS stats + thermal drift (median of first vs last minute). */
class Stats {
    private val recent = ArrayDeque<Double>()
    private val stamps = ArrayDeque<Long>()
    private val all = ArrayList<Pair<Long, Double>>()   // (ms since start, frame latency)
    private var t0 = -1L

    @Synchronized fun addFrame(ms: Double, nowMs: Long) {
        if (t0 < 0) t0 = nowMs
        recent.addLast(ms); if (recent.size > 300) recent.removeFirst()
        stamps.addLast(nowMs); while (stamps.size > 1 && nowMs - stamps.first() > 2000) stamps.removeFirst()
        all += (nowMs - t0) to ms
    }

    @Synchronized fun reset() { recent.clear(); stamps.clear(); all.clear(); t0 = -1 }

    private fun pct(xs: List<Double>, p: Double): Double {
        if (xs.isEmpty()) return Double.NaN
        val s = xs.sorted()
        val pos = p / 100 * (s.size - 1)
        val i = pos.toInt(); val f = pos - i
        return if (i + 1 < s.size) s[i] * (1 - f) + s[i + 1] * f else s[i]
    }

    @Synchronized fun p50() = pct(recent.toList(), 50.0)
    @Synchronized fun p95() = pct(recent.toList(), 95.0)
    @Synchronized fun fps(): Double =
        if (stamps.size < 2) 0.0 else (stamps.size - 1) * 1000.0 / (stamps.last() - stamps.first())

    /** % change of median latency, last 60 s vs first 60 s (needs >= 2 min of running). */
    @Synchronized fun thermalDriftPct(): Double? {
        if (all.isEmpty() || all.last().first < 120_000) return null
        val end = all.last().first
        val first = all.filter { it.first < 60_000 }.map { it.second }
        val last = all.filter { it.first > end - 60_000 }.map { it.second }
        val a = pct(first, 50.0); val b = pct(last, 50.0)
        return (b / a - 1) * 100
    }

    @Synchronized fun minutes() = if (all.isEmpty()) 0.0 else all.last().first / 60000.0
}

/**
 * Whole-device power from the battery fuel gauge (current x voltage). Only valid
 * while unplugged. Compare "ML on" vs "ML off" to estimate the pipeline's share.
 */
class PowerMeter(private val ctx: Context) {
    private val bm = ctx.getSystemService(BatteryManager::class.java)
    private val sums = DoubleArray(2); private val counts = IntArray(2)

    private fun battery(): Intent? = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

    fun pluggedIn(): Boolean = (battery()?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0

    fun tempC(): Double? = battery()?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        ?.takeIf { it != Int.MIN_VALUE }?.div(10.0)

    fun sampleMw(): Double? {
        val i = bm?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) ?: return null
        if (i == Int.MIN_VALUE || i == 0) return null
        val mv = battery()?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        if (mv <= 0) return null
        var ua = abs(i).toDouble()
        if (ua < 20_000) ua *= 1000          // some devices report mA instead of uA
        return ua / 1e6 * mv                  // A * mV = mW
    }

    /** Call ~1/s. state 1 = ML running, 0 = ML paused. */
    fun tick(mlOn: Boolean) {
        if (pluggedIn()) return
        val p = sampleMw() ?: return
        val k = if (mlOn) 1 else 0
        sums[k] += p; counts[k]++
    }

    fun avg(mlOn: Boolean): Double? { val k = if (mlOn) 1 else 0; return if (counts[k] > 0) sums[k] / counts[k] else null }
    fun reset() { sums.fill(0.0); counts.fill(0) }
}
