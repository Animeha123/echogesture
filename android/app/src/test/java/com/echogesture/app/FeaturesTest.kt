package com.echogesture.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Kotlin feature port vs Python reference (golden files written by
 * src/scripts/export_android_golden.py). Run: right-click this file -> Run 'FeaturesTest'.
 */
class FeaturesTest {

    private fun floats(name: String): FloatArray {
        val bytes = javaClass.classLoader!!.getResourceAsStream("golden/$name")!!.readBytes()
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    @Test
    fun matchesPythonReference() {
        val index = javaClass.classLoader!!.getResourceAsStream("golden/index.txt")!!
            .bufferedReader().readLines().filter { it.isNotBlank() }
        assertTrue(index.isNotEmpty())
        for (line in index) {
            val (name, nStr) = line.trim().split(" ")
            val n = nStr.toInt()
            val lm = floats("$name.lm")
            assertEquals(n * Cfg.FRAME, lm.size)
            val seq = List(n) { i -> lm.copyOfRange(i * Cfg.FRAME, (i + 1) * Cfg.FRAME) }
            val want = floats("$name.feat")
            val got = Features.compute(seq)
            assertEquals(Cfg.SEQ_LEN * Cfg.FEAT_DIM, got.size)
            var worst = 0f; var at = -1
            for (i in got.indices) {
                val d = abs(got[i] - want[i])
                if (d > worst) { worst = d; at = i }
            }
            println("$name: max |diff| = $worst at t=${at / Cfg.FEAT_DIM} dim=${at % Cfg.FEAT_DIM}")
            assertTrue("$name differs: $worst at index $at", worst < 2e-3f)
        }
    }
}
