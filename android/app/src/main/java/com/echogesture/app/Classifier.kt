package com.echogesture.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.exp

/** INT8 ONNX classifier (ONNX Runtime, CPU, 2 threads) - same model as src/main.py. */
class Classifier(ctx: Context) : AutoCloseable {

    class Prediction(val top: List<String>, val probs: List<Float>, val ms: Double)

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    val labels: List<String>

    init {
        val bytes = ctx.assets.open("echogesture_int8.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
        labels = ctx.assets.open("labels.txt").bufferedReader().readLines()
            .map { it.trim().uppercase() }.filter { it.isNotEmpty() }
    }

    fun classify(feat: FloatArray, k: Int = 5): Prediction {
        val t0 = System.nanoTime()
        val shape = longArrayOf(1, Cfg.SEQ_LEN.toLong(), Cfg.FEAT_DIM.toLong())
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(feat), shape).use { x ->
            session.run(mapOf(inputName to x)).use { r ->
                @Suppress("UNCHECKED_CAST")
                (r[0].value as Array<FloatArray>)[0]
            }
        }
        // the model may have spare output units beyond the real classes - ignore them
        val n = minOf(labels.size, logits.size)
        var mx = Float.NEGATIVE_INFINITY
        for (i in 0 until n) if (logits[i] > mx) mx = logits[i]
        val e = DoubleArray(n) { exp((logits[it] - mx).toDouble()) }
        val sum = e.sum()
        val order = (0 until n).sortedByDescending { e[it] }.take(k)
        val ms = (System.nanoTime() - t0) / 1e6
        return Prediction(order.map { labels[it] }, order.map { (e[it] / sum).toFloat() }, ms)
    }

    override fun close() { session.close() }
}
