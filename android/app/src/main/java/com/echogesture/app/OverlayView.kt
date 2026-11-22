package com.echogesture.app

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Draws the detected hand skeletons over the camera preview.
 * Mapping matches PreviewView.ScaleType.FILL_CENTER (scale to cover, centre crop).
 * The skeleton is mint while idle and coral while a sign is being recorded.
 */
class OverlayView(ctx: Context, attrs: AttributeSet?) : View(ctx, attrs) {
    private val bones = intArrayOf(0,1, 1,2, 2,3, 3,4, 0,5, 5,6, 6,7, 7,8, 5,9, 9,10, 10,11, 11,12,
        9,13, 13,14, 14,15, 15,16, 13,17, 0,17, 17,18, 18,19, 19,20)
    private val density = ctx.resources.displayMetrics.density
    private val idleColor = Color.argb(210, 61, 220, 151)   // @color/accent
    private val recColor = Color.argb(230, 255, 107, 94)    // @color/rec
    private val line = Paint().apply {
        color = idleColor; strokeWidth = 3f * density; isAntiAlias = true; strokeCap = Paint.Cap.ROUND
    }
    private val dot = Paint().apply { color = Color.argb(235, 255, 255, 255); isAntiAlias = true }

    @Volatile private var hands: List<FloatArray> = emptyList()
    @Volatile private var imgW = 1; @Volatile private var imgH = 1
    @Volatile private var mirror = true
    @Volatile var recording = false

    fun update(h: List<FloatArray>, w: Int, hgt: Int, mirrorX: Boolean, rec: Boolean) {
        hands = h; imgW = w; imgH = hgt; mirror = mirrorX; recording = rec
        postInvalidate()
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        val s = max(width.toFloat() / imgW, height.toFloat() / imgH)
        val ox = (width - imgW * s) / 2; val oy = (height - imgH * s) / 2
        fun px(x: Float) = ox + (if (mirror) 1 - x else x) * imgW * s
        fun py(y: Float) = oy + y * imgH * s
        line.color = if (recording) recColor else idleColor
        val r = 3.5f * density
        for (h in hands) {
            for (i in bones.indices step 2) {
                val a = bones[i]; val b = bones[i + 1]
                c.drawLine(px(h[a * 2]), py(h[a * 2 + 1]), px(h[b * 2]), py(h[b * 2 + 1]), line)
            }
            for (k in 0 until 21) c.drawCircle(px(h[k * 2]), py(h[k * 2 + 1]), r, dot)
        }
    }
}
