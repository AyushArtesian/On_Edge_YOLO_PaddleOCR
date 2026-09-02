package com.example.yolo_paddle_poc.util

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

object DrawingUtils {

    private const val TAG = "YOLO_NCNN"

    fun drawBoundingBoxes(
        source: Bitmap,
        detections: FloatArray
    ): Bitmap {

        val output =
            source.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val canvas = Canvas(output)

        val boxPaint =
            Paint().apply {
                style = Paint.Style.STROKE
                color = Color.RED
                strokeWidth =
                    maxOf(4f, source.width / 250f)
                isAntiAlias = true
            }

        val textPaint =
            Paint().apply {
                style = Paint.Style.FILL
                color = Color.WHITE
                isAntiAlias = true
                isFakeBoldText = true
                textSize =
                    maxOf(30f, source.width / 22f)
            }

        val labelPaint =
            Paint().apply {
                style = Paint.Style.FILL
                color = Color.RED
                isAntiAlias = true
            }

        var index = 0
        var objectNumber = 1

        while (index + 4 < detections.size) {

            val left =
                min(detections[index], detections[index + 2])

            val top =
                min(detections[index + 1], detections[index + 3])

            val right =
                max(detections[index], detections[index + 2])

            val bottom =
                max(detections[index + 1], detections[index + 3])

            canvas.drawRect(
                left,
                top,
                right,
                bottom,
                boxPaint
            )

            val label = "#$objectNumber"
            val padding =
                maxOf(6f, textPaint.textSize * 0.20f)

            val labelWidth =
                textPaint.measureText(label) +
                        padding * 2f

            val labelHeight =
                textPaint.textSize +
                        padding * 2f

            val labelLeft =
                left.coerceIn(
                    0f,
                    maxOf(
                        0f,
                        source.width.toFloat() -
                                labelWidth
                    )
                )

            val labelTop =
                if (top - labelHeight >= 0f) {
                    top - labelHeight
                } else {
                    top
                }

            val labelRight =
                min(
                    source.width.toFloat(),
                    labelLeft + labelWidth
                )

            val labelBottom =
                min(
                    source.height.toFloat(),
                    labelTop + labelHeight
                )

            canvas.drawRect(
                labelLeft,
                labelTop,
                labelRight,
                labelBottom,
                labelPaint
            )

            canvas.drawText(
                label,
                labelLeft + padding,
                labelBottom - padding,
                textPaint
            )

            Log.i(
                TAG,
                "Drew Object #$objectNumber box=[" +
                        "${String.format(Locale.US, "%.1f", left)}, " +
                        "${String.format(Locale.US, "%.1f", top)}, " +
                        "${String.format(Locale.US, "%.1f", right)}, " +
                        "${String.format(Locale.US, "%.1f", bottom)}]"
            )

            index += 5
            objectNumber++
        }

        return output
    }
}
