package com.example.yolo_paddle_poc.ocr

import android.graphics.RectF
import com.example.yolo_paddle_poc.model.ObjectBox
import com.example.yolo_paddle_poc.model.OcrTextBox
import com.paddle.ocr.model.OCRResult
import kotlin.math.max
import kotlin.math.min

object OcrMapper {

    fun parseObjectBoxes(
        detections: FloatArray
    ): List<ObjectBox> {

        val boxes = mutableListOf<ObjectBox>()
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

            boxes.add(
                ObjectBox(
                    objectIndex = objectNumber,
                    rect = RectF(left, top, right, bottom),
                    yoloConfidence = detections[index + 4]
                )
            )

            index += 5
            objectNumber++
        }

        return boxes
    }

    fun convertOcrResultToTextBox(
        result: OCRResult
    ): OcrTextBox? {

        val points = result.box.points

        if (points.size != 4) {
            return null
        }

        val left = points.minOf { it.x }
        val top = points.minOf { it.y }
        val right = points.maxOf { it.x }
        val bottom = points.maxOf { it.y }

        if (right <= left || bottom <= top) {
            return null
        }

        return OcrTextBox(
            text = result.text,
            confidence = result.confidence,
            rect = RectF(left, top, right, bottom)
        )
    }

    fun findBestObjectForOcrBox(
        ocrBox: OcrTextBox,
        objectBoxes: List<ObjectBox>,
        overlapThreshold: Float
    ): ObjectBox? {

        val ocrRect = ocrBox.rect
        val centerX = ocrRect.centerX()
        val centerY = ocrRect.centerY()

        var bestObject: ObjectBox? = null
        var bestAssociationScore = -1f

        for (objectBox in objectBoxes) {

            val objectRect = objectBox.rect

            val centerInside =
                objectRect.contains(centerX, centerY)

            val overlapRatio =
                intersectionOverOcrArea(
                    objectRect,
                    ocrRect
                )

            val isCandidate =
                centerInside ||
                        overlapRatio >= overlapThreshold

            if (!isCandidate) {
                continue
            }

            val score =
                if (centerInside) {
                    1f + overlapRatio
                } else {
                    overlapRatio
                }

            if (score > bestAssociationScore) {
                bestAssociationScore = score
                bestObject = objectBox
            }
        }

        return bestObject
    }

    private fun intersectionOverOcrArea(
        objectRect: RectF,
        ocrRect: RectF
    ): Float {

        val left = max(objectRect.left, ocrRect.left)
        val top = max(objectRect.top, ocrRect.top)
        val right = min(objectRect.right, ocrRect.right)
        val bottom = min(objectRect.bottom, ocrRect.bottom)

        if (right <= left || bottom <= top) {
            return 0f
        }

        val intersectionArea =
            (right - left) * (bottom - top)

        val ocrArea =
            max(1f, ocrRect.width() * ocrRect.height())

        return (intersectionArea / ocrArea)
            .coerceIn(0f, 1f)
    }
}
