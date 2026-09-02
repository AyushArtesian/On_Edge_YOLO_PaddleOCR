package com.example.yolo_paddle_poc.model

import android.graphics.RectF

data class DetectedItem(
    val objectIndex: Int,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val yoloConfidence: Float,
    val rawOcrText: String,
    val ocrLineCount: Int,
    val ocrTimeMs: Long,
    val matchedProductName: String?,
    val productMatchScore: Double,
    val matchMethod: String
)

data class ObjectBox(
    val objectIndex: Int,
    val rect: RectF,
    val yoloConfidence: Float
)

data class OcrTextBox(
    val text: String,
    val confidence: Float,
    val rect: RectF
)

data class ProductResolution(
    val productName: String?,
    val score: Double,
    val method: String
)

data class PipelineResult(
    val detectedItems: List<DetectedItem>,
    val detections: FloatArray,
    val yoloTimeMs: Long,
    val pipelineTimeMs: Long
)
