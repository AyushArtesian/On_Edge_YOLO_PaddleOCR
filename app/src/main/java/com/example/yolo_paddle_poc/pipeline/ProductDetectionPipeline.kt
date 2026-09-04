package com.example.yolo_paddle_poc.pipeline

import android.graphics.Bitmap
import android.util.Log
import com.example.yolo_paddle_poc.matching.HybridProductResolver
import com.example.yolo_paddle_poc.model.DetectedItem
import com.example.yolo_paddle_poc.model.OcrTextBox
import com.example.yolo_paddle_poc.model.ProductResolution
import com.example.yolo_paddle_poc.ocr.OcrMapper
import com.example.yolo_paddle_poc.ocr.ProminentTextSelector
import com.example.yolo_paddle_poc.util.BitmapUtils
import com.paddle.ocr.PaddleOCR
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

class ProductDetectionPipeline(
    private val paddleOCR: PaddleOCR,
    private val resolver: HybridProductResolver
) {

    companion object {
        private const val OCR_TAG = "PADDLE_OCR"
        private const val MAPPING_TAG = "OCR_OBJECT_MAPPING"

        private const val OCR_CONFIDENCE_THRESHOLD = 0.20f
        private const val OCR_CROP_PADDING_PERCENT = 0.12f
        private const val OCR_CROP_SCALE = 3.0f
        private const val OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50f
    }

    suspend fun runHybridOcr(
        sourceBitmap: Bitmap,
        detections: FloatArray
    ): List<DetectedItem> {

        val objectBoxes =
            OcrMapper.parseObjectBoxes(detections)

        // PASS 1: full-image OCR.
        val fullOcrStart =
            System.currentTimeMillis()

        val fullOcrResult =
            paddleOCR.recognize(sourceBitmap)

        val fullOcrElapsed =
            System.currentTimeMillis() -
                    fullOcrStart

        val fullImageOcrBoxes =
            fullOcrResult.results
                .filter {
                    it.confidence >=
                            OCR_CONFIDENCE_THRESHOLD
                }
                .mapNotNull {
                    OcrMapper.convertOcrResultToTextBox(it)
                }

        Log.i(
            OCR_TAG,
            "FULL IMAGE OCR finished: " +
                    "${fullOcrResult.results.size} raw lines, " +
                    "${fullImageOcrBoxes.size} accepted lines, " +
                    "${fullOcrElapsed}ms"
        )

        val mapped =
            mutableMapOf<Int, MutableList<OcrTextBox>>()

        for (ocrBox in fullImageOcrBoxes) {

            val bestObject =
                OcrMapper.findBestObjectForOcrBox(
                    ocrBox = ocrBox,
                    objectBoxes = objectBoxes,
                    overlapThreshold =
                        OCR_TO_OBJECT_OVERLAP_THRESHOLD
                )

            if (bestObject != null) {

                mapped
                    .getOrPut(bestObject.objectIndex) {
                        mutableListOf()
                    }
                    .add(ocrBox)

                Log.i(
                    MAPPING_TAG,
                    "FULL OCR '${ocrBox.text}' -> " +
                            "Object #${bestObject.objectIndex}"
                )

            } else {

                Log.i(
                    MAPPING_TAG,
                    "FULL OCR '${ocrBox.text}' -> UNASSIGNED"
                )
            }
        }

        val items =
            mutableListOf<DetectedItem>()

        for (objectBox in objectBoxes) {

            val mappedLines =
                mapped[objectBox.objectIndex]
                    .orEmpty()
                    .sortedWith(
                        compareBy<OcrTextBox> {
                            it.rect.top
                        }.thenBy {
                            it.rect.left
                        }
                    )

            val fullRawText =
                mappedLines
                    .joinToString(" ") {
                        it.text.trim()
                    }
                    .trim()

            val prominentFullLines =
                ProminentTextSelector.select(
                    lines = mappedLines,
                    referenceWidth =
                        objectBox.rect.width(),
                    referenceHeight =
                        objectBox.rect.height(),
                    sourceLabel =
                        "FULL_IMAGE Object #${objectBox.objectIndex}"
                )

            val fullText =
                prominentFullLines
                    .joinToString(" ") {
                        it.text.trim()
                    }
                    .trim()
                    .ifBlank {
                        fullRawText
                    }

            val fullResolution =
                if (fullText.isNotBlank()) {
                    withContext(Dispatchers.Default) {
                        resolver.resolve(fullText)
                    }
                } else {
                    ProductResolution(
                        null,
                        0.0,
                        "NO_FULL_IMAGE_TEXT"
                    )
                }

            if (fullResolution.productName != null) {

                Log.i(
                    MAPPING_TAG,
                    """
                    Object #${objectBox.objectIndex}
                    OCR source: FULL_IMAGE
                    Raw OCR text: ${if (fullRawText.isBlank()) "<none>" else fullRawText}
                    Prominent OCR text: ${if (fullText.isBlank()) "<none>" else fullText}
                    Matched: ${fullResolution.productName}
                    Score: ${String.format(Locale.US, "%.4f", fullResolution.score)}
                    """.trimIndent()
                )

                items.add(
                    DetectedItem(
                        objectIndex =
                            objectBox.objectIndex,
                        left =
                            objectBox.rect.left,
                        top =
                            objectBox.rect.top,
                        right =
                            objectBox.rect.right,
                        bottom =
                            objectBox.rect.bottom,
                        yoloConfidence =
                            objectBox.yoloConfidence,
                        rawOcrText =
                            fullText,
                        ocrLineCount =
                            prominentFullLines.size,
                        ocrTimeMs =
                            fullOcrElapsed,
                        matchedProductName =
                            fullResolution.productName,
                        productMatchScore =
                            fullResolution.score * 100.0,
                        matchMethod =
                            "FULL_IMAGE_${fullResolution.method}"
                    )
                )

                continue
            }

            // PASS 2: crop fallback.
            val cropStart =
                System.currentTimeMillis()

            var cropBitmap: Bitmap? = null
            var enlargedCrop: Bitmap? = null

            try {

                cropBitmap =
                    BitmapUtils.cropBitmapWithPadding(
                        source = sourceBitmap,
                        rect = objectBox.rect,
                        paddingPercent =
                            OCR_CROP_PADDING_PERCENT
                    )

                enlargedCrop =
                    BitmapUtils.upscaleBitmapForOcr(
                        bitmap = cropBitmap,
                        scaleFactor = OCR_CROP_SCALE
                    )

                val cropOcrResult =
                    paddleOCR.recognize(enlargedCrop)

                val cropBoxes =
                    cropOcrResult.results
                        .filter {
                            it.confidence >=
                                    OCR_CONFIDENCE_THRESHOLD
                        }
                        .mapNotNull {
                            OcrMapper.convertOcrResultToTextBox(it)
                        }

                val cropRawText =
                    cropBoxes
                        .sortedWith(
                            compareBy<OcrTextBox> {
                                it.rect.top
                            }.thenBy {
                                it.rect.left
                            }
                        )
                        .joinToString(" ") {
                            it.text.trim()
                        }
                        .trim()

                val prominentCropLines =
                    ProminentTextSelector.select(
                        lines = cropBoxes,
                        referenceWidth =
                            enlargedCrop.width.toFloat(),
                        referenceHeight =
                            enlargedCrop.height.toFloat(),
                        sourceLabel =
                            "CROP Object #${objectBox.objectIndex}"
                    )

                val cropText =
                    prominentCropLines
                        .joinToString(" ") {
                            it.text.trim()
                        }
                        .trim()
                        .ifBlank {
                            cropRawText
                        }

                val cropResolution =
                    if (cropText.isNotBlank()) {
                        withContext(Dispatchers.Default) {
                            resolver.resolve(cropText)
                        }
                    } else {
                        ProductResolution(
                            null,
                            0.0,
                            "NO_CROP_TEXT"
                        )
                    }

                val cropElapsed =
                    System.currentTimeMillis() -
                            cropStart

                val cropAccepted =
                    cropResolution.productName != null

                val selectedResolution =
                    if (cropAccepted) {
                        cropResolution
                    } else if (
                        fullResolution.score >
                        cropResolution.score
                    ) {
                        fullResolution
                    } else {
                        cropResolution
                    }

                val selectedText =
                    if (
                        cropResolution.score >=
                        fullResolution.score
                    ) {
                        cropText.ifBlank {
                            "No OCR text"
                        }
                    } else {
                        fullText.ifBlank {
                            "No OCR text"
                        }
                    }

                val selectedLineCount =
                    if (
                        cropResolution.score >=
                        fullResolution.score
                    ) {
                        prominentCropLines.size
                    } else {
                        prominentFullLines.size
                    }

                Log.i(
                    MAPPING_TAG,
                    """
                    Object #${objectBox.objectIndex}
                    OCR source: CROP_FALLBACK
                    Full-image prominent text: ${if (fullText.isBlank()) "<none>" else fullText}
                    Crop raw text: ${if (cropRawText.isBlank()) "<none>" else cropRawText}
                    Crop prominent text: ${if (cropText.isBlank()) "<none>" else cropText}
                    Final matched: ${selectedResolution.productName ?: "Unknown"}
                    Final score: ${String.format(Locale.US, "%.4f", selectedResolution.score)}
                    """.trimIndent()
                )

                items.add(
                    DetectedItem(
                        objectIndex =
                            objectBox.objectIndex,
                        left =
                            objectBox.rect.left,
                        top =
                            objectBox.rect.top,
                        right =
                            objectBox.rect.right,
                        bottom =
                            objectBox.rect.bottom,
                        yoloConfidence =
                            objectBox.yoloConfidence,
                        rawOcrText =
                            selectedText,
                        ocrLineCount =
                            selectedLineCount,
                        ocrTimeMs =
                            fullOcrElapsed +
                                    cropElapsed,
                        matchedProductName =
                            selectedResolution.productName,
                        productMatchScore =
                            selectedResolution.score * 100.0,
                        matchMethod =
                            if (cropAccepted) {
                                "CROP_FALLBACK_${cropResolution.method}"
                            } else {
                                "HYBRID_UNKNOWN"
                            }
                    )
                )

                if (selectedResolution.productName == null) {
                    Log.i(
                        MAPPING_TAG,
                        "Object #${objectBox.objectIndex} kept as UNKNOWN product; " +
                                "YOLO detection is preserved and will still be displayed."
                    )
                }

            } catch (e: Exception) {

                Log.e(
                    OCR_TAG,
                    "Crop OCR failed for Object #${objectBox.objectIndex}",
                    e
                )

                items.add(
                    DetectedItem(
                        objectIndex =
                            objectBox.objectIndex,
                        left =
                            objectBox.rect.left,
                        top =
                            objectBox.rect.top,
                        right =
                            objectBox.rect.right,
                        bottom =
                            objectBox.rect.bottom,
                        yoloConfidence =
                            objectBox.yoloConfidence,
                        rawOcrText =
                            fullText.ifBlank {
                                "OCR failed"
                            },
                        ocrLineCount =
                            prominentFullLines.size,
                        ocrTimeMs =
                            fullOcrElapsed,
                        matchedProductName =
                            null,
                        productMatchScore =
                            fullResolution.score * 100.0,
                        matchMethod =
                            "CROP_FALLBACK_FAILED"
                    )
                )

            } finally {

                if (
                    enlargedCrop != null &&
                    enlargedCrop !== cropBitmap &&
                    !enlargedCrop.isRecycled
                ) {
                    enlargedCrop.recycle()
                }

                if (
                    cropBitmap != null &&
                    !cropBitmap.isRecycled
                ) {
                    cropBitmap.recycle()
                }
            }
        }

        Log.i(
            OCR_TAG,
            "Hybrid OCR finished for ${items.size} objects"
        )

        return items
    }
}
