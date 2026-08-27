package com.example.yolo_paddle_poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.model.OCRResult
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "YOLO_NCNN"
        private const val OCR_TAG = "PADDLE_OCR"
        private const val MATCHER_TAG = "PRODUCT_MATCHER"
        private const val EMBEDDING_TAG = "TEXT_EMBEDDING_MATCHER"
        private const val HYBRID_TAG = "HYBRID_PRODUCT_RESOLVER"
        private const val OCR_MAPPING_TAG = "OCR_OBJECT_MAPPING"

        private const val MAX_IMAGE_SIZE = 1280
        private const val OCR_CONFIDENCE_THRESHOLD = 0.25f

        /*
         * OCR text is assigned to a YOLO object when:
         *
         * 1) the OCR-box center lies inside the YOLO box, OR
         * 2) at least this fraction of the OCR box overlaps the YOLO box.
         */
        private const val OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50f

        private const val TOP_K = 5

        private const val FINAL_STRONG_THRESHOLD = 0.75
        private const val FINAL_AGREEMENT_THRESHOLD = 0.50

        private const val EMBEDDING_WEIGHT = 0.55
        private const val FUZZY_WEIGHT = 0.45
        private const val AGREEMENT_BONUS = 0.10

        private const val FUZZY_STRONG_OVERRIDE = 0.88

        init {
            System.loadLibrary("yolo_native")
        }
    }

    // =====================================================
    // YOLO JNI
    // =====================================================

    external fun loadYoloModel(
        assetManager: android.content.res.AssetManager
    ): Boolean

    external fun detectObjects(
        bitmap: Bitmap
    ): FloatArray

    // =====================================================
    // RESULT MODELS
    // =====================================================

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

    private data class ObjectBox(
        val objectIndex: Int,
        val rect: RectF,
        val yoloConfidence: Float
    )

    private data class OcrTextBox(
        val text: String,
        val confidence: Float,
        val rect: RectF
    )

    private data class ProductResolution(
        val productName: String?,
        val score: Double,
        val method: String
    )

    private data class HybridCandidate(
        val displayName: String,
        val key: String,
        var embeddingScore: Double = 0.0,
        var fuzzyScore: Double = 0.0,
        var appearsInEmbedding: Boolean = false,
        var appearsInFuzzy: Boolean = false
    ) {
        fun combinedScore(): Double {
            var score =
                EMBEDDING_WEIGHT * embeddingScore +
                        FUZZY_WEIGHT * fuzzyScore

            if (
                appearsInEmbedding &&
                appearsInFuzzy
            ) {
                score += AGREEMENT_BONUS
            }

            return score.coerceIn(0.0, 1.0)
        }
    }

    // =====================================================
    // UI
    // =====================================================

    private lateinit var selectImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var resultsText: TextView

    // =====================================================
    // MODEL STATE
    // =====================================================

    private var yoloReady = false
    private var ocrReady = false
    private var embeddingReady = false

    private var paddleOCR: PaddleOCR? = null

    private lateinit var productMatcher: ProductMatcher
    private lateinit var textEmbeddingMatcher: TextEmbeddingMatcher

    // =====================================================
    // IMAGE PICKER
    // =====================================================

    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri == null) {
                return@registerForActivityResult
            }

            if (!yoloReady) {
                statusText.text = "YOLO model is still loading"
                return@registerForActivityResult
            }

            if (!ocrReady || paddleOCR == null) {
                statusText.text = "PaddleOCR is still loading"
                return@registerForActivityResult
            }

            if (!embeddingReady) {
                statusText.text = "MiniLM matcher is still loading"
                return@registerForActivityResult
            }

            lifecycleScope.launch {
                try {
                    selectImageButton.isEnabled = false
                    statusText.text = "Loading image..."
                    resultsText.text = "Processing..."

                    // =================================================
                    // LOAD IMAGE
                    // =================================================

                    val decodedBitmap =
                        withContext(Dispatchers.IO) {
                            contentResolver
                                .openInputStream(uri)
                                ?.use { inputStream ->
                                    BitmapFactory.decodeStream(inputStream)
                                }
                        }

                    if (decodedBitmap == null) {
                        statusText.text = "Unable to load image"
                        resultsText.text = "Image loading failed."
                        selectImageButton.isEnabled = true
                        return@launch
                    }

                    val rgbaBitmap =
                        decodedBitmap.copy(
                            Bitmap.Config.ARGB_8888,
                            false
                        )

                    if (rgbaBitmap == null) {
                        statusText.text = "Unable to convert image"
                        resultsText.text = "Image conversion failed."
                        selectImageButton.isEnabled = true
                        return@launch
                    }

                    if (decodedBitmap !== rgbaBitmap) {
                        decodedBitmap.recycle()
                    }

                    /*
                     * IMPORTANT:
                     *
                     * Both YOLO and PaddleOCR run on this exact same
                     * inferenceBitmap. Therefore their coordinates are
                     * already in the same coordinate system and no scaling
                     * is required while mapping OCR boxes to YOLO boxes.
                     */
                    val inferenceBitmap =
                        resizeBitmapForInference(
                            rgbaBitmap,
                            MAX_IMAGE_SIZE
                        )

                    if (inferenceBitmap !== rgbaBitmap) {
                        rgbaBitmap.recycle()
                    }

                    Log.i(
                        TAG,
                        "Inference bitmap = " +
                                "${inferenceBitmap.width}x${inferenceBitmap.height}"
                    )

                    // =================================================
                    // YOLO - ONE FULL-IMAGE RUN
                    // =================================================

                    statusText.text = "Detecting objects..."

                    val yoloStart =
                        System.currentTimeMillis()

                    val detections =
                        withContext(Dispatchers.Default) {
                            detectObjects(inferenceBitmap)
                        }

                    val yoloElapsed =
                        System.currentTimeMillis() - yoloStart

                    val detectionCount =
                        detections.size / 5

                    Log.i(
                        TAG,
                        "YOLO detected $detectionCount objects " +
                                "in ${yoloElapsed}ms"
                    )

                    if (detectionCount == 0) {
                        imageView.setImageBitmap(inferenceBitmap)
                        statusText.text = "No objects detected"
                        resultsText.text = "No objects detected."
                        selectImageButton.isEnabled = true
                        return@launch
                    }

                    // =================================================
                    // PADDLE OCR - ONE FULL-IMAGE RUN
                    // =================================================

                    statusText.text = "Running full-image OCR..."

                    val pipelineStart =
                        System.currentTimeMillis()

                    val detectedItems =
                        runFullImageOcrAndMap(
                            sourceBitmap = inferenceBitmap,
                            detections = detections
                        )

                    val pipelineElapsed =
                        System.currentTimeMillis() - pipelineStart

                    // =================================================
                    // DRAW YOLO BOXES
                    // =================================================

                    val resultBitmap =
                        drawBoundingBoxes(
                            inferenceBitmap,
                            detections
                        )

                    imageView.setImageBitmap(resultBitmap)

                    // =================================================
                    // DISPLAY / LOG
                    // =================================================

                    displayResults(detectedItems)
                    logFinalResults(detectedItems)

                    statusText.text =
                        "$detectionCount objects • " +
                                "YOLO ${yoloElapsed}ms • " +
                                "Full OCR+Map+Match ${pipelineElapsed}ms"

                    selectImageButton.isEnabled = true

                } catch (exception: Exception) {
                    Log.e(
                        TAG,
                        "Processing failed",
                        exception
                    )

                    statusText.text = "Processing failed"
                    resultsText.text = "Something went wrong."
                    selectImageButton.isEnabled = true
                }
            }
        }

    // =====================================================
    // ON CREATE
    // =====================================================

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(
            findViewById(R.id.main)
        ) { view, insets ->

            val systemBars =
                insets.getInsets(
                    WindowInsetsCompat.Type.systemBars()
                )

            view.setPadding(
                systemBars.left,
                systemBars.top,
                systemBars.right,
                systemBars.bottom
            )

            insets
        }

        selectImageButton =
            findViewById(R.id.selectImageButton)

        imageView =
            findViewById(R.id.imageView)

        statusText =
            findViewById(R.id.statusText)

        resultsText =
            findViewById(R.id.resultsText)

        selectImageButton.isEnabled = false
        resultsText.text = "Waiting for AI models..."

        productMatcher =
            ProductMatcher(this)

        selectImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        loadTextEmbeddingMatcher()
        loadYolo()
        loadPaddleOCR()
    }

    // =====================================================
    // LOAD MINILM
    // =====================================================

    private fun loadTextEmbeddingMatcher() {
        lifecycleScope.launch {
            try {
                Log.i(
                    EMBEDDING_TAG,
                    "Loading MiniLM..."
                )

                textEmbeddingMatcher =
                    withContext(Dispatchers.IO) {
                        TextEmbeddingMatcher(
                            applicationContext
                        )
                    }

                textEmbeddingMatcher.debugTokens(
                    "Amul Ginger Doodh"
                )

                embeddingReady = true

                Log.i(
                    EMBEDDING_TAG,
                    "MINILM MATCHER READY"
                )

                updateModelStatus()

            } catch (exception: Exception) {
                embeddingReady = false

                Log.e(
                    EMBEDDING_TAG,
                    "MiniLM initialization failed",
                    exception
                )

                statusText.text =
                    "MiniLM matcher failed to load"

                resultsText.text =
                    "Text embedding model initialization failed."

                selectImageButton.isEnabled = false
            }
        }
    }

    // =====================================================
    // LOAD YOLO
    // =====================================================

    private fun loadYolo() {
        Thread {
            Log.i(
                TAG,
                "Loading YOLO model..."
            )

            val success =
                loadYoloModel(assets)

            runOnUiThread {
                if (success) {
                    yoloReady = true

                    Log.i(
                        TAG,
                        "YOLO MODEL READY"
                    )

                    updateModelStatus()

                } else {
                    yoloReady = false

                    statusText.text =
                        "Failed to load YOLO model"

                    resultsText.text =
                        "YOLO initialization failed."

                    selectImageButton.isEnabled = false
                }
            }
        }.start()
    }

    // =====================================================
    // LOAD PADDLE OCR
    // =====================================================

    private fun loadPaddleOCR() {
        lifecycleScope.launch {
            try {
                Log.i(
                    OCR_TAG,
                    "Initializing OpenCV..."
                )

                val openCvReady =
                    OpenCVUtils.init(
                        this@MainActivity
                    )

                if (!openCvReady) {
                    ocrReady = false
                    statusText.text =
                        "OpenCV failed to initialize"
                    selectImageButton.isEnabled = false
                    return@launch
                }

                paddleOCR =
                    PaddleOCR.create(
                        context =
                            this@MainActivity,

                        config =
                            PaddleOCRConfig(
                                detThresh = 0.3f,
                                detBoxThresh = 0.6f,
                                recScoreThresh =
                                    OCR_CONFIDENCE_THRESHOLD,
                                recBatchSize = 1
                            ),

                        engineConfig =
                            EngineConfig(
                                numThreads = 2
                            ),

                        detModelAssetPath =
                            "models/det/inference.onnx",

                        recModelAssetPath =
                            "models/rec/inference.onnx",

                        recConfigAssetPath =
                            "models/rec/inference.yml"
                    )

                ocrReady = true

                Log.i(
                    OCR_TAG,
                    "PADDLE OCR READY"
                )

                updateModelStatus()

            } catch (exception: Exception) {
                ocrReady = false

                Log.e(
                    OCR_TAG,
                    "PaddleOCR initialization failed",
                    exception
                )

                statusText.text =
                    "PaddleOCR failed to load"

                selectImageButton.isEnabled = false
            }
        }
    }

    // =====================================================
    // MODEL STATUS
    // =====================================================

    private fun updateModelStatus() {
        if (
            yoloReady &&
            ocrReady &&
            embeddingReady
        ) {
            statusText.text =
                "YOLO + Full-image OCR + Hybrid Matcher Ready"

            resultsText.text =
                "Select an image."

            selectImageButton.isEnabled = true

        } else {
            statusText.text =
                buildString {
                    append("Loading:")

                    if (!yoloReady) {
                        append(" YOLO")
                    }

                    if (!ocrReady) {
                        append(" OCR")
                    }

                    if (!embeddingReady) {
                        append(" MiniLM")
                    }
                }

            selectImageButton.isEnabled = false
        }
    }

    // =====================================================
    // FULL IMAGE OCR + YOLO/OCR COORDINATE MAPPING
    // =====================================================

    private suspend fun runFullImageOcrAndMap(
        sourceBitmap: Bitmap,
        detections: FloatArray
    ): List<DetectedItem> {

        val ocr =
            paddleOCR
                ?: return emptyList()

        // -------------------------------------------------
        // 1. Build YOLO object boxes
        // -------------------------------------------------

        val objectBoxes =
            parseObjectBoxes(
                detections
            )

        // -------------------------------------------------
        // 2. Run PaddleOCR ONCE on the complete image
        // -------------------------------------------------

        val ocrStart =
            System.currentTimeMillis()

        val fullOcrResult =
            ocr.recognize(
                sourceBitmap
            )

        val fullOcrElapsed =
            System.currentTimeMillis() -
                    ocrStart

        Log.i(
            OCR_TAG,
            "FULL IMAGE OCR finished: " +
                    "${fullOcrResult.results.size} raw lines, " +
                    "${fullOcrElapsed}ms"
        )

        // -------------------------------------------------
        // 3. Convert PaddleOCR quadrilateral boxes to RectF
        // -------------------------------------------------

        val ocrBoxes =
            fullOcrResult.results
                .filter {
                    it.confidence >=
                            OCR_CONFIDENCE_THRESHOLD
                }
                .mapNotNull {
                    convertOcrResultToTextBox(it)
                }

        Log.i(
            OCR_TAG,
            "OCR lines after confidence filter = " +
                    "${ocrBoxes.size}"
        )

        // -------------------------------------------------
        // 4. Map each OCR box to the most likely YOLO object
        // -------------------------------------------------

        val mappedText:
                MutableMap<Int, MutableList<OcrTextBox>> =
            mutableMapOf()

        for (
        ocrBox in
        ocrBoxes
        ) {

            val bestObject =
                findBestObjectForOcrBox(
                    ocrBox = ocrBox,
                    objectBoxes = objectBoxes
                )

            if (
                bestObject !=
                null
            ) {

                mappedText
                    .getOrPut(
                        bestObject.objectIndex
                    ) {
                        mutableListOf()
                    }
                    .add(
                        ocrBox
                    )

                Log.i(
                    OCR_MAPPING_TAG,
                    "OCR '${ocrBox.text}' -> " +
                            "Object #${bestObject.objectIndex} " +
                            "OCR box=${rectToString(ocrBox.rect)} " +
                            "Object box=${rectToString(bestObject.rect)}"
                )

            } else {

                Log.i(
                    OCR_MAPPING_TAG,
                    "OCR '${ocrBox.text}' -> UNASSIGNED " +
                            "box=${rectToString(ocrBox.rect)}"
                )
            }
        }

        // -------------------------------------------------
        // 5. Build one OCR string per YOLO object
        // -------------------------------------------------

        val items =
            mutableListOf<DetectedItem>()

        for (
        objectBox in
        objectBoxes
        ) {

            val assignedLines =
                mappedText[
                    objectBox.objectIndex
                ]
                    .orEmpty()
                    /*
                     * Approximate reading order:
                     * top -> bottom, then left -> right.
                     */
                    .sortedWith(
                        compareBy<OcrTextBox> {
                            it.rect.top
                        }.thenBy {
                            it.rect.left
                        }
                    )

            val recognizedText =
                assignedLines
                    .map {
                        it.text.trim()
                    }
                    .filter {
                        it.isNotBlank()
                    }
                    .joinToString(
                        separator = " "
                    )
                    .trim()

            val rawOcrText =
                if (
                    recognizedText.isBlank()
                ) {
                    "No mapped text"
                } else {
                    recognizedText
                }

            Log.i(
                OCR_MAPPING_TAG,
                """
Object #${objectBox.objectIndex}
Mapped OCR lines: ${assignedLines.size}
Mapped OCR text: $rawOcrText
                """.trimIndent()
            )

            val productResolution =
                if (
                    recognizedText.isNotBlank()
                ) {
                    withContext(
                        Dispatchers.Default
                    ) {
                        resolveProductHybrid(
                            recognizedText
                        )
                    }
                } else {
                    ProductResolution(
                        productName = null,
                        score = 0.0,
                        method = "NO_MAPPED_TEXT"
                    )
                }

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
                        rawOcrText,

                    ocrLineCount =
                        assignedLines.size,

                    /*
                     * This is the shared full-image OCR pass time.
                     * It is intentionally the same for every object.
                     */
                    ocrTimeMs =
                        fullOcrElapsed,

                    matchedProductName =
                        productResolution.productName,

                    productMatchScore =
                        productResolution.score * 100.0,

                    matchMethod =
                        productResolution.method
                )
            )
        }

        return items
    }

    // =====================================================
    // YOLO BOX PARSING
    // =====================================================

    private fun parseObjectBoxes(
        detections: FloatArray
    ): List<ObjectBox> {

        val boxes =
            mutableListOf<ObjectBox>()

        var index =
            0

        var objectNumber =
            1

        while (
            index + 4 <
            detections.size
        ) {

            val left =
                min(
                    detections[index],
                    detections[index + 2]
                )

            val top =
                min(
                    detections[index + 1],
                    detections[index + 3]
                )

            val right =
                max(
                    detections[index],
                    detections[index + 2]
                )

            val bottom =
                max(
                    detections[index + 1],
                    detections[index + 3]
                )

            boxes.add(
                ObjectBox(
                    objectIndex =
                        objectNumber,

                    rect =
                        RectF(
                            left,
                            top,
                            right,
                            bottom
                        ),

                    yoloConfidence =
                        detections[index + 4]
                )
            )

            index += 5
            objectNumber++
        }

        return boxes
    }

    // =====================================================
    // OCR QUADRILATERAL -> AXIS-ALIGNED RectF
    // =====================================================

    private fun convertOcrResultToTextBox(
        result: OCRResult
    ): OcrTextBox? {

        val points =
            result.box.points

        if (
            points.size !=
            4
        ) {
            return null
        }

        val left =
            points.minOf {
                it.x
            }

        val top =
            points.minOf {
                it.y
            }

        val right =
            points.maxOf {
                it.x
            }

        val bottom =
            points.maxOf {
                it.y
            }

        if (
            right <= left ||
            bottom <= top
        ) {
            return null
        }

        return OcrTextBox(
            text =
                result.text,

            confidence =
                result.confidence,

            rect =
                RectF(
                    left,
                    top,
                    right,
                    bottom
                )
        )
    }

    // =====================================================
    // OCR BOX -> OBJECT BOX ASSOCIATION
    // =====================================================

    private fun findBestObjectForOcrBox(
        ocrBox: OcrTextBox,
        objectBoxes: List<ObjectBox>
    ): ObjectBox? {

        val ocrRect =
            ocrBox.rect

        val centerX =
            ocrRect.centerX()

        val centerY =
            ocrRect.centerY()

        var bestObject:
                ObjectBox? =
            null

        var bestAssociationScore =
            -1f

        for (
        objectBox in
        objectBoxes
        ) {

            val objectRect =
                objectBox.rect

            val centerInside =
                objectRect.contains(
                    centerX,
                    centerY
                )

            val overlapRatio =
                intersectionOverOcrArea(
                    objectRect =
                        objectRect,

                    ocrRect =
                        ocrRect
                )

            val isCandidate =
                centerInside ||
                        overlapRatio >=
                        OCR_TO_OBJECT_OVERLAP_THRESHOLD

            if (
                !isCandidate
            ) {
                continue
            }

            /*
             * Center-inside is the strongest rule.
             * overlapRatio then breaks ties.
             *
             * center inside => score roughly 1.0 - 2.0
             * overlap only  => score roughly 0.5 - 1.0
             */
            val associationScore =
                if (
                    centerInside
                ) {
                    1.0f + overlapRatio
                } else {
                    overlapRatio
                }

            if (
                associationScore >
                bestAssociationScore
            ) {

                bestAssociationScore =
                    associationScore

                bestObject =
                    objectBox
            }
        }

        return bestObject
    }

    private fun intersectionOverOcrArea(
        objectRect: RectF,
        ocrRect: RectF
    ): Float {

        val intersectionLeft =
            max(
                objectRect.left,
                ocrRect.left
            )

        val intersectionTop =
            max(
                objectRect.top,
                ocrRect.top
            )

        val intersectionRight =
            min(
                objectRect.right,
                ocrRect.right
            )

        val intersectionBottom =
            min(
                objectRect.bottom,
                ocrRect.bottom
            )

        if (
            intersectionRight <=
            intersectionLeft ||
            intersectionBottom <=
            intersectionTop
        ) {
            return 0f
        }

        val intersectionArea =
            (
                    intersectionRight -
                            intersectionLeft
                    ) *
                    (
                            intersectionBottom -
                                    intersectionTop
                            )

        val ocrArea =
            max(
                1f,
                ocrRect.width() *
                        ocrRect.height()
            )

        return (
                intersectionArea /
                        ocrArea
                )
            .coerceIn(
                0f,
                1f
            )
    }

    private fun rectToString(
        rect: RectF
    ): String {
        return String.format(
            Locale.US,
            "[%.1f, %.1f, %.1f, %.1f]",
            rect.left,
            rect.top,
            rect.right,
            rect.bottom
        )
    }

    // =====================================================
    // HYBRID PRODUCT RESOLVER
    // =====================================================

    private fun resolveProductHybrid(
        ocrText: String
    ): ProductResolution {

        val embeddingTop =
            textEmbeddingMatcher
                .findTopMatches(
                    text = ocrText,
                    topK = TOP_K
                )

        val fuzzyTop =
            productMatcher
                .findTopMatches(
                    ocrText = ocrText,
                    topK = TOP_K
                )

        if (
            embeddingTop.isEmpty() &&
            fuzzyTop.isEmpty()
        ) {
            return ProductResolution(
                productName = null,
                score = 0.0,
                method = "UNKNOWN"
            )
        }

        val candidates =
            LinkedHashMap<
                    String,
                    HybridCandidate
                    >()

        embeddingTop.forEach { match ->

            val key =
                canonicalKey(
                    match.productName
                )

            val candidate =
                candidates.getOrPut(
                    key
                ) {
                    HybridCandidate(
                        displayName =
                            match.productName,

                        key =
                            key
                    )
                }

            candidate.embeddingScore =
                maxOf(
                    candidate.embeddingScore,
                    match.score
                        .toDouble()
                        .coerceIn(
                            0.0,
                            1.0
                        )
                )

            candidate.appearsInEmbedding = true
        }

        fuzzyTop.forEach { match ->

            val key =
                canonicalKey(
                    match.product.name
                )

            val candidate =
                candidates.getOrPut(
                    key
                ) {
                    HybridCandidate(
                        displayName =
                            match.product.name,

                        key =
                            key
                    )
                }

            candidate.fuzzyScore =
                maxOf(
                    candidate.fuzzyScore,
                    (
                            match.score /
                                    100.0
                            )
                        .coerceIn(
                            0.0,
                            1.0
                        )
                )

            candidate.appearsInFuzzy = true
        }

        val ranked =
            candidates.values
                .sortedByDescending {
                    it.combinedScore()
                }

        val best =
            ranked.firstOrNull()
                ?: return ProductResolution(
                    productName = null,
                    score = 0.0,
                    method = "UNKNOWN"
                )

        val finalScore =
            best.combinedScore()

        val strongFuzzyOverride =
            best.fuzzyScore >=
                    FUZZY_STRONG_OVERRIDE

        val bothAgree =
            best.appearsInEmbedding &&
                    best.appearsInFuzzy

        val accepted =
            when {
                bothAgree &&
                        finalScore >=
                        FINAL_AGREEMENT_THRESHOLD ->
                    true

                finalScore >=
                        FINAL_STRONG_THRESHOLD ->
                    true

                strongFuzzyOverride ->
                    true

                else ->
                    false
            }

        val method =
            when {
                !accepted ->
                    "UNKNOWN"

                bothAgree ->
                    "HYBRID_AGREEMENT"

                strongFuzzyOverride ->
                    "FUZZY_STRONG"

                else ->
                    "HYBRID_STRONG"
            }

        Log.i(
            HYBRID_TAG,
            buildString {
                appendLine("====================================")
                appendLine("OCR:")
                appendLine(ocrText)
                appendLine()
                appendLine("HYBRID CANDIDATES:")

                ranked
                    .take(8)
                    .forEachIndexed {
                            index,
                            candidate ->

                        appendLine(
                            "${index + 1}. " +
                                    candidate.displayName
                        )

                        appendLine(
                            "   embedding=" +
                                    String.format(
                                        Locale.US,
                                        "%.4f",
                                        candidate.embeddingScore
                                    ) +
                                    " fuzzy=" +
                                    String.format(
                                        Locale.US,
                                        "%.4f",
                                        candidate.fuzzyScore
                                    ) +
                                    " both=" +
                                    (
                                            candidate.appearsInEmbedding &&
                                                    candidate.appearsInFuzzy
                                            ) +
                                    " final=" +
                                    String.format(
                                        Locale.US,
                                        "%.4f",
                                        candidate.combinedScore()
                                    )
                        )
                    }

                appendLine()
                appendLine(
                    "FINAL: " +
                            if (accepted) {
                                best.displayName
                            } else {
                                "Unknown"
                            }
                )

                appendLine(
                    "FINAL SCORE: " +
                            String.format(
                                Locale.US,
                                "%.4f",
                                finalScore
                            )
                )

                appendLine(
                    "METHOD: $method"
                )

                append("====================================")
            }
        )

        if (!accepted) {
            return ProductResolution(
                productName = null,
                score = finalScore,
                method = "UNKNOWN"
            )
        }

        return ProductResolution(
            productName = best.displayName,
            score = finalScore,
            method = method
        )
    }

    // =====================================================
    // CANONICAL PRODUCT KEY
    // =====================================================

    private fun canonicalKey(
        value: String
    ): String {

        return value
            .lowercase(Locale.ROOT)
            .replace(
                "&",
                "and"
            )
            .replace(
                "'",
                ""
            )
            .replace(
                "’",
                ""
            )
            .replace(
                "-",
                ""
            )
            .replace(
                Regex("[^a-z0-9]"),
                ""
            )
    }

    // =====================================================
    // DISPLAY
    // =====================================================

    private fun displayResults(
        detectedItems: List<DetectedItem>
    ) {

        val builder =
            StringBuilder()

        builder.append(
            "Detected Objects: " +
                    "${detectedItems.size}\n\n"
        )

        detectedItems.forEach { item ->

            builder.append(
                "Object #${item.objectIndex}\n"
            )

            builder.append(
                "Product: " +
                        "${item.matchedProductName ?: "Unknown"}\n"
            )

            if (
                item.matchedProductName !=
                null
            ) {
                builder.append(
                    "Match: " +
                            String.format(
                                Locale.US,
                                "%.1f",
                                item.productMatchScore
                            ) +
                            "%\n"
                )
            }

            builder.append(
                "Method: ${item.matchMethod}\n"
            )

            builder.append(
                "Mapped OCR (${item.ocrLineCount} lines): " +
                        "${item.rawOcrText}\n"
            )

            builder.append(
                "YOLO: " +
                        String.format(
                            Locale.US,
                            "%.2f",
                            item.yoloConfidence
                        ) +
                        "\n"
            )

            builder.append(
                "Full-image OCR time: " +
                        "${item.ocrTimeMs} ms\n"
            )

            builder.append(
                "────────────────────\n\n"
            )
        }

        resultsText.text =
            builder.toString()
    }

    private fun logFinalResults(
        detectedItems: List<DetectedItem>
    ) {

        Log.i(
            HYBRID_TAG,
            "======================================"
        )

        Log.i(
            HYBRID_TAG,
            "FINAL PRODUCT RESULTS"
        )

        detectedItems.forEach { item ->

            Log.i(
                HYBRID_TAG,
                """
Object #${item.objectIndex}
Mapped OCR: ${item.rawOcrText}
Mapped lines: ${item.ocrLineCount}
Matched: ${item.matchedProductName ?: "Unknown"}
Score: ${String.format(Locale.US, "%.2f", item.productMatchScore)}%
Method: ${item.matchMethod}
YOLO: ${item.yoloConfidence}
--------------------------------------
                """.trimIndent()
            )
        }
    }

    // =====================================================
    // RESIZE
    // =====================================================

    private fun resizeBitmapForInference(
        bitmap: Bitmap,
        maxSize: Int
    ): Bitmap {

        val width =
            bitmap.width

        val height =
            bitmap.height

        if (
            width <= maxSize &&
            height <= maxSize
        ) {
            return bitmap
        }

        val scale =
            maxSize.toFloat() /
                    maxOf(
                        width,
                        height
                    )

        val newWidth =
            maxOf(
                1,
                (
                        width *
                                scale
                        ).toInt()
            )

        val newHeight =
            maxOf(
                1,
                (
                        height *
                                scale
                        ).toInt()
            )

        return Bitmap.createScaledBitmap(
            bitmap,
            newWidth,
            newHeight,
            true
        )
    }

    // =====================================================
    // DRAW YOLO BOXES
    // =====================================================

    private fun drawBoundingBoxes(
        source: Bitmap,
        detections: FloatArray
    ): Bitmap {

        val output =
            source.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        val canvas =
            Canvas(output)

        val paint =
            Paint().apply {
                style =
                    Paint.Style.STROKE

                strokeWidth =
                    maxOf(
                        4f,
                        source.width /
                                250f
                    )

                isAntiAlias = true
            }

        var index = 0

        while (
            index + 4 <
            detections.size
        ) {

            canvas.drawRect(
                detections[index],
                detections[index + 1],
                detections[index + 2],
                detections[index + 3],
                paint
            )

            index += 5
        }

        return output
    }

    // =====================================================
    // RELEASE
    // =====================================================

    override fun onDestroy() {

        if (
            ::textEmbeddingMatcher
                .isInitialized
        ) {
            try {
                textEmbeddingMatcher.close()
            } catch (exception: Exception) {
                Log.e(
                    EMBEDDING_TAG,
                    "Failed closing MiniLM",
                    exception
                )
            }
        }

        val ocr =
            paddleOCR

        if (ocr != null) {
            lifecycleScope.launch {
                try {
                    ocr.release()
                } catch (exception: Exception) {
                    Log.e(
                        OCR_TAG,
                        "Failed releasing PaddleOCR",
                        exception
                    )
                }
            }
        }

        super.onDestroy()
    }
}
