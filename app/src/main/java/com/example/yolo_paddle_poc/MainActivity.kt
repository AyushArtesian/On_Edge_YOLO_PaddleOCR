package com.example.yolo_paddle_poc

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.net.Uri
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
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
import java.io.File
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
        private const val OCR_CONFIDENCE_THRESHOLD = 0.20f

        // Hybrid OCR settings.
        // Pass 1: run OCR once on the complete image and map OCR boxes to YOLO boxes.
        // Pass 2: only for objects that are not confidently identified from pass 1,
        // crop the YOLO object with padding, upscale it, and run OCR again.
        private const val OCR_CROP_PADDING_PERCENT = 0.12f
        private const val OCR_CROP_SCALE = 3.0f
        private const val OCR_TO_OBJECT_OVERLAP_THRESHOLD = 0.50f

        // Prominent product-title selection.
        private const val PROMINENT_MAX_LINES = 4
        private const val PROMINENT_MIN_RELATIVE_HEIGHT = 0.025f

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

    private data class ScoredOcrLine(
        val box: OcrTextBox,
        val score: Double,
        val relativeHeight: Float,
        val relativeArea: Float,
        val uppercaseRatio: Double,
        val isNoise: Boolean
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
    private lateinit var captureImageButton: Button
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
    // IMAGE INPUT: GALLERY + CAMERA
    // =====================================================

    private var pendingCameraUri: Uri? = null

    /**
     * Gallery picker.
     *
     * Both gallery images and camera images are sent through the exact same
     * processImageUri() pipeline so YOLO/OCR/matching behavior stays identical.
     */
    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->

            if (uri != null) {
                processImageUri(uri)
            }
        }

    /**
     * Opens the device camera and stores a full-resolution image in our
     * app cache through FileProvider.
     *
     * TakePicture() is preferred over the thumbnail camera contract because
     * OCR benefits heavily from the full-resolution camera image.
     */
    private val cameraLauncher =
        registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { success ->

            val uri = pendingCameraUri
            pendingCameraUri = null

            if (success && uri != null) {

                Log.i(
                    TAG,
                    "Camera image captured successfully: $uri"
                )

                processImageUri(uri)

            } else {

                Log.i(
                    TAG,
                    "Camera capture cancelled or failed"
                )
            }
        }

    /**
     * Creates a temporary full-resolution camera destination.
     *
     * The matching FileProvider path is defined in:
     * res/xml/file_paths.xml
     */
    private fun createCameraImageUri(): Uri {

        val cameraDirectory =
            File(
                cacheDir,
                "camera_images"
            )

        if (!cameraDirectory.exists()) {
            cameraDirectory.mkdirs()
        }

        val imageFile =
            File.createTempFile(
                "product_${System.currentTimeMillis()}_",
                ".jpg",
                cameraDirectory
            )

        return FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            imageFile
        )
    }

    private fun setActionButtonsEnabled(
        enabled: Boolean
    ) {
        selectImageButton.isEnabled = enabled
        captureImageButton.isEnabled = enabled
    }

    /**
     * Common image-processing pipeline used by BOTH:
     *
     * 1. Select Image
     * 2. Capture Image
     *
     * Flow:
     * image -> YOLO -> hybrid OCR -> MiniLM/fuzzy -> duplicate grouping
     */
    private fun processImageUri(
        uri: Uri
    ) {

        if (!yoloReady) {
            statusText.text =
                "YOLO model is still loading"
            return
        }

        if (!ocrReady || paddleOCR == null) {
            statusText.text =
                "PaddleOCR is still loading"
            return
        }

        if (!embeddingReady) {
            statusText.text =
                "MiniLM matcher is still loading"
            return
        }

        lifecycleScope.launch {
            try {

                setActionButtonsEnabled(false)

                statusText.text =
                    "Loading image..."

                resultsText.text =
                    "Processing..."

                // =================================================
                // LOAD IMAGE
                // =================================================

                val decodedBitmap =
                    withContext(Dispatchers.IO) {
                        contentResolver
                            .openInputStream(uri)
                            ?.use { inputStream ->
                                BitmapFactory.decodeStream(
                                    inputStream
                                )
                            }
                    }

                if (decodedBitmap == null) {

                    statusText.text =
                        "Unable to load image"

                    resultsText.text =
                        "Image loading failed."

                    setActionButtonsEnabled(true)
                    return@launch
                }

                val rgbaBitmap =
                    decodedBitmap.copy(
                        Bitmap.Config.ARGB_8888,
                        false
                    )

                if (rgbaBitmap == null) {

                    statusText.text =
                        "Unable to convert image"

                    resultsText.text =
                        "Image conversion failed."

                    setActionButtonsEnabled(true)
                    return@launch
                }

                if (decodedBitmap !== rgbaBitmap) {
                    decodedBitmap.recycle()
                }

                /*
                 * YOLO and the first OCR pass both use inferenceBitmap,
                 * so full-image OCR coordinates map directly to YOLO boxes.
                 *
                 * Large camera photos are reduced to MAX_IMAGE_SIZE before
                 * inference to keep memory/latency under control.
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

                statusText.text =
                    "Detecting objects..."

                val yoloStart =
                    System.currentTimeMillis()

                val detections =
                    withContext(
                        Dispatchers.Default
                    ) {
                        detectObjects(
                            inferenceBitmap
                        )
                    }

                val yoloElapsed =
                    System.currentTimeMillis() -
                            yoloStart

                val detectionCount =
                    detections.size / 5

                Log.i(
                    TAG,
                    "YOLO detected $detectionCount objects " +
                            "in ${yoloElapsed}ms"
                )

                if (detectionCount == 0) {

                    imageView.setImageBitmap(
                        inferenceBitmap
                    )

                    statusText.text =
                        "No objects detected"

                    resultsText.text =
                        "No objects detected."

                    setActionButtonsEnabled(true)
                    return@launch
                }

                // =================================================
                // HYBRID OCR
                //
                // Pass 1:
                // Full-image OCR + coordinate mapping.
                //
                // Pass 2:
                // Padded/upscaled per-object crop OCR only when
                // full-image OCR cannot confidently identify it.
                // =================================================

                statusText.text =
                    "Running hybrid OCR..."

                val pipelineStart =
                    System.currentTimeMillis()

                val detectedItems =
                    runHybridOcr(
                        sourceBitmap =
                            inferenceBitmap,

                        detections =
                            detections
                    )

                val pipelineElapsed =
                    System.currentTimeMillis() -
                            pipelineStart

                // =================================================
                // DRAW NUMBERED YOLO BOXES
                // =================================================

                val resultBitmap =
                    drawBoundingBoxes(
                        inferenceBitmap,
                        detections
                    )

                imageView.setImageBitmap(
                    resultBitmap
                )

                // =================================================
                // DISPLAY / LOG
                // =================================================

                displayResults(
                    detectedItems
                )

                logFinalResults(
                    detectedItems
                )

                statusText.text =
                    "$detectionCount objects • " +
                            "YOLO ${yoloElapsed}ms • " +
                            "Hybrid OCR+Match ${pipelineElapsed}ms"

                setActionButtonsEnabled(true)

            } catch (exception: Exception) {

                Log.e(
                    TAG,
                    "Processing failed",
                    exception
                )

                statusText.text =
                    "Processing failed"

                resultsText.text =
                    "Something went wrong."

                setActionButtonsEnabled(true)
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

        captureImageButton =
            findViewById(R.id.captureImageButton)

        imageView =
            findViewById(R.id.imageView)

        statusText =
            findViewById(R.id.statusText)

        resultsText =
            findViewById(R.id.resultsText)

        setActionButtonsEnabled(false)
        resultsText.text = "Waiting for AI models..."

        productMatcher =
            ProductMatcher(this)

        selectImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        captureImageButton.setOnClickListener {
            try {

                val uri =
                    createCameraImageUri()

                pendingCameraUri =
                    uri

                cameraLauncher.launch(
                    uri
                )

            } catch (exception: Exception) {

                Log.e(
                    TAG,
                    "Unable to open camera",
                    exception
                )

                pendingCameraUri =
                    null

                statusText.text =
                    "Unable to open camera"
            }
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

                setActionButtonsEnabled(false)
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

                    setActionButtonsEnabled(false)
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
                    setActionButtonsEnabled(false)
                    return@launch
                }

                paddleOCR =
                    PaddleOCR.create(
                        context =
                            this@MainActivity,

                        config =
                            PaddleOCRConfig(
                                detThresh = 0.2f,
                                detBoxThresh = 0.4f,
                                recScoreThresh =
                                    0.20f,
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

                setActionButtonsEnabled(false)
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
                "YOLO + Hybrid OCR + Hybrid Matcher Ready"

            resultsText.text =
                "Select an image or capture a photo."

            setActionButtonsEnabled(true)

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

            setActionButtonsEnabled(false)
        }
    }

    // =====================================================
    // HYBRID OCR
    // =====================================================

    private suspend fun runHybridOcr(
        sourceBitmap: Bitmap,
        detections: FloatArray
    ): List<DetectedItem> {

        val ocr =
            paddleOCR
                ?: return emptyList()

        val objectBoxes =
            parseObjectBoxes(
                detections
            )

        // -------------------------------------------------
        // PASS 1: FULL-IMAGE OCR ONCE
        // -------------------------------------------------

        val fullOcrStart =
            System.currentTimeMillis()

        val fullOcrResult =
            ocr.recognize(
                sourceBitmap
            )

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
                    convertOcrResultToTextBox(it)
                }

        Log.i(
            OCR_TAG,
            "FULL IMAGE OCR finished: " +
                    "${fullOcrResult.results.size} raw lines, " +
                    "${fullImageOcrBoxes.size} accepted lines, " +
                    "${fullOcrElapsed}ms"
        )

        // -------------------------------------------------
        // Map full-image OCR boxes to YOLO objects.
        // -------------------------------------------------

        val mappedFullImageText:
                MutableMap<Int, MutableList<OcrTextBox>> =
            mutableMapOf()

        for (ocrBox in fullImageOcrBoxes) {

            val bestObject =
                findBestObjectForOcrBox(
                    ocrBox = ocrBox,
                    objectBoxes = objectBoxes
                )

            if (bestObject != null) {

                mappedFullImageText
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
                    "FULL OCR '${ocrBox.text}' -> " +
                            "Object #${bestObject.objectIndex} " +
                            "OCR box=${rectToString(ocrBox.rect)} " +
                            "Object box=${rectToString(bestObject.rect)}"
                )

            } else {

                Log.i(
                    OCR_MAPPING_TAG,
                    "FULL OCR '${ocrBox.text}' -> UNASSIGNED " +
                            "box=${rectToString(ocrBox.rect)}"
                )
            }
        }

        // -------------------------------------------------
        // Resolve each object.
        // Full-image OCR is tried first. If it cannot produce
        // an accepted product match, crop OCR is used as fallback.
        // -------------------------------------------------

        val items =
            mutableListOf<DetectedItem>()

        for (objectBox in objectBoxes) {

            val mappedLines =
                mappedFullImageText[
                    objectBox.objectIndex
                ]
                    .orEmpty()
                    .sortedWith(
                        compareBy<OcrTextBox> {
                            it.rect.top
                        }.thenBy {
                            it.rect.left
                        }
                    )

            val fullImageRawText =
                mappedLines
                    .map { it.text.trim() }
                    .filter { it.isNotBlank() }
                    .joinToString(" ")
                    .trim()

            val prominentFullImageLines =
                selectProminentOcrLines(
                    lines = mappedLines,
                    referenceWidth = max(1f, objectBox.rect.width()),
                    referenceHeight = max(1f, objectBox.rect.height()),
                    sourceLabel = "FULL_IMAGE Object #${objectBox.objectIndex}"
                )

            val fullImageText =
                prominentFullImageLines
                    .joinToString(" ") { it.text.trim() }
                    .trim()
                    .ifBlank { fullImageRawText }

            val fullImageResolution =
                if (fullImageText.isNotBlank()) {
                    withContext(
                        Dispatchers.Default
                    ) {
                        resolveProductHybrid(
                            fullImageText
                        )
                    }
                } else {
                    ProductResolution(
                        productName = null,
                        score = 0.0,
                        method = "NO_FULL_IMAGE_TEXT"
                    )
                }

            // If full-image OCR already gives an accepted product,
            // do not crop. This preserves large/global text and saves time.
            if (fullImageResolution.productName != null) {

                Log.i(
                    OCR_MAPPING_TAG,
                    """
Object #${objectBox.objectIndex}
OCR source: FULL_IMAGE
Mapped OCR lines: ${mappedLines.size}
Raw OCR text: ${if (fullImageRawText.isBlank()) "<none>" else fullImageRawText}
Prominent OCR lines: ${prominentFullImageLines.size}
Prominent OCR text: ${if (fullImageText.isBlank()) "<none>" else fullImageText}
Matched: ${fullImageResolution.productName}
Score: ${String.format(Locale.US, "%.4f", fullImageResolution.score)}
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
                            fullImageText,

                        ocrLineCount =
                            prominentFullImageLines.size,

                        ocrTimeMs =
                            fullOcrElapsed,

                        matchedProductName =
                            fullImageResolution.productName,

                        productMatchScore =
                            fullImageResolution.score * 100.0,

                        matchMethod =
                            "FULL_IMAGE_${fullImageResolution.method}"
                    )
                )

                continue
            }

            // -------------------------------------------------
            // PASS 2: CROP OCR FALLBACK
            // -------------------------------------------------

            val cropStart =
                System.currentTimeMillis()

            var cropBitmap: Bitmap? = null
            var enlargedCrop: Bitmap? = null

            try {

                val objectCrop =
                    cropBitmapWithPadding(
                        source = sourceBitmap,
                        rect = objectBox.rect,
                        paddingPercent =
                            OCR_CROP_PADDING_PERCENT
                    )

                cropBitmap =
                    objectCrop

                val ocrInput =
                    upscaleBitmapForOcr(
                        bitmap = objectCrop,
                        scaleFactor = OCR_CROP_SCALE
                    )

                enlargedCrop =
                    ocrInput

                Log.i(
                    OCR_TAG,
                    "Object #${objectBox.objectIndex}: " +
                            "full-image match failed -> crop fallback; " +
                            "box=${rectToString(objectBox.rect)}, " +
                            "crop=${objectCrop.width}x${objectCrop.height}, " +
                            "OCR input=${ocrInput.width}x${ocrInput.height}"
                )

                val cropOcrResult =
                    ocr.recognize(
                        ocrInput
                    )

                val cropAcceptedBoxes =
                    cropOcrResult.results
                        .filter {
                            it.confidence >= OCR_CONFIDENCE_THRESHOLD
                        }
                        .mapNotNull {
                            convertOcrResultToTextBox(it)
                        }
                        .filter {
                            it.text.isNotBlank()
                        }

                val cropRawText =
                    cropAcceptedBoxes
                        .sortedWith(
                            compareBy<OcrTextBox> { it.rect.top }
                                .thenBy { it.rect.left }
                        )
                        .joinToString(" ") { it.text.trim() }
                        .trim()

                val prominentCropLines =
                    selectProminentOcrLines(
                        lines = cropAcceptedBoxes,
                        referenceWidth = max(1f, ocrInput.width.toFloat()),
                        referenceHeight = max(1f, ocrInput.height.toFloat()),
                        sourceLabel = "CROP Object #${objectBox.objectIndex}"
                    )

                val cropText =
                    prominentCropLines
                        .joinToString(" ") { it.text.trim() }
                        .trim()
                        .ifBlank { cropRawText }

                val cropResolution =
                    if (cropText.isNotBlank()) {
                        withContext(
                            Dispatchers.Default
                        ) {
                            resolveProductHybrid(
                                cropText
                            )
                        }
                    } else {
                        ProductResolution(
                            productName = null,
                            score = 0.0,
                            method = "NO_CROP_TEXT"
                        )
                    }

                val cropElapsed =
                    System.currentTimeMillis() -
                            cropStart

                // Prefer a successful crop match. If crop matching also fails,
                // retain whichever OCR text had the stronger matcher score for
                // diagnostics, while still marking the product as Unknown.
                val cropAccepted =
                    cropResolution.productName != null

                val useCropDiagnostics =
                    cropAccepted ||
                            cropResolution.score >=
                            fullImageResolution.score

                val selectedText =
                    if (useCropDiagnostics) {
                        if (cropText.isBlank()) {
                            "No OCR text"
                        } else {
                            cropText
                        }
                    } else {
                        if (fullImageText.isBlank()) {
                            "No OCR text"
                        } else {
                            fullImageText
                        }
                    }

                val selectedLineCount =
                    if (useCropDiagnostics) {
                        prominentCropLines.size
                    } else {
                        prominentFullImageLines.size
                    }

                val selectedResolution =
                    if (cropAccepted) {
                        cropResolution
                    } else if (
                        fullImageResolution.score >
                        cropResolution.score
                    ) {
                        fullImageResolution
                    } else {
                        cropResolution
                    }

                val selectedMethod =
                    if (cropAccepted) {
                        "CROP_FALLBACK_${cropResolution.method}"
                    } else {
                        "HYBRID_UNKNOWN"
                    }

                Log.i(
                    OCR_MAPPING_TAG,
                    """
Object #${objectBox.objectIndex}
OCR source: CROP_FALLBACK
Full-image raw text: ${if (fullImageRawText.isBlank()) "<none>" else fullImageRawText}
Full-image prominent text: ${if (fullImageText.isBlank()) "<none>" else fullImageText}
Crop OCR raw lines: ${cropOcrResult.results.size}
Crop accepted boxes: ${cropAcceptedBoxes.size}
Crop raw text: ${if (cropRawText.isBlank()) "<none>" else cropRawText}
Crop prominent lines: ${prominentCropLines.size}
Crop prominent text: ${if (cropText.isBlank()) "<none>" else cropText}
Final matched: ${selectedResolution.productName ?: "Unknown"}
Final score: ${String.format(Locale.US, "%.4f", selectedResolution.score)}
Crop OCR time: ${cropElapsed}ms
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

                        // Full-image OCR is shared by every object. For fallback
                        // objects, report full pass + this object's crop pass.
                        ocrTimeMs =
                            fullOcrElapsed + cropElapsed,

                        matchedProductName =
                            selectedResolution.productName,

                        productMatchScore =
                            selectedResolution.score * 100.0,

                        matchMethod =
                            selectedMethod
                    )
                )

            } catch (exception: Exception) {

                Log.e(
                    OCR_TAG,
                    "Crop OCR fallback failed for Object #${objectBox.objectIndex}",
                    exception
                )

                val fallbackText =
                    if (fullImageText.isBlank()) {
                        "OCR failed"
                    } else {
                        fullImageText
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
                            fallbackText,

                        ocrLineCount =
                            prominentFullImageLines.size,

                        ocrTimeMs =
                            fullOcrElapsed,

                        matchedProductName =
                            null,

                        productMatchScore =
                            fullImageResolution.score * 100.0,

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

    // =====================================================
    // PROMINENT / TITLE-LIKE OCR TEXT SELECTION
    // =====================================================

    private fun selectProminentOcrLines(
        lines: List<OcrTextBox>,
        referenceWidth: Float,
        referenceHeight: Float,
        sourceLabel: String
    ): List<OcrTextBox> {

        if (lines.isEmpty()) {
            return emptyList()
        }

        val safeWidth = max(1f, referenceWidth)
        val safeHeight = max(1f, referenceHeight)
        val safeArea = safeWidth * safeHeight

        val scored =
            lines.mapNotNull { box ->

                val cleanedText =
                    box.text
                        .trim()
                        .replace(Regex("\\s+"), " ")

                if (cleanedText.isBlank()) {
                    return@mapNotNull null
                }

                val relativeHeight =
                    (box.rect.height() / safeHeight)
                        .coerceIn(0f, 1f)

                val relativeArea =
                    ((box.rect.width() * box.rect.height()) / safeArea)
                        .coerceIn(0f, 1f)

                val uppercaseRatio =
                    calculateUppercaseRatio(cleanedText)

                val titleLikeBonus =
                    calculateTitleLikeBonus(cleanedText)

                val noise =
                    isPackagingNoiseText(cleanedText)

                var score =
                    0.52 * relativeHeight.toDouble() +
                            0.18 * relativeArea.toDouble() +
                            0.12 * uppercaseRatio +
                            0.10 * titleLikeBonus +
                            0.08 * box.confidence.toDouble().coerceIn(0.0, 1.0)

                if (noise) score -= 0.45
                if (looksLikeMostlyNumbers(cleanedText)) score -= 0.30
                if (cleanedText.length <= 1) score -= 0.25

                ScoredOcrLine(
                    box = box.copy(text = cleanedText),
                    score = score,
                    relativeHeight = relativeHeight,
                    relativeArea = relativeArea,
                    uppercaseRatio = uppercaseRatio,
                    isNoise = noise
                )
            }

        if (scored.isEmpty()) {
            return emptyList()
        }

        val largestRelativeHeight =
            scored.maxOfOrNull { it.relativeHeight } ?: 0f

        val dynamicHeightFloor =
            max(
                PROMINENT_MIN_RELATIVE_HEIGHT,
                largestRelativeHeight * 0.42f
            )

        var selected =
            scored
                .filter {
                    !it.isNoise &&
                            it.relativeHeight >= dynamicHeightFloor
                }
                .sortedByDescending { it.score }
                .take(PROMINENT_MAX_LINES)

        if (selected.isEmpty()) {
            selected =
                scored
                    .filter { !it.isNoise }
                    .sortedByDescending { it.score }
                    .take(min(2, PROMINENT_MAX_LINES))
        }

        val readingOrder =
            selected
                .map { it.box }
                .sortedWith(
                    compareBy<OcrTextBox> { it.rect.top }
                        .thenBy { it.rect.left }
                )

        Log.i(
            OCR_TAG,
            buildString {
                appendLine("PROMINENT OCR SELECTION [$sourceLabel]")

                scored
                    .sortedByDescending { it.score }
                    .forEach { item ->

                        val keep =
                            readingOrder.any {
                                it.text == item.box.text &&
                                        it.rect == item.box.rect
                            }

                        appendLine(
                            "${if (keep) "[KEEP]" else "[DROP]"} " +
                                    "'${item.box.text}' " +
                                    "score=${String.format(Locale.US, "%.3f", item.score)} " +
                                    "h=${String.format(Locale.US, "%.3f", item.relativeHeight)} " +
                                    "area=${String.format(Locale.US, "%.3f", item.relativeArea)} " +
                                    "upper=${String.format(Locale.US, "%.2f", item.uppercaseRatio)} " +
                                    "conf=${String.format(Locale.US, "%.2f", item.box.confidence)} " +
                                    "noise=${item.isNoise}"
                        )
                    }

                append(
                    "SELECTED TEXT: " +
                            if (readingOrder.isEmpty()) {
                                "<none>"
                            } else {
                                readingOrder.joinToString(" ") { it.text }
                            }
                )
            }
        )

        return readingOrder
    }

    private fun calculateUppercaseRatio(
        text: String
    ): Double {

        val letters =
            text.filter { it.isLetter() }

        if (letters.isEmpty()) {
            return 0.0
        }

        return letters.count { it.isUpperCase() }
            .toDouble() /
                letters.length.toDouble()
    }

    private fun calculateTitleLikeBonus(
        text: String
    ): Double {

        val words =
            text.split(Regex("\\s+"))
                .filter { word ->
                    word.any { it.isLetter() }
                }

        if (words.isEmpty()) {
            return 0.0
        }

        val titleLikeWords =
            words.count { word ->
                word.firstOrNull { it.isLetter() }
                    ?.isUpperCase() == true
            }

        return (
                titleLikeWords.toDouble() /
                        words.size.toDouble()
                ).coerceIn(0.0, 1.0)
    }

    private fun isPackagingNoiseText(
        text: String
    ): Boolean {

        val normalized =
            text.lowercase(Locale.ROOT)
                .replace(Regex("[^a-z0-9₹%./ ]"), " ")
                .replace(Regex("\\s+"), " ")
                .trim()

        if (normalized.isBlank()) {
            return true
        }

        val noisePhrases =
            listOf(
                "mrp",
                "m.r.p",
                "net qty",
                "net quantity",
                "net weight",
                "ingredients",
                "ingredient",
                "nutrition",
                "nutritional",
                "manufactured by",
                "manufactured",
                "marketed by",
                "packed by",
                "customer care",
                "consumer care",
                "batch",
                "batch no",
                "lot no",
                "mfg",
                "mfd",
                "expiry",
                "exp date",
                "best before",
                "use before",
                "fssai",
                "license no",
                "lic no",
                "email",
                "www.",
                "http",
                "telephone",
                "phone"
            )

        if (noisePhrases.any { normalized.contains(it) }) {
            return true
        }

        val quantityOrPrice =
            Regex(
                """^\s*(?:₹|rs\.?\s*)?\d+(?:[.,]\d+)?\s*(?:g|gm|gms|kg|ml|l|ltr|litre|litres|oz|lb|pcs|pc)?\.?\s*$""",
                RegexOption.IGNORE_CASE
            )

        if (quantityOrPrice.matches(normalized)) {
            return true
        }

        val digits =
            normalized.count { it.isDigit() }

        val letters =
            normalized.count { it.isLetter() }

        if (digits >= 6 && digits > letters * 2) {
            return true
        }

        return false
    }

    private fun looksLikeMostlyNumbers(
        text: String
    ): Boolean {

        val alphaNumeric =
            text.filter { it.isLetterOrDigit() }

        if (alphaNumeric.isEmpty()) {
            return true
        }

        val digitCount =
            alphaNumeric.count { it.isDigit() }

        return digitCount.toDouble() /
                alphaNumeric.length.toDouble() >= 0.70
    }

    // =====================================================
    // OCR CROP HELPERS
    // =====================================================

    private fun cropBitmapWithPadding(
        source: Bitmap,
        rect: RectF,
        paddingPercent: Float
    ): Bitmap {

        val boxWidth =
            max(
                1f,
                rect.width()
            )

        val boxHeight =
            max(
                1f,
                rect.height()
            )

        val paddingX =
            boxWidth *
                    paddingPercent

        val paddingY =
            boxHeight *
                    paddingPercent

        val left =
            (rect.left - paddingX)
                .toInt()
                .coerceIn(
                    0,
                    source.width - 1
                )

        val top =
            (rect.top - paddingY)
                .toInt()
                .coerceIn(
                    0,
                    source.height - 1
                )

        val right =
            (rect.right + paddingX)
                .toInt()
                .coerceIn(
                    left + 1,
                    source.width
                )

        val bottom =
            (rect.bottom + paddingY)
                .toInt()
                .coerceIn(
                    top + 1,
                    source.height
                )

        return Bitmap.createBitmap(
            source,
            left,
            top,
            right - left,
            bottom - top
        )
    }

    private fun upscaleBitmapForOcr(
        bitmap: Bitmap,
        scaleFactor: Float
    ): Bitmap {

        if (scaleFactor <= 1f) {
            return bitmap
        }

        val newWidth =
            max(
                1,
                (bitmap.width * scaleFactor)
                    .toInt()
            )

        val newHeight =
            max(
                1,
                (bitmap.height * scaleFactor)
                    .toInt()
            )

        return Bitmap.createScaledBitmap(
            bitmap,
            newWidth,
            newHeight,
            true
        )
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

        // =================================================
        // GROUP SAME PRODUCTS TOGETHER
        //
        // Example:
        // #1 -> Maggi
        // #2 -> Maggi
        // #3 -> Maggi
        // #4 -> Parle-G
        //
        // Display becomes:
        // Maggi   -> Quantity 3 -> Objects #1, #2, #3
        // Parle-G -> Quantity 1 -> Object  #4
        // =================================================

        val groupedProducts =
            detectedItems
                .filter {
                    !it.matchedProductName.isNullOrBlank()
                }
                .groupBy {
                    canonicalKey(
                        it.matchedProductName!!
                    )
                }

        val unknownItems =
            detectedItems.filter {
                it.matchedProductName.isNullOrBlank()
            }

        builder.append(
            "Detected Objects: ${detectedItems.size}\n"
        )

        builder.append(
            "Unique Products: ${groupedProducts.size}\n\n"
        )

        // =================================================
        // DISPLAY UNIQUE PRODUCTS
        // =================================================

        groupedProducts.values.forEachIndexed {
                index,
                items ->

            // If the same product is detected multiple times, use the
            // highest-confidence product match as the representative row.
            val bestItem =
                items.maxByOrNull {
                    it.productMatchScore
                } ?: return@forEachIndexed

            val productName =
                bestItem.matchedProductName
                    ?: "Unknown"

            val objectNumbers =
                items
                    .sortedBy {
                        it.objectIndex
                    }
                    .joinToString(", ") {
                        "#${it.objectIndex}"
                    }

            val bestYoloConfidence =
                items.maxOfOrNull {
                    it.yoloConfidence
                } ?: 0f

            val totalMappedLines =
                items.sumOf {
                    it.ocrLineCount
                }

            builder.append(
                "${index + 1}. $productName\n"
            )

            builder.append(
                "Quantity: ${items.size}\n"
            )

            builder.append(
                "Objects: $objectNumbers\n"
            )

            builder.append(
                "Best Match: " +
                        String.format(
                            Locale.US,
                            "%.1f",
                            bestItem.productMatchScore
                        ) +
                        "%\n"
            )

            builder.append(
                "Method: ${bestItem.matchMethod}\n"
            )

            builder.append(
                "Best YOLO: " +
                        String.format(
                            Locale.US,
                            "%.2f",
                            bestYoloConfidence
                        ) +
                        "\n"
            )

            builder.append(
                "Mapped OCR lines: $totalMappedLines\n"
            )

            builder.append(
                "────────────────────\n\n"
            )
        }

        // =================================================
        // UNKNOWN OBJECTS
        //
        // Keep them separate from unique-product count because they do
        // not have a reliable product identity yet.
        // =================================================

        if (unknownItems.isNotEmpty()) {

            val unknownObjectNumbers =
                unknownItems
                    .sortedBy {
                        it.objectIndex
                    }
                    .joinToString(", ") {
                        "#${it.objectIndex}"
                    }

            builder.append(
                "Unknown Objects: ${unknownItems.size}\n"
            )

            builder.append(
                "Objects: $unknownObjectNumbers\n\n"
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
            "FINAL UNIQUE PRODUCT RESULTS"
        )

        val groupedProducts =
            detectedItems
                .filter {
                    !it.matchedProductName.isNullOrBlank()
                }
                .groupBy {
                    canonicalKey(
                        it.matchedProductName!!
                    )
                }

        groupedProducts.values.forEach {
                items ->

            val bestItem =
                items.maxByOrNull {
                    it.productMatchScore
                } ?: return@forEach

            val objectNumbers =
                items
                    .sortedBy {
                        it.objectIndex
                    }
                    .joinToString(", ") {
                        "#${it.objectIndex}"
                    }

            Log.i(
                HYBRID_TAG,
                """
Product: ${bestItem.matchedProductName}
Quantity: ${items.size}
Objects: $objectNumbers
Best Score: ${String.format(Locale.US, "%.2f", bestItem.productMatchScore)}%
Method: ${bestItem.matchMethod}
--------------------------------------
                """.trimIndent()
            )
        }

        val unknownItems =
            detectedItems.filter {
                it.matchedProductName.isNullOrBlank()
            }

        if (unknownItems.isNotEmpty()) {

            val unknownNumbers =
                unknownItems
                    .sortedBy {
                        it.objectIndex
                    }
                    .joinToString(", ") {
                        "#${it.objectIndex}"
                    }

            Log.i(
                HYBRID_TAG,
                """
Unknown Objects: ${unknownItems.size}
Objects: $unknownNumbers
--------------------------------------
                """.trimIndent()
            )
        }

        Log.i(
            HYBRID_TAG,
            "Total physical objects = ${detectedItems.size}, " +
                    "unique matched products = ${groupedProducts.size}, " +
                    "unknown objects = ${unknownItems.size}"
        )

        Log.i(
            HYBRID_TAG,
            "======================================"
        )
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

        // Bounding-box paint.
        val boxPaint =
            Paint().apply {
                style = Paint.Style.STROKE
                color = Color.RED

                strokeWidth =
                    maxOf(
                        4f,
                        source.width / 250f
                    )

                isAntiAlias = true
            }

        // Text paint used for #1, #2, #3 ...
        val textPaint =
            Paint().apply {
                style = Paint.Style.FILL
                color = Color.WHITE
                isAntiAlias = true
                isFakeBoldText = true

                textSize =
                    maxOf(
                        30f,
                        source.width / 22f
                    )
            }

        // Background for the object-number label.
        val labelBackgroundPaint =
            Paint().apply {
                style = Paint.Style.FILL
                color = Color.RED
                isAntiAlias = true
            }

        var index = 0
        var objectNumber = 1

        while (
            index + 4 <
            detections.size
        ) {

            // Use min/max so the rectangle is valid even if coordinates
            // arrive in the opposite order.
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

            // Draw YOLO bounding box.
            canvas.drawRect(
                left,
                top,
                right,
                bottom,
                boxPaint
            )

            // Draw the same object number used by parseObjectBoxes() and
            // displayResults(), so #1 on the image corresponds to Object #1
            // in the result text.
            val label =
                "#$objectNumber"

            val textBounds =
                android.graphics.Rect()

            textPaint.getTextBounds(
                label,
                0,
                label.length,
                textBounds
            )

            val padding =
                maxOf(
                    6f,
                    textPaint.textSize * 0.20f
                )

            val labelWidth =
                textPaint.measureText(label) +
                        padding * 2f

            val labelHeight =
                textPaint.textSize +
                        padding * 2f

            // Prefer the label above the object. If there is no room,
            // place it inside the top edge of the bounding box.
            val labelLeft =
                left.coerceIn(
                    0f,
                    maxOf(0f, source.width.toFloat() - labelWidth)
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
                labelBackgroundPaint
            )

            val textX =
                labelLeft + padding

            val textY =
                labelBottom - padding

            canvas.drawText(
                label,
                textX,
                textY,
                textPaint
            )

            Log.i(
                TAG,
                "Drew Object #$objectNumber box=" +
                        "[${String.format(Locale.US, "%.1f", left)}, " +
                        "${String.format(Locale.US, "%.1f", top)}, " +
                        "${String.format(Locale.US, "%.1f", right)}, " +
                        "${String.format(Locale.US, "%.1f", bottom)}]"
            )

            index += 5
            objectNumber++
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
