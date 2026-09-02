package com.example.yolo_paddle_poc

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
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
import com.example.yolo_paddle_poc.matching.HybridProductResolver
import com.example.yolo_paddle_poc.model.DetectedItem
import com.example.yolo_paddle_poc.pipeline.ProductDetectionPipeline
import com.example.yolo_paddle_poc.util.BitmapUtils
import com.example.yolo_paddle_poc.util.DrawingUtils
import com.paddle.ocr.EngineConfig
import com.paddle.ocr.PaddleOCR
import com.paddle.ocr.PaddleOCRConfig
import com.paddle.ocr.util.OpenCVUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "YOLO_NCNN"
        private const val OCR_TAG = "PADDLE_OCR"
        private const val EMBEDDING_TAG = "TEXT_EMBEDDING_MATCHER"
        private const val HYBRID_TAG = "HYBRID_PRODUCT_RESOLVER"

        private const val MAX_IMAGE_SIZE = 1280

        init {
            System.loadLibrary("yolo_native")
        }
    }

    // Keep JNI functions in MainActivity because your current native JNI
    // symbols are compiled against MainActivity.
    external fun loadYoloModel(
        assetManager: android.content.res.AssetManager
    ): Boolean

    external fun detectObjects(
        bitmap: Bitmap
    ): FloatArray

    private lateinit var selectImageButton: Button
    private lateinit var captureImageButton: Button
    private lateinit var imageView: ImageView
    private lateinit var statusText: TextView
    private lateinit var resultsText: TextView

    private var yoloReady = false
    private var ocrReady = false
    private var embeddingReady = false

    private var paddleOCR: PaddleOCR? = null

    private lateinit var productMatcher: ProductMatcher
    private lateinit var textEmbeddingMatcher: TextEmbeddingMatcher
    private lateinit var hybridResolver: HybridProductResolver
    private var pipeline: ProductDetectionPipeline? = null

    private var pendingCameraUri: Uri? = null

    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->
            if (uri != null) {
                processImageUri(uri)
            }
        }

    private val cameraLauncher =
        registerForActivityResult(
            ActivityResultContracts.TakePicture()
        ) { success ->

            val uri = pendingCameraUri
            pendingCameraUri = null

            if (success && uri != null) {
                processImageUri(uri)
            }
        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(
            findViewById(R.id.main)
        ) { view, insets ->

            val bars =
                insets.getInsets(
                    WindowInsetsCompat.Type.systemBars()
                )

            view.setPadding(
                bars.left,
                bars.top,
                bars.right,
                bars.bottom
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

        productMatcher =
            ProductMatcher(this)

        selectImageButton.setOnClickListener {
            imagePicker.launch("image/*")
        }

        captureImageButton.setOnClickListener {
            try {
                val uri =
                    createCameraImageUri()

                pendingCameraUri = uri
                cameraLauncher.launch(uri)

            } catch (e: Exception) {
                Log.e(TAG, "Unable to open camera", e)
            }
        }

        loadTextEmbeddingMatcher()
        loadYolo()
        loadPaddleOCR()
    }

    private fun createCameraImageUri(): Uri {

        val directory =
            File(
                cacheDir,
                "camera_images"
            )

        if (!directory.exists()) {
            directory.mkdirs()
        }

        val imageFile =
            File.createTempFile(
                "product_${System.currentTimeMillis()}_",
                ".jpg",
                directory
            )

        return FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            imageFile
        )
    }

    private fun processImageUri(
        uri: Uri
    ) {

        val activePipeline =
            pipeline

        if (
            !yoloReady ||
            !ocrReady ||
            !embeddingReady ||
            activePipeline == null
        ) {
            statusText.text =
                "AI models are still loading"
            return
        }

        lifecycleScope.launch {
            try {

                setActionButtonsEnabled(false)

                statusText.text =
                    "Loading image..."

                resultsText.text =
                    "Processing..."

                val decoded =
                    withContext(Dispatchers.IO) {
                        BitmapUtils
                            .decodeBitmapWithCorrectOrientation(
                                contentResolver,
                                uri
                            )
                    }

                if (decoded == null) {
                    statusText.text =
                        "Unable to load image"
                    setActionButtonsEnabled(true)
                    return@launch
                }

                val rgba =
                    decoded.copy(
                        Bitmap.Config.ARGB_8888,
                        false
                    )

                if (decoded !== rgba) {
                    decoded.recycle()
                }

                if (rgba == null) {
                    statusText.text =
                        "Unable to convert image"
                    setActionButtonsEnabled(true)
                    return@launch
                }

                val inferenceBitmap =
                    BitmapUtils.resizeBitmapForInference(
                        rgba,
                        MAX_IMAGE_SIZE
                    )

                if (inferenceBitmap !== rgba) {
                    rgba.recycle()
                }

                statusText.text =
                    "Detecting objects..."

                val yoloStart =
                    System.currentTimeMillis()

                val detections =
                    withContext(Dispatchers.Default) {
                        detectObjects(inferenceBitmap)
                    }

                val yoloElapsed =
                    System.currentTimeMillis() -
                            yoloStart

                val detectionCount =
                    detections.size / 5

                if (detectionCount == 0) {

                    imageView.setImageBitmap(
                        inferenceBitmap
                    )

                    resultsText.text =
                        "No objects detected."

                    statusText.text =
                        "No objects detected"

                    setActionButtonsEnabled(true)
                    return@launch
                }

                statusText.text =
                    "Running hybrid OCR..."

                val pipelineStart =
                    System.currentTimeMillis()

                val detectedItems =
                    activePipeline.runHybridOcr(
                        sourceBitmap =
                            inferenceBitmap,
                        detections =
                            detections
                    )

                val pipelineElapsed =
                    System.currentTimeMillis() -
                            pipelineStart

                val resultBitmap =
                    DrawingUtils.drawBoundingBoxes(
                        inferenceBitmap,
                        detections
                    )

                imageView.setImageBitmap(
                    resultBitmap
                )

                displayResults(
                    detectedItems
                )

                logFinalResults(
                    detectedItems
                )

                statusText.text =
                    "$detectionCount objects • " +
                            "YOLO ${yoloElapsed}ms • " +
                            "Hybrid ${pipelineElapsed}ms"

                setActionButtonsEnabled(true)

            } catch (e: Exception) {

                Log.e(
                    TAG,
                    "Processing failed",
                    e
                )

                statusText.text =
                    "Processing failed"

                resultsText.text =
                    "Something went wrong."

                setActionButtonsEnabled(true)
            }
        }
    }

    private fun loadTextEmbeddingMatcher() {

        lifecycleScope.launch {
            try {

                textEmbeddingMatcher =
                    withContext(Dispatchers.IO) {
                        TextEmbeddingMatcher(
                            applicationContext
                        )
                    }

                embeddingReady = true

                Log.i(
                    EMBEDDING_TAG,
                    "MINILM MATCHER READY"
                )

                buildPipelineIfReady()
                updateModelStatus()

            } catch (e: Exception) {

                embeddingReady = false

                Log.e(
                    EMBEDDING_TAG,
                    "MiniLM initialization failed",
                    e
                )
            }
        }
    }

    private fun loadYolo() {

        Thread {

            val success =
                loadYoloModel(assets)

            runOnUiThread {

                yoloReady = success

                if (success) {
                    Log.i(
                        TAG,
                        "YOLO MODEL READY"
                    )
                }

                updateModelStatus()
            }

        }.start()
    }

    private fun loadPaddleOCR() {

        lifecycleScope.launch {
            try {

                val openCvReady =
                    OpenCVUtils.init(
                        this@MainActivity
                    )

                if (!openCvReady) {
                    ocrReady = false
                    updateModelStatus()
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
                                recScoreThresh = 0.20f,
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

                buildPipelineIfReady()
                updateModelStatus()

            } catch (e: Exception) {

                ocrReady = false

                Log.e(
                    OCR_TAG,
                    "PaddleOCR initialization failed",
                    e
                )
            }
        }
    }

    private fun buildPipelineIfReady() {

        if (
            ocrReady &&
            embeddingReady &&
            paddleOCR != null &&
            ::textEmbeddingMatcher.isInitialized
        ) {

            hybridResolver =
                HybridProductResolver(
                    productMatcher,
                    textEmbeddingMatcher
                )

            pipeline =
                ProductDetectionPipeline(
                    paddleOCR = paddleOCR!!,
                    resolver = hybridResolver
                )
        }
    }

    private fun updateModelStatus() {

        if (
            yoloReady &&
            ocrReady &&
            embeddingReady &&
            pipeline != null
        ) {
            statusText.text =
                "YOLO + Hybrid OCR + Matcher Ready"

            resultsText.text =
                "Select an image or capture a photo."

            setActionButtonsEnabled(true)

        } else {

            statusText.text =
                "Loading AI models..."

            setActionButtonsEnabled(false)
        }
    }

    private fun setActionButtonsEnabled(
        enabled: Boolean
    ) {
        selectImageButton.isEnabled = enabled
        captureImageButton.isEnabled = enabled
    }

    private fun displayResults(
        detectedItems: List<DetectedItem>
    ) {

        val grouped =
            detectedItems
                .filter {
                    !it.matchedProductName.isNullOrBlank()
                }
                .groupBy {
                    hybridResolver.canonicalKey(
                        it.matchedProductName!!
                    )
                }

        val unknown =
            detectedItems.filter {
                it.matchedProductName.isNullOrBlank()
            }

        resultsText.text =
            buildString {

                appendLine(
                    "Detected Objects: ${detectedItems.size}"
                )

                appendLine(
                    "Unique Products: ${grouped.size}"
                )

                appendLine()

                grouped.values
                    .forEachIndexed { index, items ->

                        val best =
                            items.maxByOrNull {
                                it.productMatchScore
                            } ?: return@forEachIndexed

                        val objectNumbers =
                            items.sortedBy {
                                it.objectIndex
                            }
                                .joinToString(", ") {
                                    "#${it.objectIndex}"
                                }

                        appendLine(
                            "${index + 1}. ${best.matchedProductName}"
                        )

                        appendLine(
                            "Quantity: ${items.size}"
                        )

                        appendLine(
                            "Objects: $objectNumbers"
                        )

                        appendLine(
                            "Best Match: " +
                                    String.format(
                                        Locale.US,
                                        "%.1f",
                                        best.productMatchScore
                                    ) +
                                    "%"
                        )

                        appendLine(
                            "Method: ${best.matchMethod}"
                        )

                        appendLine()
                    }

                if (unknown.isNotEmpty()) {

                    appendLine(
                        "Unknown Objects: ${unknown.size}"
                    )

                    appendLine(
                        "Objects: " +
                                unknown
                                    .sortedBy {
                                        it.objectIndex
                                    }
                                    .joinToString(", ") {
                                        "#${it.objectIndex}"
                                    }
                    )
                }
            }
    }

    private fun logFinalResults(
        detectedItems: List<DetectedItem>
    ) {

        val grouped =
            detectedItems
                .filter {
                    !it.matchedProductName.isNullOrBlank()
                }
                .groupBy {
                    hybridResolver.canonicalKey(
                        it.matchedProductName!!
                    )
                }

        Log.i(
            HYBRID_TAG,
            "FINAL UNIQUE PRODUCT RESULTS"
        )

        grouped.values.forEach { items ->

            val best =
                items.maxByOrNull {
                    it.productMatchScore
                } ?: return@forEach

            Log.i(
                HYBRID_TAG,
                """
                Product: ${best.matchedProductName}
                Quantity: ${items.size}
                Objects: ${
                    items
                        .sortedBy { it.objectIndex }
                        .joinToString(", ") {
                            "#${it.objectIndex}"
                        }
                }
                Best Score: ${String.format(Locale.US, "%.2f", best.productMatchScore)}%
                Method: ${best.matchMethod}
                """.trimIndent()
            )
        }
    }

    override fun onDestroy() {

        if (
            ::textEmbeddingMatcher.isInitialized
        ) {
            try {
                textEmbeddingMatcher.close()
            } catch (e: Exception) {
                Log.e(
                    EMBEDDING_TAG,
                    "Failed closing MiniLM",
                    e
                )
            }
        }

        val ocr = paddleOCR

        if (ocr != null) {
            lifecycleScope.launch {
                try {
                    ocr.release()
                } catch (e: Exception) {
                    Log.e(
                        OCR_TAG,
                        "Failed releasing PaddleOCR",
                        e
                    )
                }
            }
        }

        super.onDestroy()
    }
}
