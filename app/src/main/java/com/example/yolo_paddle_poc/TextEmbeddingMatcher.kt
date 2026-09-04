package com.example.yolo_paddle_poc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import java.util.Locale
import kotlin.math.sqrt

class TextEmbeddingMatcher(
    private val context: Context
) {

    companion object {
        private const val TAG = "TEXT_EMBEDDING_MATCHER"

        private const val MODEL_PATH =
            "text_matcher/model_qint8_arm64.onnx"

        private const val VOCAB_PATH =
            "text_matcher/vocab.txt"

        private const val PRODUCT_EMBEDDINGS_PATH =
            "text_matcher/product_embeddings.bin"

        private const val METADATA_PATH =
            "text_matcher/product_embeddings_metadata.json"

        private const val MAX_LENGTH = 64

        private const val CLS_TOKEN = "[CLS]"
        private const val SEP_TOKEN = "[SEP]"
        private const val UNK_TOKEN = "[UNK]"
    }

    data class MatchResult(
        val productName: String,
        val score: Float,
        val productId: String = "",
        val brand: String = "",
        val category: String = "",
        val matchedVariant: String = "",
        val variantType: String = ""
    )

    private data class EmbeddingEntry(
        val productId: String,
        val productName: String,
        val brand: String,
        val category: String,
        val text: String,
        val variantType: String
    )

    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private lateinit var session: OrtSession

    private val vocab =
        HashMap<String, Int>()

    private var clsId = 101
    private var sepId = 102
    private var unkId = 100

    private val embeddingEntries =
        mutableListOf<EmbeddingEntry>()

    private lateinit var productEmbeddings:
            Array<FloatArray>

    private var embeddingDimension = 384
    private var metadataNormalized = true

    init {
        loadVocab()
        loadMetadataAndEntries()
        loadProductEmbeddings()
        loadModel()

        val canonicalProducts =
            embeddingEntries
                .map { it.productId }
                .distinct()
                .size

        Log.i(
            TAG,
            "TextEmbeddingMatcher READY: " +
                    "${embeddingEntries.size} embedding variants, " +
                    "$canonicalProducts canonical products, " +
                    "${embeddingDimension}D, normalized=$metadataNormalized"
        )
    }

    // =====================================================
    // PUBLIC API
    // =====================================================

    fun findBestMatch(
        text: String
    ): MatchResult? {
        return findTopMatches(
            text = text,
            topK = 1
        ).firstOrNull()
    }

    fun findTopMatches(
        text: String,
        topK: Int = 5
    ): List<MatchResult> {

        if (text.isBlank()) {
            return emptyList()
        }

        val queryEmbedding =
            embedText(text)

        // One product can have many embedding rows:
        // name, brand, aliases and search_text.
        // Keep only the strongest row for each canonical product_id.
        val bestByProduct =
            LinkedHashMap<String, MatchResult>()

        for (index in embeddingEntries.indices) {

            val entry =
                embeddingEntries[index]

            val score =
                dotProduct(
                    queryEmbedding,
                    productEmbeddings[index]
                )

            val current =
                bestByProduct[entry.productId]

            if (
                current == null ||
                score > current.score
            ) {
                bestByProduct[entry.productId] =
                    MatchResult(
                        productName = entry.productName,
                        score = score,
                        productId = entry.productId,
                        brand = entry.brand,
                        category = entry.category,
                        matchedVariant = entry.text,
                        variantType = entry.variantType
                    )
            }
        }

        val top =
            bestByProduct
                .values
                .sortedByDescending {
                    it.score
                }
                .take(
                    topK.coerceAtLeast(1)
                )

        Log.i(
            TAG,
            buildString {
                appendLine("====================================")
                appendLine("OCR:")
                appendLine(text)
                appendLine()
                appendLine("TOP CANONICAL EMBEDDING CANDIDATES:")

                top.forEachIndexed { index, candidate ->
                    appendLine(
                        "${index + 1}. ${candidate.productName} = " +
                                String.format(
                                    Locale.US,
                                    "%.4f",
                                    candidate.score
                                )
                    )
                    appendLine(
                        "   productId=${candidate.productId} " +
                                "brand=${candidate.brand.ifBlank { "<none>" }} " +
                                "category=${candidate.category}"
                    )
                    appendLine(
                        "   bestVariant='${candidate.matchedVariant}' " +
                                "type=${candidate.variantType}"
                    )
                }

                append("====================================")
            }
        )

        return top
    }

    fun debugTokens(
        text: String
    ) {

        val encoded =
            tokenize(text)

        Log.i(
            TAG,
            "TOKENS for \"$text\" = " +
                    encoded.inputIds.joinToString(
                        separator = ", "
                    )
        )
    }

    // =====================================================
    // LOAD ASSETS
    // =====================================================

    private fun loadVocab() {

        context.assets
            .open(VOCAB_PATH)
            .bufferedReader()
            .useLines { lines ->

                lines.forEachIndexed {
                        index,
                        token ->

                    vocab[token] =
                        index
                }
            }

        clsId =
            vocab[CLS_TOKEN]
                ?: 101

        sepId =
            vocab[SEP_TOKEN]
                ?: 102

        unkId =
            vocab[UNK_TOKEN]
                ?: 100

        Log.i(
            TAG,
            "Vocabulary loaded: ${vocab.size}"
        )
    }

    private fun loadMetadataAndEntries() {

        val json =
            context.assets
                .open(METADATA_PATH)
                .bufferedReader()
                .use {
                    JSONObject(
                        it.readText()
                    )
                }

        embeddingDimension =
            when {
                json.has("embedding_dimension") ->
                    json.getInt("embedding_dimension")

                // Backward-compatible fallback for old metadata.
                json.has("dimension") ->
                    json.getInt("dimension")

                else ->
                    384
            }

        metadataNormalized =
            if (json.has("normalized")) {
                json.optBoolean("normalized", true)
            } else {
                true
            }

        val entries =
            json.optJSONArray("entries")
                ?: throw IllegalStateException(
                    "New embedding metadata must contain an 'entries' array."
                )

        embeddingEntries.clear()

        for (index in 0 until entries.length()) {

            val item =
                entries.getJSONObject(index)

            val productId =
                item.optString("product_id").trim()

            val productName =
                item.optString("product_name").trim()

            val variantText =
                item.optString("text").trim()

            require(productId.isNotBlank()) {
                "Metadata entry $index has empty product_id"
            }

            require(productName.isNotBlank()) {
                "Metadata entry $index has empty product_name"
            }

            require(variantText.isNotBlank()) {
                "Metadata entry $index has empty text"
            }

            embeddingEntries.add(
                EmbeddingEntry(
                    productId = productId,
                    productName = productName,
                    brand = item.optString("brand"),
                    category = item.optString("category"),
                    text = variantText,
                    variantType =
                        item.optString(
                            "variant_type",
                            "unknown"
                        )
                )
            )
        }

        val declaredTotal =
            json.optInt(
                "total_embeddings",
                embeddingEntries.size
            )

        require(
            declaredTotal == embeddingEntries.size
        ) {
            "Metadata count mismatch. " +
                    "Declared $declaredTotal entries but parsed ${embeddingEntries.size}."
        }

        Log.i(
            TAG,
            "Embedding metadata loaded: " +
                    "${embeddingEntries.size} variants, " +
                    "${embeddingDimension}D"
        )
    }

    private fun loadProductEmbeddings() {

        val bytes =
            context.assets
                .open(
                    PRODUCT_EMBEDDINGS_PATH
                )
                .readBytes()

        val expectedFloatCount =
            embeddingEntries.size *
                    embeddingDimension

        val actualFloatCount =
            bytes.size /
                    4

        require(
            actualFloatCount ==
                    expectedFloatCount
        ) {
            "Embedding file size mismatch. " +
                    "Expected $expectedFloatCount floats " +
                    "but found $actualFloatCount."
        }

        val buffer =
            ByteBuffer
                .wrap(bytes)
                .order(
                    ByteOrder.LITTLE_ENDIAN
                )

        productEmbeddings =
            Array(
                embeddingEntries.size
            ) {

                FloatArray(
                    embeddingDimension
                )
            }

        for (
        productIndex in
        embeddingEntries.indices
        ) {

            for (
            dimensionIndex in
            0 until embeddingDimension
            ) {

                productEmbeddings[
                    productIndex
                ][
                    dimensionIndex
                ] =
                    buffer.float
            }
        }

        Log.i(
            TAG,
            "Product embeddings loaded: " +
                    "${productEmbeddings.size} x " +
                    embeddingDimension
        )
    }

    private fun loadModel() {

        val modelBytes =
            context.assets
                .open(MODEL_PATH)
                .readBytes()

        val options =
            OrtSession.SessionOptions()

        options.setIntraOpNumThreads(
            2
        )

        session =
            environment.createSession(
                modelBytes,
                options
            )

        Log.i(
            TAG,
            "MiniLM ONNX model loaded"
        )
    }

    // =====================================================
    // INFERENCE
    // =====================================================

    private fun embedText(
        text: String
    ): FloatArray {

        val encoded =
            tokenize(text)

        val inputIdsTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(
                    encoded.inputIds
                )
            )

        val attentionMaskTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(
                    encoded.attentionMask
                )
            )

        val tokenTypeIdsTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(
                    encoded.tokenTypeIds
                )
            )

        try {

            val inputs:
                    Map<String, OnnxTensor> =
                mapOf(
                    "input_ids" to
                            inputIdsTensor,

                    "attention_mask" to
                            attentionMaskTensor,

                    "token_type_ids" to
                            tokenTypeIdsTensor
                )

            session.run(
                inputs
            ).use { result ->

                @Suppress("UNCHECKED_CAST")
                val hiddenState =
                    result[0].value
                            as Array<Array<FloatArray>>

                val sentenceEmbedding =
                    meanPool(
                        hiddenStates =
                            hiddenState[0],

                        attentionMask =
                            encoded.attentionMask
                    )

                normalizeL2(
                    sentenceEmbedding
                )

                return sentenceEmbedding
            }

        } finally {

            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
        }
    }

    // =====================================================
    // TOKENIZER
    // =====================================================

    private data class TokenizedInput(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray
    )

    private fun tokenize(
        text: String
    ): TokenizedInput {

        val basicTokens =
            basicTokenize(text)

        val tokenIds =
            mutableListOf<Int>()

        tokenIds.add(
            clsId
        )

        outer@
        for (
        token in
        basicTokens
        ) {

            val pieces =
                wordPieceTokenize(
                    token
                )

            for (
            pieceId in
            pieces
            ) {

                if (
                    tokenIds.size >=
                    MAX_LENGTH - 1
                ) {

                    break@outer
                }

                tokenIds.add(
                    pieceId
                )
            }
        }

        tokenIds.add(
            sepId
        )

        val inputIds =
            LongArray(
                tokenIds.size
            )

        val attentionMask =
            LongArray(
                tokenIds.size
            )

        val tokenTypeIds =
            LongArray(
                tokenIds.size
            )

        for (
        index in
        tokenIds.indices
        ) {

            inputIds[index] =
                tokenIds[index]
                    .toLong()

            attentionMask[index] =
                1L

            tokenTypeIds[index] =
                0L
        }

        return TokenizedInput(
            inputIds =
                inputIds,

            attentionMask =
                attentionMask,

            tokenTypeIds =
                tokenTypeIds
        )
    }

    /*
     * Lightweight BERT-style basic tokenizer:
     * - lowercase
     * - remove accents
     * - split punctuation into separate tokens
     *
     * This is much closer to Hugging Face's BERT tokenizer
     * than simply deleting punctuation.
     */
    private fun basicTokenize(
        text: String
    ): List<String> {

        val normalized =
            stripAccents(
                text
                    .lowercase(
                        Locale.ROOT
                    )
            )

        val tokens =
            mutableListOf<String>()

        val current =
            StringBuilder()

        fun flushCurrent() {

            if (
                current.isNotEmpty()
            ) {

                tokens.add(
                    current.toString()
                )

                current.clear()
            }
        }

        normalized.forEach { char ->

            when {

                char.isWhitespace() -> {
                    flushCurrent()
                }

                isPunctuation(char) -> {

                    flushCurrent()

                    tokens.add(
                        char.toString()
                    )
                }

                char.code <= 0x1F ||
                        char.code == 0x7F -> {
                    // Skip control characters.
                }

                else -> {
                    current.append(char)
                }
            }
        }

        flushCurrent()

        return tokens
    }

    private fun stripAccents(
        text: String
    ): String {

        val normalized =
            Normalizer.normalize(
                text,
                Normalizer.Form.NFD
            )

        return buildString {

            normalized.forEach { char ->

                val type =
                    Character.getType(char)

                if (
                    type !=
                    Character.NON_SPACING_MARK.toInt()
                ) {

                    append(char)
                }
            }
        }
    }

    private fun isPunctuation(
        char: Char
    ): Boolean {

        val code =
            char.code

        if (
            code in 33..47 ||
            code in 58..64 ||
            code in 91..96 ||
            code in 123..126
        ) {

            return true
        }

        return when (
            Character.getType(char)
        ) {

            Character.CONNECTOR_PUNCTUATION.toInt(),
            Character.DASH_PUNCTUATION.toInt(),
            Character.START_PUNCTUATION.toInt(),
            Character.END_PUNCTUATION.toInt(),
            Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
            Character.FINAL_QUOTE_PUNCTUATION.toInt(),
            Character.OTHER_PUNCTUATION.toInt() ->
                true

            else ->
                false
        }
    }

    private fun wordPieceTokenize(
        token: String
    ): List<Int> {

        val direct =
            vocab[token]

        if (
            direct != null
        ) {

            return listOf(
                direct
            )
        }

        if (
            token.length >
            100
        ) {

            return listOf(
                unkId
            )
        }

        val result =
            mutableListOf<Int>()

        var start =
            0

        var badToken =
            false

        while (
            start <
            token.length
        ) {

            var end =
                token.length

            var currentId:
                    Int? =
                null

            var currentEnd =
                -1

            while (
                start <
                end
            ) {

                var piece =
                    token.substring(
                        start,
                        end
                    )

                if (
                    start >
                    0
                ) {

                    piece =
                        "##$piece"
                }

                val id =
                    vocab[piece]

                if (
                    id != null
                ) {

                    currentId =
                        id

                    currentEnd =
                        end

                    break
                }

                end--
            }

            if (
                currentId ==
                null
            ) {

                badToken =
                    true

                break
            }

            result.add(
                currentId
            )

            start =
                currentEnd
        }

        if (
            badToken
        ) {

            return listOf(
                unkId
            )
        }

        return result
    }

    // =====================================================
    // POOLING / VECTOR MATH
    // =====================================================

    private fun meanPool(
        hiddenStates:
        Array<FloatArray>,

        attentionMask:
        LongArray
    ): FloatArray {

        val embedding =
            FloatArray(
                embeddingDimension
            )

        var tokenCount =
            0f

        for (
        tokenIndex in
        hiddenStates.indices
        ) {

            if (
                tokenIndex >=
                attentionMask.size
            ) {

                break
            }

            if (
                attentionMask[tokenIndex] ==
                0L
            ) {

                continue
            }

            tokenCount +=
                1f

            val tokenEmbedding =
                hiddenStates[tokenIndex]

            for (
            dimensionIndex in
            0 until embeddingDimension
            ) {

                embedding[dimensionIndex] +=
                    tokenEmbedding[dimensionIndex]
            }
        }

        if (
            tokenCount >
            0f
        ) {

            for (
            dimensionIndex in
            embedding.indices
            ) {

                embedding[dimensionIndex] /=
                    tokenCount
            }
        }

        return embedding
    }

    private fun normalizeL2(
        vector: FloatArray
    ) {

        var sum =
            0.0

        for (
        value in
        vector
        ) {

            sum +=
                value *
                        value
        }

        val norm =
            sqrt(sum)
                .toFloat()

        if (
            norm <=
            0f
        ) {

            return
        }

        for (
        index in
        vector.indices
        ) {

            vector[index] /=
                norm
        }
    }

    private fun dotProduct(
        first: FloatArray,
        second: FloatArray
    ): Float {

        var dot =
            0f

        val size =
            minOf(
                first.size,
                second.size
            )

        for (
        index in
        0 until size
        ) {

            dot +=
                first[index] *
                        second[index]
        }

        return dot
    }

    // =====================================================
    // RELEASE
    // =====================================================

    fun close() {

        if (
            ::session.isInitialized
        ) {

            try {

                session.close()

            } catch (
                exception: Exception
            ) {

                Log.e(
                    TAG,
                    "Failed closing MiniLM session",
                    exception
                )
            }
        }
    }
}
