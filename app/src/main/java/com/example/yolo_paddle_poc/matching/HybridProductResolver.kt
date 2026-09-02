package com.example.yolo_paddle_poc.matching

import android.util.Log
import com.example.yolo_paddle_poc.ProductMatcher
import com.example.yolo_paddle_poc.TextEmbeddingMatcher
import com.example.yolo_paddle_poc.model.ProductResolution
import java.util.Locale

class HybridProductResolver(
    private val productMatcher: ProductMatcher,
    private val textEmbeddingMatcher: TextEmbeddingMatcher
) {

    companion object {
        private const val TAG = "HYBRID_PRODUCT_RESOLVER"

        private const val TOP_K = 5
        private const val FINAL_STRONG_THRESHOLD = 0.75
        private const val FINAL_AGREEMENT_THRESHOLD = 0.50

        private const val EMBEDDING_WEIGHT = 0.55
        private const val FUZZY_WEIGHT = 0.45
        private const val AGREEMENT_BONUS = 0.10
        private const val FUZZY_STRONG_OVERRIDE = 0.88
    }

    private data class Candidate(
        val displayName: String,
        val key: String,
        var embeddingScore: Double = 0.0,
        var fuzzyScore: Double = 0.0,
        var inEmbedding: Boolean = false,
        var inFuzzy: Boolean = false
    ) {
        fun combinedScore(): Double {
            var score =
                EMBEDDING_WEIGHT * embeddingScore +
                        FUZZY_WEIGHT * fuzzyScore

            if (inEmbedding && inFuzzy) {
                score += AGREEMENT_BONUS
            }

            return score.coerceIn(0.0, 1.0)
        }
    }

    fun resolve(
        ocrText: String
    ): ProductResolution {

        val embeddingTop =
            textEmbeddingMatcher.findTopMatches(
                text = ocrText,
                topK = TOP_K
            )

        val fuzzyTop =
            productMatcher.findTopMatches(
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
            LinkedHashMap<String, Candidate>()

        embeddingTop.forEach { match ->

            val key =
                canonicalKey(match.productName)

            val candidate =
                candidates.getOrPut(key) {
                    Candidate(
                        displayName = match.productName,
                        key = key
                    )
                }

            candidate.embeddingScore =
                maxOf(
                    candidate.embeddingScore,
                    match.score.toDouble()
                        .coerceIn(0.0, 1.0)
                )

            candidate.inEmbedding = true
        }

        fuzzyTop.forEach { match ->

            val key =
                canonicalKey(match.product.name)

            val candidate =
                candidates.getOrPut(key) {
                    Candidate(
                        displayName = match.product.name,
                        key = key
                    )
                }

            candidate.fuzzyScore =
                maxOf(
                    candidate.fuzzyScore,
                    (match.score / 100.0)
                        .coerceIn(0.0, 1.0)
                )

            candidate.inFuzzy = true
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

        val strongFuzzy =
            best.fuzzyScore >=
                    FUZZY_STRONG_OVERRIDE

        val bothAgree =
            best.inEmbedding &&
                    best.inFuzzy

        val accepted =
            when {
                bothAgree &&
                        finalScore >= FINAL_AGREEMENT_THRESHOLD ->
                    true

                finalScore >= FINAL_STRONG_THRESHOLD ->
                    true

                strongFuzzy ->
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

                strongFuzzy ->
                    "FUZZY_STRONG"

                else ->
                    "HYBRID_STRONG"
            }

        Log.i(
            TAG,
            buildString {
                appendLine("====================================")
                appendLine("OCR:")
                appendLine(ocrText)
                appendLine()
                appendLine("HYBRID CANDIDATES:")

                ranked.take(8)
                    .forEachIndexed { index, c ->
                        appendLine("${index + 1}. ${c.displayName}")
                        appendLine(
                            "   embedding=" +
                                    String.format(Locale.US, "%.4f", c.embeddingScore) +
                                    " fuzzy=" +
                                    String.format(Locale.US, "%.4f", c.fuzzyScore) +
                                    " both=" +
                                    (c.inEmbedding && c.inFuzzy) +
                                    " final=" +
                                    String.format(Locale.US, "%.4f", c.combinedScore())
                        )
                    }

                appendLine()
                appendLine(
                    "FINAL: " +
                            if (accepted) best.displayName else "Unknown"
                )
                appendLine(
                    "FINAL SCORE: " +
                            String.format(Locale.US, "%.4f", finalScore)
                )
                appendLine("METHOD: $method")
                append("====================================")
            }
        )

        return if (accepted) {
            ProductResolution(
                productName = best.displayName,
                score = finalScore,
                method = method
            )
        } else {
            ProductResolution(
                productName = null,
                score = finalScore,
                method = "UNKNOWN"
            )
        }
    }

    fun canonicalKey(
        value: String
    ): String =
        value
            .lowercase(Locale.ROOT)
            .replace("&", "and")
            .replace("'", "")
            .replace("’", "")
            .replace("-", "")
            .replace(Regex("[^a-z0-9]"), "")
}
