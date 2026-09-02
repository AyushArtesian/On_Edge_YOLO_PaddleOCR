package com.example.yolo_paddle_poc.ocr

import android.util.Log
import com.example.yolo_paddle_poc.model.OcrTextBox
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

object ProminentTextSelector {

    private const val TAG = "PADDLE_OCR"
    private const val MAX_LINES = 4
    private const val MIN_RELATIVE_HEIGHT = 0.025f

    private data class ScoredLine(
        val box: OcrTextBox,
        val score: Double,
        val relativeHeight: Float,
        val relativeArea: Float,
        val uppercaseRatio: Double,
        val isNoise: Boolean
    )

    fun select(
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

                val cleaned =
                    box.text
                        .trim()
                        .replace(Regex("\\s+"), " ")

                if (cleaned.isBlank()) {
                    return@mapNotNull null
                }

                val relativeHeight =
                    (box.rect.height() / safeHeight)
                        .coerceIn(0f, 1f)

                val relativeArea =
                    (
                            box.rect.width() *
                                    box.rect.height() /
                                    safeArea
                            ).coerceIn(0f, 1f)

                val uppercaseRatio =
                    uppercaseRatio(cleaned)

                val titleBonus =
                    titleLikeBonus(cleaned)

                val noise =
                    isPackagingNoise(cleaned)

                var score =
                    0.52 * relativeHeight +
                            0.18 * relativeArea +
                            0.12 * uppercaseRatio +
                            0.10 * titleBonus +
                            0.08 * box.confidence
                        .toDouble()
                        .coerceIn(0.0, 1.0)

                if (noise) score -= 0.45
                if (mostlyNumbers(cleaned)) score -= 0.30
                if (cleaned.length <= 1) score -= 0.25

                ScoredLine(
                    box = box.copy(text = cleaned),
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

        val largestHeight =
            scored.maxOfOrNull {
                it.relativeHeight
            } ?: 0f

        val dynamicHeightFloor =
            max(
                MIN_RELATIVE_HEIGHT,
                largestHeight * 0.42f
            )

        var selected =
            scored
                .filter {
                    !it.isNoise &&
                            it.relativeHeight >= dynamicHeightFloor
                }
                .sortedByDescending { it.score }
                .take(MAX_LINES)

        if (selected.isEmpty()) {
            selected =
                scored
                    .filter { !it.isNoise }
                    .sortedByDescending { it.score }
                    .take(min(2, MAX_LINES))
        }

        val readingOrder =
            selected
                .map { it.box }
                .sortedWith(
                    compareBy<OcrTextBox> { it.rect.top }
                        .thenBy { it.rect.left }
                )

        Log.i(
            TAG,
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

    private fun uppercaseRatio(
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

    private fun titleLikeBonus(
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

        val titleWords =
            words.count { word ->
                word.firstOrNull { it.isLetter() }
                    ?.isUpperCase() == true
            }

        return (
                titleWords.toDouble() /
                        words.size.toDouble()
                ).coerceIn(0.0, 1.0)
    }

    private fun isPackagingNoise(
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

        if (
            noisePhrases.any {
                normalized.contains(it)
            }
        ) {
            return true
        }

        val quantityOrPrice =
            Regex(
                "^\\s*(?:₹|rs\\.?\\s*)?\\d+(?:[.,]\\d+)?\\s*" +
                        "(?:g|gm|gms|kg|ml|l|ltr|litre|litres|oz|lb|pcs|pc)?\\.?\\s*$",
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

    private fun mostlyNumbers(
        text: String
    ): Boolean {

        val value =
            text.filter { it.isLetterOrDigit() }

        if (value.isEmpty()) {
            return true
        }

        return value.count { it.isDigit() }
            .toDouble() /
                value.length.toDouble() >= 0.70
    }
}
