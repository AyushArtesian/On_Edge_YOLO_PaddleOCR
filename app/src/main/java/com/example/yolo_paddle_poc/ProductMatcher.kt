package com.example.yolo_paddle_poc

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.util.Locale
import kotlin.math.max

class ProductMatcher(
    context: Context
) {

    companion object {
        private const val TAG = "PRODUCT_MATCHER"

        /*
         * IMPORTANT:
         * This is now the SAME metadata used by TextEmbeddingMatcher.
         *
         * It guarantees that fuzzy aliases and embedding aliases resolve
         * to the same canonical product_id.
         */
        private const val PRODUCT_METADATA_ASSET =
            "text_matcher/product_embeddings_metadata.json"
    }

    data class Product(
        val id: String,
        val name: String,
        val brand: String = "",
        val category: String = ""
    )

    data class MatchResult(
        val product: Product,
        val score: Double,
        val matchedAlias: String
    )

    private data class AliasEntry(
        val product: Product,
        val alias: String,
        val variantType: String
    )

    private val productsById =
        LinkedHashMap<String, Product>()

    private val productIdByCanonicalName =
        HashMap<String, String>()

    private val aliases =
        ArrayList<AliasEntry>()

    init {
        loadCanonicalProducts(context)
    }

    private fun loadCanonicalProducts(
        context: Context
    ) {
        val jsonText =
            context.assets
                .open(PRODUCT_METADATA_ASSET)
                .bufferedReader()
                .use {
                    it.readText()
                }

        val root =
            JSONObject(jsonText)

        val entries =
            root.getJSONArray("entries")

        /*
         * Prevent duplicate aliases for the same canonical product.
         */
        val seenAliases =
            HashSet<String>()

        for (index in 0 until entries.length()) {

            val item =
                entries.getJSONObject(index)

            val productId =
                item.optString("product_id")
                    .trim()

            val productName =
                item.optString("product_name")
                    .trim()

            val brand =
                item.optString("brand")
                    .trim()

            val category =
                item.optString("category")
                    .trim()

            val alias =
                item.optString("text")
                    .trim()

            val variantType =
                item.optString("variant_type")
                    .trim()

            if (
                productId.isBlank() ||
                productName.isBlank()
            ) {
                continue
            }

            val product =
                productsById.getOrPut(productId) {
                    Product(
                        id = productId,
                        name = productName,
                        brand = brand,
                        category = category
                    )
                }

            productIdByCanonicalName[
                canonicalNameKey(productName)
            ] = productId

            /*
             * search_text is intentionally NOT used as a fuzzy alias.
             * It is useful for embeddings, but it contains many words and
             * would make fuzzy matching overly permissive.
             */
            if (
                alias.isBlank() ||
                variantType.equals(
                    "search_text",
                    ignoreCase = true
                )
            ) {
                continue
            }

            val dedupeKey =
                productId + "|" + normalize(alias)

            if (seenAliases.add(dedupeKey)) {
                aliases.add(
                    AliasEntry(
                        product = product,
                        alias = alias,
                        variantType = variantType
                    )
                )
            }
        }

        Log.i(
            TAG,
            "Loaded ${productsById.size} canonical products " +
                    "with ${aliases.size} fuzzy aliases from shared embedding metadata"
        )
    }

    /*
     * Used by HybridProductResolver to map the canonical name returned by
     * TextEmbeddingMatcher back to the exact product_id.
     */
    fun productIdForCanonicalName(
        canonicalName: String
    ): String? =
        productIdByCanonicalName[
            canonicalNameKey(canonicalName)
        ]

    fun findBestMatch(
        ocrText: String,
        threshold: Double = 50.0
    ): MatchResult? {

        val best =
            findTopMatches(
                ocrText = ocrText,
                topK = 1
            ).firstOrNull()
                ?: return null

        return if (best.score >= threshold) {
            best
        } else {
            null
        }
    }

    /*
     * Match OCR against ALL aliases, but return only ONE result for each
     * canonical product_id.  The best alias score becomes that product's
     * fuzzy score.
     */
    fun findTopMatches(
        ocrText: String,
        topK: Int = 5
    ): List<MatchResult> {

        val normalizedOcr =
            normalize(ocrText)

        if (normalizedOcr.isBlank()) {
            return emptyList()
        }

        val compactOcr =
            compact(normalizedOcr)

        val bestByProductId =
            LinkedHashMap<String, MatchResult>()

        for (entry in aliases) {

            val score =
                calculateProductScore(
                    normalizedOcr = normalizedOcr,
                    compactOcr = compactOcr,
                    productName = entry.alias
                )

            val existing =
                bestByProductId[
                    entry.product.id
                ]

            if (
                existing == null ||
                score > existing.score
            ) {
                bestByProductId[
                    entry.product.id
                ] =
                    MatchResult(
                        product = entry.product,
                        score = score,
                        matchedAlias = entry.alias
                    )
            }
        }

        val top =
            bestByProductId
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
                appendLine("==============================")
                appendLine("OCR:")
                appendLine(ocrText)
                appendLine()
                appendLine("TOP CANONICAL FUZZY CANDIDATES:")

                top.forEachIndexed {
                        index,
                        candidate ->

                    appendLine(
                        "${index + 1}. " +
                                "${candidate.product.name} = " +
                                String.format(
                                    Locale.US,
                                    "%.2f",
                                    candidate.score
                                ) +
                                "%"
                    )

                    appendLine(
                        "   productId=${candidate.product.id} " +
                                "matchedAlias='${candidate.matchedAlias}'"
                    )
                }

                append("==============================")
            }
        )

        return top
    }

    private fun canonicalNameKey(
        value: String
    ): String =
        normalize(value)
            .replace(
                Regex("[^a-z0-9]"),
                ""
            )

    private fun calculateProductScore(
        normalizedOcr: String,
        compactOcr: String,
        productName: String
    ): Double {

        val normalizedProduct =
            normalize(
                productName
            )

        val compactProduct =
            compact(
                normalizedProduct
            )

        if (
            normalizedProduct.isBlank()
        ) {

            return 0.0
        }

        // -------------------------------------------------
        // EXACT / COMPACT EXACT
        // -------------------------------------------------

        if (
            normalizedOcr ==
            normalizedProduct
        ) {

            return 100.0
        }

        if (
            compactOcr.isNotBlank() &&
            compactOcr ==
            compactProduct
        ) {

            return 100.0
        }

        // -------------------------------------------------
        // STRONG CONTAINMENT
        // -------------------------------------------------

        if (
            compactProduct.length >=
            5 &&
            compactOcr.contains(
                compactProduct
            )
        ) {

            return 96.0
        }

        if (
            compactOcr.length >=
            5 &&
            compactProduct.contains(
                compactOcr
            )
        ) {

            return 92.0
        }

        val fullScore =
            similarityPercent(
                normalizedOcr,
                normalizedProduct
            )

        val compactScore =
            similarityPercent(
                compactOcr,
                compactProduct
            )

        val tokenScore =
            tokenCoverageScore(
                normalizedOcr,
                normalizedProduct
            )

        val partialScore =
            partialSimilarityPercent(
                compactOcr,
                compactProduct
            )

        var finalScore =
            0.20 *
                    fullScore +
                    0.20 *
                    compactScore +
                    0.45 *
                    tokenScore +
                    0.15 *
                    partialScore

        // -------------------------------------------------
        // MULTI-TOKEN AGREEMENT BONUS
        // -------------------------------------------------

        val strongTokenMatches =
            countStrongTokenMatches(
                normalizedOcr,
                normalizedProduct
            )

        finalScore +=
            when {

                strongTokenMatches >=
                        3 ->
                    8.0

                strongTokenMatches ==
                        2 ->
                    4.0

                else ->
                    0.0
            }

        // -------------------------------------------------
        // PENALIZE GENERIC ONE-WORD PRODUCT IF OCR HAS
        // SEVERAL WORDS.
        // -------------------------------------------------

        val ocrWordCount =
            words(
                normalizedOcr
            ).size

        val productWordCount =
            words(
                normalizedProduct
            ).size

        if (
            ocrWordCount >=
            2 &&
            productWordCount ==
            1
        ) {

            finalScore *=
                0.72
        }

        return finalScore
            .coerceIn(
                0.0,
                100.0
            )
    }

    // =====================================================
    // TOKEN COVERAGE
    // =====================================================

    private fun tokenCoverageScore(
        normalizedOcr: String,
        normalizedProduct: String
    ): Double {

        val ocrWords =
            words(
                normalizedOcr
            )

        val productWords =
            words(
                normalizedProduct
            )

        if (
            ocrWords.isEmpty() ||
            productWords.isEmpty()
        ) {

            return 0.0
        }

        var total =
            0.0

        var weightSum =
            0.0

        for (
        productWord in
        productWords
        ) {

            val weight =
                max(
                    1,
                    productWord.length
                )
                    .toDouble()

            val best =
                ocrWords
                    .maxOfOrNull { ocrWord ->

                        similarityPercent(
                            compact(ocrWord),
                            compact(productWord)
                        )
                    }
                    ?: 0.0

            total +=
                best *
                        weight

            weightSum +=
                weight
        }

        if (
            weightSum ==
            0.0
        ) {

            return 0.0
        }

        return total /
                weightSum
    }

    private fun countStrongTokenMatches(
        normalizedOcr: String,
        normalizedProduct: String
    ): Int {

        val ocrWords =
            words(
                normalizedOcr
            )

        val productWords =
            words(
                normalizedProduct
            )

        var count =
            0

        for (
        productWord in
        productWords
        ) {

            val best =
                ocrWords
                    .maxOfOrNull { ocrWord ->

                        similarityPercent(
                            compact(ocrWord),
                            compact(productWord)
                        )
                    }
                    ?: 0.0

            if (
                best >=
                60.0
            ) {

                count++
            }
        }

        return count
    }

    // =====================================================
    // PARTIAL SIMILARITY
    // =====================================================

    private fun partialSimilarityPercent(
        first: String,
        second: String
    ): Double {

        if (
            first.isBlank() ||
            second.isBlank()
        ) {

            return 0.0
        }

        if (
            first.contains(second) ||
            second.contains(first)
        ) {

            val shorter =
                minOf(
                    first.length,
                    second.length
                )

            val longer =
                maxOf(
                    first.length,
                    second.length
                )

            return (
                    shorter.toDouble() /
                            longer.toDouble()
                    ) *
                    100.0
        }

        val shorter =
            if (
                first.length <=
                second.length
            ) {

                first

            } else {

                second
            }

        val longer =
            if (
                first.length >
                second.length
            ) {

                first

            } else {

                second
            }

        if (
            shorter.length <
            2
        ) {

            return similarityPercent(
                first,
                second
            )
        }

        var best =
            0.0

        val windowSize =
            shorter.length

        if (
            longer.length <
            windowSize
        ) {

            return similarityPercent(
                first,
                second
            )
        }

        for (
        start in
        0..(
                longer.length -
                        windowSize
                )
        ) {

            val window =
                longer.substring(
                    start,
                    start +
                            windowSize
                )

            val score =
                similarityPercent(
                    shorter,
                    window
                )

            if (
                score >
                best
            ) {

                best =
                    score
            }
        }

        return best
    }

    // =====================================================
    // LEVENSHTEIN
    // =====================================================

    private fun similarityPercent(
        first: String,
        second: String
    ): Double {

        if (
            first ==
            second
        ) {

            return 100.0
        }

        if (
            first.isEmpty() ||
            second.isEmpty()
        ) {

            return 0.0
        }

        val distance =
            levenshteinDistance(
                first,
                second
            )

        val maxLength =
            maxOf(
                first.length,
                second.length
            )

        return (
                1.0 -
                        distance.toDouble() /
                        maxLength.toDouble()
                ) *
                100.0
    }

    private fun levenshteinDistance(
        first: String,
        second: String
    ): Int {

        val previous =
            IntArray(
                second.length +
                        1
            ) {
                it
            }

        val current =
            IntArray(
                second.length +
                        1
            )

        for (
        firstIndex in
        1..first.length
        ) {

            current[0] =
                firstIndex

            for (
            secondIndex in
            1..second.length
            ) {

                val insertCost =
                    current[
                        secondIndex -
                                1
                    ] +
                            1

                val deleteCost =
                    previous[
                        secondIndex
                    ] +
                            1

                val replaceCost =
                    previous[
                        secondIndex -
                                1
                    ] +
                            if (
                                first[
                                    firstIndex -
                                            1
                                ] ==
                                second[
                                    secondIndex -
                                            1
                                ]
                            ) {

                                0

                            } else {

                                1
                            }

                current[secondIndex] =
                    minOf(
                        insertCost,
                        deleteCost,
                        replaceCost
                    )
            }

            for (
            index in
            previous.indices
            ) {

                previous[index] =
                    current[index]
            }
        }

        return previous[
            second.length
        ]
    }

    // =====================================================
    // NORMALIZATION
    // =====================================================

    private fun normalize(
        value: String
    ): String {

        return value
            .lowercase(
                Locale.ROOT
            )
            .replace(
                "&",
                " and "
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
                " "
            )
            .replace(
                Regex(
                    "[^a-z0-9 ]+"
                ),
                " "
            )
            .replace(
                Regex(
                    "\\s+"
                ),
                " "
            )
            .trim()
    }

    private fun compact(
        value: String
    ): String {

        return value
            .replace(
                Regex(
                    "[^a-z0-9]"
                ),
                ""
            )
    }

    private fun words(
        value: String
    ): List<String> {

        return value
            .split(
                Regex(
                    "\\s+"
                )
            )
            .filter {
                it.isNotBlank()
            }
    }
}
