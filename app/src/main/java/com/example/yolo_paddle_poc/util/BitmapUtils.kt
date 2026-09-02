package com.example.yolo_paddle_poc.util

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.RectF
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import kotlin.math.max

object BitmapUtils {

    private const val TAG = "YOLO_NCNN"

    fun decodeBitmapWithCorrectOrientation(
        contentResolver: ContentResolver,
        uri: Uri
    ): Bitmap? {

        val orientation =
            try {
                contentResolver.openInputStream(uri)?.use { input ->
                    ExifInterface(input).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                } ?: ExifInterface.ORIENTATION_NORMAL
            } catch (e: Exception) {
                Log.w(TAG, "Could not read EXIF orientation", e)
                ExifInterface.ORIENTATION_NORMAL
            }

        val decoded =
            contentResolver.openInputStream(uri)?.use { input ->
                BitmapFactory.decodeStream(input)
            } ?: return null

        val matrix = Matrix()

        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL ->
                matrix.setScale(-1f, 1f)

            ExifInterface.ORIENTATION_ROTATE_180 ->
                matrix.setRotate(180f)

            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.setRotate(180f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.setRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_ROTATE_90 ->
                matrix.setRotate(90f)

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.setRotate(-90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_ROTATE_270 ->
                matrix.setRotate(-90f)
        }

        val needsTransform =
            orientation != ExifInterface.ORIENTATION_NORMAL &&
                    orientation != ExifInterface.ORIENTATION_UNDEFINED

        if (!needsTransform) {
            return decoded
        }

        return try {
            val corrected =
                Bitmap.createBitmap(
                    decoded,
                    0,
                    0,
                    decoded.width,
                    decoded.height,
                    matrix,
                    true
                )

            if (corrected !== decoded) {
                decoded.recycle()
            }

            corrected
        } catch (e: Exception) {
            Log.e(TAG, "EXIF correction failed", e)
            decoded
        }
    }

    fun resizeBitmapForInference(
        bitmap: Bitmap,
        maxSize: Int
    ): Bitmap {

        if (
            bitmap.width <= maxSize &&
            bitmap.height <= maxSize
        ) {
            return bitmap
        }

        val scale =
            maxSize.toFloat() /
                    maxOf(bitmap.width, bitmap.height)

        return Bitmap.createScaledBitmap(
            bitmap,
            maxOf(1, (bitmap.width * scale).toInt()),
            maxOf(1, (bitmap.height * scale).toInt()),
            true
        )
    }

    fun cropBitmapWithPadding(
        source: Bitmap,
        rect: RectF,
        paddingPercent: Float
    ): Bitmap {

        val boxWidth = max(1f, rect.width())
        val boxHeight = max(1f, rect.height())

        val paddingX = boxWidth * paddingPercent
        val paddingY = boxHeight * paddingPercent

        val left =
            (rect.left - paddingX).toInt()
                .coerceIn(0, source.width - 1)

        val top =
            (rect.top - paddingY).toInt()
                .coerceIn(0, source.height - 1)

        val right =
            (rect.right + paddingX).toInt()
                .coerceIn(left + 1, source.width)

        val bottom =
            (rect.bottom + paddingY).toInt()
                .coerceIn(top + 1, source.height)

        return Bitmap.createBitmap(
            source,
            left,
            top,
            right - left,
            bottom - top
        )
    }

    fun upscaleBitmapForOcr(
        bitmap: Bitmap,
        scaleFactor: Float
    ): Bitmap {

        if (scaleFactor <= 1f) {
            return bitmap
        }

        return Bitmap.createScaledBitmap(
            bitmap,
            maxOf(1, (bitmap.width * scaleFactor).toInt()),
            maxOf(1, (bitmap.height * scaleFactor).toInt()),
            true
        )
    }
}
