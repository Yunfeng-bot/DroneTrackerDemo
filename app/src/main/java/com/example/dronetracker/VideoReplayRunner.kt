package com.example.dronetracker

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.SystemClock
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.Locale

class VideoReplayRunner(
    private val videoPath: String,
    private val loop: Boolean = true,
    private val previewEveryNFrames: Int = 3,
    private val maxFrameDim: Int = 1280,
    private val onFrame: (Bitmap) -> Unit,
    private val onMatFrame: (Mat) -> Unit,
    private val onError: (String, Throwable?) -> Unit
) : Runnable {

    @Volatile
    private var running = true

    fun stop() {
        running = false
    }

    override fun run() {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoPath)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: throw IllegalStateException("invalid video duration")

            val fps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                ?.toDoubleOrNull()
                ?.takeIf { it >= 1.0 }
                ?: DEFAULT_FPS

            val stepMs = (1000.0 / fps)
                .toLong()
                .coerceIn(MIN_STEP_MS, MAX_STEP_MS)

            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull()
                ?: 0

            Log.i(
                TAG,
                String.format(
                    Locale.US,
                    "replay start: path=%s durationMs=%d fps=%.2f stepMs=%d rotation=%d loop=%s",
                    videoPath,
                    durationMs,
                    fps,
                    stepMs,
                    rotation,
                    loop
                )
            )

            var ptsMs = 0L
            var frameIndex = 0L
            var timelineStartMs = SystemClock.elapsedRealtime()
            val previewStride = previewEveryNFrames.coerceAtLeast(1)

            while (running) {
                val targetWall = timelineStartMs + ptsMs
                val now = SystemClock.elapsedRealtime()
                if (targetWall > now) {
                    Thread.sleep((targetWall - now).coerceAtMost(MAX_SLEEP_MS))
                }

                val source = retriever.getFrameAtTime(ptsMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                if (source != null) {
                    val display = if (rotation == 0) source else rotateBitmap(source, rotation)
                    if (display !== source) {
                        source.recycle()
                    }

                    val processed = scaleDownIfNeeded(display, maxFrameDim)
                    if (processed !== display && !display.isRecycled) {
                        display.recycle()
                    }

                    val shouldPreview = frameIndex % previewStride == 0L
                    if (shouldPreview) {
                        onFrame(processed)
                    }

                    val mat = Mat()
                    try {
                        Utils.bitmapToMat(processed, mat)
                        when (mat.channels()) {
                            4 -> Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
                            1 -> Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2RGB)
                        }
                        onMatFrame(mat)
                    } finally {
                        mat.release()
                    }
                    if (!shouldPreview && !processed.isRecycled) {
                        processed.recycle()
                    }
                }

                ptsMs += stepMs
                frameIndex++
                if (ptsMs >= durationMs) {
                    if (!loop) break
                    ptsMs = 0L
                    frameIndex = 0L
                    timelineStartMs = SystemClock.elapsedRealtime()
                }
            }
        } catch (t: Throwable) {
            onError("video replay failed: $videoPath", t)
        } finally {
            retriever.release()
            Log.i(TAG, "replay stopped: path=$videoPath")
        }
    }

    private fun rotateBitmap(source: Bitmap, rotation: Int): Bitmap {
        if (rotation == 0) return source
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun scaleDownIfNeeded(source: Bitmap, targetMaxDim: Int): Bitmap {
        if (targetMaxDim <= 0) return source
        val maxDim = kotlin.math.max(source.width, source.height)
        if (maxDim <= targetMaxDim) return source

        val scale = targetMaxDim.toDouble() / maxDim.toDouble()
        val targetW = (source.width * scale).toInt().coerceAtLeast(1)
        val targetH = (source.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(source, targetW, targetH, true)
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val DEFAULT_FPS = 15.0
        private const val MIN_STEP_MS = 30L
        private const val MAX_STEP_MS = 200L
        private const val MAX_SLEEP_MS = 50L
    }
}
