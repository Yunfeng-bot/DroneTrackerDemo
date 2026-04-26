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
    private val startOffsetMs: Long = 0L,
    private val loop: Boolean = true,
    private val enableCatchup: Boolean = false,
    private val fpsOverride: Double = 0.0,
    private val previewEveryNFrames: Int = 1,
    private val maxFrameDim: Int = 1280,
    private val onFrame: (Bitmap) -> Unit,
    private val onMatFrame: (Mat, Long) -> Unit,
    private val onError: (String, Throwable?) -> Unit
) : Runnable {

    @Volatile
    private var running = true

    fun stop() {
        running = false
    }

    override fun run() {
        val retriever = MediaMetadataRetriever()
        var decodedFrames = 0L
        var nullFrames = 0L
        var skippedFrames = 0L
        var lastVideoPtsMs = 0L
        var lastReplayPtsMs = 0L
        try {
            retriever.setDataSource(videoPath)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.coerceAtLeast(1L)
                ?: throw IllegalStateException("invalid video duration")
            val clampedStartMs = startOffsetMs
                .coerceAtLeast(0L)
                .coerceAtMost((durationMs - 1L).coerceAtLeast(0L))

            val captureFps = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)
                ?.toDoubleOrNull()
                ?.takeIf { it >= 1.0 }
            val frameCount = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
                ?.toLongOrNull()
                ?.takeIf { it > 0L }
            val fpsByFrameCount = frameCount
                ?.let { it.toDouble() * 1000.0 / durationMs.toDouble() }
                ?.takeIf { it >= 1.0 }
            val fps = when {
                fpsOverride >= 1.0 -> fpsOverride
                captureFps != null -> captureFps
                fpsByFrameCount != null -> fpsByFrameCount
                else -> DEFAULT_FPS
            }
            val fpsSource = when {
                fpsOverride >= 1.0 -> "override"
                captureFps != null -> "capture_meta"
                fpsByFrameCount != null -> "frame_count"
                else -> "default"
            }

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
            Log.w(
                TRACKER_TAG,
                "EVAL_EVENT type=REPLAY state=start path=$videoPath durationMs=$durationMs " +
                    "fps=${String.format(Locale.US, "%.2f", fps)} stepMs=$stepMs rotation=$rotation " +
                    "loop=$loop startOffsetMs=$clampedStartMs catchup=$enableCatchup " +
                    "fpsSource=$fpsSource frameCount=${frameCount ?: -1}"
            )

            var videoPtsMs = clampedStartMs
            var frameIndex = 0L
            var timelineStartMs = SystemClock.elapsedRealtime() - clampedStartMs
            val previewStride = previewEveryNFrames.coerceAtLeast(1)

            while (running) {
                val targetWall = timelineStartMs + videoPtsMs
                val now = SystemClock.elapsedRealtime()
                if (targetWall > now) {
                    Thread.sleep((targetWall - now).coerceAtMost(MAX_SLEEP_MS))
                }

                val source = retriever.getFrameAtTime(videoPtsMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST)
                val replayPtsMs = (videoPtsMs - clampedStartMs).coerceAtLeast(0L)
                if (source != null) {
                    decodedFrames++
                    lastVideoPtsMs = videoPtsMs
                    lastReplayPtsMs = replayPtsMs
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
                    if (decodedFrames == 1L) {
                        Log.w(
                            TRACKER_TAG,
                            "EVAL_EVENT type=REPLAY state=first_frame replayPtsMs=$replayPtsMs " +
                                "videoPtsMs=$videoPtsMs frameIndex=$frameIndex size=${processed.width}x${processed.height}"
                        )
                    }

                    val mat = Mat()
                    try {
                        Utils.bitmapToMat(processed, mat)
                        when (mat.channels()) {
                            4 -> Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)
                            1 -> Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2RGB)
                        }
                        try {
                            onMatFrame(mat, replayPtsMs)
                        } catch (t: Throwable) {
                            Log.w(
                                TRACKER_TAG,
                                "EVAL_EVENT type=REPLAY state=analyze_error replayPtsMs=$replayPtsMs " +
                                    "videoPtsMs=$videoPtsMs ex=${t.javaClass.simpleName}"
                            )
                            throw t
                        }
                    } finally {
                        mat.release()
                    }
                    if (!shouldPreview && !processed.isRecycled) {
                        processed.recycle()
                    }
                } else {
                    nullFrames++
                    if (nullFrames <= 5L || nullFrames % 60L == 0L) {
                        Log.w(
                            TRACKER_TAG,
                            "EVAL_EVENT type=REPLAY state=null_frame replayPtsMs=$replayPtsMs " +
                                "videoPtsMs=$videoPtsMs nullCount=$nullFrames"
                        )
                    }
                }

                videoPtsMs += stepMs
                val lagMs = SystemClock.elapsedRealtime() - (timelineStartMs + videoPtsMs)
                if (enableCatchup && lagMs > stepMs * CATCHUP_TRIGGER_STEPS) {
                    val extraSteps = (lagMs / stepMs).coerceIn(1L, MAX_CATCHUP_STEPS.toLong())
                    videoPtsMs += extraSteps * stepMs
                    skippedFrames += extraSteps
                    if (skippedFrames <= 5L || skippedFrames % 60L == 0L) {
                        val replayPtsAfterCatchup = (videoPtsMs - clampedStartMs).coerceAtLeast(0L)
                        Log.w(
                            TRACKER_TAG,
                            "EVAL_EVENT type=REPLAY state=catchup lagMs=$lagMs skipped=$extraSteps " +
                                "totalSkipped=$skippedFrames replayPtsMs=$replayPtsAfterCatchup videoPtsMs=$videoPtsMs"
                        )
                    }
                }
                frameIndex++
                if (videoPtsMs >= durationMs) {
                    if (!loop) break
                    videoPtsMs = clampedStartMs
                    frameIndex = 0L
                    timelineStartMs = SystemClock.elapsedRealtime() - clampedStartMs
                }
            }
        } catch (t: Throwable) {
            Log.w(
                TRACKER_TAG,
                "EVAL_EVENT type=REPLAY state=fatal path=$videoPath decoded=$decodedFrames nullCount=$nullFrames " +
                    "lastReplayPtsMs=$lastReplayPtsMs lastVideoPtsMs=$lastVideoPtsMs skipped=$skippedFrames " +
                    "ex=${t.javaClass.simpleName}"
            )
            onError("video replay failed: $videoPath", t)
        } finally {
            retriever.release()
            Log.w(
                TRACKER_TAG,
                "EVAL_EVENT type=REPLAY state=stop path=$videoPath decoded=$decodedFrames " +
                "nullCount=$nullFrames lastReplayPtsMs=$lastReplayPtsMs " +
                    "lastVideoPtsMs=$lastVideoPtsMs skipped=$skippedFrames running=$running"
            )
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
        private const val TAG = "VideoReplay"
        private const val TRACKER_TAG = "Tracker"
        private const val DEFAULT_FPS = 15.0
        private const val MIN_STEP_MS = 30L
        private const val MAX_STEP_MS = 200L
        private const val MAX_SLEEP_MS = 50L
        private const val CATCHUP_TRIGGER_STEPS = 2L
        private const val MAX_CATCHUP_STEPS = 8
    }
}
