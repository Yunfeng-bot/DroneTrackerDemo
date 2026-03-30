package com.example.dronetracker

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.tracking.TrackerCSRT
import java.util.Locale

class OpenCVTrackerAnalyzer(
    private val overlayView: TrackingOverlayView
) : ImageAnalysis.Analyzer {

    private enum class TrackerMode {
        BASELINE,
        ENHANCED
    }

    private data class MatchCandidate(
        val box: Rect,
        val score: Double
    )

    private var trackerMode = TrackerMode.ENHANCED
    private var tracker: TrackerCSRT? = null
    private var pendingInitBox: Rect? = null
    private var isTracking = false

    private var templateBase: Mat? = null
    private var adaptiveTemplate: Mat? = null
    private var lastTrackedBox: Rect? = null

    private var scaleX = 1f
    private var scaleY = 1f
    private var frameCounter = 0L
    private var trackerInitFrame = 0L

    private var consecutiveTrackerFailures = 0
    private var consecutiveRefineFailures = 0

    private var nv21Buffer: ByteArray? = null

    private var metricsSessionStartMs = SystemClock.elapsedRealtime()
    private var metricsFrames = 0L
    private var metricsTotalProcessMs = 0L
    private var metricsTrackingFrames = 0L
    private var metricsSearchingFrames = 0L
    private var metricsLockCount = 0
    private var metricsLostCount = 0
    private var metricsRecoverLocalCount = 0
    private var metricsRecoverGlobalCount = 0
    private var metricsFirstLockMs = -1L
    private var metricsCurrentTrackingStreak = 0
    private var metricsMaxTrackingStreak = 0

    private var globalMatchThreshold = DEFAULT_GLOBAL_MATCH_THRESHOLD
    private var localMatchThreshold = DEFAULT_LOCAL_MATCH_THRESHOLD
    private var baselineMatchThreshold = DEFAULT_BASELINE_MATCH_THRESHOLD
    private var templateUpdateScore = DEFAULT_TEMPLATE_UPDATE_SCORE
    private var templateUpdateAlpha = DEFAULT_TEMPLATE_UPDATE_ALPHA
    private var localSearchExpandFactor = DEFAULT_LOCAL_SEARCH_EXPAND_FACTOR
    private var enhancedRefineIntervalFrames = DEFAULT_ENHANCED_REFINE_INTERVAL_FRAMES
    private var globalDownscaleFactor = DEFAULT_GLOBAL_DOWNSCALE_FACTOR
    private var downscaleMinScoreRelax = DEFAULT_DOWNSCALE_MIN_SCORE_RELAX

    fun setTrackerMode(mode: String?) {
        trackerMode = when (mode?.trim()?.lowercase(Locale.US)) {
            "baseline" -> TrackerMode.BASELINE
            else -> TrackerMode.ENHANCED
        }
        Log.w(TAG, "EVAL_EVENT type=MODE mode=${trackerMode.name.lowercase(Locale.US)}")
    }

    fun currentTrackerMode(): String = trackerMode.name.lowercase(Locale.US)

    fun applyRuntimeOverrides(raw: String?) {
        if (raw.isNullOrBlank()) {
            logEffectiveParams("default")
            return
        }

        val entries = raw.split(',', ';')
            .map { it.trim() }
            .filter { it.isNotEmpty() && it.contains('=') }

        for (entry in entries) {
            val parts = entry.split('=', limit = 2)
            if (parts.size != 2) continue
            val key = parts[0].trim().lowercase(Locale.US)
            val value = parts[1].trim()

            when (key) {
                "global_threshold" -> value.toDoubleOrNull()?.let {
                    globalMatchThreshold = it.coerceIn(0.30, 0.95)
                }
                "local_threshold" -> value.toDoubleOrNull()?.let {
                    localMatchThreshold = it.coerceIn(0.30, 0.95)
                }
                "baseline_threshold" -> value.toDoubleOrNull()?.let {
                    baselineMatchThreshold = it.coerceIn(0.30, 0.95)
                }
                "template_update_score" -> value.toDoubleOrNull()?.let {
                    templateUpdateScore = it.coerceIn(0.45, 0.98)
                }
                "template_update_alpha" -> value.toDoubleOrNull()?.let {
                    templateUpdateAlpha = it.coerceIn(0.01, 0.40)
                }
                "local_expand" -> value.toDoubleOrNull()?.let {
                    localSearchExpandFactor = it.coerceIn(1.2, 4.0)
                }
                "refine_interval" -> value.toLongOrNull()?.let {
                    enhancedRefineIntervalFrames = it.coerceIn(1L, 12L)
                }
                "global_downscale_factor" -> value.toDoubleOrNull()?.let {
                    globalDownscaleFactor = it.coerceIn(0.35, 1.0)
                }
                "downscale_relax" -> value.toDoubleOrNull()?.let {
                    downscaleMinScoreRelax = it.coerceIn(0.0, 0.12)
                }
            }
        }

        logEffectiveParams("override")
    }

    private fun logEffectiveParams(source: String) {
        Log.w(
            TAG,
            "EVAL_EVENT type=PARAMS source=$source mode=${trackerMode.name.lowercase(Locale.US)} " +
                "global=${fmt(globalMatchThreshold)} local=${fmt(localMatchThreshold)} baseline=${fmt(baselineMatchThreshold)} " +
                "refine=${enhancedRefineIntervalFrames} localExpand=${fmt(localSearchExpandFactor)} " +
                "tplScore=${fmt(templateUpdateScore)} tplAlpha=${fmt(templateUpdateAlpha)} " +
                "downscale=${fmt(globalDownscaleFactor)} relax=${fmt(downscaleMinScoreRelax)}"
        )
    }

    fun setInitialTarget(viewRect: RectF, viewWidth: Int, viewHeight: Int) {
        val sx = if (scaleX > 0f) scaleX else 1f
        val sy = if (scaleY > 0f) scaleY else 1f
        pendingInitBox = Rect(
            (viewRect.left / sx).toInt(),
            (viewRect.top / sy).toInt(),
            (viewRect.width() / sx).toInt(),
            (viewRect.height() / sy).toInt()
        )
        Log.d(
            TAG,
            "manual target selected: overlay=${viewRect.width().toInt()}x${viewRect.height().toInt()} view=${viewWidth}x${viewHeight}"
        )
    }

    fun setTemplateImage(bitmap: Bitmap) {
        val tempMat = Mat()
        Utils.bitmapToMat(bitmap, tempMat)
        Imgproc.cvtColor(tempMat, tempMat, Imgproc.COLOR_RGBA2RGB)

        val normalized = normalizeTemplateSize(tempMat)
        tempMat.release()

        templateBase?.release()
        adaptiveTemplate?.release()

        templateBase = normalized
        adaptiveTemplate = normalized.clone()

        resetTracking(logSummary = false)
        pendingInitBox = null
        Log.i(TAG, "template loaded: ${normalized.cols()}x${normalized.rows()}")
        Log.w(TAG, "EVAL_EVENT type=TEMPLATE_READY width=${normalized.cols()} height=${normalized.rows()}")
    }

    fun resetTracking(logSummary: Boolean = true) {
        if (logSummary) {
            logEvalSummary("reset")
        }
        tracker = null
        isTracking = false
        pendingInitBox = null
        lastTrackedBox = null
        consecutiveTrackerFailures = 0
        consecutiveRefineFailures = 0
        overlayView.post { overlayView.reset() }
    }

    fun logEvalSummary(reason: String = "manual") {
        val elapsed = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(1L)
        val avgMs = if (metricsFrames > 0) metricsTotalProcessMs.toDouble() / metricsFrames else 0.0
        val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
        val trackingRatio = if (metricsFrames > 0) metricsTrackingFrames.toDouble() / metricsFrames else 0.0

        val summary = String.format(
            Locale.US,
            "EVAL_SUMMARY reason=%s mode=%s elapsedMs=%d frames=%d avgFrameMs=%.2f locks=%d lost=%d firstLockSec=%.3f recoverLocal=%d recoverGlobal=%d trackRatio=%.3f maxTrackStreak=%d",
            reason,
            trackerMode.name.lowercase(Locale.US),
            elapsed,
            metricsFrames,
            avgMs,
            metricsLockCount,
            metricsLostCount,
            firstLockSec,
            metricsRecoverLocalCount,
            metricsRecoverGlobalCount,
            trackingRatio,
            metricsMaxTrackingStreak
        )
        Log.w(TAG, summary)
    }

    override fun analyze(image: ImageProxy) {
        val startMs = SystemClock.elapsedRealtime()
        val frame = imageToMat(image)

        try {
            updateScaleFactors(frame.cols(), frame.rows())
            processFrame(frame)
        } catch (t: Throwable) {
            Log.e(TAG, "analyze failed", t)
        } finally {
            frame.release()
            image.close()
            val processMs = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L)
            updateMetrics(processMs)
        }
    }

    fun analyzeReplayFrame(frame: Mat) {
        val startMs = SystemClock.elapsedRealtime()
        try {
            updateScaleFactors(frame.cols(), frame.rows())
            processFrame(frame)
        } catch (t: Throwable) {
            Log.e(TAG, "analyze replay frame failed", t)
        } finally {
            val processMs = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L)
            updateMetrics(processMs)
        }
    }

    private fun processFrame(frame: Mat) {
        frameCounter++
        when {
            attemptManualInitialization(frame) -> Unit
            isTracking && trackerMode == TrackerMode.BASELINE -> trackFrameBaseline(frame)
            isTracking -> trackFrameEnhanced(frame)
            trackerMode == TrackerMode.BASELINE -> searchFrameBaseline(frame)
            else -> searchFrameEnhanced(frame)
        }
    }

    private fun attemptManualInitialization(frame: Mat): Boolean {
        val requested = pendingInitBox ?: return false
        pendingInitBox = null

        val safe = clampRect(requested, frame.cols(), frame.rows())
        if (safe == null) {
            Log.w(TAG, "manual init box out of bounds: $requested")
            return false
        }

        initializeTracker(frame, safe, "manual")
        return true
    }

    private fun searchFrameBaseline(frame: Mat) {
        val template = templateBase ?: adaptiveTemplate ?: return
        val match = findGlobalMatch(
            frame = frame,
            template = template,
            scales = BASELINE_GLOBAL_SCALES,
            minScore = baselineMatchThreshold
        )
        if (match != null) {
            initializeTracker(frame, match.box, "search_baseline score=${fmt(match.score)}")
        }
    }

    private fun searchFrameEnhanced(frame: Mat) {
        val template = activeTemplate() ?: return

        val localRecovery = lastTrackedBox?.let {
            val candidate = findLocalMatch(frame, template, it)
            if (candidate != null) {
                Log.d(TAG, "local recover match: score=${fmt(candidate.score)}")
            }
            candidate
        }

        val match = localRecovery ?: findGlobalMatch(
            frame = frame,
            template = template,
            scales = GLOBAL_SCALES,
            minScore = globalMatchThreshold
        )

        if (match != null) {
            initializeTracker(frame, match.box, "search score=${fmt(match.score)}")
        }
    }

    private fun trackFrameBaseline(frame: Mat) {
        val trackingRect = Rect()
        val trackingOk = tracker?.update(frame, trackingRect) == true
        if (!trackingOk) {
            onLost("baseline_update_fail")
            return
        }

        val safeTracked = clampRect(trackingRect, frame.cols(), frame.rows())
        if (safeTracked == null) {
            onLost("baseline_box_invalid")
            return
        }

        lastTrackedBox = safeTracked
        dispatchTrackedRect(safeTracked)
    }

    private fun trackFrameEnhanced(frame: Mat) {
        val trackingRect = Rect()
        val trackingOk = tracker?.update(frame, trackingRect) == true
        if (!trackingOk) {
            handleTrackingDropEnhanced(frame, "csrt_update_fail")
            return
        }

        val safeTracked = clampRect(trackingRect, frame.cols(), frame.rows())
        if (safeTracked == null) {
            handleTrackingDropEnhanced(frame, "tracked_box_invalid")
            return
        }

        var finalBox = safeTracked
        val template = activeTemplate()
        val shouldRefine = frameCounter % enhancedRefineIntervalFrames == 0L || consecutiveRefineFailures > 0
        var refineScore = -1.0

        if (template != null && shouldRefine) {
            val refineMatch = findLocalMatch(frame, template, safeTracked)
            if (refineMatch != null) {
                finalBox = refineMatch.box
                refineScore = refineMatch.score
                consecutiveRefineFailures = 0
                if (refineScore >= templateUpdateScore) {
                    updateAdaptiveTemplate(frame, finalBox, force = false)
                }
            } else {
                consecutiveRefineFailures++
            }
        }

        consecutiveTrackerFailures = 0
        lastTrackedBox = finalBox
        dispatchTrackedRect(finalBox)

        val shouldReinit = frameCounter - trackerInitFrame >= TRACKER_REINIT_INTERVAL_FRAMES ||
            (refineScore in 0.0..REINIT_WEAK_SCORE)
        if (shouldReinit) {
            reinitializeTracker(frame, finalBox)
        }

        if (consecutiveRefineFailures >= MAX_REFINE_FAILURES) {
            handleTrackingDropEnhanced(frame, "local_refine_fail_streak")
        }
    }

    private fun handleTrackingDropEnhanced(frame: Mat, reason: String) {
        consecutiveTrackerFailures++
        Log.w(TAG, "tracking drop: reason=$reason failStreak=$consecutiveTrackerFailures")

        val template = activeTemplate()

        if (template != null) {
            val localRecovery = lastTrackedBox?.let { findLocalMatch(frame, template, it) }
            if (localRecovery != null) {
                initializeTracker(
                    frame,
                    localRecovery.box,
                    "recover_local score=${fmt(localRecovery.score)}"
                )
                return
            }

            if (consecutiveTrackerFailures >= MAX_TRACKER_FAILURES) {
                val globalRecovery = findGlobalMatch(
                    frame = frame,
                    template = template,
                    scales = GLOBAL_SCALES,
                    minScore = globalMatchThreshold
                )
                if (globalRecovery != null) {
                    initializeTracker(
                        frame,
                        globalRecovery.box,
                        "recover_global score=${fmt(globalRecovery.score)}"
                    )
                    return
                }
            }
        }

        if (consecutiveTrackerFailures < MAX_TRACKER_FAILURES) {
            lastTrackedBox?.let { dispatchTrackedRect(it) }
            return
        }

        onLost(reason)
    }

    private fun findGlobalMatch(
        frame: Mat,
        template: Mat,
        scales: DoubleArray,
        minScore: Double
    ): MatchCandidate? {
        val downscale = computeGlobalDownscale(frame.cols(), frame.rows())
        if (downscale >= 0.999) {
            return findBestTemplateMatch(
                frame = frame,
                template = template,
                searchRegion = null,
                scales = scales,
                minScore = minScore,
                referenceBox = null
            )
        }

        val smallFrame = Mat()
        val smallTemplate = Mat()
        try {
            Imgproc.resize(
                frame,
                smallFrame,
                Size(),
                downscale,
                downscale,
                Imgproc.INTER_AREA
            )

            val targetW = (template.cols() * downscale).toInt().coerceAtLeast(MIN_TEMPLATE_DIM)
            val targetH = (template.rows() * downscale).toInt().coerceAtLeast(MIN_TEMPLATE_DIM)
            Imgproc.resize(template, smallTemplate, Size(targetW.toDouble(), targetH.toDouble()))

            val adjustedMinScore = (minScore - downscaleMinScoreRelax).coerceAtLeast(MIN_FALLBACK_SCORE)
            val candidate = findBestTemplateMatch(
                frame = smallFrame,
                template = smallTemplate,
                searchRegion = null,
                scales = scales,
                minScore = adjustedMinScore,
                referenceBox = null
            ) ?: return null

            val mapped = Rect(
                (candidate.box.x / downscale).toInt(),
                (candidate.box.y / downscale).toInt(),
                (candidate.box.width / downscale).toInt(),
                (candidate.box.height / downscale).toInt()
            )
            val safe = clampRect(mapped, frame.cols(), frame.rows()) ?: return null
            return MatchCandidate(safe, candidate.score)
        } finally {
            smallTemplate.release()
            smallFrame.release()
        }
    }

    private fun computeGlobalDownscale(frameW: Int, frameH: Int): Double {
        val maxDim = kotlin.math.max(frameW, frameH)
        if (maxDim <= GLOBAL_DOWNSCALE_TRIGGER_DIM) {
            return 1.0
        }
        return globalDownscaleFactor
    }

    private fun onLost(reason: String) {
        tracker = null
        isTracking = false
        consecutiveRefineFailures = 0
        consecutiveTrackerFailures = 0
        metricsLostCount++
        Log.w(TAG, "EVAL_EVENT type=LOST reason=$reason lost=$metricsLostCount")
        logEvalSummary("lost_$reason")
        overlayView.post { overlayView.reset() }
    }

    private fun initializeTracker(frame: Mat, box: Rect, reason: String) {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return
        val wasTracking = isTracking

        val csrt = TrackerCSRT.create()
        csrt.init(frame, safe)
        tracker = csrt
        isTracking = true

        trackerInitFrame = frameCounter
        consecutiveTrackerFailures = 0
        consecutiveRefineFailures = 0
        lastTrackedBox = safe

        if (templateBase == null) {
            templateBase = extractTemplate(frame, safe)
            adaptiveTemplate?.release()
            adaptiveTemplate = templateBase?.clone()
        } else if (trackerMode == TrackerMode.ENHANCED) {
            updateAdaptiveTemplate(frame, safe, force = true)
        }

        dispatchTrackedRect(safe)

        if (!wasTracking) {
            metricsLockCount++
            if (metricsFirstLockMs < 0L) {
                metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
            }
            if (reason.startsWith("recover_local")) {
                metricsRecoverLocalCount++
            }
            if (reason.startsWith("recover_global")) {
                metricsRecoverGlobalCount++
            }
            Log.w(
                TAG,
                "EVAL_EVENT type=LOCK reason=$reason locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)}"
            )
        }

        Log.i(TAG, "tracker initialized ($reason): ${safe.x},${safe.y},${safe.width}x${safe.height}")
    }

    private fun reinitializeTracker(frame: Mat, box: Rect) {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return
        val csrt = TrackerCSRT.create()
        csrt.init(frame, safe)
        tracker = csrt
        trackerInitFrame = frameCounter
    }

    private fun findLocalMatch(frame: Mat, template: Mat, anchor: Rect): MatchCandidate? {
        val region = expandRect(anchor, frame.cols(), frame.rows(), localSearchExpandFactor) ?: return null
        return findBestTemplateMatch(
            frame = frame,
            template = template,
            searchRegion = region,
            scales = LOCAL_SCALES,
            minScore = localMatchThreshold,
            referenceBox = anchor
        )
    }

    private fun findBestTemplateMatch(
        frame: Mat,
        template: Mat,
        searchRegion: Rect?,
        scales: DoubleArray,
        minScore: Double,
        referenceBox: Rect?
    ): MatchCandidate? {
        val searchMat = if (searchRegion != null) frame.submat(searchRegion) else frame

        try {
            var bestAdjusted = Double.NEGATIVE_INFINITY
            var bestRaw = Double.NEGATIVE_INFINITY
            var bestBox: Rect? = null

            for (scale in scales) {
                val targetW = (template.cols() * scale).toInt()
                val targetH = (template.rows() * scale).toInt()

                if (targetW < MIN_TEMPLATE_DIM || targetH < MIN_TEMPLATE_DIM) continue
                if (targetW >= searchMat.cols() || targetH >= searchMat.rows()) continue

                val scaledTemplate = Mat()
                val result = Mat()
                try {
                    Imgproc.resize(template, scaledTemplate, Size(targetW.toDouble(), targetH.toDouble()))
                    Imgproc.matchTemplate(searchMat, scaledTemplate, result, Imgproc.TM_CCOEFF_NORMED)

                    val mm = Core.minMaxLoc(result)
                    val rawScore = mm.maxVal

                    val scalePenalty = if (referenceBox == null) 0.0 else {
                        val refW = referenceBox.width.toDouble().coerceAtLeast(1.0)
                        val refH = referenceBox.height.toDouble().coerceAtLeast(1.0)
                        val dw = kotlin.math.abs(targetW - refW) / refW
                        val dh = kotlin.math.abs(targetH - refH) / refH
                        (dw + dh) * 0.05
                    }
                    val adjustedScore = rawScore - scalePenalty

                    if (adjustedScore > bestAdjusted) {
                        val localX = mm.maxLoc.x.toInt()
                        val localY = mm.maxLoc.y.toInt()
                        val offsetX = searchRegion?.x ?: 0
                        val offsetY = searchRegion?.y ?: 0

                        bestAdjusted = adjustedScore
                        bestRaw = rawScore
                        bestBox = Rect(offsetX + localX, offsetY + localY, targetW, targetH)
                    }
                } finally {
                    result.release()
                    scaledTemplate.release()
                }
            }

            if (bestRaw < minScore || bestBox == null) {
                return null
            }

            val clipped = clampRect(bestBox, frame.cols(), frame.rows()) ?: return null
            return MatchCandidate(clipped, bestRaw)
        } finally {
            if (searchRegion != null) {
                searchMat.release()
            }
        }
    }

    private fun updateAdaptiveTemplate(frame: Mat, box: Rect, force: Boolean) {
        val base = templateBase ?: return
        val patch = extractTemplate(frame, box, base.cols(), base.rows()) ?: return

        if (adaptiveTemplate == null || force) {
            adaptiveTemplate?.release()
            adaptiveTemplate = patch.clone()
        } else {
            Core.addWeighted(
                adaptiveTemplate,
                1.0 - templateUpdateAlpha,
                patch,
                templateUpdateAlpha,
                0.0,
                adaptiveTemplate
            )
        }
        patch.release()
    }

    private fun extractTemplate(frame: Mat, box: Rect, targetW: Int? = null, targetH: Int? = null): Mat? {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return null
        val roi = frame.submat(safe)
        return try {
            val out = Mat()
            val desiredW = targetW ?: safe.width.coerceAtMost(MAX_TEMPLATE_DIM.toInt())
            val desiredH = targetH ?: safe.height.coerceAtMost(MAX_TEMPLATE_DIM.toInt())
            Imgproc.resize(roi, out, Size(desiredW.toDouble(), desiredH.toDouble()))
            out
        } finally {
            roi.release()
        }
    }

    private fun normalizeTemplateSize(template: Mat): Mat {
        val out = Mat()
        val w = template.cols().toDouble()
        val h = template.rows().toDouble()
        val maxDim = kotlin.math.max(w, h)
        if (maxDim <= MAX_TEMPLATE_DIM) {
            template.copyTo(out)
            return out
        }

        val scale = MAX_TEMPLATE_DIM / maxDim
        Imgproc.resize(template, out, Size(w * scale, h * scale))
        return out
    }

    private fun activeTemplate(): Mat? = adaptiveTemplate ?: templateBase

    private fun dispatchTrackedRect(box: Rect) {
        val rect = RectF(
            box.x * scaleX,
            box.y * scaleY,
            (box.x + box.width) * scaleX,
            (box.y + box.height) * scaleY
        )
        overlayView.post { overlayView.updateTrackedObject(rect) }
    }

    private fun updateScaleFactors(frameWidth: Int, frameHeight: Int) {
        if (overlayView.width == 0 || overlayView.height == 0) return

        scaleX = overlayView.width.toFloat() / frameWidth.toFloat()
        scaleY = overlayView.height.toFloat() / frameHeight.toFloat()
    }

    private fun updateMetrics(processMs: Long) {
        metricsFrames++
        metricsTotalProcessMs += processMs

        if (isTracking) {
            metricsTrackingFrames++
            metricsCurrentTrackingStreak++
            if (metricsCurrentTrackingStreak > metricsMaxTrackingStreak) {
                metricsMaxTrackingStreak = metricsCurrentTrackingStreak
            }
        } else {
            metricsSearchingFrames++
            metricsCurrentTrackingStreak = 0
        }

        if (metricsFrames % PERF_LOG_INTERVAL_FRAMES == 0L) {
            val avgMs = metricsTotalProcessMs.toDouble() / metricsFrames
            val trackRatio = metricsTrackingFrames.toDouble() / metricsFrames
            val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
            Log.w(
                TAG,
                "EVAL_PERF mode=${trackerMode.name.lowercase(Locale.US)} frames=$metricsFrames avgFrameMs=${fmt(avgMs)} trackRatio=${fmt(trackRatio)} locks=$metricsLockCount lost=$metricsLostCount firstLockSec=${fmt(firstLockSec)}"
            )
        }

        if (metricsFrames % SUMMARY_LOG_INTERVAL_FRAMES == 0L) {
            logEvalSummary("periodic")
        }
    }

    private fun clampRect(rect: Rect, frameW: Int, frameH: Int): Rect? {
        val x = rect.x.coerceIn(0, frameW - 1)
        val y = rect.y.coerceIn(0, frameH - 1)
        val maxW = frameW - x
        val maxH = frameH - y
        val w = rect.width.coerceAtMost(maxW)
        val h = rect.height.coerceAtMost(maxH)
        if (w < MIN_BOX_DIM || h < MIN_BOX_DIM) return null
        return Rect(x, y, w, h)
    }

    private fun expandRect(box: Rect, frameW: Int, frameH: Int, factor: Double): Rect? {
        val cx = box.x + box.width / 2.0
        val cy = box.y + box.height / 2.0
        val w = (box.width * factor).toInt()
        val h = (box.height * factor).toInt()

        val x = (cx - w / 2.0).toInt()
        val y = (cy - h / 2.0).toInt()
        return clampRect(Rect(x, y, w, h), frameW, frameH)
    }

    private fun ensureNv21Buffer(width: Int, height: Int): ByteArray {
        val needed = width * height * 3 / 2
        val existing = nv21Buffer
        if (existing != null && existing.size == needed) {
            return existing
        }
        return ByteArray(needed).also { nv21Buffer = it }
    }

    private fun imageToMat(image: ImageProxy): Mat {
        require(image.format == ImageFormat.YUV_420_888) { "Invalid image format" }

        val width = image.width
        val height = image.height
        val data = ensureNv21Buffer(width, height)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        var outIndex = 0

        for (row in 0 until height) {
            var yIndex = row * yRowStride
            for (col in 0 until width) {
                data[outIndex++] = yBuffer.get(yIndex)
                yIndex += yPixelStride
            }
        }

        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            var uIndex = row * uRowStride
            var vIndex = row * vRowStride
            for (col in 0 until chromaWidth) {
                data[outIndex++] = vBuffer.get(vIndex)
                data[outIndex++] = uBuffer.get(uIndex)
                uIndex += uPixelStride
                vIndex += vPixelStride
            }
        }

        val yuvMat = Mat(height + height / 2, width, CvType.CV_8UC1)
        yuvMat.put(0, 0, data)

        val rgbMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21, 3)

        when (image.imageInfo.rotationDegrees) {
            90 -> Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE)
            270 -> Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_COUNTERCLOCKWISE)
        }

        yuvMat.release()
        return rgbMat
    }

    private fun fmt(value: Double): String = String.format(Locale.US, "%.3f", value)

    companion object {
        private const val TAG = "Tracker"

        private const val MIN_BOX_DIM = 16
        private const val MIN_TEMPLATE_DIM = 12
        private const val MAX_TEMPLATE_DIM = 180.0

        private const val DEFAULT_GLOBAL_MATCH_THRESHOLD = 0.62
        private const val DEFAULT_LOCAL_MATCH_THRESHOLD = 0.56
        private const val DEFAULT_BASELINE_MATCH_THRESHOLD = 0.60

        private const val DEFAULT_TEMPLATE_UPDATE_SCORE = 0.74
        private const val DEFAULT_TEMPLATE_UPDATE_ALPHA = 0.08

        private const val TRACKER_REINIT_INTERVAL_FRAMES = 12
        private const val MAX_TRACKER_FAILURES = 3
        private const val MAX_REFINE_FAILURES = 5
        private const val REINIT_WEAK_SCORE = 0.60
        private const val GLOBAL_DOWNSCALE_TRIGGER_DIM = 1280
        private const val DEFAULT_GLOBAL_DOWNSCALE_FACTOR = 0.60
        private const val DEFAULT_DOWNSCALE_MIN_SCORE_RELAX = 0.03
        private const val MIN_FALLBACK_SCORE = 0.45

        private const val DEFAULT_LOCAL_SEARCH_EXPAND_FACTOR = 2.0
        private const val DEFAULT_ENHANCED_REFINE_INTERVAL_FRAMES = 4L
        private const val PERF_LOG_INTERVAL_FRAMES = 10
        private const val SUMMARY_LOG_INTERVAL_FRAMES = 30

        private val BASELINE_GLOBAL_SCALES = doubleArrayOf(0.5, 0.75, 1.0, 1.25, 1.5)
        private val GLOBAL_SCALES = doubleArrayOf(0.35, 0.5, 0.7, 0.9, 1.0, 1.15, 1.35, 1.6, 1.9)
        private val LOCAL_SCALES = doubleArrayOf(0.75, 0.9, 1.0, 1.1, 1.25, 1.4)
    }
}
