package com.example.dronetracker

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc
import org.opencv.tracking.TrackerKCF
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class OpenCVTrackerAnalyzer(
    private val overlayView: TrackingOverlayView
) : ImageAnalysis.Analyzer {

    private enum class TrackerMode { BASELINE, ENHANCED }

    private data class OrbMatchCandidate(
        val box: Rect,
        val goodMatches: Int,
        val inlierCount: Int,
        val confidence: Double,
        val usedHomography: Boolean,
        val searchScale: Double,
        val fallbackReason: String?,
        val matcherType: String,
        val isStrong: Boolean
    )

    private data class BoxMatchQuality(
        val goodMatches: Int,
        val inlierCount: Int,
        val confidence: Double
    )

    private data class TrackVerifyDecision(
        val action: String,
        val box: Rect?,
        val reason: String
    )

    private data class SearchLevelResult(
        val candidate: OrbMatchCandidate?,
        val reason: String,
        val flannGood: Int,
        val bfGood: Int,
        val selectedGood: Int,
        val selectedInliers: Int,
        val matcherType: String,
        val detail: String
    )

    private data class ClusterEstimate(
        val center: Point,
        val sideOriginal: Int
    )

    private class TemplateLevel(
        val scale: Double,
        val gray: Mat,
        val keypoints: MatOfKeyPoint,
        val descriptors8U: Mat,
        val descriptors32F: Mat,
        val corners: MatOfPoint2f,
        val keypointCount: Int,
        val textureScore: Double
    ) {
        fun release() {
            corners.release()
            descriptors32F.release()
            descriptors8U.release()
            keypoints.release()
            gray.release()
        }
    }

    private var trackerMode = TrackerMode.ENHANCED
    private var tracker: TrackerKCF? = null
    private var pendingInitBox: Rect? = null
    private var isTracking = false
    private var lastTrackedBox: Rect? = null
    private var lastTrackerGcMs = 0L

    private val orb: ORB = ORB.create(DEFAULT_ORB_FEATURES)
    private val flannMatcher: DescriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED)
    private val bfMatcher: DescriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
    private val clahe: CLAHE = Imgproc.createCLAHE(DEFAULT_CLAHE_CLIP_LIMIT, Size(DEFAULT_CLAHE_TILE_SIZE, DEFAULT_CLAHE_TILE_SIZE))

    private var templateGray: Mat? = null
    private var templateKeypoints: MatOfKeyPoint? = null
    private var templateDescriptors8U: Mat? = null
    private var templateDescriptors32F: Mat? = null
    private var templateCorners: MatOfPoint2f? = null
    private val templateSourceGrays = ArrayList<Mat>()
    private val templatePyramidLevels = ArrayList<TemplateLevel>()
    private var templateLibrarySize = 0
    private var templateKeypointCount = 0
    private var templateTextureScore = 0.0

    private var scaleX = 1f
    private var scaleY = 1f
    private var frameCounter = 0L
    private var consecutiveTrackerFailures = 0
    private var lastTemplateReadyState = true

    private var nv21Buffer: ByteArray? = null

    private var metricsSessionStartMs = SystemClock.elapsedRealtime()
    private var metricsFrames = 0L
    private var metricsTotalProcessMs = 0L
    private var metricsTrackingFrames = 0L
    private var metricsSearchingFrames = 0L
    private var metricsLockCount = 0
    private var metricsLostCount = 0
    private var metricsFirstLockMs = -1L
    private var metricsCurrentTrackingStreak = 0
    private var metricsMaxTrackingStreak = 0

    private var orbMaxFeatures = DEFAULT_ORB_FEATURES
    private var orbLoweRatio = DEFAULT_ORB_LOWE_RATIO
    private var orbMinGoodMatches = DEFAULT_ORB_MIN_GOOD_MATCHES
    private var orbMinInliers = DEFAULT_ORB_MIN_INLIERS
    private var orbSoftMinGoodMatches = DEFAULT_ORB_SOFT_MIN_GOOD_MATCHES
    private var orbSoftMinInliers = DEFAULT_ORB_SOFT_MIN_INLIERS
    private var orbRansacThreshold = DEFAULT_ORB_RANSAC_THRESHOLD
    private var orbUseClahe = DEFAULT_ORB_USE_CLAHE

    private var searchShortEdge = DEFAULT_SEARCH_SHORT_EDGE
    private var searchHighResMissStreak = DEFAULT_SEARCH_HIGH_RES_MISS_STREAK
    private var searchHighResShortEdge = DEFAULT_SEARCH_HIGH_RES_SHORT_EDGE
    private var searchUltraHighResMissStreak = DEFAULT_SEARCH_ULTRA_HIGH_RES_MISS_STREAK
    private var searchUltraHighResShortEdge = DEFAULT_SEARCH_ULTRA_HIGH_RES_SHORT_EDGE
    private var initBoxSize = DEFAULT_INIT_BOX_SIZE
    private var fallbackMaxBoxSize = DEFAULT_FALLBACK_MAX_BOX_SIZE
    private var homographyMaxDistortion = DEFAULT_HOMOGRAPHY_MAX_DISTORTION
    private var homographyScaleMin = DEFAULT_HOMOGRAPHY_SCALE_MIN
    private var homographyScaleMax = DEFAULT_HOMOGRAPHY_SCALE_MAX
    private var homographyMinJacobianDet = DEFAULT_HOMOGRAPHY_MIN_JACOBIAN_DET
    private var templateMinTextureScore = DEFAULT_TEMPLATE_MIN_TEXTURE_SCORE
    private var orbFarBoostFeatures = DEFAULT_ORB_FAR_BOOST_FEATURES

    private var kcfMaxFailStreak = DEFAULT_KCF_MAX_FAIL_STREAK
    private var forceTrackerGcOnDrop = DEFAULT_FORCE_TRACKER_GC_ON_DROP
    private var searchMissStreak = 0

    private var trackVerifyIntervalFrames = DEFAULT_TRACK_VERIFY_INTERVAL_FRAMES
    private var trackVerifyGlobalIntervalFrames = DEFAULT_TRACK_VERIFY_GLOBAL_INTERVAL_FRAMES
    private var trackVerifyLocalExpandFactor = DEFAULT_TRACK_VERIFY_LOCAL_EXPAND_FACTOR
    private var trackVerifyMinGoodMatches = DEFAULT_TRACK_VERIFY_MIN_GOOD_MATCHES
    private var trackVerifyMinInliers = DEFAULT_TRACK_VERIFY_MIN_INLIERS
    private var trackVerifyFailTolerance = DEFAULT_TRACK_VERIFY_FAIL_TOLERANCE
    private var trackVerifyRecenterPx = DEFAULT_TRACK_VERIFY_RECENTER_PX
    private var trackVerifyMinIou = DEFAULT_TRACK_VERIFY_MIN_IOU
    private var trackVerifySwitchConfidenceMargin = DEFAULT_TRACK_VERIFY_SWITCH_CONFIDENCE_MARGIN
    private var trackVerifyFailStreak = 0
    private var trackVerifyHardDriftStreak = 0
    private var trackVerifyHardDriftTolerance = DEFAULT_TRACK_VERIFY_HARD_DRIFT_TOLERANCE

    private var firstLockStableFrames = DEFAULT_FIRST_LOCK_STABLE_FRAMES
    private var firstLockStableMs = DEFAULT_FIRST_LOCK_STABLE_MS
    private var firstLockStablePx = DEFAULT_FIRST_LOCK_STABLE_PX
    private var firstLockMinIou = DEFAULT_FIRST_LOCK_MIN_IOU
    private var firstLockGapMs = DEFAULT_FIRST_LOCK_GAP_MS
    private var allowFallbackLock = DEFAULT_ALLOW_FALLBACK_LOCK
    private var fallbackRefineExpandFactor = DEFAULT_FALLBACK_REFINE_EXPAND_FACTOR
    private var fallbackRefineMinGoodMatches = DEFAULT_FALLBACK_REFINE_MIN_GOOD_MATCHES
    private var fallbackRefineMinInliers = DEFAULT_FALLBACK_REFINE_MIN_INLIERS
    private var fallbackRefineMinConfidence = DEFAULT_FALLBACK_REFINE_MIN_CONFIDENCE
    private var firstLockCandidateBox: Rect? = null
    private var firstLockCandidateCenter = Point(0.0, 0.0)
    private var firstLockCandidateFrames = 0
    private var firstLockCandidateStartMs = 0L
    private var firstLockCandidateLastMs = 0L
    private var firstLockCandidateBestGood = 0
    private var firstLockCandidateBestInliers = 0

    fun setTrackerMode(mode: String?) {
        trackerMode = when (mode?.trim()?.lowercase(Locale.US)) {
            "baseline" -> TrackerMode.BASELINE
            else -> TrackerMode.ENHANCED
        }
        Log.w(TAG, "EVAL_EVENT type=MODE mode=${trackerMode.name.lowercase(Locale.US)}")
    }

    fun currentTrackerMode(): String = trackerMode.name.lowercase(Locale.US)

    fun hasReadyTemplate(): Boolean = lastTemplateReadyState

    fun applyRuntimeOverrides(raw: String?) {
        if (raw.isNullOrBlank()) {
            orb.setMaxFeatures(orbMaxFeatures)
            logEffectiveParams("default")
            return
        }

        var shouldRefreshTemplate = false
        val entries = raw.split(',', ';')
            .map { it.trim() }
            .filter { it.isNotEmpty() && it.contains('=') }

        for (entry in entries) {
            val parts = entry.split('=', limit = 2)
            if (parts.size != 2) continue
            val key = parts[0].trim().lowercase(Locale.US)
            val value = parts[1].trim()

            when (key) {
                "orb_features" -> value.toIntOrNull()?.let {
                    orbMaxFeatures = it.coerceIn(200, 2500)
                    shouldRefreshTemplate = true
                }
                "orb_ratio" -> value.toDoubleOrNull()?.let { orbLoweRatio = it.coerceIn(0.50, 0.95) }
                "orb_min_matches" -> value.toIntOrNull()?.let { orbMinGoodMatches = it.coerceIn(8, 400) }
                "orb_min_inliers", "min_match_count" -> value.toIntOrNull()?.let { orbMinInliers = it.coerceIn(8, 300) }
                "orb_soft_min_matches" -> value.toIntOrNull()?.let { orbSoftMinGoodMatches = it.coerceIn(4, 300) }
                "orb_soft_min_inliers" -> value.toIntOrNull()?.let { orbSoftMinInliers = it.coerceIn(3, 200) }
                "orb_ransac" -> value.toDoubleOrNull()?.let { orbRansacThreshold = it.coerceIn(1.0, 12.0) }
                "search_short_edge" -> value.toIntOrNull()?.let { searchShortEdge = it.coerceIn(240, 720) }
                "search_high_res_miss" -> value.toIntOrNull()?.let { searchHighResMissStreak = it.coerceIn(4, 400) }
                "search_high_res_short_edge" -> value.toIntOrNull()?.let { searchHighResShortEdge = it.coerceIn(360, 960) }
                "search_ultra_high_res_miss" -> value.toIntOrNull()?.let { searchUltraHighResMissStreak = it.coerceIn(8, 600) }
                "search_ultra_high_res_short_edge" -> value.toIntOrNull()?.let { searchUltraHighResShortEdge = it.coerceIn(480, 1400) }
                "init_box" -> value.toIntOrNull()?.let { initBoxSize = it.coerceIn(48, 360) }
                "fallback_box_max" -> value.toIntOrNull()?.let { fallbackMaxBoxSize = it.coerceIn(100, 420) }
                "homography_distortion" -> value.toDoubleOrNull()?.let { homographyMaxDistortion = it.coerceIn(0.05, 0.60) }
                "homography_scale_min" -> value.toDoubleOrNull()?.let { homographyScaleMin = it.coerceIn(0.40, 2.0) }
                "homography_scale_max" -> value.toDoubleOrNull()?.let { homographyScaleMax = it.coerceIn(0.60, 3.0) }
                "homography_min_det" -> value.toDoubleOrNull()?.let { homographyMinJacobianDet = it.coerceIn(1e-7, 1e-2) }
                "template_min_texture" -> value.toDoubleOrNull()?.let { templateMinTextureScore = it.coerceIn(0.0, 200.0) }
                "orb_clahe" -> parseBoolean(value)?.let { orbUseClahe = it }
                "orb_far_features" -> value.toIntOrNull()?.let { orbFarBoostFeatures = it.coerceIn(700, 2800) }
                "first_lock_stable_frames" -> value.toIntOrNull()?.let { firstLockStableFrames = it.coerceIn(3, 90) }
                "first_lock_stable_ms" -> value.toLongOrNull()?.let { firstLockStableMs = it.coerceIn(120L, 3_500L) }
                "first_lock_stable_px" -> value.toDoubleOrNull()?.let { firstLockStablePx = it.coerceIn(8.0, 240.0) }
                "first_lock_min_iou" -> value.toDoubleOrNull()?.let { firstLockMinIou = it.coerceIn(0.40, 0.98) }
                "first_lock_gap_ms" -> value.toLongOrNull()?.let { firstLockGapMs = it.coerceIn(80L, 1_200L) }
                "allow_fallback_lock" -> parseBoolean(value)?.let { allowFallbackLock = it }
                "fallback_refine_expand" -> value.toDoubleOrNull()?.let { fallbackRefineExpandFactor = it.coerceIn(1.1, 3.0) }
                "fallback_refine_min_good" -> value.toIntOrNull()?.let { fallbackRefineMinGoodMatches = it.coerceIn(4, 120) }
                "fallback_refine_min_inliers" -> value.toIntOrNull()?.let { fallbackRefineMinInliers = it.coerceIn(3, 80) }
                "fallback_refine_min_conf" -> value.toDoubleOrNull()?.let { fallbackRefineMinConfidence = it.coerceIn(0.0, 1.0) }
                "track_verify_interval" -> value.toIntOrNull()?.let { trackVerifyIntervalFrames = it.coerceIn(4, 120) }
                "track_verify_global_interval" -> value.toIntOrNull()?.let { trackVerifyGlobalIntervalFrames = it.coerceIn(10, 300) }
                "track_verify_expand" -> value.toDoubleOrNull()?.let { trackVerifyLocalExpandFactor = it.coerceIn(1.2, 4.0) }
                "track_verify_min_good" -> value.toIntOrNull()?.let {
                    trackVerifyMinGoodMatches = it.coerceIn(TRACK_VERIFY_HARD_MIN_GOOD_MATCHES, 120)
                }
                "track_verify_min_inliers" -> value.toIntOrNull()?.let { trackVerifyMinInliers = it.coerceIn(3, 80) }
                "track_verify_fail_tol" -> value.toIntOrNull()?.let { trackVerifyFailTolerance = it.coerceIn(1, 12) }
                "track_verify_hard_tol" -> value.toIntOrNull()?.let { trackVerifyHardDriftTolerance = it.coerceIn(1, 6) }
                "track_verify_recenter_px" -> value.toDoubleOrNull()?.let { trackVerifyRecenterPx = it.coerceIn(8.0, 360.0) }
                "track_verify_min_iou" -> value.toDoubleOrNull()?.let { trackVerifyMinIou = it.coerceIn(0.0, 0.95) }
                "track_verify_conf_margin" -> value.toDoubleOrNull()?.let { trackVerifySwitchConfidenceMargin = it.coerceIn(0.0, 0.60) }
                "kcf_failures" -> value.toIntOrNull()?.let { kcfMaxFailStreak = it.coerceIn(1, 10) }
                "tracker_gc" -> parseBoolean(value)?.let { forceTrackerGcOnDrop = it }
            }
        }

        if (homographyScaleMin > homographyScaleMax) {
            val t = homographyScaleMin
            homographyScaleMin = homographyScaleMax
            homographyScaleMax = t
        }
        if (searchHighResShortEdge < searchShortEdge) {
            searchHighResShortEdge = searchShortEdge
        }
        if (searchUltraHighResShortEdge < searchHighResShortEdge) {
            searchUltraHighResShortEdge = searchHighResShortEdge
        }
        if (searchUltraHighResMissStreak < searchHighResMissStreak) {
            searchUltraHighResMissStreak = searchHighResMissStreak
        }
        if (trackVerifyGlobalIntervalFrames < trackVerifyIntervalFrames) {
            trackVerifyGlobalIntervalFrames = trackVerifyIntervalFrames
        }
        orbSoftMinGoodMatches = orbSoftMinGoodMatches.coerceAtMost(orbMinGoodMatches).coerceAtLeast(4)
        orbSoftMinInliers = orbSoftMinInliers.coerceAtMost(orbMinInliers).coerceAtLeast(3)
        trackVerifyMinInliers = trackVerifyMinInliers.coerceAtMost(trackVerifyMinGoodMatches).coerceAtLeast(3)
        fallbackRefineMinInliers = fallbackRefineMinInliers.coerceAtMost(fallbackRefineMinGoodMatches).coerceAtLeast(3)

        orb.setMaxFeatures(orbMaxFeatures)
        if (shouldRefreshTemplate && templateSourceGrays.isNotEmpty()) {
            rebuildTemplatePyramid(templateSourceGrays)
        }
        logEffectiveParams("override")
    }

    private fun parseBoolean(value: String): Boolean? {
        return when (value.trim().lowercase(Locale.US)) {
            "1", "true", "yes", "on" -> true
            "0", "false", "no", "off" -> false
            else -> null
        }
    }

    private fun logEffectiveParams(source: String) {
        Log.w(
            TAG,
            "EVAL_EVENT type=PARAMS source=$source mode=${trackerMode.name.lowercase(Locale.US)} " +
                "orbFeatures=$orbMaxFeatures ratio=${fmt(orbLoweRatio)} minMatches=$orbMinGoodMatches minInliers=$orbMinInliers " +
                "softMatches=$orbSoftMinGoodMatches softInliers=$orbSoftMinInliers ransac=${fmt(orbRansacThreshold)} " +
                "searchShort=$searchShortEdge searchHiMiss=$searchHighResMissStreak searchHiShort=$searchHighResShortEdge " +
                "searchUltraMiss=$searchUltraHighResMissStreak searchUltraShort=$searchUltraHighResShortEdge " +
                "initBox=$initBoxSize fallbackMax=$fallbackMaxBoxSize " +
                "hDist=${fmt(homographyMaxDistortion)} hScaleMin=${fmt(homographyScaleMin)} hScaleMax=${fmt(homographyScaleMax)} " +
                "hDet=${fmt(homographyMinJacobianDet)} tplTextureMin=${fmt(templateMinTextureScore)} clahe=$orbUseClahe " +
                "farBoostFeatures=$orbFarBoostFeatures " +
                "firstLockFrames=$firstLockStableFrames firstLockMs=$firstLockStableMs firstLockPx=${fmt(firstLockStablePx)} " +
                "firstLockIou=${fmt(firstLockMinIou)} allowFallbackLock=$allowFallbackLock " +
                "fallbackRefineExpand=${fmt(fallbackRefineExpandFactor)} fallbackRefineGood=$fallbackRefineMinGoodMatches " +
                "fallbackRefineInliers=$fallbackRefineMinInliers fallbackRefineConf=${fmt(fallbackRefineMinConfidence)} " +
                "firstLockGapMs=$firstLockGapMs verifyInt=$trackVerifyIntervalFrames verifyGlobal=$trackVerifyGlobalIntervalFrames " +
                "verifyExpand=${fmt(trackVerifyLocalExpandFactor)} verifyGood=$trackVerifyMinGoodMatches " +
                "verifyInliers=$trackVerifyMinInliers verifyTol=$trackVerifyFailTolerance verifyHardTol=$trackVerifyHardDriftTolerance " +
                "verifyRecenter=${fmt(trackVerifyRecenterPx)} " +
                "verifyIou=${fmt(trackVerifyMinIou)} verifyConfMargin=${fmt(trackVerifySwitchConfidenceMargin)} " +
                "kcfFail=$kcfMaxFailStreak trackerGc=$forceTrackerGcOnDrop"
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

    fun setTemplateImage(bitmap: Bitmap): Boolean = setTemplateImages(listOf(bitmap))

    fun setTemplateImages(bitmaps: List<Bitmap>): Boolean {
        if (bitmaps.isEmpty()) {
            clearTemplateSources()
            clearTemplateFeatures()
            templateGray?.release()
            templateGray = null
            lastTemplateReadyState = false
            return false
        }

        val pendingSources = ArrayList<Mat>(bitmaps.size)
        try {
            for (bitmap in bitmaps) {
                val rgba = Mat()
                val gray = Mat()
                try {
                    Utils.bitmapToMat(bitmap, rgba)
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
                    val normalized = normalizeTemplateSize(gray)
                    pendingSources += normalized
                } finally {
                    gray.release()
                    rgba.release()
                }
            }
            if (pendingSources.isEmpty()) {
                lastTemplateReadyState = false
                return false
            }

            clearTemplateSources()
            templateSourceGrays.addAll(pendingSources)
            templateLibrarySize = templateSourceGrays.size
            pendingSources.clear()

            templateGray?.release()
            templateGray = if (templateSourceGrays.isNotEmpty()) {
                templateSourceGrays.first().clone()
            } else {
                null
            }

            val ok = rebuildTemplatePyramid(templateSourceGrays)
            if (!ok) {
                Log.w(TAG, "template loaded but ORB template quality is weak")
                Log.w(
                    TAG,
                    "EVAL_EVENT type=TEMPLATE_WEAK kp=$templateKeypointCount texture=${fmt(templateTextureScore)} " +
                        "textureMin=${fmt(templateMinTextureScore)} templates=$templateLibrarySize"
                )
            }
            lastTemplateReadyState = ok
            clearFirstLockCandidate("template_changed")
            searchMissStreak = 0

            resetTracking(logSummary = false)
            pendingInitBox = null
            val first = templateSourceGrays.firstOrNull()
            val firstW = first?.cols() ?: 0
            val firstH = first?.rows() ?: 0
            Log.i(TAG, "template loaded: ${templateLibrarySize} templates first=${firstW}x${firstH}")
            Log.w(
                TAG,
                "EVAL_EVENT type=TEMPLATE_READY templates=$templateLibrarySize width=$firstW height=$firstH " +
                    "kp=$templateKeypointCount texture=${fmt(templateTextureScore)} ready=$ok"
            )
            return ok
        } finally {
            for (pending in pendingSources) {
                pending.release()
            }
            pendingSources.clear()
        }
    }

    private fun rebuildTemplatePyramid(grayTemplates: List<Mat>): Boolean {
        clearTemplateFeatures()
        if (grayTemplates.isEmpty()) return false

        val built = ArrayList<TemplateLevel>(TEMPLATE_PYRAMID_SCALES.size * grayTemplates.size)
        for (grayTemplate in grayTemplates) {
            for (scale in TEMPLATE_PYRAMID_SCALES) {
                val scaled = Mat()
                if (scale >= 0.999) {
                    grayTemplate.copyTo(scaled)
                } else {
                    Imgproc.resize(grayTemplate, scaled, Size(), scale, scale, Imgproc.INTER_AREA)
                }
                if (scaled.cols() < MIN_TEMPLATE_LEVEL_DIM || scaled.rows() < MIN_TEMPLATE_LEVEL_DIM) {
                    scaled.release()
                    continue
                }
                val level = buildTemplateLevel(scaled, scale)
                if (level != null) {
                    built += level
                } else {
                    scaled.release()
                }
            }
        }

        if (built.isEmpty()) {
            templateKeypointCount = 0
            templateTextureScore = 0.0
            templateKeypoints = null
            templateDescriptors8U = null
            templateDescriptors32F = null
            templateCorners = null
            return false
        }

        built.sortByDescending { it.keypointCount }
        if (built.size > MAX_ACTIVE_TEMPLATE_LEVELS) {
            val dropped = built.subList(MAX_ACTIVE_TEMPLATE_LEVELS, built.size).toList()
            for (level in dropped) {
                level.release()
            }
            built.subList(MAX_ACTIVE_TEMPLATE_LEVELS, built.size).clear()
        }
        templatePyramidLevels.addAll(built)

        val primary = templatePyramidLevels.first()
        templateKeypoints = primary.keypoints
        templateDescriptors8U = primary.descriptors8U
        templateDescriptors32F = primary.descriptors32F
        templateCorners = primary.corners
        templateKeypointCount = primary.keypointCount
        templateTextureScore = primary.textureScore

        val bestLevelKp = templatePyramidLevels.maxOf { it.keypointCount }
        templateKeypointCount = bestLevelKp
        return bestLevelKp >= MIN_TEMPLATE_USABLE_KEYPOINTS &&
            templateTextureScore >= templateMinTextureScore &&
            bestLevelKp >= MIN_TEMPLATE_LEVEL_KEYPOINTS
    }

    private fun clearTemplateSources() {
        for (source in templateSourceGrays) {
            source.release()
        }
        templateSourceGrays.clear()
        templateLibrarySize = 0
    }

    private fun buildTemplateLevel(grayScaled: Mat, scale: Double): TemplateLevel? {
        val enhanced = Mat()
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        val descriptors32F = Mat()
        val mask = Mat()
        try {
            if (orbUseClahe) {
                clahe.apply(grayScaled, enhanced)
            } else {
                grayScaled.copyTo(enhanced)
            }
            orb.setMaxFeatures(max(orbMaxFeatures, orbFarBoostFeatures))
            orb.detectAndCompute(enhanced, mask, keypoints, descriptors, false)
            if (descriptors.empty()) return null

            val kpCount = keypoints.toArray().size
            if (kpCount < MIN_TEMPLATE_LEVEL_KEYPOINTS) return null

            descriptors.convertTo(descriptors32F, CvType.CV_32F)
            val texture = computeTextureScore(enhanced)
            val corners = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(grayScaled.cols().toDouble(), 0.0),
                Point(grayScaled.cols().toDouble(), grayScaled.rows().toDouble()),
                Point(0.0, grayScaled.rows().toDouble())
            )

            return TemplateLevel(
                scale = scale,
                gray = grayScaled,
                keypoints = keypoints,
                descriptors8U = descriptors,
                descriptors32F = descriptors32F,
                corners = corners,
                keypointCount = kpCount,
                textureScore = texture
            )
        } finally {
            mask.release()
            enhanced.release()
        }
    }

    private fun clearTemplateFeatures() {
        for (level in templatePyramidLevels) {
            level.release()
        }
        templatePyramidLevels.clear()
        templateKeypoints = null
        templateDescriptors8U = null
        templateDescriptors32F = null
        templateCorners = null
        templateKeypointCount = 0
        templateTextureScore = 0.0
    }

    fun resetTracking(logSummary: Boolean = true) {
        if (logSummary) {
            logEvalSummary("reset")
        }
        releaseTracker("reset", requestGc = true)
        pendingInitBox = null
        lastTrackedBox = null
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchMissStreak = 0
        clearFirstLockCandidate("reset")
        overlayView.post { overlayView.reset() }
    }

    private fun releaseTracker(reason: String, requestGc: Boolean) {
        val hadTracker = tracker != null
        tracker = null
        isTracking = false
        if (!hadTracker) return

        Log.d(TAG, "EVAL_EVENT type=TRACKER_RELEASE reason=$reason")

        if (!requestGc || !forceTrackerGcOnDrop) return
        val now = SystemClock.elapsedRealtime()
        if (now - lastTrackerGcMs < TRACKER_GC_MIN_INTERVAL_MS) return
        lastTrackerGcMs = now
        runCatching {
            System.gc()
            System.runFinalization()
        }
    }

    fun logEvalSummary(reason: String = "manual") {
        val elapsed = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(1L)
        val avgMs = if (metricsFrames > 0) metricsTotalProcessMs.toDouble() / metricsFrames else 0.0
        val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
        val trackingRatio = if (metricsFrames > 0) metricsTrackingFrames.toDouble() / metricsFrames else 0.0
        Log.w(
            TAG,
            String.format(
                Locale.US,
                "EVAL_SUMMARY reason=%s mode=%s elapsedMs=%d frames=%d avgFrameMs=%.2f locks=%d lost=%d firstLockSec=%.3f trackRatio=%.3f maxTrackStreak=%d",
                reason,
                trackerMode.name.lowercase(Locale.US),
                elapsed,
                metricsFrames,
                avgMs,
                metricsLockCount,
                metricsLostCount,
                firstLockSec,
                trackingRatio,
                metricsMaxTrackingStreak
            )
        )
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
            updateMetrics((SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L))
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
            updateMetrics((SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L))
        }
    }

    private fun processFrame(frame: Mat) {
        frameCounter++
        when {
            attemptManualInitialization(frame) -> Unit
            isTracking -> trackFrame(frame)
            else -> searchFrameByOrb(frame)
        }
    }

    private fun attemptManualInitialization(frame: Mat): Boolean {
        val requested = pendingInitBox ?: return false
        pendingInitBox = null
        val safe = clampRect(requested, frame.cols(), frame.rows()) ?: return false
        initializeTracker(frame, safe, "manual")
        return true
    }

    private fun searchFrameByOrb(frame: Mat) {
        val templateReady = lastTemplateReadyState && templatePyramidLevels.isNotEmpty()
        if (!templateReady) {
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_SKIPPED reason=template_not_ready kp=$templateKeypointCount " +
                        "texture=${fmt(templateTextureScore)} textureMin=${fmt(templateMinTextureScore)}"
                )
            }
            lastTemplateReadyState = false
            clearFirstLockCandidate("template_not_ready")
            return
        }
        lastTemplateReadyState = true

        val candidate = findOrbMatch(frame)
        if (candidate == null) {
            searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
            expireFirstLockCandidateIfNeeded()
            return
        }
        val confirmed = maybeRefineFallbackCandidate(frame, candidate)

        if (!isCandidateEligibleForTemporal(confirmed)) {
            searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
            clearFirstLockCandidate("candidate_not_lockable")
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_STABLE state=reject " +
                        "good=${confirmed.goodMatches} inliers=${confirmed.inlierCount} " +
                        "conf=${fmt(confirmed.confidence)} h=${confirmed.usedHomography} " +
                        "fallback=${confirmed.fallbackReason ?: "none"}"
                )
            }
            return
        }

        searchMissStreak = if (confirmed.isStrong) {
            (searchMissStreak - 2).coerceAtLeast(0)
        } else {
            (searchMissStreak - 1).coerceAtLeast(0)
        }
        val promoted = accumulateFirstLockCandidate(confirmed, frame.cols(), frame.rows())
        if (promoted != null) {
            if (!isCandidateLockableForInit(promoted)) {
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_STABLE state=final_reject " +
                            "good=${promoted.goodMatches} inliers=${promoted.inlierCount} " +
                            "conf=${fmt(promoted.confidence)} h=${promoted.usedHomography} " +
                            "fallback=${promoted.fallbackReason ?: "none"}"
                    )
                }
                searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
                clearFirstLockCandidate("promoted_not_lockable")
                return
            }
            searchMissStreak = 0
            clearFirstLockCandidate("promoted_candidate")
            initializeTracker(
                frame,
                promoted.box,
                "orb_temporal_confirm good=${promoted.goodMatches} inliers=${promoted.inlierCount} " +
                    "conf=${fmt(promoted.confidence)} h=${promoted.usedHomography} ds=${fmt(promoted.searchScale)} " +
                    "fallback=${promoted.fallbackReason ?: "none"} matcher=${promoted.matcherType}"
            )
        }
    }

    private fun trackFrame(frame: Mat) {
        val tracked = Rect()
        val ok = tracker?.update(frame, tracked) == true
        if (!ok) {
            consecutiveTrackerFailures++
            Log.w(TAG, "KCF update failed: streak=$consecutiveTrackerFailures")
            if (consecutiveTrackerFailures < kcfMaxFailStreak) {
                lastTrackedBox?.let { dispatchTrackedRect(it) }
                return
            }
            onLost("kcf_update_fail")
            return
        }

        val safe = clampRect(tracked, frame.cols(), frame.rows())
        if (safe == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < kcfMaxFailStreak) {
                lastTrackedBox?.let { dispatchTrackedRect(it) }
                return
            }
            onLost("kcf_invalid_box")
            return
        }

        consecutiveTrackerFailures = 0
        val verifyDecision = maybeVerifyTracking(frame, safe)
        if (verifyDecision != null) {
            when (verifyDecision.action) {
                "realign" -> {
                    verifyDecision.box?.let {
                        initializeTracker(frame, it, verifyDecision.reason)
                        return
                    }
                }
                "drop" -> {
                    onLost(verifyDecision.reason)
                    return
                }
            }
        }

        lastTrackedBox = safe
        dispatchTrackedRect(safe)
    }

    private fun onLost(reason: String) {
        releaseTracker("lost_$reason", requestGc = true)
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        lastTrackedBox = null
        metricsLostCount++
        clearFirstLockCandidate("lost")
        Log.w(TAG, "EVAL_EVENT type=LOST reason=$reason lost=$metricsLostCount")
        overlayView.post { overlayView.reset() }
    }

    private fun initializeTracker(frame: Mat, box: Rect, reason: String) {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return
        val wasTracking = isTracking
        releaseTracker("reinit_$reason", requestGc = false)

        val freshTracker = runCatching {
            val kcf = TrackerKCF.create()
            kcf.init(frame, safe)
            kcf
        }.getOrElse { err ->
            Log.e(TAG, "KCF init failed", err)
            releaseTracker("reinit_failed", requestGc = true)
            return
        }

        tracker = freshTracker
        isTracking = true
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchMissStreak = 0
        lastTrackedBox = safe
        dispatchTrackedRect(safe)

        if (!wasTracking) {
            metricsLockCount++
            if (metricsFirstLockMs < 0L) {
                metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
            }
            Log.w(
                TAG,
                "EVAL_EVENT type=LOCK reason=$reason locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)}"
            )
        }

        Log.i(TAG, "KCF initialized ($reason): ${safe.x},${safe.y},${safe.width}x${safe.height}")
    }

    private fun maybeVerifyTracking(frame: Mat, trackedBox: Rect): TrackVerifyDecision? {
        if (trackVerifyIntervalFrames <= 0 || frameCounter % trackVerifyIntervalFrames.toLong() != 0L) {
            return null
        }

        val trackedQuality = evaluateBoxMatch(frame, trackedBox)
        val hardDrift = trackedQuality.goodMatches < TRACK_VERIFY_HARD_MIN_GOOD_MATCHES
        val trackedHealthy =
            trackedQuality.goodMatches >= trackVerifyMinGoodMatches &&
                trackedQuality.inlierCount >= trackVerifyMinInliers

        val localRegion = expandRectWithFactor(trackedBox, trackVerifyLocalExpandFactor, frame.cols(), frame.rows()) ?: trackedBox
        val localCandidate = findOrbMatchInRoi(frame, localRegion)
        val localHealthy =
            localCandidate != null &&
                localCandidate.goodMatches >= trackVerifyMinGoodMatches &&
                localCandidate.inlierCount >= trackVerifyMinInliers
        if (localCandidate != null && localHealthy) {
            val shift = pointDistance(rectCenter(trackedBox), rectCenter(localCandidate.box))
            val iou = rectIou(trackedBox, localCandidate.box)
            val shouldRealign = shift > trackVerifyRecenterPx || iou < trackVerifyMinIou
            trackVerifyHardDriftStreak = 0
            if (shouldRealign) {
                trackVerifyFailStreak = 0
                return TrackVerifyDecision(
                    action = "realign",
                    box = localCandidate.box,
                    reason = "verify_local_realign shift=${fmt(shift)} iou=${fmt(iou)} " +
                        "good=${localCandidate.goodMatches} inliers=${localCandidate.inlierCount}"
                )
            }
            trackVerifyFailStreak = 0
            return null
        }

        if (hardDrift) {
            trackVerifyHardDriftStreak++
        } else {
            trackVerifyHardDriftStreak = 0
        }

        if (hardDrift) {
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=TRACK_VERIFY state=hard_drift " +
                        "good=${trackedQuality.goodMatches} inliers=${trackedQuality.inlierCount} " +
                        "conf=${fmt(trackedQuality.confidence)} hardStreak=$trackVerifyHardDriftStreak/$trackVerifyHardDriftTolerance"
                )
            }
            if (trackVerifyHardDriftStreak >= trackVerifyHardDriftTolerance) {
                trackVerifyFailStreak = 0
                return TrackVerifyDecision(
                    action = "drop",
                    box = null,
                    reason = "verify_hard_drift good=${trackedQuality.goodMatches} inliers=${trackedQuality.inlierCount} conf=${fmt(trackedQuality.confidence)}"
                )
            }
        }

        if (trackedHealthy) {
            trackVerifyFailStreak = 0
        } else {
            trackVerifyFailStreak++
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=TRACK_VERIFY state=weak good=${trackedQuality.goodMatches} inliers=${trackedQuality.inlierCount} " +
                        "conf=${fmt(trackedQuality.confidence)} streak=$trackVerifyFailStreak/$trackVerifyFailTolerance"
                )
            }
            if (trackVerifyFailStreak >= trackVerifyFailTolerance) {
                trackVerifyHardDriftStreak = 0
                return TrackVerifyDecision(
                    action = "drop",
                    box = null,
                    reason = "verify_drift good=${trackedQuality.goodMatches} inliers=${trackedQuality.inlierCount} conf=${fmt(trackedQuality.confidence)}"
                )
            }
        }

        if (trackVerifyGlobalIntervalFrames > 0 && frameCounter % trackVerifyGlobalIntervalFrames.toLong() == 0L) {
            val globalCandidate = findOrbMatch(frame)
            val globalHealthy =
                globalCandidate != null &&
                    globalCandidate.goodMatches >= trackVerifyMinGoodMatches &&
                    globalCandidate.inlierCount >= trackVerifyMinInliers
            if (globalCandidate != null && globalHealthy) {
                val shift = pointDistance(rectCenter(trackedBox), rectCenter(globalCandidate.box))
                val iou = rectIou(trackedBox, globalCandidate.box)
                val confidenceDelta = globalCandidate.confidence - trackedQuality.confidence
                val shouldSwitch =
                    shift > trackVerifyRecenterPx &&
                        (
                            iou < trackVerifyMinIou ||
                                confidenceDelta > trackVerifySwitchConfidenceMargin ||
                                !trackedHealthy
                            )
                if (shouldSwitch) {
                    trackVerifyFailStreak = 0
                    trackVerifyHardDriftStreak = 0
                    return TrackVerifyDecision(
                        action = "realign",
                        box = globalCandidate.box,
                        reason = "verify_global_switch shift=${fmt(shift)} iou=${fmt(iou)} " +
                            "good=${globalCandidate.goodMatches} inliers=${globalCandidate.inlierCount} " +
                            "confDelta=${fmt(confidenceDelta)}"
                    )
                }
            }
        }

        return null
    }

    private fun findOrbMatchInRoi(frame: Mat, roiRect: Rect): OrbMatchCandidate? {
        val safeRoi = clampRect(roiRect, frame.cols(), frame.rows()) ?: return null
        val sub = frame.submat(safeRoi)
        try {
            val local = findOrbMatch(sub) ?: return null
            val shifted = Rect(
                local.box.x + safeRoi.x,
                local.box.y + safeRoi.y,
                local.box.width,
                local.box.height
            )
            val safeGlobal = clampRect(shifted, frame.cols(), frame.rows()) ?: return null
            return local.copy(box = safeGlobal)
        } finally {
            sub.release()
        }
    }

    private fun evaluateBoxMatch(frame: Mat, box: Rect): BoxMatchQuality {
        if (templatePyramidLevels.isEmpty()) return BoxMatchQuality(0, 0, 0.0)
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return BoxMatchQuality(0, 0, 0.0)

        val roi = frame.submat(safe)
        val gray = Mat()
        val input = Mat()
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        val mask = Mat()
        val knnMatches = ArrayList<MatOfDMatch>()
        val srcMat = MatOfPoint2f()
        val dstMat = MatOfPoint2f()
        val inlierMask = Mat()

        try {
            Imgproc.cvtColor(roi, gray, Imgproc.COLOR_RGB2GRAY)
            if (orbUseClahe) {
                clahe.apply(gray, input)
            } else {
                gray.copyTo(input)
            }

            orb.setMaxFeatures(max(300, orbMaxFeatures / 2))
            orb.detectAndCompute(input, mask, keypoints, descriptors, false)
            if (descriptors.empty() || descriptors.rows() < 2) {
                return BoxMatchQuality(0, 0, 0.0)
            }

            var best = BoxMatchQuality(0, 0, 0.0)
            val roiPoints = keypoints.toArray()
            for (level in templatePyramidLevels) {
                if (level.descriptors8U.rows() < 2) continue
                clearKnnMatches(knnMatches)
                bfMatcher.knnMatch(level.descriptors8U, descriptors, knnMatches, 2)
                val goodMatches = collectGoodMatches(knnMatches)
                if (goodMatches.isEmpty()) continue
                if (goodMatches.size < 4) {
                    if (goodMatches.size > best.goodMatches) {
                        best = BoxMatchQuality(goodMatches.size, 0, 0.0)
                    }
                    continue
                }

                val srcList = ArrayList<Point>(goodMatches.size)
                val dstList = ArrayList<Point>(goodMatches.size)
                val templatePoints = level.keypoints.toArray()
                fillMatchPointPairs(goodMatches, templatePoints, roiPoints, srcList, dstList)
                if (srcList.size < 4 || dstList.size < 4) continue

                srcMat.fromList(srcList)
                dstMat.fromList(dstList)
                inlierMask.release()
                inlierMask.create(0, 0, CvType.CV_8U)
                Calib3d.findHomography(srcMat, dstMat, Calib3d.RANSAC, orbRansacThreshold, inlierMask)?.release()
                val inliers = if (inlierMask.empty()) 0 else Core.countNonZero(inlierMask)
                val confidence = inliers.toDouble() / goodMatches.size.toDouble()
                if (
                    inliers > best.inlierCount ||
                    (inliers == best.inlierCount && goodMatches.size > best.goodMatches) ||
                    (inliers == best.inlierCount && goodMatches.size == best.goodMatches && confidence > best.confidence)
                ) {
                    best = BoxMatchQuality(goodMatches.size, inliers, confidence)
                }
            }
            return best
        } catch (_: Throwable) {
            return BoxMatchQuality(0, 0, 0.0)
        } finally {
            inlierMask.release()
            dstMat.release()
            srcMat.release()
            clearKnnMatches(knnMatches)
            mask.release()
            descriptors.release()
            keypoints.release()
            input.release()
            gray.release()
            roi.release()
        }
    }

    private fun accumulateFirstLockCandidate(candidate: OrbMatchCandidate, frameW: Int, frameH: Int): OrbMatchCandidate? {
        if (!isCandidateEligibleForTemporal(candidate)) return null
        val now = SystemClock.elapsedRealtime()
        val candidateCenter = Point(
            candidate.box.x + candidate.box.width * 0.5,
            candidate.box.y + candidate.box.height * 0.5
        )

        val previous = firstLockCandidateBox
        val iouWithPrevious = if (previous != null) rectIou(previous, candidate.box) else 1.0
        val minSide = min(candidate.box.width, candidate.box.height).toDouble()
        val useCenterRule =
            minSide <= FIRST_LOCK_SMALL_BOX_SIDE_PX ||
                (candidate.box.width.toDouble() * candidate.box.height.toDouble()) <= FIRST_LOCK_SMALL_BOX_AREA_PX
        val adaptiveIou = computeAdaptiveFirstLockIou(candidate.box, frameW, frameH)
        val centerDrift = pointDistance(firstLockCandidateCenter, candidateCenter)
        val centerRulePass = centerDrift <= FIRST_LOCK_SMALL_BOX_CENTER_DRIFT_PX
        val needReset =
            previous == null ||
                (now - firstLockCandidateLastMs) > firstLockGapMs ||
                pointDistance(firstLockCandidateCenter, candidateCenter) > firstLockStablePx ||
                if (useCenterRule) !centerRulePass else iouWithPrevious < adaptiveIou

        if (needReset) {
            firstLockCandidateBox = candidate.box
            firstLockCandidateCenter = candidateCenter
            firstLockCandidateFrames = 1
            firstLockCandidateStartMs = now
            firstLockCandidateLastMs = now
            firstLockCandidateBestGood = candidate.goodMatches
            firstLockCandidateBestInliers = candidate.inlierCount
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_STABLE state=seed good=${candidate.goodMatches} inliers=${candidate.inlierCount} " +
                        "iou=${fmt(iouWithPrevious)} minIou=${fmt(adaptiveIou)} centerDrift=${fmt(centerDrift)} " +
                        "centerRule=$useCenterRule box=${candidate.box.x},${candidate.box.y},${candidate.box.width}x${candidate.box.height}"
                )
            }
            return null
        }

        firstLockCandidateFrames++
        firstLockCandidateLastMs = now
        firstLockCandidateBox = candidate.box
        firstLockCandidateCenter.x = (firstLockCandidateCenter.x + candidateCenter.x) * 0.5
        firstLockCandidateCenter.y = (firstLockCandidateCenter.y + candidateCenter.y) * 0.5
        firstLockCandidateBestGood = max(firstLockCandidateBestGood, candidate.goodMatches)
        firstLockCandidateBestInliers = max(firstLockCandidateBestInliers, candidate.inlierCount)

        val stableMs = now - firstLockCandidateStartMs
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=SEARCH_STABLE state=accum frames=$firstLockCandidateFrames stableMs=$stableMs " +
                    "bestGood=$firstLockCandidateBestGood bestInliers=$firstLockCandidateBestInliers " +
                    "iou=${fmt(iouWithPrevious)} minIou=${fmt(adaptiveIou)} centerDrift=${fmt(centerDrift)} " +
                    "centerRule=$useCenterRule needFrames=$firstLockStableFrames needMs=$firstLockStableMs"
            )
        }

        val ready = firstLockCandidateFrames >= firstLockStableFrames && stableMs >= firstLockStableMs
        if (!ready) return null

        val promotedBox = firstLockCandidateBox ?: return null
        return OrbMatchCandidate(
            box = promotedBox,
            goodMatches = max(candidate.goodMatches, firstLockCandidateBestGood),
            inlierCount = max(candidate.inlierCount, firstLockCandidateBestInliers),
            confidence = candidate.confidence,
            usedHomography = candidate.usedHomography,
            searchScale = candidate.searchScale,
            fallbackReason = (candidate.fallbackReason ?: "soft_stable"),
            matcherType = candidate.matcherType,
            isStrong = true
        )
    }

    private fun clearFirstLockCandidate(reason: String) {
        if (firstLockCandidateFrames > 0 && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(TAG, "EVAL_EVENT type=SEARCH_STABLE state=clear reason=$reason frames=$firstLockCandidateFrames")
        }
        firstLockCandidateBox = null
        firstLockCandidateCenter = Point(0.0, 0.0)
        firstLockCandidateFrames = 0
        firstLockCandidateStartMs = 0L
        firstLockCandidateLastMs = 0L
        firstLockCandidateBestGood = 0
        firstLockCandidateBestInliers = 0
    }

    private fun expireFirstLockCandidateIfNeeded() {
        if (firstLockCandidateFrames <= 0) return
        val now = SystemClock.elapsedRealtime()
        if (now - firstLockCandidateLastMs > firstLockGapMs) {
            clearFirstLockCandidate("timeout")
        }
    }

    private fun findOrbMatch(frame: Mat): OrbMatchCandidate? {
        if (templatePyramidLevels.isEmpty()) return null

        val frameGray = Mat()
        var searchGray: Mat? = null
        var orbInput: Mat? = null
        val frameKeypoints = MatOfKeyPoint()
        val frameDescriptors = Mat()
        val frameDescriptors32F = Mat()
        val detectionMask = Mat()

        try {
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_RGB2GRAY)

            val effectiveSearchShortEdge = when {
                searchMissStreak >= searchUltraHighResMissStreak -> max(searchHighResShortEdge, searchUltraHighResShortEdge)
                searchMissStreak >= searchHighResMissStreak -> max(searchShortEdge, searchHighResShortEdge)
                else -> searchShortEdge
            }
            val searchScale = computeSearchScale(frameGray.cols(), frameGray.rows(), effectiveSearchShortEdge)
            searchGray = if (searchScale < 0.999) Mat() else frameGray
            if (searchScale < 0.999) {
                Imgproc.resize(frameGray, searchGray, Size(), searchScale, searchScale, Imgproc.INTER_AREA)
            }

            orbInput = if (orbUseClahe) Mat() else searchGray
            if (orbUseClahe) {
                clahe.apply(searchGray, orbInput)
            }

            val effectiveMaxFeatures = if (searchMissStreak >= searchHighResMissStreak) {
                max(orbMaxFeatures, orbFarBoostFeatures)
            } else {
                orbMaxFeatures
            }
            orb.setMaxFeatures(effectiveMaxFeatures)
            orb.detectAndCompute(orbInput, detectionMask, frameKeypoints, frameDescriptors, false)
            if (frameDescriptors.empty()) {
                logSearchDiag(
                    reason = "frame_desc_empty",
                    kpFrame = 0,
                    flannGood = 0,
                    bfGood = 0,
                    selectedGood = 0,
                    selectedInliers = 0,
                    matcher = "none",
                    detail = "searchShort=$effectiveSearchShortEdge orbFeatures=$effectiveMaxFeatures"
                )
                return null
            }
            if (frameDescriptors.rows() < 2) return null

            frameDescriptors.convertTo(frameDescriptors32F, CvType.CV_32F)
            val framePoints = frameKeypoints.toArray()
            val searchMat = requireNotNull(searchGray)

            var bestCandidate: OrbMatchCandidate? = null
            var bestDiag: SearchLevelResult? = null
            for (level in templatePyramidLevels) {
                val levelResult = findOrbMatchForLevel(
                    frame = frame,
                    searchGray = searchMat,
                    searchScale = searchScale,
                    level = level,
                    framePoints = framePoints,
                    frameDescriptors8U = frameDescriptors,
                    frameDescriptors32F = frameDescriptors32F
                )
                val candidate = levelResult.candidate
                if (candidate != null && isBetterOrbCandidate(candidate, bestCandidate)) {
                    bestCandidate = candidate
                }
                val currentDiag = bestDiag
                if (
                    currentDiag == null ||
                    levelResult.selectedInliers > currentDiag.selectedInliers ||
                    (
                        levelResult.selectedInliers == currentDiag.selectedInliers &&
                            levelResult.selectedGood > currentDiag.selectedGood
                        )
                ) {
                    bestDiag = levelResult
                }
            }

            if (bestCandidate != null) return bestCandidate

            val diag = bestDiag
            logSearchDiag(
                reason = diag?.reason ?: "search_no_candidate",
                kpFrame = framePoints.size,
                flannGood = diag?.flannGood ?: 0,
                bfGood = diag?.bfGood ?: 0,
                selectedGood = diag?.selectedGood ?: 0,
                selectedInliers = diag?.selectedInliers ?: 0,
                matcher = diag?.matcherType ?: "none",
                detail = (diag?.detail ?: "none") +
                    " searchShort=$effectiveSearchShortEdge orbFeatures=$effectiveMaxFeatures"
            )
            return null
        } catch (t: Throwable) {
            Log.e(TAG, "ORB/FLANN search failed", t)
            logSearchDiag(
                reason = "search_exception",
                kpFrame = -1,
                flannGood = 0,
                bfGood = 0,
                selectedGood = 0,
                selectedInliers = 0,
                matcher = "none",
                detail = t.javaClass.simpleName ?: "unknown"
            )
            return null
        } finally {
            detectionMask.release()
            frameDescriptors32F.release()
            frameDescriptors.release()
            frameKeypoints.release()
            if (orbInput != null && orbInput.nativeObj != searchGray?.nativeObj) {
                orbInput.release()
            }
            if (searchGray != null && searchGray.nativeObj != frameGray.nativeObj) {
                searchGray.release()
            }
            frameGray.release()
        }
    }

    private fun findOrbMatchForLevel(
        frame: Mat,
        searchGray: Mat,
        searchScale: Double,
        level: TemplateLevel,
        framePoints: Array<KeyPoint>,
        frameDescriptors8U: Mat,
        frameDescriptors32F: Mat
    ): SearchLevelResult {
        if (level.descriptors8U.rows() < 2 || frameDescriptors8U.rows() < 2) {
            return SearchLevelResult(
                candidate = null,
                reason = "descriptor_rows_small",
                flannGood = 0,
                bfGood = 0,
                selectedGood = 0,
                selectedInliers = 0,
                matcherType = "none",
                detail = "lvl=${fmt(level.scale)} tplRows=${level.descriptors8U.rows()} frmRows=${frameDescriptors8U.rows()}"
            )
        }

        val knnMatches = ArrayList<MatOfDMatch>()
        val srcMat = MatOfPoint2f()
        val dstMat = MatOfPoint2f()
        val inlierMask = Mat()
        val projectedCorners = MatOfPoint2f()
        var homography: Mat? = null

        try {
            var flannGoodMatches: List<DMatch> = emptyList()
            var bfGoodMatches: List<DMatch> = emptyList()
            var flannError: String? = null

            if (level.descriptors32F.rows() >= 2 && frameDescriptors32F.rows() >= 2) {
                runCatching {
                    flannMatcher.knnMatch(level.descriptors32F, frameDescriptors32F, knnMatches, 2)
                    flannGoodMatches = collectGoodMatches(knnMatches)
                }.onFailure {
                    flannError = it.javaClass.simpleName ?: "flann_exception"
                }
            }
            clearKnnMatches(knnMatches)

            runCatching {
                bfMatcher.knnMatch(level.descriptors8U, frameDescriptors8U, knnMatches, 2)
                bfGoodMatches = collectGoodMatches(knnMatches)
            }.onFailure {
                return SearchLevelResult(
                    candidate = null,
                    reason = "bf_exception",
                    flannGood = flannGoodMatches.size,
                    bfGood = 0,
                    selectedGood = flannGoodMatches.size,
                    selectedInliers = 0,
                    matcherType = "flann",
                    detail = "lvl=${fmt(level.scale)} err=${it.javaClass.simpleName ?: "bf_exception"}"
                )
            }
            clearKnnMatches(knnMatches)

            var bestMatches = flannGoodMatches
            var matcherType = "flann"
            if (bfGoodMatches.size > bestMatches.size) {
                bestMatches = bfGoodMatches
                matcherType = "bf_hamming_fallback"
            } else if (bestMatches.isEmpty() && bfGoodMatches.isNotEmpty()) {
                bestMatches = bfGoodMatches
                matcherType = "bf_hamming"
            }

            val baseDetail =
                "lvl=${fmt(level.scale)} tplKp=${level.keypointCount} tpl=${level.gray.cols()}x${level.gray.rows()}"
            if (bestMatches.size < orbSoftMinGoodMatches) {
                return SearchLevelResult(
                    candidate = null,
                    reason = "few_matches",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = 0,
                    matcherType = matcherType,
                    detail = "$baseDetail needSoft=$orbSoftMinGoodMatches flannErr=${flannError ?: "none"}"
                )
            }

            val srcList = ArrayList<Point>(bestMatches.size)
            val dstList = ArrayList<Point>(bestMatches.size)
            val templatePoints = level.keypoints.toArray()
            fillMatchPointPairs(bestMatches, templatePoints, framePoints, srcList, dstList)
            if (srcList.size < orbSoftMinGoodMatches || dstList.size < orbSoftMinGoodMatches) {
                return SearchLevelResult(
                    candidate = null,
                    reason = "pair_count_small",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = 0,
                    matcherType = matcherType,
                    detail = "$baseDetail src=${srcList.size} dst=${dstList.size}"
                )
            }

            if (srcList.size < 4 || dstList.size < 4) {
                val soft = buildSoftCandidate(
                    frame = frame,
                    points = dstList,
                    searchScale = searchScale,
                    goodMatches = bestMatches.size,
                    inliers = 0,
                    matcherType = matcherType,
                    fallbackReason = "soft_pairs"
                )
                return SearchLevelResult(
                    candidate = soft,
                    reason = if (soft == null) "soft_pairs_invalid" else "soft_pairs",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = 0,
                    matcherType = matcherType,
                    detail = baseDetail
                )
            }

            srcMat.fromList(srcList)
            dstMat.fromList(dstList)
            homography = Calib3d.findHomography(srcMat, dstMat, Calib3d.RANSAC, orbRansacThreshold, inlierMask)

            val inliers = if (inlierMask.empty()) 0 else Core.countNonZero(inlierMask)
            val inlierDstPoints = selectInlierPoints(dstList, inlierMask)
            if (inlierDstPoints.size < orbSoftMinInliers || inliers < orbSoftMinInliers) {
                val soft = buildSoftCandidate(
                    frame = frame,
                    points = dstList,
                    searchScale = searchScale,
                    goodMatches = bestMatches.size,
                    inliers = inliers,
                    matcherType = matcherType,
                    fallbackReason = "few_inliers"
                )
                return SearchLevelResult(
                    candidate = soft,
                    reason = "few_inliers",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = inliers,
                    matcherType = matcherType,
                    detail = "$baseDetail needSoft=$orbSoftMinInliers filtered=${inlierDstPoints.size}"
                )
            }

            var finalBox: Rect? = null
            var usedHomography = false
            var fallbackReason: String? = null
            if (homography != null && !homography.empty()) {
                Core.perspectiveTransform(level.corners, projectedCorners, homography)
                val projected = projectedCorners.toArray()
                val validationReason = validateHomography(
                    homography = homography,
                    corners = projected,
                    templateWidth = level.gray.cols().toDouble(),
                    templateHeight = level.gray.rows().toDouble(),
                    searchWidth = searchGray.cols(),
                    searchHeight = searchGray.rows()
                )
                if (validationReason == null) {
                    finalBox = buildProjectedBoundingRect(projected, searchScale, frame.cols(), frame.rows())
                    usedHomography = finalBox != null
                    if (finalBox == null) {
                        fallbackReason = "homography_rect_invalid"
                    }
                } else {
                    fallbackReason = validationReason
                }
            } else {
                fallbackReason = "homography_empty"
            }

            if (finalBox == null) {
                val pointsForFallback = if (inlierDstPoints.isNotEmpty()) inlierDstPoints else dstList
                val cluster = estimateFallbackCluster(pointsForFallback, searchScale)
                val centerOriginal = scalePointToOriginal(cluster.center, searchScale)
                val square = buildCenteredSquare(centerOriginal, cluster.sideOriginal)
                finalBox = clampRect(square, frame.cols(), frame.rows())
                if (finalBox == null) {
                    return SearchLevelResult(
                        candidate = null,
                        reason = "fallback_box_invalid",
                        flannGood = flannGoodMatches.size,
                        bfGood = bfGoodMatches.size,
                        selectedGood = bestMatches.size,
                        selectedInliers = inliers,
                        matcherType = matcherType,
                        detail = "$baseDetail fallback=${fallbackReason ?: "none"}"
                    )
                }
            }

            val confidence = inliers.toDouble() / bestMatches.size.toDouble()
            val isStrong =
                bestMatches.size >= orbMinGoodMatches &&
                    inliers >= orbMinInliers &&
                    inlierDstPoints.size >= orbMinInliers

            val candidate = OrbMatchCandidate(
                box = finalBox,
                goodMatches = bestMatches.size,
                inlierCount = inliers,
                confidence = confidence,
                usedHomography = usedHomography,
                searchScale = searchScale,
                fallbackReason = fallbackReason,
                matcherType = matcherType,
                isStrong = isStrong
            )
            return SearchLevelResult(
                candidate = candidate,
                reason = if (isStrong) "ok" else "soft_candidate",
                flannGood = flannGoodMatches.size,
                bfGood = bfGoodMatches.size,
                selectedGood = bestMatches.size,
                selectedInliers = inliers,
                matcherType = matcherType,
                detail = "$baseDetail fallback=${fallbackReason ?: "none"} usedH=$usedHomography"
            )
        } finally {
            homography?.release()
            projectedCorners.release()
            inlierMask.release()
            dstMat.release()
            srcMat.release()
            clearKnnMatches(knnMatches)
        }
    }

    private fun isBetterOrbCandidate(candidate: OrbMatchCandidate, current: OrbMatchCandidate?): Boolean {
        if (current == null) return true
        if (candidate.isStrong != current.isStrong) return candidate.isStrong
        if (candidate.inlierCount != current.inlierCount) return candidate.inlierCount > current.inlierCount
        if (candidate.goodMatches != current.goodMatches) return candidate.goodMatches > current.goodMatches
        if (abs(candidate.confidence - current.confidence) > 1e-6) return candidate.confidence > current.confidence
        if (candidate.usedHomography != current.usedHomography) return candidate.usedHomography
        return false
    }

    private fun isCandidateLockableForInit(candidate: OrbMatchCandidate): Boolean {
        if (!candidate.isStrong) return false
        if (candidate.usedHomography) return true
        if (!allowFallbackLock) return false
        val minGoodFallback =
            (orbMinGoodMatches - FALLBACK_LOCK_GOOD_RELAX).coerceAtLeast(orbSoftMinGoodMatches + 2)
        val minInliersFallback =
            (orbMinInliers - FALLBACK_LOCK_INLIER_RELAX).coerceAtLeast(orbSoftMinInliers + 1)
        return candidate.goodMatches >= minGoodFallback &&
            candidate.inlierCount >= minInliersFallback &&
            candidate.confidence >= FALLBACK_LOCK_MIN_CONFIDENCE
    }

    private fun isCandidateEligibleForTemporal(candidate: OrbMatchCandidate): Boolean {
        if (candidate.goodMatches < orbSoftMinGoodMatches) return false
        val minInliers = if (candidate.usedHomography) {
            orbSoftMinInliers
        } else {
            max(orbSoftMinInliers, fallbackRefineMinInliers - 1)
        }
        if (candidate.inlierCount < minInliers) return false
        if (!candidate.usedHomography && candidate.confidence < min(0.35, fallbackRefineMinConfidence)) return false
        return true
    }

    private fun maybeRefineFallbackCandidate(frame: Mat, candidate: OrbMatchCandidate): OrbMatchCandidate {
        if (candidate.usedHomography) return candidate

        val refineRegion =
            expandRectWithFactor(candidate.box, fallbackRefineExpandFactor, frame.cols(), frame.rows())
                ?: candidate.box
        val localCandidate = findOrbMatchInRoi(frame, refineRegion) ?: return candidate

        val merged = if (isBetterOrbCandidate(localCandidate, candidate)) localCandidate else candidate
        val passesRefine =
            localCandidate.goodMatches >= fallbackRefineMinGoodMatches &&
                localCandidate.inlierCount >= fallbackRefineMinInliers &&
                localCandidate.confidence >= fallbackRefineMinConfidence

        if (!passesRefine) return merged

        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=SEARCH_REFINE state=pass " +
                    "good=${localCandidate.goodMatches} inliers=${localCandidate.inlierCount} " +
                    "conf=${fmt(localCandidate.confidence)} h=${localCandidate.usedHomography}"
            )
        }
        return merged.copy(
            goodMatches = max(merged.goodMatches, localCandidate.goodMatches),
            inlierCount = max(merged.inlierCount, localCandidate.inlierCount),
            confidence = max(merged.confidence, localCandidate.confidence),
            usedHomography = merged.usedHomography || localCandidate.usedHomography,
            fallbackReason = "refined_${merged.fallbackReason ?: "none"}",
            isStrong = true
        )
    }

    private fun computeAdaptiveFirstLockIou(box: Rect, frameW: Int, frameH: Int): Double {
        if (frameW <= 0 || frameH <= 0) return firstLockMinIou
        val frameArea = frameW.toDouble() * frameH.toDouble()
        val boxArea = box.width.toDouble() * box.height.toDouble()
        val areaRatio = (boxArea / frameArea).coerceAtLeast(1e-5)
        val normalized = sqrt((areaRatio / FIRST_LOCK_BASE_AREA_RATIO).coerceAtMost(1.0))
        return (firstLockMinIou * normalized).coerceIn(FIRST_LOCK_MIN_IOU_FLOOR, firstLockMinIou)
    }

    private fun computeSearchScale(frameWidth: Int, frameHeight: Int, shortEdgeTarget: Int = searchShortEdge): Double {
        val shortEdge = min(frameWidth, frameHeight).toDouble()
        if (shortEdge <= shortEdgeTarget.toDouble()) return 1.0
        return (shortEdgeTarget.toDouble() / shortEdge).coerceIn(0.10, 1.0)
    }

    private fun clearKnnMatches(knnMatches: MutableList<MatOfDMatch>) {
        for (pair in knnMatches) {
            pair.release()
        }
        knnMatches.clear()
    }

    private fun buildSoftCandidate(
        frame: Mat,
        points: List<Point>,
        searchScale: Double,
        goodMatches: Int,
        inliers: Int,
        matcherType: String,
        fallbackReason: String
    ): OrbMatchCandidate? {
        if (points.size < orbSoftMinGoodMatches) return null
        val cluster = estimateFallbackCluster(points, searchScale)
        val centerSearch = cluster.center
        val sideOriginal = cluster.sideOriginal
        val centerOriginal = scalePointToOriginal(centerSearch, searchScale)
        val square = buildCenteredSquare(centerOriginal, sideOriginal)
        val safe = clampRect(square, frame.cols(), frame.rows()) ?: return null
        val confidence = if (goodMatches > 0) inliers.toDouble() / goodMatches.toDouble() else 0.0
        return OrbMatchCandidate(
            box = safe,
            goodMatches = goodMatches,
            inlierCount = inliers,
            confidence = confidence,
            usedHomography = false,
            searchScale = searchScale,
            fallbackReason = fallbackReason,
            matcherType = matcherType,
            isStrong = false
        )
    }

    private fun logSearchDiag(
        reason: String,
        kpFrame: Int,
        flannGood: Int,
        bfGood: Int,
        selectedGood: Int,
        selectedInliers: Int,
        matcher: String,
        detail: String
    ) {
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES != 0L) return
        Log.w(
            TAG,
            "EVAL_EVENT type=SEARCH_DIAG reason=$reason kpTpl=$templateKeypointCount kpFrm=$kpFrame " +
                "good=$selectedGood inliers=$selectedInliers flann=$flannGood bf=$bfGood matcher=$matcher " +
                "minGood=$orbMinGoodMatches minInliers=$orbMinInliers softGood=$orbSoftMinGoodMatches " +
                "softInliers=$orbSoftMinInliers miss=$searchMissStreak stable=$firstLockCandidateFrames/${firstLockStableFrames} " +
                "detail=$detail"
        )
    }

    private fun collectGoodMatches(knnMatches: List<MatOfDMatch>): List<DMatch> {
        val out = ArrayList<DMatch>(knnMatches.size)
        for (pair in knnMatches) {
            val m = pair.toArray()
            if (m.size < 2) continue
            if (m[0].distance < orbLoweRatio * m[1].distance) out += m[0]
        }
        return out
    }

    private fun fillMatchPointPairs(
        matches: List<DMatch>,
        templatePoints: Array<KeyPoint>,
        framePoints: Array<KeyPoint>,
        outTemplatePoints: MutableList<Point>,
        outFramePoints: MutableList<Point>
    ) {
        for (m in matches) {
            if (m.queryIdx !in templatePoints.indices || m.trainIdx !in framePoints.indices) continue
            outTemplatePoints += templatePoints[m.queryIdx].pt
            outFramePoints += framePoints[m.trainIdx].pt
        }
    }

    private fun selectInlierPoints(points: List<Point>, inlierMask: Mat): List<Point> {
        if (points.isEmpty() || inlierMask.empty()) return emptyList()
        val out = ArrayList<Point>(points.size)
        for (i in points.indices) {
            val value = when {
                inlierMask.rows() == 1 && i < inlierMask.cols() -> inlierMask.get(0, i)
                i < inlierMask.rows() -> inlierMask.get(i, 0)
                else -> null
            }
            if (value != null && value.isNotEmpty() && value[0] > 0.0) out += points[i]
        }
        return out
    }

    private fun validateHomography(
        homography: Mat,
        corners: Array<Point>,
        templateWidth: Double,
        templateHeight: Double,
        searchWidth: Int,
        searchHeight: Int
    ): String? {
        if (corners.size != 4) return "corner_count"
        val maxX = searchWidth * HOMOGRAPHY_COORD_BOUND_FACTOR
        val maxY = searchHeight * HOMOGRAPHY_COORD_BOUND_FACTOR
        for (p in corners) {
            if (!p.x.isFinite() || !p.y.isFinite()) return "corner_non_finite"
            if (p.x < -maxX || p.x > maxX || p.y < -maxY || p.y > maxY) return "corner_oob"
        }

        val signedArea = polygonSignedArea(corners)
        val polygonArea = abs(signedArea)
        if (polygonArea < MIN_HOMOGRAPHY_POLYGON_AREA) return "area_small"
        val frameArea = (searchWidth.toDouble() * searchHeight.toDouble()).coerceAtLeast(1.0)
        if (polygonArea / frameArea > MAX_HOMOGRAPHY_AREA_RATIO) return "area_ratio_high"
        if (signedArea * (templateWidth * templateHeight) <= 0.0) return "flip"

        val edges = edgeLengths(corners)
        if (edges.any { !it.isFinite() || it < MIN_HOMOGRAPHY_EDGE_PX }) return "edge_degenerate"
        val distortion = max(
            abs(edges[0] - edges[2]) / max(edges[0], edges[2]),
            abs(edges[1] - edges[3]) / max(edges[1], edges[3])
        )
        if (distortion > homographyMaxDistortion) return "distortion_high"

        val avgW = (edges[0] + edges[2]) * 0.5
        val avgH = (edges[1] + edges[3]) * 0.5
        val projectedAspect = avgW / avgH
        val templateAspect = (templateWidth / templateHeight).coerceAtLeast(1e-6)
        if (abs(projectedAspect / templateAspect - 1.0) > MAX_HOMOGRAPHY_ASPECT_DRIFT) return "aspect_drift"

        val det = computeHomographyJacobianDeterminant(
            homography = homography,
            x = templateWidth * 0.5,
            y = templateHeight * 0.5
        ) ?: return "jacobian_nan"
        if (det <= homographyMinJacobianDet) return "jacobian_det"

        return null
    }

    private fun computeHomographyJacobianDeterminant(homography: Mat, x: Double, y: Double): Double? {
        if (homography.rows() < 3 || homography.cols() < 3) return null
        val h00 = homography.get(0, 0)?.getOrNull(0) ?: return null
        val h01 = homography.get(0, 1)?.getOrNull(0) ?: return null
        val h02 = homography.get(0, 2)?.getOrNull(0) ?: return null
        val h10 = homography.get(1, 0)?.getOrNull(0) ?: return null
        val h11 = homography.get(1, 1)?.getOrNull(0) ?: return null
        val h12 = homography.get(1, 2)?.getOrNull(0) ?: return null
        val h20 = homography.get(2, 0)?.getOrNull(0) ?: return null
        val h21 = homography.get(2, 1)?.getOrNull(0) ?: return null
        val h22 = homography.get(2, 2)?.getOrNull(0) ?: return null

        val u = h00 * x + h01 * y + h02
        val v = h10 * x + h11 * y + h12
        val w = h20 * x + h21 * y + h22
        if (!w.isFinite() || abs(w) < 1e-9) return null

        val w2 = w * w
        val dxDx = (h00 * w - u * h20) / w2
        val dxDy = (h01 * w - u * h21) / w2
        val dyDx = (h10 * w - v * h20) / w2
        val dyDy = (h11 * w - v * h21) / w2
        val det = dxDx * dyDy - dxDy * dyDx
        return if (det.isFinite()) det else null
    }

    private fun polygonSignedArea(points: Array<Point>): Double {
        var sum = 0.0
        for (i in points.indices) {
            val p1 = points[i]
            val p2 = points[(i + 1) % points.size]
            sum += p1.x * p2.y - p1.y * p2.x
        }
        return 0.5 * sum
    }

    private fun edgeLengths(points: Array<Point>): DoubleArray {
        val out = DoubleArray(4)
        for (i in 0 until 4) {
            val p1 = points[i]
            val p2 = points[(i + 1) % 4]
            val dx = p1.x - p2.x
            val dy = p1.y - p2.y
            out[i] = sqrt(dx * dx + dy * dy)
        }
        return out
    }

    private fun buildProjectedBoundingRect(
        corners: Array<Point>,
        searchScale: Double,
        frameW: Int,
        frameH: Int
    ): Rect? {
        if (corners.size < 4) return null
        var minX = Double.POSITIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY
        for (p in corners) {
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }
        if (!minX.isFinite() || !minY.isFinite() || !maxX.isFinite() || !maxY.isFinite()) return null
        if (maxX - minX < MIN_HOMOGRAPHY_EDGE_PX || maxY - minY < MIN_HOMOGRAPHY_EDGE_PX) return null

        val topLeft = scalePointToOriginal(Point(minX, minY), searchScale)
        val bottomRight = scalePointToOriginal(Point(maxX, maxY), searchScale)
        val width = abs(bottomRight.x - topLeft.x).roundToInt().coerceAtLeast(MIN_BOX_DIM)
        val height = abs(bottomRight.y - topLeft.y).roundToInt().coerceAtLeast(MIN_BOX_DIM)
        val longEdge = max(width, height).toDouble()
        val shortEdge = min(width, height).toDouble().coerceAtLeast(1.0)
        val aspect = longEdge / shortEdge
        if (aspect > MAX_PROJECTED_BOX_ASPECT_RATIO) return null

        val frameArea = frameW.toDouble().coerceAtLeast(1.0) * frameH.toDouble().coerceAtLeast(1.0)
        val areaRatio = (width.toDouble() * height.toDouble()) / frameArea
        if (areaRatio > MAX_PROJECTED_BOX_AREA_RATIO) return null

        val rect = Rect(
            min(topLeft.x, bottomRight.x).roundToInt(),
            min(topLeft.y, bottomRight.y).roundToInt(),
            width,
            height
        )
        return clampRect(rect, frameW, frameH)
    }

    private fun pointDistance(a: Point, b: Point): Double {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun rectCenter(rect: Rect): Point {
        return Point(rect.x + rect.width * 0.5, rect.y + rect.height * 0.5)
    }

    private fun rectIou(a: Rect, b: Rect): Double {
        val left = max(a.x, b.x)
        val top = max(a.y, b.y)
        val right = min(a.x + a.width, b.x + b.width)
        val bottom = min(a.y + a.height, b.y + b.height)
        val interW = (right - left).coerceAtLeast(0)
        val interH = (bottom - top).coerceAtLeast(0)
        val interArea = interW.toDouble() * interH.toDouble()
        if (interArea <= 0.0) return 0.0
        val union = a.width.toDouble() * a.height.toDouble() + b.width.toDouble() * b.height.toDouble() - interArea
        if (union <= 1e-6) return 0.0
        return (interArea / union).coerceIn(0.0, 1.0)
    }

    private fun expandRectWithFactor(rect: Rect, factor: Double, frameW: Int, frameH: Int): Rect? {
        if (factor <= 1.0) return clampRect(rect, frameW, frameH)
        val cx = rect.x + rect.width * 0.5
        val cy = rect.y + rect.height * 0.5
        val halfW = rect.width * factor * 0.5
        val halfH = rect.height * factor * 0.5
        val expanded = Rect(
            (cx - halfW).roundToInt(),
            (cy - halfH).roundToInt(),
            (halfW * 2.0).roundToInt(),
            (halfH * 2.0).roundToInt()
        )
        return clampRect(expanded, frameW, frameH)
    }

    private fun computeTextureScore(gray: Mat): Double {
        if (gray.empty()) return 0.0
        val lap = Mat()
        val mean = MatOfDouble()
        val std = MatOfDouble()
        return try {
            Imgproc.Laplacian(gray, lap, CvType.CV_64F)
            Core.meanStdDev(lap, mean, std)
            val sigma = std.get(0, 0)?.getOrNull(0) ?: 0.0
            sigma * sigma
        } catch (_: Throwable) {
            0.0
        } finally {
            std.release()
            mean.release()
            lap.release()
        }
    }

    private fun scalePointToOriginal(point: Point, scale: Double): Point {
        if (scale >= 0.999) return point
        val inv = 1.0 / scale.coerceAtLeast(1e-6)
        return Point(point.x * inv, point.y * inv)
    }

    private fun estimateFallbackCluster(points: List<Point>, searchScale: Double): ClusterEstimate {
        if (points.isEmpty()) return ClusterEstimate(Point(0.0, 0.0), initBoxSize)
        val xs = points.map { it.x }.sorted()
        val ys = points.map { it.y }.sorted()
        val size = points.size
        val lo = ((size - 1) * CLUSTER_TRIM_LOW_RATIO).roundToInt().coerceIn(0, size - 1)
        val hi = ((size - 1) * CLUSTER_TRIM_HIGH_RATIO).roundToInt().coerceIn(lo, size - 1)

        val minX = xs[lo]
        val maxX = xs[hi]
        val minY = ys[lo]
        val maxY = ys[hi]

        val center = Point((minX + maxX) * 0.5, (minY + maxY) * 0.5)
        val spanSearch = max(maxX - minX, maxY - minY).coerceAtLeast(MIN_HOMOGRAPHY_EDGE_PX)
        val sideSearch = max(spanSearch * FALLBACK_SPAN_TO_SIDE_FACTOR, MIN_HOMOGRAPHY_EDGE_PX * 2.0)
        val sideOriginal = (sideSearch / searchScale.coerceAtLeast(1e-6)).roundToInt()
        val maxSide = max(fallbackMaxBoxSize, initBoxSize)
        return ClusterEstimate(center, sideOriginal.coerceIn(initBoxSize, maxSide))
    }

    private fun buildCenteredSquare(center: Point, side: Int): Rect {
        val safeSide = side.coerceAtLeast(MIN_BOX_DIM)
        val half = safeSide / 2
        return Rect(
            (center.x - half).roundToInt(),
            (center.y - half).roundToInt(),
            safeSide,
            safeSide
        )
    }

    private fun normalizeTemplateSize(template: Mat): Mat {
        val out = Mat()
        val w = template.cols().toDouble()
        val h = template.rows().toDouble()
        val maxDim = max(w, h)
        if (maxDim <= MAX_TEMPLATE_DIM) {
            template.copyTo(out)
            return out
        }
        val scale = MAX_TEMPLATE_DIM / maxDim
        Imgproc.resize(template, out, Size(w * scale, h * scale), 0.0, 0.0, Imgproc.INTER_AREA)
        return out
    }

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
                "EVAL_PERF mode=${trackerMode.name.lowercase(Locale.US)} frames=$metricsFrames " +
                    "avgFrameMs=${fmt(avgMs)} trackRatio=${fmt(trackRatio)} locks=$metricsLockCount " +
                    "lost=$metricsLostCount firstLockSec=${fmt(firstLockSec)}"
            )
        }

        if (metricsFrames % SUMMARY_LOG_INTERVAL_FRAMES == 0L) {
            logEvalSummary("periodic")
        }
    }

    private fun clampRect(rect: Rect, frameW: Int, frameH: Int): Rect? {
        if (frameW <= 0 || frameH <= 0) return null
        val x = rect.x.coerceIn(0, frameW - 1)
        val y = rect.y.coerceIn(0, frameH - 1)
        val maxW = frameW - x
        val maxH = frameH - y
        val w = rect.width.coerceAtMost(maxW)
        val h = rect.height.coerceAtMost(maxH)
        if (w < MIN_BOX_DIM || h < MIN_BOX_DIM) return null
        return Rect(x, y, w, h)
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
        private const val MIN_TEMPLATE_LEVEL_DIM = 20
        private const val MIN_TEMPLATE_LEVEL_KEYPOINTS = 10
        private const val MIN_TEMPLATE_USABLE_KEYPOINTS = 50
        private const val MAX_ACTIVE_TEMPLATE_LEVELS = 12
        private const val MAX_TEMPLATE_DIM = 480.0
        private val TEMPLATE_PYRAMID_SCALES = doubleArrayOf(1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125)

        private const val MIN_HOMOGRAPHY_EDGE_PX = 6.0
        private const val MIN_HOMOGRAPHY_POLYGON_AREA = 48.0
        private const val MAX_HOMOGRAPHY_AREA_RATIO = 0.45
        private const val MAX_HOMOGRAPHY_ASPECT_DRIFT = 0.35
        private const val MAX_PROJECTED_BOX_AREA_RATIO = 0.30
        private const val MAX_PROJECTED_BOX_ASPECT_RATIO = 5.0
        private const val HOMOGRAPHY_COORD_BOUND_FACTOR = 2.0
        private const val FALLBACK_SPAN_TO_SIDE_FACTOR = 2.8
        private const val CLUSTER_TRIM_LOW_RATIO = 0.15
        private const val CLUSTER_TRIM_HIGH_RATIO = 0.85
        private const val FIRST_LOCK_SMALL_BOX_SIDE_PX = 50.0
        private const val FIRST_LOCK_SMALL_BOX_AREA_PX = 2_500.0
        private const val FIRST_LOCK_SMALL_BOX_CENTER_DRIFT_PX = 20.0
        private const val FIRST_LOCK_BASE_AREA_RATIO = 0.015
        private const val FIRST_LOCK_MIN_IOU_FLOOR = 0.55
        private const val FALLBACK_LOCK_GOOD_RELAX = 2
        private const val FALLBACK_LOCK_INLIER_RELAX = 2
        private const val FALLBACK_LOCK_MIN_CONFIDENCE = 0.38

        private const val DEFAULT_ORB_FEATURES = 900
        private const val DEFAULT_ORB_LOWE_RATIO = 0.75
        private const val DEFAULT_ORB_MIN_GOOD_MATCHES = 10
        private const val DEFAULT_ORB_MIN_INLIERS = 6
        private const val DEFAULT_ORB_SOFT_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_ORB_SOFT_MIN_INLIERS = 3
        private const val DEFAULT_ORB_RANSAC_THRESHOLD = 4.0
        private const val DEFAULT_ORB_USE_CLAHE = true
        private const val DEFAULT_CLAHE_CLIP_LIMIT = 2.0
        private const val DEFAULT_CLAHE_TILE_SIZE = 8.0
        private const val DEFAULT_ORB_FAR_BOOST_FEATURES = 1600

        private const val DEFAULT_SEARCH_SHORT_EDGE = 600
        private const val DEFAULT_SEARCH_HIGH_RES_MISS_STREAK = 8
        private const val DEFAULT_SEARCH_HIGH_RES_SHORT_EDGE = 860
        private const val DEFAULT_SEARCH_ULTRA_HIGH_RES_MISS_STREAK = 24
        private const val DEFAULT_SEARCH_ULTRA_HIGH_RES_SHORT_EDGE = 1080
        private const val DEFAULT_INIT_BOX_SIZE = 100
        private const val DEFAULT_FALLBACK_MAX_BOX_SIZE = 180
        private const val DEFAULT_HOMOGRAPHY_MAX_DISTORTION = 0.20
        private const val DEFAULT_HOMOGRAPHY_SCALE_MIN = 0.60
        private const val DEFAULT_HOMOGRAPHY_SCALE_MAX = 1.80
        private const val DEFAULT_HOMOGRAPHY_MIN_JACOBIAN_DET = 1e-5
        private const val DEFAULT_TEMPLATE_MIN_TEXTURE_SCORE = 15.0

        private const val DEFAULT_FIRST_LOCK_STABLE_FRAMES = 2
        private const val DEFAULT_FIRST_LOCK_STABLE_MS = 220L
        private const val DEFAULT_FIRST_LOCK_STABLE_PX = 54.0
        private const val DEFAULT_FIRST_LOCK_MIN_IOU = 0.70
        private const val DEFAULT_FIRST_LOCK_GAP_MS = 380L
        private const val DEFAULT_ALLOW_FALLBACK_LOCK = true
        private const val DEFAULT_FALLBACK_REFINE_EXPAND_FACTOR = 1.8
        private const val DEFAULT_FALLBACK_REFINE_MIN_GOOD_MATCHES = 8
        private const val DEFAULT_FALLBACK_REFINE_MIN_INLIERS = 5
        private const val DEFAULT_FALLBACK_REFINE_MIN_CONFIDENCE = 0.45

        private const val DEFAULT_KCF_MAX_FAIL_STREAK = 3
        private const val DEFAULT_FORCE_TRACKER_GC_ON_DROP = true
        private const val TRACKER_GC_MIN_INTERVAL_MS = 1_200L

        private const val DEFAULT_TRACK_VERIFY_INTERVAL_FRAMES = 30
        private const val DEFAULT_TRACK_VERIFY_GLOBAL_INTERVAL_FRAMES = 60
        private const val DEFAULT_TRACK_VERIFY_LOCAL_EXPAND_FACTOR = 2.2
        private const val DEFAULT_TRACK_VERIFY_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_MIN_INLIERS = 3
        private const val DEFAULT_TRACK_VERIFY_FAIL_TOLERANCE = 1
        private const val DEFAULT_TRACK_VERIFY_RECENTER_PX = 48.0
        private const val DEFAULT_TRACK_VERIFY_MIN_IOU = 0.35
        private const val DEFAULT_TRACK_VERIFY_SWITCH_CONFIDENCE_MARGIN = 0.15
        private const val TRACK_VERIFY_HARD_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_HARD_DRIFT_TOLERANCE = 2

        private const val SEARCH_DIAG_INTERVAL_FRAMES = 20
        private const val PERF_LOG_INTERVAL_FRAMES = 10
        private const val SUMMARY_LOG_INTERVAL_FRAMES = 30
    }
}
