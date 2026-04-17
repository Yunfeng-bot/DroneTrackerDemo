package com.example.dronetracker

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.dronetracker.nativebridge.NativeTrackerBridge
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

    private enum class TrackBackend {
        KCF,
        NATIVE_NCNN,
        NATIVE_RKNN
    }

    private enum class TrackingStage {
        ACQUIRE,
        TRACK
    }

    private enum class NativeConfidenceAction {
        ACCEPT,
        HOLD,
        DROP_HARD,
        DROP_SOFT_STREAK,
        DROP_MIN_CONF
    }

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

    private data class MatchThresholds(
        val minGoodMatches: Int,
        val minInliers: Int
    )

    private data class SoftGateThresholds(
        val minGoodMatches: Int,
        val minInliers: Int,
        val relaxed: Boolean
    )

    private data class OrbThresholdConfig(
        val minGoodMatches: Int,
        val minInliers: Int,
        val softMinGoodMatches: Int,
        val softMinInliers: Int,
        val loweRatio: Double
    )

    private data class SmallTargetConfig(
        val areaRatio: Double,
        val minGoodMatches: Int,
        val minInliers: Int,
        val scaleThreshold: Double
    )

    private data class WeakFallbackConfig(
        val maxMatches: Int,
        val maxSpanPx: Double,
        val maxAreaPx: Double,
        val requireRefine: Boolean,
        val rescueEnabled: Boolean,
        val rescueRatio: Double,
        val rescueMinGood: Int,
        val relaxMissStreak: Int,
        val relaxFactor: Double,
        val coreRescueEnabled: Boolean,
        val coreMaxSpanPx: Double,
        val coreMaxAreaPx: Double
    )

    private data class SoftRelaxConfig(
        val enabled: Boolean,
        val missStreak: Int,
        val minGoodMatches: Int,
        val scaleThreshold: Double,
        val maxRatio: Double
    )

    private data class FallbackRefineConfig(
        val expandFactor: Double,
        val minGoodMatches: Int,
        val minInliers: Int,
        val minConfidence: Double,
        val loweRatio: Double
    )

    private data class FirstLockConfig(
        val stableFrames: Int,
        val stableMs: Long,
        val stablePx: Double,
        val minIou: Double,
        val gapMs: Long,
        val holdOnTemporalReject: Boolean,
        val smallCenterDriftPx: Double,
        val smallCenterDriftRelaxedPx: Double,
        val smallStablePx: Double,
        val smallRelaxMissStreak: Int,
        val smallDynamicCenterFactor: Double,
        val smallRelaxedIouFloor: Double,
        val outlierHoldMax: Int,
        val allowFallbackLock: Boolean
    )

    private data class FirstLockAdaptiveConfig(
        val missRelaxHighStreak: Int,
        val missRelaxHighFactor: Double,
        val missRelaxMidStreak: Int,
        val missRelaxMidFactor: Double,
        val missRelaxLowStreak: Int,
        val missRelaxLowFactor: Double,
        val resetRelaxMinFrames: Int,
        val resetRelaxMinIouResets: Int,
        val resetRelaxFactor: Double,
        val iouFloorStrongGood: Int,
        val iouFloorStrongInliers: Int,
        val iouFloorStrong: Double,
        val iouFloorMediumGood: Int,
        val iouFloorMediumInliers: Int,
        val iouFloorMedium: Double,
        val iouFloorBase: Double,
        val centerRelaxGood: Int,
        val centerRelaxInliers: Int,
        val centerRelaxStrongFactor: Double,
        val centerRelaxMissStreak: Int,
        val centerRelaxMissFactor: Double,
        val centerRelaxCapFactor: Double,
        val confStrongGood: Int,
        val confStrongInliers: Int,
        val confStrongMin: Double,
        val confMediumGood: Int,
        val confMediumInliers: Int,
        val confMediumMin: Double,
        val confSmallMin: Double,
        val confBaseMin: Double
    )

    private data class TrackVerifyConfig(
        val intervalFrames: Int,
        val localExpandFactor: Double,
        val minGoodMatches: Int,
        val minInliers: Int,
        val failTolerance: Int,
        val hardDriftTolerance: Int,
        val recenterPx: Double,
        val minIou: Double,
        val switchConfidenceMargin: Double,
        val nativeBypassConfidence: Double
    )

    private data class NativeGateConfig(
        val maxFailStreak: Int,
        val minConfidence: Double,
        val fuseSoftConfidence: Double,
        val fuseHardConfidence: Double,
        val fuseSoftStreak: Int,
        val fuseWarmupFrames: Int,
        val holdLastOnSoftReject: Boolean
    )

    private data class AutoInitVerifyConfig(
        val strongCandidateMinGood: Int,
        val strongCandidateMinInliers: Int,
        val strongCandidateMinConfidence: Double,
        val strongAppearanceMinLive: Double,
        val strongAppearanceMinReplay: Double,
        val localRoiExpandFactor: Double,
        val localCenterFactorLive: Double,
        val localCenterFactorReplay: Double,
        val localMinIouLive: Double,
        val localMinIouReplay: Double,
        val localMinConfScaleLive: Double,
        val localMinConfScaleReplay: Double,
        val localMinConfFloorLive: Double,
        val localMinConfFloorReplay: Double,
        val appearanceMinStrongGood: Int,
        val appearanceMinStrongInliers: Int,
        val appearanceMinStrong: Double,
        val appearanceMinMediumGood: Int,
        val appearanceMinMediumInliers: Int,
        val appearanceMinMedium: Double,
        val appearanceMinBase: Double,
        val appearanceLiveBias: Double
    )

    private data class TemporalGateConfig(
        val highGoodMatches: Int,
        val highInliers: Int,
        val minConfidenceHighGood: Double,
        val mediumGoodMatches: Int,
        val minConfidenceMedium: Double,
        val minConfidenceSmallRefined: Double,
        val minConfidenceBase: Double,
        val liveConfidenceRelax: Double,
        val liveConfidenceFloor: Double
    )

    private data class TrackGuardConfig(
        val maxCenterJumpFactor: Double,
        val minAreaRatio: Double,
        val maxAreaRatio: Double,
        val dropStreak: Int,
        val appearanceCheckIntervalFrames: Long,
        val minAppearanceScore: Double
    )

    private data class HeuristicConfig(
        val orb: OrbThresholdConfig,
        val smallTarget: SmallTargetConfig,
        val weakFallback: WeakFallbackConfig,
        val softRelax: SoftRelaxConfig,
        val fallbackRefine: FallbackRefineConfig,
        val firstLock: FirstLockConfig,
        val firstLockAdaptive: FirstLockAdaptiveConfig,
        val trackVerify: TrackVerifyConfig,
        val nativeGate: NativeGateConfig,
        val autoInitVerify: AutoInitVerifyConfig,
        val temporalGate: TemporalGateConfig,
        val trackGuard: TrackGuardConfig
    )

    private data class KalmanConfig(
        val enabled: Boolean,
        val processNoise: Double,
        val measurementNoise: Double,
        val maxPredictMs: Long,
        val usePredictedHold: Boolean,
        val dynamicMeasurementNoise: Boolean,
        val highConfidenceThreshold: Double,
        val lowConfidenceThreshold: Double,
        val highConfidenceNoiseScale: Double,
        val lowConfidenceNoiseScale: Double,
        val occlusionNoiseScale: Double,
        val feedNativePrior: Boolean,
        val preTrackPriorFeed: Boolean,
        val priorOnlyOnUncertain: Boolean,
        val priorMinIou: Double,
        val priorStaleMs: Long
    )

    private data class WeakFallbackStats(
        val spanX: Double,
        val spanY: Double,
        val area: Double
    )

    private data class KnnPair(
        val first: DMatch,
        val second: DMatch
    )

    private data class ClusterEstimate(
        val center: Point,
        val sideOriginal: Int
    )

    data class PredictionSnapshot(
        val frameId: Long,
        val x: Int,
        val y: Int,
        val width: Int,
        val height: Int,
        val confidence: Double,
        val tracking: Boolean
    )

    data class EvalMetricsSnapshot(
        val frames: Long,
        val locks: Int,
        val lost: Int,
        val trackRatio: Double,
        val avgFrameMs: Double,
        val firstLockSec: Double,
        val kalmanPredHold: Int,
        val kalmanPredHoldRatio: Double
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

    private class BoxKalmanPredictor {
        private data class State(
            var x: Double,
            var y: Double,
            var w: Double,
            var h: Double,
            var vx: Double = 0.0,
            var vy: Double = 0.0,
            var vw: Double = 0.0,
            var vh: Double = 0.0,
            var timestampMs: Long = 0L
        )

        private var state: State? = null

        fun reset() {
            state = null
        }

        fun seed(box: Rect, timestampMs: Long) {
            state = State(
                x = box.x.toDouble(),
                y = box.y.toDouble(),
                w = box.width.toDouble(),
                h = box.height.toDouble(),
                timestampMs = timestampMs
            )
        }

        fun predict(timestampMs: Long, cfg: KalmanConfig): Rect? {
            val s = state ?: return null
            val dt = computeDeltaSeconds(timestampMs, s.timestampMs)
            if (dt <= 0.0) return toRect(s)
            val damp = (1.0 - cfg.processNoise.coerceIn(0.001, 0.99) * 0.18).coerceIn(0.70, 0.995)
            s.x += s.vx * dt
            s.y += s.vy * dt
            s.w = (s.w + s.vw * dt).coerceAtLeast(MIN_BOX_DIM.toDouble())
            s.h = (s.h + s.vh * dt).coerceAtLeast(MIN_BOX_DIM.toDouble())
            s.vx *= damp
            s.vy *= damp
            s.vw *= damp
            s.vh *= damp
            s.timestampMs = timestampMs
            return toRect(s)
        }

        fun correct(box: Rect, timestampMs: Long, cfg: KalmanConfig, measurementNoiseOverride: Double? = null): Rect {
            val current = state
            if (current == null) {
                seed(box, timestampMs)
                return box
            }
            val dt = computeDeltaSeconds(timestampMs, current.timestampMs).coerceAtLeast(1e-3)
            predict(timestampMs, cfg)
            val s = state ?: return box

            val measurementNoise = (measurementNoiseOverride ?: cfg.measurementNoise).coerceAtLeast(1e-6)
            val gainBase =
                (cfg.processNoise / (cfg.processNoise + measurementNoise).coerceAtLeast(1e-6))
                    .coerceIn(0.08, 0.88)

            fun blend(measured: Double, predicted: Double): Double {
                return predicted + gainBase * (measured - predicted)
            }

            val mx = box.x.toDouble()
            val my = box.y.toDouble()
            val mw = box.width.toDouble().coerceAtLeast(MIN_BOX_DIM.toDouble())
            val mh = box.height.toDouble().coerceAtLeast(MIN_BOX_DIM.toDouble())

            val nextX = blend(mx, s.x)
            val nextY = blend(my, s.y)
            val nextW = blend(mw, s.w).coerceAtLeast(MIN_BOX_DIM.toDouble())
            val nextH = blend(mh, s.h).coerceAtLeast(MIN_BOX_DIM.toDouble())

            val velGain = (gainBase / dt) * 0.42
            s.vx += (mx - s.x) * velGain
            s.vy += (my - s.y) * velGain
            s.vw += (mw - s.w) * velGain
            s.vh += (mh - s.h) * velGain

            s.x = nextX
            s.y = nextY
            s.w = nextW
            s.h = nextH
            s.timestampMs = timestampMs
            return toRect(s)
        }

        private fun toRect(s: State): Rect {
            return Rect(
                s.x.roundToInt(),
                s.y.roundToInt(),
                s.w.roundToInt().coerceAtLeast(MIN_BOX_DIM),
                s.h.roundToInt().coerceAtLeast(MIN_BOX_DIM)
            )
        }

        private fun computeDeltaSeconds(nowMs: Long, lastMs: Long): Double {
            if (lastMs <= 0L || nowMs <= lastMs) return 0.0
            return ((nowMs - lastMs).coerceAtMost(200L)).toDouble() / 1000.0
        }
    }

    private var trackerMode = TrackerMode.ENHANCED
    private var tracker: TrackerKCF? = null
    private var pendingInitBox: Rect? = null
    private var isTracking = false
    private var trackingStage = TrackingStage.ACQUIRE
    private var lastTrackedBox: Rect? = null
    private var lastMeasuredTrackBox: Rect? = null
    private var lastTrackerGcMs = 0L
    private var trackMismatchStreak = 0
    private var trackAppearanceLowStreak = 0
    private var preferredTrackBackend = TrackBackend.NATIVE_NCNN
    private var activeTrackBackend = TrackBackend.KCF

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
    private var offsetX = 0f
    private var offsetY = 0f
    private var frameCounter = 0L
    private var consecutiveTrackerFailures = 0
    private var lastTemplateReadyState = true
    @Volatile
    private var latestPredictionFrame = 0L
    @Volatile
    private var latestPredictionBox: Rect? = null
    @Volatile
    private var latestPredictionConfidence = 0.0
    @Volatile
    private var latestPredictionTracking = false

    private var nv21Buffer: ByteArray? = null
    private var nativeRgbBuffer: ByteArray? = null
    private val nativeRgbMat = Mat()
    private var latestSearchFrame: Mat? = null
    private var isReplayInput = false

    private var metricsSessionStartMs = SystemClock.elapsedRealtime()
    private var diagSessionId = metricsSessionStartMs
    private var metricsFrames = 0L
    private var metricsTotalProcessMs = 0L
    private var metricsTrackingFrames = 0L
    private var metricsSearchingFrames = 0L
    private var metricsLockCount = 0
    private var metricsLostCount = 0
    private var metricsFirstLockMs = -1L
    private var metricsCurrentTrackingStreak = 0
    private var metricsMaxTrackingStreak = 0
    private var metricsSearchCandidateCount = 0
    private var metricsSearchMissCount = 0
    private var metricsSearchTemplateSkipCount = 0
    private var metricsSearchStrideSkipCount = 0
    private var metricsSearchBudgetSkipCount = 0
    private var metricsSearchBudgetTripCount = 0
    private var metricsSearchTemporalRejectCount = 0
    private var metricsSearchPromoteRejectCount = 0
    private var metricsSearchRefinePassCount = 0
    private var metricsSearchRefineRejectCount = 0
    private var metricsSearchStableSeedCount = 0
    private var metricsSearchStableAccumCount = 0
    private var metricsSearchPromoteCount = 0
    private var metricsSearchStableOutlierHoldCount = 0
    private var metricsSearchTemporalHoldCount = 0
    private var metricsSearchResetNoSeedCount = 0
    private var metricsSearchResetGapCount = 0
    private var metricsSearchResetStableDriftCount = 0
    private var metricsSearchResetCenterRuleCount = 0
    private var metricsSearchResetIouCount = 0
    private var metricsSearchResetConfidenceCount = 0
    private var metricsNativeFuseSoftCount = 0
    private var metricsNativeFuseHardCount = 0
    private var metricsNativeLowConfHoldCount = 0
    private var metricsKalmanPredHoldCount = 0
    private var metricsNativeConfSamples = 0L
    private var metricsNativeConfSum = 0.0
    private var metricsNativeConfMin = 1.0
    private var metricsNativeConfMax = 0.0
    private var metricsNativeSimSamples = 0L
    private var metricsNativeSimSum = 0.0
    private var metricsNativeSimMin = 1.0
    private var metricsNativeSimMax = 0.0
    private var metricsSearchLastReason = "none"

    private var orbMaxFeatures = DEFAULT_ORB_FEATURES
    private var orbFeatureHardCap = DEFAULT_ORB_FEATURE_HARD_CAP
    private var orbScaleFactor = DEFAULT_ORB_SCALE_FACTOR
    private var orbNLevels = DEFAULT_ORB_N_LEVELS
    private var orbFastThreshold = DEFAULT_ORB_FAST_THRESHOLD
    private var orbLoweRatio = DEFAULT_ORB_LOWE_RATIO
    private var orbMinGoodMatches = DEFAULT_ORB_MIN_GOOD_MATCHES
    private var orbMinInliers = DEFAULT_ORB_MIN_INLIERS
    private var orbSoftMinGoodMatches = DEFAULT_ORB_SOFT_MIN_GOOD_MATCHES
    private var orbSoftMinInliers = DEFAULT_ORB_SOFT_MIN_INLIERS
    private var orbRansacThreshold = DEFAULT_ORB_RANSAC_THRESHOLD
    private var orbUseClahe = DEFAULT_ORB_USE_CLAHE

    private var searchShortEdge = DEFAULT_SEARCH_SHORT_EDGE
    private var searchStrideFrames = DEFAULT_SEARCH_STRIDE_FRAMES
    private var searchOverBudgetMs = DEFAULT_SEARCH_OVER_BUDGET_MS
    private var searchOverBudgetSkipFrames = DEFAULT_SEARCH_OVER_BUDGET_SKIP_FRAMES
    private var searchHighResMissStreak = DEFAULT_SEARCH_HIGH_RES_MISS_STREAK
    private var searchHighResShortEdge = DEFAULT_SEARCH_HIGH_RES_SHORT_EDGE
    private var searchUltraHighResMissStreak = DEFAULT_SEARCH_ULTRA_HIGH_RES_MISS_STREAK
    private var searchUltraHighResShortEdge = DEFAULT_SEARCH_ULTRA_HIGH_RES_SHORT_EDGE
    private var searchMaxLongEdge = DEFAULT_SEARCH_MAX_LONG_EDGE
    private var searchMultiTemplateMaxLongEdge = DEFAULT_SEARCH_MULTI_TEMPLATE_MAX_LONG_EDGE
    private var initBoxSize = DEFAULT_INIT_BOX_SIZE
    private var fallbackMinBoxSize = DEFAULT_FALLBACK_MIN_BOX_SIZE
    private var fallbackMaxBoxSize = DEFAULT_FALLBACK_MAX_BOX_SIZE
    private var homographyMaxDistortion = DEFAULT_HOMOGRAPHY_MAX_DISTORTION
    private var homographyScaleMin = DEFAULT_HOMOGRAPHY_SCALE_MIN
    private var homographyScaleMax = DEFAULT_HOMOGRAPHY_SCALE_MAX
    private var homographyMinJacobianDet = DEFAULT_HOMOGRAPHY_MIN_JACOBIAN_DET
    private var templateMinTextureScore = DEFAULT_TEMPLATE_MIN_TEXTURE_SCORE
    private var orbFarBoostFeatures = DEFAULT_ORB_FAR_BOOST_FEATURES
    private var orbFarBoostMultiTemplateCap = DEFAULT_ORB_FAR_BOOST_MULTI_TEMPLATE_CAP

    private var kcfMaxFailStreak = DEFAULT_KCF_MAX_FAIL_STREAK
    private var nativeMaxFailStreak = DEFAULT_NATIVE_MAX_FAIL_STREAK
    private var nativeMinConfidence = DEFAULT_NATIVE_MIN_CONFIDENCE
    private var nativeFuseSoftConfidence = DEFAULT_NATIVE_FUSE_SOFT_CONFIDENCE
    private var nativeFuseHardConfidence = DEFAULT_NATIVE_FUSE_HARD_CONFIDENCE
    private var nativeFuseSoftStreak = DEFAULT_NATIVE_FUSE_SOFT_STREAK
    private var nativeFuseWarmupFrames = DEFAULT_NATIVE_FUSE_WARMUP_FRAMES
    private var nativeHoldLastOnSoftReject = DEFAULT_NATIVE_HOLD_LAST_ON_SOFT_REJECT
    private var nativeScoreLogIntervalFrames = DEFAULT_NATIVE_SCORE_LOG_INTERVAL_FRAMES
    private var nativeModelParamPathOverride: String? = DEFAULT_NATIVE_MODEL_PARAM_PATH
    private var nativeModelBinPathOverride: String? = DEFAULT_NATIVE_MODEL_BIN_PATH
    private var forceTrackerGcOnDrop = DEFAULT_FORCE_TRACKER_GC_ON_DROP
    private var searchMissStreak = 0
    private var searchBudgetCooldownFrames = 0
    private var nativeLowConfidenceStreak = 0
    private var nativeFuseWarmupRemaining = 0
    private var nativeOrbVerifyEnabled = DEFAULT_NATIVE_ORB_VERIFY_ENABLED
    private var lockHoldFrames = DEFAULT_LOCK_HOLD_FRAMES
    private var lockHoldFramesRemaining = 0
    private var lastNativeAcceptMs = 0L
    private var fastFirstLockFrames = DEFAULT_FAST_FIRST_LOCK_FRAMES
    private var fastFirstLockRemaining = 0
    private var lostOverlayHoldMs = DEFAULT_LOST_OVERLAY_HOLD_MS
    private var overlayResetToken = 0L

    private var trackVerifyIntervalFrames = DEFAULT_TRACK_VERIFY_INTERVAL_FRAMES
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
    private var trackVerifyNativeBypassConfidence = DEFAULT_TRACK_VERIFY_NATIVE_BYPASS_CONFIDENCE
    private var trackGuardMaxCenterJumpFactor = DEFAULT_TRACK_GUARD_MAX_CENTER_JUMP_FACTOR
    private var trackGuardMinAreaRatio = DEFAULT_TRACK_GUARD_MIN_AREA_RATIO
    private var trackGuardMaxAreaRatio = DEFAULT_TRACK_GUARD_MAX_AREA_RATIO
    private var trackGuardDropStreak = DEFAULT_TRACK_GUARD_DROP_STREAK
    private var trackGuardAppearanceCheckInterval = DEFAULT_TRACK_GUARD_APPEARANCE_CHECK_INTERVAL
    private var trackGuardMinAppearanceScore = DEFAULT_TRACK_GUARD_MIN_APPEARANCE_SCORE
    private var trackGuardAnchorEnabled = DEFAULT_TRACK_GUARD_ANCHOR_ENABLED
    private var trackGuardAnchorMaxDrop = DEFAULT_TRACK_GUARD_ANCHOR_MAX_DROP
    private var trackGuardAnchorMinScore = DEFAULT_TRACK_GUARD_ANCHOR_MIN_SCORE
    private var trackAnchorAppearanceScore = Double.NaN

    private var firstLockStableFrames = DEFAULT_FIRST_LOCK_STABLE_FRAMES
    private var firstLockStableMs = DEFAULT_FIRST_LOCK_STABLE_MS
    private var firstLockStablePx = DEFAULT_FIRST_LOCK_STABLE_PX
    private var firstLockMinIou = DEFAULT_FIRST_LOCK_MIN_IOU
    private var firstLockGapMs = DEFAULT_FIRST_LOCK_GAP_MS
    private var firstLockSmallCenterDriftPx = DEFAULT_FIRST_LOCK_SMALL_CENTER_DRIFT_PX
    private var firstLockSmallCenterDriftRelaxedPx = DEFAULT_FIRST_LOCK_SMALL_CENTER_DRIFT_RELAXED_PX
    private var firstLockSmallStablePx = DEFAULT_FIRST_LOCK_SMALL_STABLE_PX
    private var firstLockSmallRelaxMissStreak = DEFAULT_FIRST_LOCK_SMALL_RELAX_MISS_STREAK
    private var firstLockSmallDynamicCenterFactor = DEFAULT_FIRST_LOCK_SMALL_DYNAMIC_CENTER_FACTOR
    private var firstLockSmallRelaxedIouFloor = DEFAULT_FIRST_LOCK_SMALL_RELAXED_IOU_FLOOR
    private var firstLockHoldOnTemporalReject = DEFAULT_FIRST_LOCK_HOLD_ON_TEMPORAL_REJECT
    private var firstLockOutlierHoldMax = DEFAULT_FIRST_LOCK_OUTLIER_HOLD_MAX
    private var allowFallbackLock = DEFAULT_ALLOW_FALLBACK_LOCK
    private var fallbackRefineExpandFactor = DEFAULT_FALLBACK_REFINE_EXPAND_FACTOR
    private var fallbackRefineMinGoodMatches = DEFAULT_FALLBACK_REFINE_MIN_GOOD_MATCHES
    private var fallbackRefineMinInliers = DEFAULT_FALLBACK_REFINE_MIN_INLIERS
    private var fallbackRefineMinConfidence = DEFAULT_FALLBACK_REFINE_MIN_CONFIDENCE
    private var fallbackRefineLoweRatio = DEFAULT_FALLBACK_REFINE_LOWE_RATIO
    private var smallTargetAreaRatio = DEFAULT_SMALL_TARGET_AREA_RATIO
    private var smallTargetMinGoodMatches = DEFAULT_SMALL_TARGET_MIN_GOOD_MATCHES
    private var smallTargetMinInliers = DEFAULT_SMALL_TARGET_MIN_INLIERS
    private var smallTargetScaleThreshold = DEFAULT_SMALL_TARGET_SCALE_THRESHOLD
    private var weakFallbackMaxMatches = DEFAULT_WEAK_FALLBACK_MAX_MATCHES
    private var weakFallbackMaxSpanPx = DEFAULT_WEAK_FALLBACK_MAX_SPAN_PX
    private var weakFallbackMaxAreaPx = DEFAULT_WEAK_FALLBACK_MAX_AREA_PX
    private var weakFallbackRequireRefine = DEFAULT_WEAK_FALLBACK_REQUIRE_REFINE
    private var weakFallbackRescueEnabled = DEFAULT_WEAK_FALLBACK_RESCUE_ENABLED
    private var weakFallbackRescueRatio = DEFAULT_WEAK_FALLBACK_RESCUE_RATIO
    private var weakFallbackRescueMinGood = DEFAULT_WEAK_FALLBACK_RESCUE_MIN_GOOD
    private var weakFallbackRelaxMissStreak = DEFAULT_WEAK_FALLBACK_RELAX_MISS_STREAK
    private var weakFallbackRelaxFactor = DEFAULT_WEAK_FALLBACK_RELAX_FACTOR
    private var weakFallbackCoreRescueEnabled = DEFAULT_WEAK_FALLBACK_CORE_RESCUE_ENABLED
    private var weakFallbackCoreMaxSpanPx = DEFAULT_WEAK_FALLBACK_CORE_MAX_SPAN_PX
    private var weakFallbackCoreMaxAreaPx = DEFAULT_WEAK_FALLBACK_CORE_MAX_AREA_PX
    private var softRelaxEnabled = DEFAULT_SOFT_RELAX_ENABLED
    private var softRelaxMissStreak = DEFAULT_SOFT_RELAX_MISS_STREAK
    private var softRelaxMinGoodMatches = DEFAULT_SOFT_RELAX_MIN_GOOD_MATCHES
    private var softRelaxScaleThreshold = DEFAULT_SOFT_RELAX_SCALE_THRESHOLD
    private var softRelaxMaxRatio = DEFAULT_SOFT_RELAX_MAX_RATIO
    private var firstLockMissRelaxHighStreak = DEFAULT_FIRST_LOCK_MISS_RELAX_HIGH_STREAK
    private var firstLockMissRelaxHighFactor = DEFAULT_FIRST_LOCK_MISS_RELAX_HIGH_FACTOR
    private var firstLockMissRelaxMidStreak = DEFAULT_FIRST_LOCK_MISS_RELAX_MID_STREAK
    private var firstLockMissRelaxMidFactor = DEFAULT_FIRST_LOCK_MISS_RELAX_MID_FACTOR
    private var firstLockMissRelaxLowStreak = DEFAULT_FIRST_LOCK_MISS_RELAX_LOW_STREAK
    private var firstLockMissRelaxLowFactor = DEFAULT_FIRST_LOCK_MISS_RELAX_LOW_FACTOR
    private var firstLockResetRelaxMinFrames = DEFAULT_FIRST_LOCK_RESET_RELAX_MIN_FRAMES
    private var firstLockResetRelaxMinIouResets = DEFAULT_FIRST_LOCK_RESET_RELAX_MIN_IOU_RESETS
    private var firstLockResetRelaxFactor = DEFAULT_FIRST_LOCK_RESET_RELAX_FACTOR
    private var firstLockIouFloorStrongGood = DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG_GOOD
    private var firstLockIouFloorStrongInliers = DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG_INLIERS
    private var firstLockIouFloorStrong = DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG
    private var firstLockIouFloorMediumGood = DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM_GOOD
    private var firstLockIouFloorMediumInliers = DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM_INLIERS
    private var firstLockIouFloorMedium = DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM
    private var firstLockIouFloorBase = FIRST_LOCK_MIN_IOU_FLOOR
    private var firstLockCenterRelaxStrongGood = DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_GOOD
    private var firstLockCenterRelaxStrongInliers = DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_INLIERS
    private var firstLockCenterRelaxStrongFactor = DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_FACTOR
    private var firstLockCenterRelaxMissStreak = DEFAULT_FIRST_LOCK_CENTER_RELAX_MISS_STREAK
    private var firstLockCenterRelaxMissFactor = DEFAULT_FIRST_LOCK_CENTER_RELAX_MISS_FACTOR
    private var firstLockCenterRelaxCapFactor = DEFAULT_FIRST_LOCK_CENTER_RELAX_CAP_FACTOR
    private var firstLockConfStrongGood = DEFAULT_FIRST_LOCK_CONF_STRONG_GOOD
    private var firstLockConfStrongInliers = DEFAULT_FIRST_LOCK_CONF_STRONG_INLIERS
    private var firstLockConfStrongMin = DEFAULT_FIRST_LOCK_CONF_STRONG_MIN
    private var firstLockConfMediumGood = DEFAULT_FIRST_LOCK_CONF_MEDIUM_GOOD
    private var firstLockConfMediumInliers = DEFAULT_FIRST_LOCK_CONF_MEDIUM_INLIERS
    private var firstLockConfMediumMin = DEFAULT_FIRST_LOCK_CONF_MEDIUM_MIN
    private var firstLockConfSmallMin = DEFAULT_FIRST_LOCK_CONF_SMALL_MIN
    private var firstLockConfBaseMin = DEFAULT_FIRST_LOCK_CONF_BASE_MIN
    private var autoVerifyStrongMinGood = DEFAULT_AUTO_VERIFY_STRONG_MIN_GOOD
    private var autoVerifyStrongMinInliers = DEFAULT_AUTO_VERIFY_STRONG_MIN_INLIERS
    private var autoVerifyStrongMinConfidence = DEFAULT_AUTO_VERIFY_STRONG_MIN_CONFIDENCE
    private var autoVerifyStrongAppearanceMinLive = DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_LIVE
    private var autoVerifyStrongAppearanceMinReplay = DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_REPLAY
    private var autoVerifyLocalRoiExpandFactor = DEFAULT_AUTO_VERIFY_LOCAL_ROI_EXPAND_FACTOR
    private var autoVerifyLocalCenterFactorLive = DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_LIVE
    private var autoVerifyLocalCenterFactorReplay = DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_REPLAY
    private var autoVerifyLocalMinIouLive = DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_LIVE
    private var autoVerifyLocalMinIouReplay = DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_REPLAY
    private var autoVerifyLocalMinConfScaleLive = DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_LIVE
    private var autoVerifyLocalMinConfScaleReplay = DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_REPLAY
    private var autoVerifyLocalMinConfFloorLive = DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_LIVE
    private var autoVerifyLocalMinConfFloorReplay = DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_REPLAY
    private var autoVerifyAppearanceStrongGood = DEFAULT_AUTO_VERIFY_APPEAR_STRONG_GOOD
    private var autoVerifyAppearanceStrongInliers = DEFAULT_AUTO_VERIFY_APPEAR_STRONG_INLIERS
    private var autoVerifyAppearanceMinStrong = DEFAULT_AUTO_VERIFY_APPEAR_MIN_STRONG
    private var autoVerifyAppearanceMediumGood = DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_GOOD
    private var autoVerifyAppearanceMediumInliers = DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_INLIERS
    private var autoVerifyAppearanceMinMedium = DEFAULT_AUTO_VERIFY_APPEAR_MIN_MEDIUM
    private var autoVerifyAppearanceMinBase = DEFAULT_AUTO_VERIFY_APPEAR_MIN_BASE
    private var autoVerifyAppearanceLiveBias = DEFAULT_AUTO_VERIFY_APPEAR_LIVE_BIAS
    private var temporalHighGoodMatches = TEMPORAL_HIGH_GOOD_MATCHES
    private var temporalHighInliers = TEMPORAL_HIGH_INLIERS
    private var temporalMinConfidenceHighGood = TEMPORAL_MIN_CONFIDENCE_HIGH_GOOD
    private var temporalMediumGoodMatches = TEMPORAL_MEDIUM_GOOD_MATCHES
    private var temporalMinConfidenceMedium = TEMPORAL_MIN_CONFIDENCE_MEDIUM
    private var temporalMinConfidenceSmallRefined = DEFAULT_TEMPORAL_MIN_CONFIDENCE_SMALL_REFINED
    private var temporalMinConfidenceBase = DEFAULT_TEMPORAL_MIN_CONFIDENCE_BASE
    private var temporalLiveConfidenceRelax = DEFAULT_TEMPORAL_LIVE_CONFIDENCE_RELAX
    private var temporalLiveConfidenceFloor = DEFAULT_TEMPORAL_LIVE_CONFIDENCE_FLOOR
    private var firstLockCandidateBox: Rect? = null
    private var firstLockCandidateCenter = Point(0.0, 0.0)
    private var firstLockCandidateFrames = 0
    private var firstLockCandidateStartMs = 0L
    private var firstLockCandidateLastMs = 0L
    private var firstLockCandidateBestGood = 0
    private var firstLockCandidateBestInliers = 0
    private var firstLockCandidateBestScore = 0.0
    private var firstLockCandidateAppearanceSum = 0.0
    private var firstLockCandidateAppearanceMin = 1.0
    private var firstLockCandidateAppearanceLast = 0.0
    private var firstLockOutlierHoldStreak = 0
    private var lastSearchDiagReason = "none"

    @Volatile
    private var heuristicConfig: HeuristicConfig = buildHeuristicConfig()
    private var kalmanConfig = KalmanConfig(
        enabled = DEFAULT_KALMAN_ENABLED,
        processNoise = DEFAULT_KALMAN_PROCESS_NOISE,
        measurementNoise = DEFAULT_KALMAN_MEASUREMENT_NOISE,
        maxPredictMs = DEFAULT_KALMAN_MAX_PREDICT_MS,
        usePredictedHold = DEFAULT_KALMAN_USE_PREDICTED_HOLD,
        dynamicMeasurementNoise = DEFAULT_KALMAN_DYNAMIC_MEASUREMENT_NOISE,
        highConfidenceThreshold = DEFAULT_KALMAN_HIGH_CONFIDENCE_THRESHOLD,
        lowConfidenceThreshold = DEFAULT_KALMAN_LOW_CONFIDENCE_THRESHOLD,
        highConfidenceNoiseScale = DEFAULT_KALMAN_HIGH_CONFIDENCE_NOISE_SCALE,
        lowConfidenceNoiseScale = DEFAULT_KALMAN_LOW_CONFIDENCE_NOISE_SCALE,
        occlusionNoiseScale = DEFAULT_KALMAN_OCCLUSION_NOISE_SCALE,
        feedNativePrior = DEFAULT_KALMAN_FEED_NATIVE_PRIOR,
        preTrackPriorFeed = DEFAULT_KALMAN_PRETRACK_PRIOR_FEED,
        priorOnlyOnUncertain = DEFAULT_KALMAN_PRIOR_ONLY_ON_UNCERTAIN,
        priorMinIou = DEFAULT_KALMAN_PRIOR_MIN_IOU,
        priorStaleMs = DEFAULT_KALMAN_PRIOR_STALE_MS
    )
    private val kalmanPredictor = BoxKalmanPredictor()
    private var latestKalmanPrediction: Rect? = null
    private var lastKalmanMeasureMs = 0L

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
            configureOrbDetector(orbMaxFeatures)
            refreshHeuristicConfig()
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
                "orb_feature_cap", "orb_max_feature_cap", "orb_budget" -> value.toIntOrNull()?.let {
                    orbFeatureHardCap = it.coerceIn(200, 1200)
                    shouldRefreshTemplate = true
                }
                "orb_scale_factor" -> value.toDoubleOrNull()?.let {
                    orbScaleFactor = it.coerceIn(1.01, 1.40)
                    shouldRefreshTemplate = true
                }
                "orb_nlevels" -> value.toIntOrNull()?.let {
                    orbNLevels = it.coerceIn(4, 24)
                    shouldRefreshTemplate = true
                }
                "orb_fast_threshold" -> value.toIntOrNull()?.let {
                    orbFastThreshold = it.coerceIn(4, 60)
                    shouldRefreshTemplate = true
                }
                "orb_ratio" -> value.toDoubleOrNull()?.let { orbLoweRatio = it.coerceIn(0.50, 0.95) }
                "orb_min_matches" -> value.toIntOrNull()?.let { orbMinGoodMatches = it.coerceIn(8, 400) }
                "orb_min_inliers", "min_match_count" -> value.toIntOrNull()?.let { orbMinInliers = it.coerceIn(8, 300) }
                "orb_soft_min_matches" -> value.toIntOrNull()?.let { orbSoftMinGoodMatches = it.coerceIn(3, 300) }
                "orb_soft_min_inliers" -> value.toIntOrNull()?.let { orbSoftMinInliers = it.coerceIn(3, 200) }
                "orb_ransac" -> value.toDoubleOrNull()?.let { orbRansacThreshold = it.coerceIn(1.0, 12.0) }
                "search_short_edge" -> value.toIntOrNull()?.let { searchShortEdge = it.coerceIn(240, 720) }
                "search_stride", "search_stride_frames" -> value.toIntOrNull()?.let { searchStrideFrames = it.coerceIn(1, 8) }
                "search_over_budget_ms", "search_budget_ms" ->
                    value.toLongOrNull()?.let { searchOverBudgetMs = it.coerceIn(0L, 240L) }
                "search_over_budget_skip_frames", "search_budget_skip_frames", "search_budget_skip" ->
                    value.toIntOrNull()?.let { searchOverBudgetSkipFrames = it.coerceIn(0, 8) }
                "search_high_res_miss" -> value.toIntOrNull()?.let { searchHighResMissStreak = it.coerceIn(4, 400) }
                "search_high_res_short_edge" -> value.toIntOrNull()?.let { searchHighResShortEdge = it.coerceIn(360, 960) }
                "search_ultra_high_res_miss" -> value.toIntOrNull()?.let { searchUltraHighResMissStreak = it.coerceIn(8, 600) }
                "search_ultra_high_res_short_edge" -> value.toIntOrNull()?.let { searchUltraHighResShortEdge = it.coerceIn(480, 1400) }
                "search_max_long_edge" -> value.toIntOrNull()?.let { searchMaxLongEdge = it.coerceIn(320, 1400) }
                "search_multi_max_long_edge" -> value.toIntOrNull()?.let { searchMultiTemplateMaxLongEdge = it.coerceIn(320, 1200) }
                "init_box" -> value.toIntOrNull()?.let { initBoxSize = it.coerceIn(48, 360) }
                "fallback_box_min" -> value.toIntOrNull()?.let { fallbackMinBoxSize = it.coerceIn(32, 240) }
                "fallback_box_max" -> value.toIntOrNull()?.let { fallbackMaxBoxSize = it.coerceIn(64, 420) }
                "homography_distortion" -> value.toDoubleOrNull()?.let { homographyMaxDistortion = it.coerceIn(0.05, 0.60) }
                "homography_scale_min" -> value.toDoubleOrNull()?.let { homographyScaleMin = it.coerceIn(0.40, 2.0) }
                "homography_scale_max" -> value.toDoubleOrNull()?.let { homographyScaleMax = it.coerceIn(0.60, 3.0) }
                "homography_min_det" -> value.toDoubleOrNull()?.let { homographyMinJacobianDet = it.coerceIn(1e-7, 1e-2) }
                "template_min_texture" -> value.toDoubleOrNull()?.let { templateMinTextureScore = it.coerceIn(0.0, 200.0) }
                "orb_clahe" -> parseBoolean(value)?.let { orbUseClahe = it }
                "orb_far_features" -> value.toIntOrNull()?.let { orbFarBoostFeatures = it.coerceIn(700, 2800) }
                "orb_far_features_multi_cap" -> value.toIntOrNull()?.let { orbFarBoostMultiTemplateCap = it.coerceIn(600, 2600) }
                "first_lock_stable_frames" -> value.toIntOrNull()?.let { firstLockStableFrames = it.coerceIn(1, 90) }
                "first_lock_stable_ms" -> value.toLongOrNull()?.let { firstLockStableMs = it.coerceIn(120L, 3_500L) }
                "first_lock_stable_px" -> value.toDoubleOrNull()?.let { firstLockStablePx = it.coerceIn(8.0, 240.0) }
                "first_lock_min_iou" -> value.toDoubleOrNull()?.let { firstLockMinIou = it.coerceIn(0.40, 0.98) }
                "first_lock_gap_ms" -> value.toLongOrNull()?.let { firstLockGapMs = it.coerceIn(80L, 1_200L) }
                "first_lock_small_center_px" -> value.toDoubleOrNull()?.let { firstLockSmallCenterDriftPx = it.coerceIn(8.0, 120.0) }
                "first_lock_small_center_relaxed_px" ->
                    value.toDoubleOrNull()?.let { firstLockSmallCenterDriftRelaxedPx = it.coerceIn(12.0, 200.0) }
                "first_lock_small_stable_px" -> value.toDoubleOrNull()?.let { firstLockSmallStablePx = it.coerceIn(16.0, 260.0) }
                "first_lock_small_relax_miss" -> value.toIntOrNull()?.let { firstLockSmallRelaxMissStreak = it.coerceIn(2, 300) }
                "first_lock_small_dynamic_factor" ->
                    value.toDoubleOrNull()?.let { firstLockSmallDynamicCenterFactor = it.coerceIn(0.30, 2.20) }
                "first_lock_small_relaxed_iou" ->
                    value.toDoubleOrNull()?.let { firstLockSmallRelaxedIouFloor = it.coerceIn(0.0, 0.60) }
                "first_lock_temporal_hold" -> parseBoolean(value)?.let { firstLockHoldOnTemporalReject = it }
                "first_lock_outlier_hold_max" -> value.toIntOrNull()?.let { firstLockOutlierHoldMax = it.coerceIn(1, 30) }
                "allow_fallback_lock" -> parseBoolean(value)?.let { allowFallbackLock = it }
                "first_lock_miss_relax_high_streak" -> value.toIntOrNull()?.let { firstLockMissRelaxHighStreak = it.coerceIn(1, 600) }
                "first_lock_miss_relax_high_factor" -> value.toDoubleOrNull()?.let { firstLockMissRelaxHighFactor = it.coerceIn(0.40, 1.20) }
                "first_lock_miss_relax_mid_streak" -> value.toIntOrNull()?.let { firstLockMissRelaxMidStreak = it.coerceIn(1, 600) }
                "first_lock_miss_relax_mid_factor" -> value.toDoubleOrNull()?.let { firstLockMissRelaxMidFactor = it.coerceIn(0.40, 1.20) }
                "first_lock_miss_relax_low_streak" -> value.toIntOrNull()?.let { firstLockMissRelaxLowStreak = it.coerceIn(1, 600) }
                "first_lock_miss_relax_low_factor" -> value.toDoubleOrNull()?.let { firstLockMissRelaxLowFactor = it.coerceIn(0.40, 1.20) }
                "first_lock_reset_relax_min_frames" -> value.toIntOrNull()?.let { firstLockResetRelaxMinFrames = it.coerceIn(1, 30) }
                "first_lock_reset_relax_min_iou_resets" -> value.toIntOrNull()?.let { firstLockResetRelaxMinIouResets = it.coerceIn(1, 100) }
                "first_lock_reset_relax_factor" -> value.toDoubleOrNull()?.let { firstLockResetRelaxFactor = it.coerceIn(0.40, 1.20) }
                "first_lock_iou_floor_strong_good" -> value.toIntOrNull()?.let { firstLockIouFloorStrongGood = it.coerceIn(4, 80) }
                "first_lock_iou_floor_strong_inliers" -> value.toIntOrNull()?.let { firstLockIouFloorStrongInliers = it.coerceIn(3, 80) }
                "first_lock_iou_floor_strong" -> value.toDoubleOrNull()?.let { firstLockIouFloorStrong = it.coerceIn(0.0, 1.0) }
                "first_lock_iou_floor_medium_good" -> value.toIntOrNull()?.let { firstLockIouFloorMediumGood = it.coerceIn(4, 80) }
                "first_lock_iou_floor_medium_inliers" -> value.toIntOrNull()?.let { firstLockIouFloorMediumInliers = it.coerceIn(3, 80) }
                "first_lock_iou_floor_medium" -> value.toDoubleOrNull()?.let { firstLockIouFloorMedium = it.coerceIn(0.0, 1.0) }
                "first_lock_iou_floor_base" -> value.toDoubleOrNull()?.let { firstLockIouFloorBase = it.coerceIn(0.0, 1.0) }
                "first_lock_center_relax_strong_good" -> value.toIntOrNull()?.let { firstLockCenterRelaxStrongGood = it.coerceIn(4, 80) }
                "first_lock_center_relax_strong_inliers" -> value.toIntOrNull()?.let { firstLockCenterRelaxStrongInliers = it.coerceIn(3, 80) }
                "first_lock_center_relax_strong_factor" -> value.toDoubleOrNull()?.let { firstLockCenterRelaxStrongFactor = it.coerceIn(1.0, 2.0) }
                "first_lock_center_relax_miss_streak" -> value.toIntOrNull()?.let { firstLockCenterRelaxMissStreak = it.coerceIn(1, 600) }
                "first_lock_center_relax_miss_factor" -> value.toDoubleOrNull()?.let { firstLockCenterRelaxMissFactor = it.coerceIn(1.0, 2.0) }
                "first_lock_center_relax_cap_factor" -> value.toDoubleOrNull()?.let { firstLockCenterRelaxCapFactor = it.coerceIn(1.0, 2.0) }
                "first_lock_conf_strong_good" -> value.toIntOrNull()?.let { firstLockConfStrongGood = it.coerceIn(4, 80) }
                "first_lock_conf_strong_inliers" -> value.toIntOrNull()?.let { firstLockConfStrongInliers = it.coerceIn(3, 80) }
                "first_lock_conf_strong_min" -> value.toDoubleOrNull()?.let { firstLockConfStrongMin = it.coerceIn(0.0, 1.0) }
                "first_lock_conf_medium_good" -> value.toIntOrNull()?.let { firstLockConfMediumGood = it.coerceIn(4, 80) }
                "first_lock_conf_medium_inliers" -> value.toIntOrNull()?.let { firstLockConfMediumInliers = it.coerceIn(3, 80) }
                "first_lock_conf_medium_min" -> value.toDoubleOrNull()?.let { firstLockConfMediumMin = it.coerceIn(0.0, 1.0) }
                "first_lock_conf_small_min" -> value.toDoubleOrNull()?.let { firstLockConfSmallMin = it.coerceIn(0.0, 1.0) }
                "first_lock_conf_base_min" -> value.toDoubleOrNull()?.let { firstLockConfBaseMin = it.coerceIn(0.0, 1.0) }
                "fallback_refine_expand" -> value.toDoubleOrNull()?.let { fallbackRefineExpandFactor = it.coerceIn(1.1, 3.0) }
                "fallback_refine_min_good" -> value.toIntOrNull()?.let { fallbackRefineMinGoodMatches = it.coerceIn(4, 120) }
                "fallback_refine_min_inliers" -> value.toIntOrNull()?.let { fallbackRefineMinInliers = it.coerceIn(3, 80) }
                "fallback_refine_min_conf" -> value.toDoubleOrNull()?.let { fallbackRefineMinConfidence = it.coerceIn(0.0, 1.0) }
                "fallback_refine_ratio" -> value.toDoubleOrNull()?.let { fallbackRefineLoweRatio = it.coerceIn(0.60, 0.95) }
                "auto_verify_strong_min_good" -> value.toIntOrNull()?.let { autoVerifyStrongMinGood = it.coerceIn(4, 120) }
                "auto_verify_strong_min_inliers" -> value.toIntOrNull()?.let { autoVerifyStrongMinInliers = it.coerceIn(3, 80) }
                "auto_verify_strong_min_conf" -> value.toDoubleOrNull()?.let { autoVerifyStrongMinConfidence = it.coerceIn(0.0, 1.0) }
                "auto_verify_strong_appearance_live" -> value.toDoubleOrNull()?.let { autoVerifyStrongAppearanceMinLive = it.coerceIn(-1.0, 1.0) }
                "auto_verify_strong_appearance_replay" -> value.toDoubleOrNull()?.let { autoVerifyStrongAppearanceMinReplay = it.coerceIn(-1.0, 1.0) }
                "auto_verify_local_roi_expand" -> value.toDoubleOrNull()?.let { autoVerifyLocalRoiExpandFactor = it.coerceIn(1.1, 4.0) }
                "auto_verify_local_center_live" -> value.toDoubleOrNull()?.let { autoVerifyLocalCenterFactorLive = it.coerceIn(0.10, 2.0) }
                "auto_verify_local_center_replay" -> value.toDoubleOrNull()?.let { autoVerifyLocalCenterFactorReplay = it.coerceIn(0.10, 2.0) }
                "auto_verify_local_min_iou_live" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinIouLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_min_iou_replay" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinIouReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_min_conf_scale_live" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinConfScaleLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_min_conf_scale_replay" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinConfScaleReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_min_conf_floor_live" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinConfFloorLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_min_conf_floor_replay" -> value.toDoubleOrNull()?.let { autoVerifyLocalMinConfFloorReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_appear_strong_good" -> value.toIntOrNull()?.let { autoVerifyAppearanceStrongGood = it.coerceIn(4, 120) }
                "auto_verify_appear_strong_inliers" -> value.toIntOrNull()?.let { autoVerifyAppearanceStrongInliers = it.coerceIn(3, 80) }
                "auto_verify_appear_min_strong" -> value.toDoubleOrNull()?.let { autoVerifyAppearanceMinStrong = it.coerceIn(-1.0, 1.0) }
                "auto_verify_appear_medium_good" -> value.toIntOrNull()?.let { autoVerifyAppearanceMediumGood = it.coerceIn(4, 120) }
                "auto_verify_appear_medium_inliers" -> value.toIntOrNull()?.let { autoVerifyAppearanceMediumInliers = it.coerceIn(3, 80) }
                "auto_verify_appear_min_medium" -> value.toDoubleOrNull()?.let { autoVerifyAppearanceMinMedium = it.coerceIn(-1.0, 1.0) }
                "auto_verify_appear_min_base" -> value.toDoubleOrNull()?.let { autoVerifyAppearanceMinBase = it.coerceIn(-1.0, 1.0) }
                "auto_verify_appear_live_bias" -> value.toDoubleOrNull()?.let { autoVerifyAppearanceLiveBias = it.coerceIn(-1.0, 1.0) }
                "temporal_high_good" -> value.toIntOrNull()?.let { temporalHighGoodMatches = it.coerceIn(4, 120) }
                "temporal_high_inliers" -> value.toIntOrNull()?.let { temporalHighInliers = it.coerceIn(3, 80) }
                "temporal_min_conf_high" -> value.toDoubleOrNull()?.let { temporalMinConfidenceHighGood = it.coerceIn(0.0, 1.0) }
                "temporal_medium_good" -> value.toIntOrNull()?.let { temporalMediumGoodMatches = it.coerceIn(4, 120) }
                "temporal_min_conf_medium" -> value.toDoubleOrNull()?.let { temporalMinConfidenceMedium = it.coerceIn(0.0, 1.0) }
                "temporal_min_conf_small_refined" -> value.toDoubleOrNull()?.let { temporalMinConfidenceSmallRefined = it.coerceIn(0.0, 1.0) }
                "temporal_min_conf_base" -> value.toDoubleOrNull()?.let { temporalMinConfidenceBase = it.coerceIn(0.0, 1.0) }
                "temporal_live_conf_relax" -> value.toDoubleOrNull()?.let { temporalLiveConfidenceRelax = it.coerceIn(0.0, 1.0) }
                "temporal_live_conf_floor" -> value.toDoubleOrNull()?.let { temporalLiveConfidenceFloor = it.coerceIn(0.0, 1.0) }
                "small_target_area_ratio" -> value.toDoubleOrNull()?.let { smallTargetAreaRatio = it.coerceIn(0.005, 0.20) }
                "small_target_min_good" -> value.toIntOrNull()?.let { smallTargetMinGoodMatches = it.coerceIn(4, 120) }
                "small_target_min_inliers" -> value.toIntOrNull()?.let { smallTargetMinInliers = it.coerceIn(3, 80) }
                "small_target_scale" -> value.toDoubleOrNull()?.let { smallTargetScaleThreshold = it.coerceIn(0.30, 0.95) }
                "weak_fallback_max_matches" -> value.toIntOrNull()?.let { weakFallbackMaxMatches = it.coerceIn(4, 12) }
                "weak_fallback_max_span" -> value.toDoubleOrNull()?.let { weakFallbackMaxSpanPx = it.coerceIn(32.0, 280.0) }
                "weak_fallback_max_area" -> value.toDoubleOrNull()?.let { weakFallbackMaxAreaPx = it.coerceIn(1_024.0, 80_000.0) }
                "weak_fallback_relax_miss" -> value.toIntOrNull()?.let { weakFallbackRelaxMissStreak = it.coerceIn(2, 300) }
                "weak_fallback_relax_factor" -> value.toDoubleOrNull()?.let { weakFallbackRelaxFactor = it.coerceIn(1.0, 2.0) }
                "weak_fallback_require_refine" -> parseBoolean(value)?.let { weakFallbackRequireRefine = it }
                "weak_fallback_rescue" -> parseBoolean(value)?.let { weakFallbackRescueEnabled = it }
                "weak_fallback_rescue_ratio" -> value.toDoubleOrNull()?.let { weakFallbackRescueRatio = it.coerceIn(0.55, 0.90) }
                "weak_fallback_rescue_min_good" -> value.toIntOrNull()?.let { weakFallbackRescueMinGood = it.coerceIn(3, 12) }
                "weak_fallback_core_rescue" -> parseBoolean(value)?.let { weakFallbackCoreRescueEnabled = it }
                "weak_fallback_core_max_span" -> value.toDoubleOrNull()?.let { weakFallbackCoreMaxSpanPx = it.coerceIn(24.0, 220.0) }
                "weak_fallback_core_max_area" -> value.toDoubleOrNull()?.let { weakFallbackCoreMaxAreaPx = it.coerceIn(400.0, 20_000.0) }
                "soft_relax_enable" -> parseBoolean(value)?.let { softRelaxEnabled = it }
                "soft_relax_miss" -> value.toIntOrNull()?.let { softRelaxMissStreak = it.coerceIn(4, 600) }
                "soft_relax_min_good" -> value.toIntOrNull()?.let { softRelaxMinGoodMatches = it.coerceIn(3, 12) }
                "soft_relax_scale" -> value.toDoubleOrNull()?.let { softRelaxScaleThreshold = it.coerceIn(0.30, 0.98) }
                "soft_relax_max_ratio" -> value.toDoubleOrNull()?.let { softRelaxMaxRatio = it.coerceIn(0.55, 0.90) }
                "track_verify_interval" -> value.toIntOrNull()?.let { trackVerifyIntervalFrames = it.coerceIn(4, 120) }
                // Deprecated: global ORB fallback was removed to avoid costly full-frame rescans.
                "track_verify_global_interval" -> Unit
                "track_verify_expand" -> value.toDoubleOrNull()?.let { trackVerifyLocalExpandFactor = it.coerceIn(1.2, 4.0) }
                "track_verify_min_good" -> value.toIntOrNull()?.let {
                    trackVerifyMinGoodMatches = it.coerceIn(TRACK_VERIFY_HARD_MIN_GOOD_MATCHES, 120)
                }
                "track_verify_min_inliers" -> value.toIntOrNull()?.let { trackVerifyMinInliers = it.coerceIn(3, 80) }
                "track_verify_fail_tol" -> value.toIntOrNull()?.let { trackVerifyFailTolerance = it.coerceIn(1, 12) }
                "track_verify_hard_tol" -> value.toIntOrNull()?.let { trackVerifyHardDriftTolerance = it.coerceIn(1, 6) }
                "track_verify_native_bypass_conf", "native_verify_bypass_conf" ->
                    value.toDoubleOrNull()?.let { trackVerifyNativeBypassConfidence = it.coerceIn(0.0, 1.0) }
                "track_verify_recenter_px" -> value.toDoubleOrNull()?.let { trackVerifyRecenterPx = it.coerceIn(8.0, 360.0) }
                "track_verify_min_iou" -> value.toDoubleOrNull()?.let { trackVerifyMinIou = it.coerceIn(0.0, 0.95) }
                "track_verify_conf_margin" -> value.toDoubleOrNull()?.let { trackVerifySwitchConfidenceMargin = it.coerceIn(0.0, 0.60) }
                // Deprecated: global ORB fallback was removed to avoid costly full-frame rescans.
                "track_verify_global_fallback", "track_verify_enable_global_fallback" -> Unit
                "track_guard_max_jump" ->
                    value.toDoubleOrNull()?.let { trackGuardMaxCenterJumpFactor = it.coerceIn(0.20, 5.0) }
                "track_guard_min_area_ratio" ->
                    value.toDoubleOrNull()?.let { trackGuardMinAreaRatio = it.coerceIn(0.05, 1.0) }
                "track_guard_max_area_ratio" ->
                    value.toDoubleOrNull()?.let { trackGuardMaxAreaRatio = it.coerceIn(1.0, 10.0) }
                "track_guard_drop_streak" ->
                    value.toIntOrNull()?.let { trackGuardDropStreak = it.coerceIn(1, 12) }
                "track_guard_appearance_interval" ->
                    value.toLongOrNull()?.let { trackGuardAppearanceCheckInterval = it.coerceIn(1L, 120L) }
                "track_guard_min_appearance" ->
                    value.toDoubleOrNull()?.let { trackGuardMinAppearanceScore = it.coerceIn(-1.0, 1.0) }
                "track_guard_anchor_enable", "track_guard_anchor_enabled" ->
                    parseBoolean(value)?.let { trackGuardAnchorEnabled = it }
                "track_guard_anchor_max_drop" ->
                    value.toDoubleOrNull()?.let { trackGuardAnchorMaxDrop = it.coerceIn(0.05, 1.50) }
                "track_guard_anchor_min_score" ->
                    value.toDoubleOrNull()?.let { trackGuardAnchorMinScore = it.coerceIn(-1.0, 1.0) }
                "track_backend", "tracker_backend" -> parseTrackBackend(value)?.let { preferredTrackBackend = it }
                "native_failures" -> value.toIntOrNull()?.let { nativeMaxFailStreak = it.coerceIn(1, 10) }
                "native_min_confidence", "native_min_conf" ->
                    value.toDoubleOrNull()?.let { nativeMinConfidence = it.coerceIn(0.05, 0.95) }
                "native_fuse_soft_confidence", "native_fuse_soft_conf" ->
                    value.toDoubleOrNull()?.let { nativeFuseSoftConfidence = it.coerceIn(0.05, 0.95) }
                "native_fuse_hard_confidence", "native_fuse_hard_conf" ->
                    value.toDoubleOrNull()?.let { nativeFuseHardConfidence = it.coerceIn(0.01, 0.80) }
                "native_fuse_soft_streak", "native_fuse_streak" ->
                    value.toIntOrNull()?.let { nativeFuseSoftStreak = it.coerceIn(1, 12) }
                "native_fuse_warmup_frames", "native_fuse_warmup" ->
                    value.toIntOrNull()?.let { nativeFuseWarmupFrames = it.coerceIn(0, 60) }
                "native_hold_last_on_soft_reject", "native_hold_last" ->
                    parseBoolean(value)?.let { nativeHoldLastOnSoftReject = it }
                "native_orb_verify", "native_verify" ->
                    parseBoolean(value)?.let { nativeOrbVerifyEnabled = it }
                "native_score_log_interval", "native_score_log_stride" ->
                    value.toIntOrNull()?.let { nativeScoreLogIntervalFrames = it.coerceIn(1, 240) }
                "lock_hold_frames", "track_lock_hold_frames" ->
                    value.toIntOrNull()?.let { lockHoldFrames = it.coerceIn(0, 180) }
                "lost_overlay_hold_ms", "track_lost_overlay_hold_ms" ->
                    value.toLongOrNull()?.let { lostOverlayHoldMs = it.coerceIn(0L, 3000L) }
                "fast_first_lock_frames", "fast_lock_frames" ->
                    value.toIntOrNull()?.let { fastFirstLockFrames = it.coerceIn(0, 300) }
                "kalman_enable", "kalman_enabled" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(enabled = it) }
                "kalman_process_noise", "kalman_q" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(processNoise = it.coerceIn(0.001, 1.0))
                    }
                "kalman_measurement_noise", "kalman_r" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(measurementNoise = it.coerceIn(0.001, 1.0))
                    }
                "kalman_max_predict_ms" ->
                    value.toLongOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(maxPredictMs = it.coerceIn(30L, 2000L))
                    }
                "kalman_use_predicted_hold", "kalman_loss_hold" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(usePredictedHold = it) }
                "kalman_dynamic_measurement", "kalman_dynamic_r" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(dynamicMeasurementNoise = it) }
                "kalman_conf_high", "kalman_high_confidence_threshold" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(highConfidenceThreshold = it.coerceIn(0.05, 1.0))
                    }
                "kalman_conf_low", "kalman_low_confidence_threshold" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(lowConfidenceThreshold = it.coerceIn(0.0, 0.95))
                    }
                "kalman_r_scale_high", "kalman_high_conf_noise_scale" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(highConfidenceNoiseScale = it.coerceIn(0.01, 10.0))
                    }
                "kalman_r_scale_low", "kalman_low_conf_noise_scale" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(lowConfidenceNoiseScale = it.coerceIn(0.10, 100.0))
                    }
                "kalman_r_scale_occlusion", "kalman_occlusion_noise_scale" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(occlusionNoiseScale = it.coerceIn(0.10, 200.0))
                    }
                "kalman_feed_native_prior", "kalman_prior_feedback" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(feedNativePrior = it) }
                "kalman_pretrack_prior_feed", "kalman_prior_pretrack_feed" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(preTrackPriorFeed = it) }
                "kalman_prior_only_on_uncertain", "kalman_prior_uncertain_only" ->
                    parseBoolean(value)?.let { kalmanConfig = kalmanConfig.copy(priorOnlyOnUncertain = it) }
                "kalman_prior_min_iou" ->
                    value.toDoubleOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(priorMinIou = it.coerceIn(0.0, 1.0))
                    }
                "kalman_prior_stale_ms" ->
                    value.toLongOrNull()?.let {
                        kalmanConfig = kalmanConfig.copy(priorStaleMs = it.coerceIn(0L, 3000L))
                    }
                "native_model_param", "native_param_path", "ncnn_param_path" -> {
                    nativeModelParamPathOverride = normalizeOptionalPath(value)
                }
                "native_model_bin", "native_bin_path", "ncnn_bin_path" -> {
                    nativeModelBinPathOverride = normalizeOptionalPath(value)
                }
                "kcf_failures" -> value.toIntOrNull()?.let { kcfMaxFailStreak = it.coerceIn(1, 10) }
                "tracker_gc" -> parseBoolean(value)?.let { forceTrackerGcOnDrop = it }
            }
        }

        orbFeatureHardCap = orbFeatureHardCap.coerceIn(200, 1200)
        if (homographyScaleMin > homographyScaleMax) {
            val t = homographyScaleMin
            homographyScaleMin = homographyScaleMax
            homographyScaleMax = t
        }
        if (searchHighResShortEdge < searchShortEdge) {
            searchHighResShortEdge = searchShortEdge
        }
        searchStrideFrames = searchStrideFrames.coerceAtLeast(1)
        searchOverBudgetMs = searchOverBudgetMs.coerceIn(0L, 240L)
        searchOverBudgetSkipFrames = searchOverBudgetSkipFrames.coerceIn(0, 8)
        if (searchOverBudgetMs <= 0L) {
            searchOverBudgetSkipFrames = 0
        }
        if (searchUltraHighResShortEdge < searchHighResShortEdge) {
            searchUltraHighResShortEdge = searchHighResShortEdge
        }
        if (searchUltraHighResMissStreak < searchHighResMissStreak) {
            searchUltraHighResMissStreak = searchHighResMissStreak
        }
        if (searchMultiTemplateMaxLongEdge > searchMaxLongEdge) {
            searchMultiTemplateMaxLongEdge = searchMaxLongEdge
        }
        if (fallbackMaxBoxSize < fallbackMinBoxSize) {
            fallbackMaxBoxSize = fallbackMinBoxSize
        }
        trackGuardMaxCenterJumpFactor = trackGuardMaxCenterJumpFactor.coerceIn(0.20, 5.0)
        trackGuardMinAreaRatio = trackGuardMinAreaRatio.coerceIn(0.05, 1.0)
        trackGuardMaxAreaRatio = trackGuardMaxAreaRatio.coerceIn(1.0, 10.0)
        if (trackGuardMinAreaRatio > trackGuardMaxAreaRatio) {
            trackGuardMinAreaRatio = trackGuardMaxAreaRatio
        }
        trackGuardDropStreak = trackGuardDropStreak.coerceIn(1, 12)
        trackGuardAppearanceCheckInterval = trackGuardAppearanceCheckInterval.coerceIn(1L, 120L)
        trackGuardMinAppearanceScore = trackGuardMinAppearanceScore.coerceIn(-1.0, 1.0)
        trackGuardAnchorMaxDrop = trackGuardAnchorMaxDrop.coerceIn(0.05, 1.50)
        trackGuardAnchorMinScore = trackGuardAnchorMinScore.coerceIn(-1.0, 1.0)
        nativeScoreLogIntervalFrames = nativeScoreLogIntervalFrames.coerceIn(1, 240)
        if (firstLockSmallCenterDriftRelaxedPx < firstLockSmallCenterDriftPx) {
            firstLockSmallCenterDriftRelaxedPx = firstLockSmallCenterDriftPx
        }
        firstLockSmallDynamicCenterFactor = firstLockSmallDynamicCenterFactor.coerceAtLeast(0.30)
        firstLockSmallRelaxedIouFloor = firstLockSmallRelaxedIouFloor.coerceIn(0.0, 0.60)
        firstLockOutlierHoldMax = firstLockOutlierHoldMax.coerceIn(1, 30)
        firstLockMissRelaxHighStreak = firstLockMissRelaxHighStreak.coerceIn(1, 600)
        firstLockMissRelaxMidStreak = firstLockMissRelaxMidStreak.coerceIn(1, 600)
        firstLockMissRelaxLowStreak = firstLockMissRelaxLowStreak.coerceIn(1, 600)
        if (firstLockMissRelaxMidStreak > firstLockMissRelaxHighStreak) {
            firstLockMissRelaxMidStreak = firstLockMissRelaxHighStreak
        }
        if (firstLockMissRelaxLowStreak > firstLockMissRelaxMidStreak) {
            firstLockMissRelaxLowStreak = firstLockMissRelaxMidStreak
        }
        firstLockMissRelaxHighFactor = firstLockMissRelaxHighFactor.coerceIn(0.40, 1.20)
        firstLockMissRelaxMidFactor = firstLockMissRelaxMidFactor.coerceIn(0.40, 1.20)
        firstLockMissRelaxLowFactor = firstLockMissRelaxLowFactor.coerceIn(0.40, 1.20)
        firstLockResetRelaxMinFrames = firstLockResetRelaxMinFrames.coerceIn(1, 30)
        firstLockResetRelaxMinIouResets = firstLockResetRelaxMinIouResets.coerceIn(1, 100)
        firstLockResetRelaxFactor = firstLockResetRelaxFactor.coerceIn(0.40, 1.20)
        firstLockIouFloorStrongGood = firstLockIouFloorStrongGood.coerceIn(4, 80)
        firstLockIouFloorStrongInliers = firstLockIouFloorStrongInliers.coerceIn(3, 80)
        firstLockIouFloorMediumGood = firstLockIouFloorMediumGood.coerceIn(4, 80)
        firstLockIouFloorMediumInliers = firstLockIouFloorMediumInliers.coerceIn(3, 80)
        firstLockIouFloorStrong = firstLockIouFloorStrong.coerceIn(0.0, 1.0)
        firstLockIouFloorMedium = firstLockIouFloorMedium.coerceIn(0.0, 1.0)
        firstLockIouFloorBase = firstLockIouFloorBase.coerceIn(0.0, 1.0)
        if (firstLockIouFloorStrong > firstLockIouFloorMedium) {
            firstLockIouFloorStrong = firstLockIouFloorMedium
        }
        if (firstLockIouFloorMedium > firstLockIouFloorBase) {
            firstLockIouFloorMedium = firstLockIouFloorBase
        }
        firstLockCenterRelaxStrongGood = firstLockCenterRelaxStrongGood.coerceIn(4, 80)
        firstLockCenterRelaxStrongInliers = firstLockCenterRelaxStrongInliers.coerceIn(3, 80)
        firstLockCenterRelaxStrongFactor = firstLockCenterRelaxStrongFactor.coerceIn(1.0, 2.0)
        firstLockCenterRelaxMissStreak = firstLockCenterRelaxMissStreak.coerceIn(1, 600)
        firstLockCenterRelaxMissFactor = firstLockCenterRelaxMissFactor.coerceIn(1.0, 2.0)
        firstLockCenterRelaxCapFactor = firstLockCenterRelaxCapFactor.coerceIn(1.0, 2.0)
        firstLockConfStrongGood = firstLockConfStrongGood.coerceIn(4, 80)
        firstLockConfStrongInliers = firstLockConfStrongInliers.coerceIn(3, 80)
        firstLockConfMediumGood = firstLockConfMediumGood.coerceIn(4, 80)
        firstLockConfMediumInliers = firstLockConfMediumInliers.coerceIn(3, 80)
        firstLockConfStrongMin = firstLockConfStrongMin.coerceIn(0.0, 1.0)
        firstLockConfMediumMin = firstLockConfMediumMin.coerceIn(0.0, 1.0)
        firstLockConfSmallMin = firstLockConfSmallMin.coerceIn(0.0, 1.0)
        firstLockConfBaseMin = firstLockConfBaseMin.coerceIn(0.0, 1.0)
        if (firstLockConfStrongMin > firstLockConfMediumMin) {
            firstLockConfStrongMin = firstLockConfMediumMin
        }
        if (firstLockConfMediumMin > firstLockConfSmallMin) {
            firstLockConfMediumMin = firstLockConfSmallMin
        }
        if (firstLockConfSmallMin > firstLockConfBaseMin) {
            firstLockConfSmallMin = firstLockConfBaseMin
        }
        autoVerifyStrongMinGood = autoVerifyStrongMinGood.coerceIn(4, 120)
        autoVerifyStrongMinInliers = autoVerifyStrongMinInliers.coerceIn(3, 80)
        autoVerifyStrongMinConfidence = autoVerifyStrongMinConfidence.coerceIn(0.0, 1.0)
        autoVerifyStrongAppearanceMinLive = autoVerifyStrongAppearanceMinLive.coerceIn(-1.0, 1.0)
        autoVerifyStrongAppearanceMinReplay = autoVerifyStrongAppearanceMinReplay.coerceIn(-1.0, 1.0)
        autoVerifyLocalRoiExpandFactor = autoVerifyLocalRoiExpandFactor.coerceIn(1.1, 4.0)
        autoVerifyLocalCenterFactorLive = autoVerifyLocalCenterFactorLive.coerceIn(0.10, 2.0)
        autoVerifyLocalCenterFactorReplay = autoVerifyLocalCenterFactorReplay.coerceIn(0.10, 2.0)
        autoVerifyLocalMinIouLive = autoVerifyLocalMinIouLive.coerceIn(0.0, 1.0)
        autoVerifyLocalMinIouReplay = autoVerifyLocalMinIouReplay.coerceIn(0.0, 1.0)
        autoVerifyLocalMinConfScaleLive = autoVerifyLocalMinConfScaleLive.coerceIn(0.0, 1.0)
        autoVerifyLocalMinConfScaleReplay = autoVerifyLocalMinConfScaleReplay.coerceIn(0.0, 1.0)
        autoVerifyLocalMinConfFloorLive = autoVerifyLocalMinConfFloorLive.coerceIn(0.0, 1.0)
        autoVerifyLocalMinConfFloorReplay = autoVerifyLocalMinConfFloorReplay.coerceIn(0.0, 1.0)
        autoVerifyAppearanceStrongGood = autoVerifyAppearanceStrongGood.coerceIn(4, 120)
        autoVerifyAppearanceStrongInliers = autoVerifyAppearanceStrongInliers.coerceIn(3, 80)
        autoVerifyAppearanceMediumGood = autoVerifyAppearanceMediumGood.coerceIn(4, 120)
        autoVerifyAppearanceMediumInliers = autoVerifyAppearanceMediumInliers.coerceIn(3, 80)
        autoVerifyAppearanceMinStrong = autoVerifyAppearanceMinStrong.coerceIn(-1.0, 1.0)
        autoVerifyAppearanceMinMedium = autoVerifyAppearanceMinMedium.coerceIn(-1.0, 1.0)
        autoVerifyAppearanceMinBase = autoVerifyAppearanceMinBase.coerceIn(-1.0, 1.0)
        autoVerifyAppearanceLiveBias = autoVerifyAppearanceLiveBias.coerceIn(-1.0, 1.0)
        if (autoVerifyAppearanceMinStrong > autoVerifyAppearanceMinMedium) {
            autoVerifyAppearanceMinStrong = autoVerifyAppearanceMinMedium
        }
        if (autoVerifyAppearanceMinMedium > autoVerifyAppearanceMinBase) {
            autoVerifyAppearanceMinMedium = autoVerifyAppearanceMinBase
        }
        temporalHighGoodMatches = temporalHighGoodMatches.coerceIn(4, 120)
        temporalHighInliers = temporalHighInliers.coerceIn(3, 80)
        temporalMediumGoodMatches = temporalMediumGoodMatches.coerceIn(4, temporalHighGoodMatches)
        temporalMinConfidenceHighGood = temporalMinConfidenceHighGood.coerceIn(0.0, 1.0)
        temporalMinConfidenceMedium = temporalMinConfidenceMedium.coerceIn(0.0, 1.0)
        temporalMinConfidenceSmallRefined = temporalMinConfidenceSmallRefined.coerceIn(0.0, 1.0)
        temporalMinConfidenceBase = temporalMinConfidenceBase.coerceIn(0.0, 1.0)
        temporalLiveConfidenceRelax = temporalLiveConfidenceRelax.coerceIn(0.0, 1.0)
        temporalLiveConfidenceFloor = temporalLiveConfidenceFloor.coerceIn(0.0, 1.0)
        if (temporalMinConfidenceHighGood > temporalMinConfidenceMedium) {
            temporalMinConfidenceHighGood = temporalMinConfidenceMedium
        }
        if (temporalMinConfidenceMedium > temporalMinConfidenceSmallRefined) {
            temporalMinConfidenceMedium = temporalMinConfidenceSmallRefined
        }
        if (temporalMinConfidenceSmallRefined > temporalMinConfidenceBase) {
            temporalMinConfidenceSmallRefined = temporalMinConfidenceBase
        }
        if (temporalLiveConfidenceFloor > temporalMinConfidenceBase) {
            temporalLiveConfidenceFloor = temporalMinConfidenceBase
        }
        orbSoftMinGoodMatches = orbSoftMinGoodMatches.coerceAtMost(orbMinGoodMatches).coerceAtLeast(3)
        orbSoftMinInliers = orbSoftMinInliers.coerceAtMost(orbMinInliers).coerceAtLeast(3)
        trackVerifyMinInliers = trackVerifyMinInliers.coerceAtMost(trackVerifyMinGoodMatches).coerceAtLeast(3)
        fallbackRefineMinInliers = fallbackRefineMinInliers.coerceAtMost(fallbackRefineMinGoodMatches).coerceAtLeast(3)
        smallTargetMinInliers = smallTargetMinInliers.coerceAtMost(smallTargetMinGoodMatches).coerceAtLeast(3)
        fallbackRefineLoweRatio = fallbackRefineLoweRatio.coerceAtLeast(orbLoweRatio)
        weakFallbackRescueMinGood = weakFallbackRescueMinGood.coerceAtLeast(orbSoftMinGoodMatches)
        weakFallbackRelaxFactor = weakFallbackRelaxFactor.coerceAtLeast(1.0)
        weakFallbackCoreMaxSpanPx = weakFallbackCoreMaxSpanPx.coerceAtMost(weakFallbackMaxSpanPx)
        weakFallbackCoreMaxAreaPx = weakFallbackCoreMaxAreaPx.coerceAtMost(weakFallbackMaxAreaPx)
        softRelaxMinGoodMatches = softRelaxMinGoodMatches.coerceAtMost(orbSoftMinGoodMatches).coerceAtLeast(3)
        softRelaxMaxRatio = softRelaxMaxRatio.coerceAtMost(orbLoweRatio).coerceAtLeast(0.55)
        nativeMaxFailStreak = nativeMaxFailStreak.coerceIn(1, 10)
        nativeMinConfidence = nativeMinConfidence.coerceIn(0.05, 0.95)
        nativeFuseHardConfidence = nativeFuseHardConfidence.coerceIn(0.01, 0.80)
        nativeFuseSoftConfidence = nativeFuseSoftConfidence.coerceIn(0.05, 0.95)
        if (nativeFuseSoftConfidence < nativeFuseHardConfidence) {
            nativeFuseSoftConfidence = nativeFuseHardConfidence
        }
        nativeFuseSoftStreak = nativeFuseSoftStreak.coerceIn(1, 12)
        nativeFuseWarmupFrames = nativeFuseWarmupFrames.coerceIn(0, 60)
        if (nativeMinConfidence < nativeFuseHardConfidence) {
            nativeMinConfidence = nativeFuseHardConfidence
        }
        kalmanConfig = kalmanConfig.copy(
            processNoise = kalmanConfig.processNoise.coerceIn(0.001, 1.0),
            measurementNoise = kalmanConfig.measurementNoise.coerceIn(0.001, 1.0),
            maxPredictMs = kalmanConfig.maxPredictMs.coerceIn(30L, 2000L),
            highConfidenceThreshold = kalmanConfig.highConfidenceThreshold.coerceIn(0.05, 1.0),
            lowConfidenceThreshold = kalmanConfig.lowConfidenceThreshold.coerceIn(0.0, 0.95),
            highConfidenceNoiseScale = kalmanConfig.highConfidenceNoiseScale.coerceIn(0.01, 10.0),
            lowConfidenceNoiseScale = kalmanConfig.lowConfidenceNoiseScale.coerceIn(0.10, 100.0),
            occlusionNoiseScale = kalmanConfig.occlusionNoiseScale.coerceIn(0.10, 200.0),
            priorMinIou = kalmanConfig.priorMinIou.coerceIn(0.0, 1.0),
            priorStaleMs = kalmanConfig.priorStaleMs.coerceIn(0L, 3000L)
        )
        if (kalmanConfig.lowConfidenceThreshold > kalmanConfig.highConfidenceThreshold) {
            kalmanConfig = kalmanConfig.copy(lowConfidenceThreshold = kalmanConfig.highConfidenceThreshold)
        }
        if (kalmanConfig.highConfidenceNoiseScale > kalmanConfig.lowConfidenceNoiseScale) {
            kalmanConfig = kalmanConfig.copy(highConfidenceNoiseScale = kalmanConfig.lowConfidenceNoiseScale)
        }
        lockHoldFrames = lockHoldFrames.coerceIn(0, 180)
        lostOverlayHoldMs = lostOverlayHoldMs.coerceIn(0L, 3000L)
        fastFirstLockFrames = fastFirstLockFrames.coerceIn(0, 300)

        configureOrbDetector(orbMaxFeatures)
        refreshHeuristicConfig()
        if (shouldRefreshTemplate && templateSourceGrays.isNotEmpty()) {
            rebuildTemplatePyramid(templateSourceGrays)
        }
        logEffectiveParams("override")
    }

    private fun buildHeuristicConfig(): HeuristicConfig {
        return HeuristicConfig(
            orb = OrbThresholdConfig(
                minGoodMatches = orbMinGoodMatches,
                minInliers = orbMinInliers,
                softMinGoodMatches = orbSoftMinGoodMatches,
                softMinInliers = orbSoftMinInliers,
                loweRatio = orbLoweRatio
            ),
            smallTarget = SmallTargetConfig(
                areaRatio = smallTargetAreaRatio,
                minGoodMatches = smallTargetMinGoodMatches,
                minInliers = smallTargetMinInliers,
                scaleThreshold = smallTargetScaleThreshold
            ),
            weakFallback = WeakFallbackConfig(
                maxMatches = weakFallbackMaxMatches,
                maxSpanPx = weakFallbackMaxSpanPx,
                maxAreaPx = weakFallbackMaxAreaPx,
                requireRefine = weakFallbackRequireRefine,
                rescueEnabled = weakFallbackRescueEnabled,
                rescueRatio = weakFallbackRescueRatio,
                rescueMinGood = weakFallbackRescueMinGood,
                relaxMissStreak = weakFallbackRelaxMissStreak,
                relaxFactor = weakFallbackRelaxFactor,
                coreRescueEnabled = weakFallbackCoreRescueEnabled,
                coreMaxSpanPx = weakFallbackCoreMaxSpanPx,
                coreMaxAreaPx = weakFallbackCoreMaxAreaPx
            ),
            softRelax = SoftRelaxConfig(
                enabled = softRelaxEnabled,
                missStreak = softRelaxMissStreak,
                minGoodMatches = softRelaxMinGoodMatches,
                scaleThreshold = softRelaxScaleThreshold,
                maxRatio = softRelaxMaxRatio
            ),
            fallbackRefine = FallbackRefineConfig(
                expandFactor = fallbackRefineExpandFactor,
                minGoodMatches = fallbackRefineMinGoodMatches,
                minInliers = fallbackRefineMinInliers,
                minConfidence = fallbackRefineMinConfidence,
                loweRatio = fallbackRefineLoweRatio
            ),
            firstLock = FirstLockConfig(
                stableFrames = firstLockStableFrames,
                stableMs = firstLockStableMs,
                stablePx = firstLockStablePx,
                minIou = firstLockMinIou,
                gapMs = firstLockGapMs,
                holdOnTemporalReject = firstLockHoldOnTemporalReject,
                smallCenterDriftPx = firstLockSmallCenterDriftPx,
                smallCenterDriftRelaxedPx = firstLockSmallCenterDriftRelaxedPx,
                smallStablePx = firstLockSmallStablePx,
                smallRelaxMissStreak = firstLockSmallRelaxMissStreak,
                smallDynamicCenterFactor = firstLockSmallDynamicCenterFactor,
                smallRelaxedIouFloor = firstLockSmallRelaxedIouFloor,
                outlierHoldMax = firstLockOutlierHoldMax,
                allowFallbackLock = allowFallbackLock
            ),
            firstLockAdaptive = FirstLockAdaptiveConfig(
                missRelaxHighStreak = firstLockMissRelaxHighStreak,
                missRelaxHighFactor = firstLockMissRelaxHighFactor,
                missRelaxMidStreak = firstLockMissRelaxMidStreak,
                missRelaxMidFactor = firstLockMissRelaxMidFactor,
                missRelaxLowStreak = firstLockMissRelaxLowStreak,
                missRelaxLowFactor = firstLockMissRelaxLowFactor,
                resetRelaxMinFrames = firstLockResetRelaxMinFrames,
                resetRelaxMinIouResets = firstLockResetRelaxMinIouResets,
                resetRelaxFactor = firstLockResetRelaxFactor,
                iouFloorStrongGood = firstLockIouFloorStrongGood,
                iouFloorStrongInliers = firstLockIouFloorStrongInliers,
                iouFloorStrong = firstLockIouFloorStrong,
                iouFloorMediumGood = firstLockIouFloorMediumGood,
                iouFloorMediumInliers = firstLockIouFloorMediumInliers,
                iouFloorMedium = firstLockIouFloorMedium,
                iouFloorBase = firstLockIouFloorBase,
                centerRelaxGood = firstLockCenterRelaxStrongGood,
                centerRelaxInliers = firstLockCenterRelaxStrongInliers,
                centerRelaxStrongFactor = firstLockCenterRelaxStrongFactor,
                centerRelaxMissStreak = firstLockCenterRelaxMissStreak,
                centerRelaxMissFactor = firstLockCenterRelaxMissFactor,
                centerRelaxCapFactor = firstLockCenterRelaxCapFactor,
                confStrongGood = firstLockConfStrongGood,
                confStrongInliers = firstLockConfStrongInliers,
                confStrongMin = firstLockConfStrongMin,
                confMediumGood = firstLockConfMediumGood,
                confMediumInliers = firstLockConfMediumInliers,
                confMediumMin = firstLockConfMediumMin,
                confSmallMin = firstLockConfSmallMin,
                confBaseMin = firstLockConfBaseMin
            ),
            trackVerify = TrackVerifyConfig(
                intervalFrames = trackVerifyIntervalFrames,
                localExpandFactor = trackVerifyLocalExpandFactor,
                minGoodMatches = trackVerifyMinGoodMatches,
                minInliers = trackVerifyMinInliers,
                failTolerance = trackVerifyFailTolerance,
                hardDriftTolerance = trackVerifyHardDriftTolerance,
                recenterPx = trackVerifyRecenterPx,
                minIou = trackVerifyMinIou,
                switchConfidenceMargin = trackVerifySwitchConfidenceMargin,
                nativeBypassConfidence = trackVerifyNativeBypassConfidence
            ),
            nativeGate = NativeGateConfig(
                maxFailStreak = nativeMaxFailStreak,
                minConfidence = nativeMinConfidence,
                fuseSoftConfidence = nativeFuseSoftConfidence,
                fuseHardConfidence = nativeFuseHardConfidence,
                fuseSoftStreak = nativeFuseSoftStreak,
                fuseWarmupFrames = nativeFuseWarmupFrames,
                holdLastOnSoftReject = nativeHoldLastOnSoftReject
            ),
            autoInitVerify = AutoInitVerifyConfig(
                strongCandidateMinGood = autoVerifyStrongMinGood,
                strongCandidateMinInliers = autoVerifyStrongMinInliers,
                strongCandidateMinConfidence = autoVerifyStrongMinConfidence,
                strongAppearanceMinLive = autoVerifyStrongAppearanceMinLive,
                strongAppearanceMinReplay = autoVerifyStrongAppearanceMinReplay,
                localRoiExpandFactor = autoVerifyLocalRoiExpandFactor,
                localCenterFactorLive = autoVerifyLocalCenterFactorLive,
                localCenterFactorReplay = autoVerifyLocalCenterFactorReplay,
                localMinIouLive = autoVerifyLocalMinIouLive,
                localMinIouReplay = autoVerifyLocalMinIouReplay,
                localMinConfScaleLive = autoVerifyLocalMinConfScaleLive,
                localMinConfScaleReplay = autoVerifyLocalMinConfScaleReplay,
                localMinConfFloorLive = autoVerifyLocalMinConfFloorLive,
                localMinConfFloorReplay = autoVerifyLocalMinConfFloorReplay,
                appearanceMinStrongGood = autoVerifyAppearanceStrongGood,
                appearanceMinStrongInliers = autoVerifyAppearanceStrongInliers,
                appearanceMinStrong = autoVerifyAppearanceMinStrong,
                appearanceMinMediumGood = autoVerifyAppearanceMediumGood,
                appearanceMinMediumInliers = autoVerifyAppearanceMediumInliers,
                appearanceMinMedium = autoVerifyAppearanceMinMedium,
                appearanceMinBase = autoVerifyAppearanceMinBase,
                appearanceLiveBias = autoVerifyAppearanceLiveBias
            ),
            temporalGate = TemporalGateConfig(
                highGoodMatches = temporalHighGoodMatches,
                highInliers = temporalHighInliers,
                minConfidenceHighGood = temporalMinConfidenceHighGood,
                mediumGoodMatches = temporalMediumGoodMatches,
                minConfidenceMedium = temporalMinConfidenceMedium,
                minConfidenceSmallRefined = temporalMinConfidenceSmallRefined,
                minConfidenceBase = temporalMinConfidenceBase,
                liveConfidenceRelax = temporalLiveConfidenceRelax,
                liveConfidenceFloor = temporalLiveConfidenceFloor
            ),
            trackGuard = TrackGuardConfig(
                maxCenterJumpFactor = trackGuardMaxCenterJumpFactor,
                minAreaRatio = trackGuardMinAreaRatio,
                maxAreaRatio = trackGuardMaxAreaRatio,
                dropStreak = trackGuardDropStreak,
                appearanceCheckIntervalFrames = trackGuardAppearanceCheckInterval,
                minAppearanceScore = trackGuardMinAppearanceScore
            )
        )
    }

    private fun refreshHeuristicConfig() {
        heuristicConfig = buildHeuristicConfig()
    }

    private fun updateKalmanPrediction(frameTsMs: Long) {
        if (!kalmanConfig.enabled) {
            latestKalmanPrediction = null
            return
        }
        val predicted = kalmanPredictor.predict(frameTsMs, kalmanConfig)
        if (predicted == null) {
            latestKalmanPrediction = null
            return
        }
        if (lastKalmanMeasureMs > 0L && frameTsMs - lastKalmanMeasureMs > kalmanConfig.maxPredictMs) {
            latestKalmanPrediction = null
            return
        }
        latestKalmanPrediction = predicted
    }

    private fun resetKalman(reason: String) {
        lastKalmanMeasureMs = 0L
        latestKalmanPrediction = null
        kalmanPredictor.reset()
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag("KALMAN", "session=$diagSessionId action=reset reason=$reason")
        }
    }

    private fun resolveKalmanMeasurementNoise(confidence: Double, occluded: Boolean): Double {
        val cfg = kalmanConfig
        var scale = 1.0
        if (cfg.dynamicMeasurementNoise) {
            val c = confidence.coerceIn(0.0, 1.0)
            val high = cfg.highConfidenceThreshold.coerceAtLeast(cfg.lowConfidenceThreshold + 1e-6)
            val low = cfg.lowConfidenceThreshold.coerceIn(0.0, high)
            scale = when {
                c >= high -> cfg.highConfidenceNoiseScale
                c <= low -> cfg.lowConfidenceNoiseScale
                else -> {
                    val t = (c - low) / (high - low)
                    cfg.lowConfidenceNoiseScale + (cfg.highConfidenceNoiseScale - cfg.lowConfidenceNoiseScale) * t
                }
            }
        }
        if (occluded) {
            scale *= cfg.occlusionNoiseScale
        }
        return (cfg.measurementNoise * scale).coerceAtLeast(1e-6)
    }

    private fun correctKalman(box: Rect, confidence: Double = 1.0, occluded: Boolean = false): Rect {
        if (!kalmanConfig.enabled) return box
        val now = SystemClock.elapsedRealtime()
        val measurementNoise = resolveKalmanMeasurementNoise(confidence, occluded)
        val corrected = kalmanPredictor.correct(box, now, kalmanConfig, measurementNoise)
        lastKalmanMeasureMs = now
        latestKalmanPrediction = corrected
        return corrected
    }

    private fun commitTrackingObservation(measured: Rect, confidence: Double, markNativeAccept: Boolean): Rect {
        val fused = correctKalman(measured, confidence = confidence)
        lastMeasuredTrackBox = measured
        lastTrackedBox = fused
        if (markNativeAccept) {
            lastNativeAcceptMs = SystemClock.elapsedRealtime()
        }
        dispatchTrackedRect(fused)
        updateLatestPrediction(fused, confidence.coerceIn(0.0, 1.0), tracking = true)
        return fused
    }

    private fun feedNativePriorBox(prior: Rect, reason: String) {
        if (!kalmanConfig.enabled || !kalmanConfig.feedNativePrior) return
        if (!isTracking || activeTrackBackend == TrackBackend.KCF) return
        if (prior.width < MIN_BOX_DIM || prior.height < MIN_BOX_DIM) return
        if (kalmanConfig.priorOnlyOnUncertain && reason.startsWith("pre_track_")) {
            val now = SystemClock.elapsedRealtime()
            val sinceAccept = if (lastNativeAcceptMs > 0L) now - lastNativeAcceptMs else Long.MAX_VALUE
            val uncertain =
                consecutiveTrackerFailures > 0 ||
                    nativeLowConfidenceStreak > 0 ||
                    lockHoldFramesRemaining in 0 until lockHoldFrames ||
                    sinceAccept > kalmanConfig.priorStaleMs
            if (!uncertain) return
            val measured = lastMeasuredTrackBox
            if (measured != null && sinceAccept <= kalmanConfig.priorStaleMs) {
                val iou = rectIou(prior, measured)
                if (iou < kalmanConfig.priorMinIou) {
                    if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                        logDiag(
                            "KALMAN",
                            "session=$diagSessionId action=prior_skip reason=$reason why=iou_gate iou=${fmt(iou)} min=${fmt(kalmanConfig.priorMinIou)}"
                        )
                    }
                    return
                }
            }
        }
        val nativeRect = android.graphics.Rect(prior.x, prior.y, prior.x + prior.width, prior.y + prior.height)
        val ok = NativeTrackerBridge.setPriorBbox(nativeRect)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag(
                "KALMAN",
                "session=$diagSessionId action=prior_feed reason=$reason ok=$ok box=${prior.x},${prior.y},${prior.width}x${prior.height}"
            )
        }
    }

    private fun feedKalmanPriorToNative(reason: String) {
        if (!kalmanConfig.preTrackPriorFeed) return
        val prior = latestKalmanPrediction ?: lastMeasuredTrackBox ?: lastTrackedBox ?: return
        feedNativePriorBox(prior, reason)
    }

    private fun parseBoolean(value: String): Boolean? {
        return when (value.trim().lowercase(Locale.US)) {
            "1", "true", "yes", "on" -> true
            "0", "false", "no", "off" -> false
            else -> null
        }
    }

    private fun parseTrackBackend(value: String): TrackBackend? {
        return when (value.trim().lowercase(Locale.US)) {
            "kcf", "opencv" -> TrackBackend.KCF
            "native_ncnn", "ncnn", "native" -> TrackBackend.NATIVE_NCNN
            "native_rknn", "rknn" -> TrackBackend.NATIVE_RKNN
            else -> null
        }
    }

    private fun normalizeOptionalPath(raw: String): String? {
        val normalized = raw.trim()
        if (normalized.isEmpty()) return null
        return when (normalized.lowercase(Locale.US)) {
            "auto", "default", "null", "none" -> null
            else -> normalized
        }
    }

    private fun summarizePath(path: String?): String {
        val normalized = path?.trim()?.replace('\\', '/') ?: return "auto"
        if (normalized.isEmpty()) return "auto"
        return normalized.substringAfterLast('/')
    }

    private fun configureOrbDetector(maxFeatures: Int, enforceBudgetCap: Boolean = false) {
        val boundedFeatures = if (enforceBudgetCap) {
            min(maxFeatures, orbFeatureHardCap).coerceAtLeast(64)
        } else {
            maxFeatures.coerceAtLeast(64)
        }
        orb.setMaxFeatures(boundedFeatures)
        orb.setScaleFactor(orbScaleFactor)
        orb.setNLevels(orbNLevels)
        orb.setFastThreshold(orbFastThreshold)
    }

    private fun logEffectiveParams(source: String) {
        val cfg = heuristicConfig
        Log.w(
            TAG,
            "EVAL_EVENT type=PARAMS source=$source mode=${trackerMode.name.lowercase(Locale.US)} " +
                "orbFeatures=$orbMaxFeatures featureCap=$orbFeatureHardCap ratio=${fmt(cfg.orb.loweRatio)} minMatches=${cfg.orb.minGoodMatches} minInliers=${cfg.orb.minInliers} " +
                "orbScale=${fmt(orbScaleFactor)} orbLevels=$orbNLevels orbFast=$orbFastThreshold " +
                "softMatches=${cfg.orb.softMinGoodMatches} softInliers=${cfg.orb.softMinInliers} ransac=${fmt(orbRansacThreshold)} " +
                "searchShort=$searchShortEdge searchHiMiss=$searchHighResMissStreak searchHiShort=$searchHighResShortEdge " +
                "searchUltraMiss=$searchUltraHighResMissStreak searchUltraShort=$searchUltraHighResShortEdge " +
                "searchStride=$searchStrideFrames searchBudgetMs=$searchOverBudgetMs searchBudgetSkip=$searchOverBudgetSkipFrames " +
                "searchMaxLong=$searchMaxLongEdge searchMultiMaxLong=$searchMultiTemplateMaxLongEdge " +
                "initBox=$initBoxSize fallbackMin=$fallbackMinBoxSize fallbackMax=$fallbackMaxBoxSize " +
                "hDist=${fmt(homographyMaxDistortion)} hScaleMin=${fmt(homographyScaleMin)} hScaleMax=${fmt(homographyScaleMax)} " +
                "hDet=${fmt(homographyMinJacobianDet)} tplTextureMin=${fmt(templateMinTextureScore)} clahe=$orbUseClahe " +
                "farBoostFeatures=$orbFarBoostFeatures farBoostMultiCap=$orbFarBoostMultiTemplateCap " +
                "firstLockFrames=${cfg.firstLock.stableFrames} firstLockMs=${cfg.firstLock.stableMs} firstLockPx=${fmt(cfg.firstLock.stablePx)} " +
                "firstLockIou=${fmt(cfg.firstLock.minIou)} allowFallbackLock=${cfg.firstLock.allowFallbackLock} " +
                "firstLockSmallCenter=${fmt(cfg.firstLock.smallCenterDriftPx)} " +
                "firstLockSmallCenterRelax=${fmt(cfg.firstLock.smallCenterDriftRelaxedPx)} " +
                "firstLockSmallStable=${fmt(cfg.firstLock.smallStablePx)} firstLockSmallRelaxMiss=${cfg.firstLock.smallRelaxMissStreak} " +
                "firstLockSmallDyn=${fmt(cfg.firstLock.smallDynamicCenterFactor)} " +
                "firstLockSmallRelaxIou=${fmt(cfg.firstLock.smallRelaxedIouFloor)} " +
                "firstLockTemporalHold=${cfg.firstLock.holdOnTemporalReject} " +
                "firstLockOutlierHoldMax=${cfg.firstLock.outlierHoldMax} " +
                "flMissRelaxHigh=${cfg.firstLockAdaptive.missRelaxHighStreak}/${fmt(cfg.firstLockAdaptive.missRelaxHighFactor)} " +
                "flMissRelaxMid=${cfg.firstLockAdaptive.missRelaxMidStreak}/${fmt(cfg.firstLockAdaptive.missRelaxMidFactor)} " +
                "flMissRelaxLow=${cfg.firstLockAdaptive.missRelaxLowStreak}/${fmt(cfg.firstLockAdaptive.missRelaxLowFactor)} " +
                "flIouFloor=${fmt(cfg.firstLockAdaptive.iouFloorStrong)}/${fmt(cfg.firstLockAdaptive.iouFloorMedium)}/${fmt(cfg.firstLockAdaptive.iouFloorBase)} " +
                "flConfFloor=${fmt(cfg.firstLockAdaptive.confStrongMin)}/${fmt(cfg.firstLockAdaptive.confMediumMin)}/${fmt(cfg.firstLockAdaptive.confSmallMin)}/${fmt(cfg.firstLockAdaptive.confBaseMin)} " +
                "fallbackRefineExpand=${fmt(cfg.fallbackRefine.expandFactor)} fallbackRefineGood=${cfg.fallbackRefine.minGoodMatches} " +
                "fallbackRefineInliers=${cfg.fallbackRefine.minInliers} fallbackRefineConf=${fmt(cfg.fallbackRefine.minConfidence)} " +
                "fallbackRefineRatio=${fmt(cfg.fallbackRefine.loweRatio)} " +
                "autoVerifyStrong=${cfg.autoInitVerify.strongCandidateMinGood}/${cfg.autoInitVerify.strongCandidateMinInliers}/${fmt(cfg.autoInitVerify.strongCandidateMinConfidence)} " +
                "autoVerifyLocal=${fmt(cfg.autoInitVerify.localCenterFactorLive)}/${fmt(cfg.autoInitVerify.localCenterFactorReplay)}@${fmt(cfg.autoInitVerify.localMinIouLive)}/${fmt(cfg.autoInitVerify.localMinIouReplay)} " +
                "autoVerifyAppear=${fmt(cfg.autoInitVerify.appearanceMinStrong)}/${fmt(cfg.autoInitVerify.appearanceMinMedium)}/${fmt(cfg.autoInitVerify.appearanceMinBase)}+${fmt(cfg.autoInitVerify.appearanceLiveBias)} " +
                "temporalGate=${cfg.temporalGate.highGoodMatches}/${cfg.temporalGate.highInliers}/${fmt(cfg.temporalGate.minConfidenceHighGood)} " +
                "temporalMedium=${cfg.temporalGate.mediumGoodMatches}/${fmt(cfg.temporalGate.minConfidenceMedium)} " +
                "temporalLow=${fmt(cfg.temporalGate.minConfidenceSmallRefined)}/${fmt(cfg.temporalGate.minConfidenceBase)} liveRelax=${fmt(cfg.temporalGate.liveConfidenceRelax)} floor=${fmt(cfg.temporalGate.liveConfidenceFloor)} " +
                "smallArea=${fmt(cfg.smallTarget.areaRatio)} smallGood=${cfg.smallTarget.minGoodMatches} " +
                "smallInliers=${cfg.smallTarget.minInliers} smallScale=${fmt(cfg.smallTarget.scaleThreshold)} " +
                "weakFallbackMaxMatches=${cfg.weakFallback.maxMatches} weakFallbackMaxSpan=${fmt(cfg.weakFallback.maxSpanPx)} " +
                "weakFallbackMaxArea=${fmt(cfg.weakFallback.maxAreaPx)} weakFallbackRequireRefine=${cfg.weakFallback.requireRefine} " +
                "weakFallbackRescue=${cfg.weakFallback.rescueEnabled} weakFallbackRescueRatio=${fmt(cfg.weakFallback.rescueRatio)} " +
                "weakFallbackRescueGood=${cfg.weakFallback.rescueMinGood} " +
                "weakFallbackRelaxMiss=${cfg.weakFallback.relaxMissStreak} weakFallbackRelaxFactor=${fmt(cfg.weakFallback.relaxFactor)} " +
                "weakFallbackCoreRescue=${cfg.weakFallback.coreRescueEnabled} " +
                "weakFallbackCoreSpan=${fmt(cfg.weakFallback.coreMaxSpanPx)} " +
                "weakFallbackCoreArea=${fmt(cfg.weakFallback.coreMaxAreaPx)} " +
                "softRelaxEnable=${cfg.softRelax.enabled} softRelaxMiss=${cfg.softRelax.missStreak} " +
                "softRelaxGood=${cfg.softRelax.minGoodMatches} softRelaxScale=${fmt(cfg.softRelax.scaleThreshold)} " +
                "softRelaxRatio=${fmt(cfg.softRelax.maxRatio)} " +
                "firstLockGapMs=${cfg.firstLock.gapMs} verifyInt=${cfg.trackVerify.intervalFrames} " +
                "verifyExpand=${fmt(cfg.trackVerify.localExpandFactor)} verifyGood=${cfg.trackVerify.minGoodMatches} " +
                "verifyInliers=${cfg.trackVerify.minInliers} verifyTol=${cfg.trackVerify.failTolerance} verifyHardTol=${cfg.trackVerify.hardDriftTolerance} " +
                "verifyNativeBypassConf=${fmt(cfg.trackVerify.nativeBypassConfidence)} " +
                "verifyRecenter=${fmt(cfg.trackVerify.recenterPx)} " +
                "verifyIou=${fmt(cfg.trackVerify.minIou)} verifyConfMargin=${fmt(cfg.trackVerify.switchConfidenceMargin)} " +
                "trackGuardJump=${fmt(cfg.trackGuard.maxCenterJumpFactor)} " +
                "trackGuardArea=${fmt(cfg.trackGuard.minAreaRatio)}/${fmt(cfg.trackGuard.maxAreaRatio)} " +
                "trackGuardDrop=${cfg.trackGuard.dropStreak} " +
                "trackGuardAppearInt=${cfg.trackGuard.appearanceCheckIntervalFrames} " +
                "trackGuardAppearMin=${fmt(cfg.trackGuard.minAppearanceScore)} " +
                "trackGuardAnchor=$trackGuardAnchorEnabled/${fmt(trackGuardAnchorMaxDrop)}/${fmt(trackGuardAnchorMinScore)} " +
                "trackBackend=${preferredTrackBackend.name.lowercase(Locale.US)} " +
                "kcfFail=$kcfMaxFailStreak nativeFail=${cfg.nativeGate.maxFailStreak} nativeMinConf=${fmt(cfg.nativeGate.minConfidence)} " +
                "nativeFuseSoft=${fmt(cfg.nativeGate.fuseSoftConfidence)} nativeFuseHard=${fmt(cfg.nativeGate.fuseHardConfidence)} " +
                "nativeFuseStreak=${cfg.nativeGate.fuseSoftStreak} nativeFuseWarmup=${cfg.nativeGate.fuseWarmupFrames} " +
                "nativeHoldLast=${cfg.nativeGate.holdLastOnSoftReject} nativeOrbVerify=$nativeOrbVerifyEnabled nativeScoreLogInt=$nativeScoreLogIntervalFrames " +
                "lockHoldFrames=$lockHoldFrames lostOverlayHoldMs=$lostOverlayHoldMs " +
                "fastFirstLockFrames=$fastFirstLockFrames " +
                "nativeParam=${summarizePath(nativeModelParamPathOverride)} " +
                "nativeBin=${summarizePath(nativeModelBinPathOverride)} " +
                "trackerGc=$forceTrackerGcOnDrop " +
                "kalmanEnable=${kalmanConfig.enabled} kalmanQ=${fmt(kalmanConfig.processNoise)} " +
                "kalmanR=${fmt(kalmanConfig.measurementNoise)} kalmanMaxMs=${kalmanConfig.maxPredictMs} " +
                "kalmanPredHold=${kalmanConfig.usePredictedHold} " +
                "kalmanDynR=${kalmanConfig.dynamicMeasurementNoise} " +
                "kalmanConf=${fmt(kalmanConfig.lowConfidenceThreshold)}/${fmt(kalmanConfig.highConfidenceThreshold)} " +
                "kalmanRScale=${fmt(kalmanConfig.highConfidenceNoiseScale)}/${fmt(kalmanConfig.lowConfidenceNoiseScale)}/${fmt(kalmanConfig.occlusionNoiseScale)} " +
                "kalmanFeedNativePrior=${kalmanConfig.feedNativePrior} " +
                "kalmanPreTrackPriorFeed=${kalmanConfig.preTrackPriorFeed} " +
                "kalmanPriorUncertainOnly=${kalmanConfig.priorOnlyOnUncertain} " +
                "kalmanPriorMinIou=${fmt(kalmanConfig.priorMinIou)} " +
                "kalmanPriorStaleMs=${kalmanConfig.priorStaleMs}"
        )
    }

    fun setInitialTarget(viewRect: RectF, viewWidth: Int, viewHeight: Int) {
        val sx = if (scaleX > 0f) scaleX else 1f
        val sy = if (scaleY > 0f) scaleY else 1f
        pendingInitBox = Rect(
            ((viewRect.left - offsetX) / sx).toInt(),
            ((viewRect.top - offsetY) / sy).toInt(),
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
            logDiag(
                "TEMPLATE",
                "session=$diagSessionId templates=$templateLibrarySize size=${firstW}x${firstH} kp=$templateKeypointCount texture=${fmt(templateTextureScore)} minTexture=${fmt(templateMinTextureScore)} ready=$ok"
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
            configureOrbDetector(max(orbMaxFeatures, orbFarBoostFeatures))
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
        logDiag("TRACK", "session=$diagSessionId action=reset_tracking")
        releaseTracker("reset", requestGc = true)
        pendingInitBox = null
        lastTrackedBox = null
        lastMeasuredTrackBox = null
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchMissStreak = 0
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = 0
        fastFirstLockRemaining = fastFirstLockFrames
        lastNativeAcceptMs = 0L
        overlayResetToken++
        clearFirstLockCandidate("reset")
        latestSearchFrame?.release()
        latestSearchFrame = null
        resetKalman("reset_tracking")
        updateLatestPrediction(null, 0.0, tracking = false)
        overlayView.post { overlayView.reset() }
    }

    private fun releaseTracker(reason: String, requestGc: Boolean) {
        logDiag("TRACK", "session=$diagSessionId action=release reason=$reason requestGc=$requestGc isTracking=$isTracking backend=${activeTrackBackend.name.lowercase(Locale.US)}")
        val hadTracker = tracker != null
        val hadNativeTracking = isTracking && activeTrackBackend != TrackBackend.KCF
        tracker = null
        if (hadNativeTracking) {
            NativeTrackerBridge.reset()
        }
        isTracking = false
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        activeTrackBackend = TrackBackend.KCF
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lastNativeAcceptMs = 0L
        if (!hadTracker && !hadNativeTracking) return

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
                "EVAL_SUMMARY reason=%s mode=%s elapsedMs=%d frames=%d avgFrameMs=%.2f locks=%d lost=%d firstLockSec=%.3f trackRatio=%.3f maxTrackStreak=%d searchSkipStride=%d searchSkipBudget=%d budgetTrips=%d nativeFuseSoft=%d nativeFuseHard=%d nativeHold=%d kalmanPredHold=%d",
                reason,
                trackerMode.name.lowercase(Locale.US),
                elapsed,
                metricsFrames,
                avgMs,
                metricsLockCount,
                metricsLostCount,
                firstLockSec,
                trackingRatio,
                metricsMaxTrackingStreak,
                metricsSearchStrideSkipCount,
                metricsSearchBudgetSkipCount,
                metricsSearchBudgetTripCount,
                metricsNativeFuseSoftCount,
                metricsNativeFuseHardCount,
                metricsNativeLowConfHoldCount,
                metricsKalmanPredHoldCount
            )
        )
    }
    @Synchronized
    fun latestPredictionSnapshot(): PredictionSnapshot {
        val box = latestPredictionBox
        return PredictionSnapshot(
            frameId = latestPredictionFrame,
            x = box?.x ?: -1,
            y = box?.y ?: -1,
            width = box?.width ?: -1,
            height = box?.height ?: -1,
            confidence = latestPredictionConfidence,
            tracking = latestPredictionTracking
        )
    }

    @Synchronized
    fun evalMetricsSnapshot(): EvalMetricsSnapshot {
        val frames = metricsFrames
        val trackRatio = if (frames > 0) metricsTrackingFrames.toDouble() / frames else 0.0
        val avgFrameMs = if (frames > 0) metricsTotalProcessMs.toDouble() / frames else 0.0
        val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
        val kalmanPredHoldRatio = if (frames > 0) metricsKalmanPredHoldCount.toDouble() / frames else 0.0
        return EvalMetricsSnapshot(
            frames = frames,
            locks = metricsLockCount,
            lost = metricsLostCount,
            trackRatio = trackRatio,
            avgFrameMs = avgFrameMs,
            firstLockSec = firstLockSec,
            kalmanPredHold = metricsKalmanPredHoldCount,
            kalmanPredHoldRatio = kalmanPredHoldRatio
        )
    }
    override fun analyze(image: ImageProxy) {
        val startMs = SystemClock.elapsedRealtime()
        var frame: Mat? = null
        try {
            isReplayInput = false
            val logicalSize = logicalFrameSize(image)
            updateScaleFactors(logicalSize.first, logicalSize.second)
            if (isTracking && activeTrackBackend != TrackBackend.KCF && pendingInitBox == null) {
                frameCounter++
                updateKalmanPrediction(SystemClock.elapsedRealtime())
                trackFrameNativeImage(image, logicalSize.first, logicalSize.second)
            } else {
                frame = imageToMat(image)
                processFrame(frame, image)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "analyze failed", t)
        } finally {
            frame?.release()
            image.close()
            updateMetrics((SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L))
        }
    }

    fun analyzeReplayFrame(frame: Mat) {
        val startMs = SystemClock.elapsedRealtime()
        try {
            isReplayInput = true
            updateScaleFactors(frame.cols(), frame.rows())
            processFrame(frame, image = null)
        } catch (t: Throwable) {
            Log.e(TAG, "analyze replay frame failed", t)
        } finally {
            updateMetrics((SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L))
        }
    }

    private fun processFrame(frame: Mat, image: ImageProxy? = null) {
        frameCounter++
        updateKalmanPrediction(SystemClock.elapsedRealtime())
        if (!isTracking && fastFirstLockRemaining > 0) {
            fastFirstLockRemaining--
        }
        when {
            attemptManualInitialization(frame, image) -> Unit
            isTracking -> trackFrame(frame)
            else -> {
                latestSearchFrame?.release()
                latestSearchFrame = Mat()
                frame.copyTo(latestSearchFrame)
                val skipReason = consumeSearchSkipReason()
                if (skipReason != null) {
                    onSearchFrameSkipped(skipReason)
                } else {
                    searchFrameByOrb(frame, image)
                }
            }
        }
    }

    private fun consumeSearchSkipReason(): String? {
        if (searchBudgetCooldownFrames > 0) {
            searchBudgetCooldownFrames--
            return "budget_cooldown"
        }
        if (searchStrideFrames > 1 && frameCounter % searchStrideFrames.toLong() != 0L) {
            return "stride"
        }
        return null
    }

    private fun onSearchFrameSkipped(reason: String) {
        when (reason) {
            "budget_cooldown" -> {
                metricsSearchBudgetSkipCount++
                metricsSearchLastReason = "search_skip_budget"
            }
            else -> {
                metricsSearchStrideSkipCount++
                metricsSearchLastReason = "search_skip_stride"
            }
        }
        expireFirstLockCandidateIfNeeded()
        updateLatestPrediction(null, 0.0, tracking = false)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=SEARCH_SKIPPED reason=$reason stride=$searchStrideFrames " +
                    "budgetMs=$searchOverBudgetMs budgetSkip=$searchOverBudgetSkipFrames cooldown=$searchBudgetCooldownFrames"
            )
        }
    }

    private fun attemptManualInitialization(frame: Mat, image: ImageProxy?): Boolean {
        val requested = pendingInitBox ?: return false
        pendingInitBox = null
        val safe = clampRect(requested, frame.cols(), frame.rows()) ?: return false
        initializeTracker(frame, safe, "manual", image)
        return true
    }

    private fun searchFrameByOrb(frame: Mat, image: ImageProxy? = null) {
        val templateReady = templatePyramidLevels.isNotEmpty()
        if (!templateReady) {
            logDiag(
                "SEARCH",
                "session=$diagSessionId stage=template_check ready=false kp=$templateKeypointCount texture=${fmt(templateTextureScore)} minTexture=${fmt(templateMinTextureScore)}"
            )
            metricsSearchTemplateSkipCount++
            metricsSearchLastReason = "template_not_ready"
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_SKIPPED reason=template_not_ready kp=$templateKeypointCount " +
                        "texture=${fmt(templateTextureScore)} textureMin=${fmt(templateMinTextureScore)}"
                )
            }
            lastTemplateReadyState = false
            clearFirstLockCandidate("template_not_ready")
            updateLatestPrediction(null, 0.0, tracking = false)
            return
        }
        lastTemplateReadyState = true

        val candidate = findOrbMatch(frame)
        if (candidate == null) {
            metricsSearchMissCount++
            metricsSearchLastReason = lastSearchDiagReason
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                logDiag(
                    "SEARCH",
                    "session=$diagSessionId stage=no_candidate reason=$lastSearchDiagReason miss=$searchMissStreak tplKp=$templateKeypointCount"
                )
            }
            searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
            expireFirstLockCandidateIfNeeded()
            updateLatestPrediction(null, 0.0, tracking = false)
            return
        }
        metricsSearchCandidateCount++
        metricsSearchLastReason = "candidate"
        val confirmed = maybeRefineFallbackCandidate(frame, candidate)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag(
                "SEARCH",
                "session=$diagSessionId stage=candidate good=${confirmed.goodMatches} inliers=${confirmed.inlierCount} conf=${fmt(confirmed.confidence)} box=${confirmed.box.x},${confirmed.box.y},${confirmed.box.width}x${confirmed.box.height} strong=${confirmed.isStrong} matcher=${confirmed.matcherType} fallback=${confirmed.fallbackReason ?: "none"}"
            )
        }
        updateLatestPrediction(confirmed.box, confirmed.confidence, tracking = false)

        if (shouldAutoInitWithoutManual(confirmed, frame.cols(), frame.rows())) {
            if (canDirectAutoInit(confirmed) && passesAutoInitVerification(frame, confirmed)) {
                searchMissStreak = 0
                clearFirstLockCandidate("auto_init")
                metricsSearchLastReason = "auto_lock_init"
                initializeTracker(
                    frame,
                    confirmed.box,
                    "orb_auto_init good=${confirmed.goodMatches} inliers=${confirmed.inlierCount} " +
                        "conf=${fmt(confirmed.confidence)} h=${confirmed.usedHomography} ds=${fmt(confirmed.searchScale)} " +
                        "fallback=${confirmed.fallbackReason ?: "none"} matcher=${confirmed.matcherType}",
                    image
                )
                return
            }
            metricsSearchLastReason = "auto_lock_buffer"
        }

        if (!isCandidateEligibleForTemporal(confirmed)) {
            metricsSearchTemporalRejectCount++
            searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
            if (heuristicConfig.firstLock.holdOnTemporalReject && firstLockCandidateFrames > 0) {
                metricsSearchTemporalHoldCount++
                metricsSearchLastReason = "temporal_reject_hold"
                firstLockCandidateLastMs = SystemClock.elapsedRealtime()
                expireFirstLockCandidateIfNeeded()
            } else {
                metricsSearchLastReason = "temporal_reject"
                clearFirstLockCandidate("candidate_not_lockable")
            }
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
            if (!isCandidateLockableForInit(promoted, frame.cols(), frame.rows())) {
                metricsSearchPromoteRejectCount++
                metricsSearchLastReason = "promoted_not_lockable"
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
            if (!passesAutoInitVerification(frame, promoted)) {
                logDiag(
                    "LOCK_GATE",
                    "session=$diagSessionId stage=promoted_verify pass=false good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
                )
                metricsSearchPromoteRejectCount++
                metricsSearchLastReason = "promoted_verify_reject"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_STABLE state=final_reject reason=promoted_verify_reject " +
                            "good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
                    )
                }
                searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
                clearFirstLockCandidate("promoted_verify_reject")
                return
            }
            searchMissStreak = 0
            clearFirstLockCandidate("promoted_candidate")
            metricsSearchLastReason = "lock_init"
            logDiag(
                "LOCK_GATE",
                "session=$diagSessionId stage=promote pass=true good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
            )
            initializeTracker(
                frame,
                promoted.box,
                "orb_temporal_confirm good=${promoted.goodMatches} inliers=${promoted.inlierCount} " +
                    "conf=${fmt(promoted.confidence)} h=${promoted.usedHomography} ds=${fmt(promoted.searchScale)} " +
                    "fallback=${promoted.fallbackReason ?: "none"} matcher=${promoted.matcherType}",
                image
            )
        }
    }

    private fun trackFrame(frame: Mat) {
        if (activeTrackBackend != TrackBackend.KCF) {
            trackFrameNative(frame)
            return
        }

        val tracked = Rect()
        val ok = tracker?.update(frame, tracked) == true
        if (!ok) {
            consecutiveTrackerFailures++
            Log.w(TAG, "KCF update failed: streak=$consecutiveTrackerFailures")
            if (consecutiveTrackerFailures < kcfMaxFailStreak) {
                lastTrackedBox?.let {
                    val held = correctKalman(it, confidence = 0.55, occluded = true)
                    lastTrackedBox = held
                    dispatchTrackedRect(held)
                    updateLatestPrediction(held, 0.55, tracking = true)
                }
                return
            }
            onLost("kcf_update_fail")
            return
        }

        val safe = clampRect(tracked, frame.cols(), frame.rows())
        if (safe == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < kcfMaxFailStreak) {
                lastTrackedBox?.let {
                    val held = correctKalman(it, confidence = 0.55, occluded = true)
                    lastTrackedBox = held
                    dispatchTrackedRect(held)
                    updateLatestPrediction(held, 0.55, tracking = true)
                }
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
                    if (holdCurrentBoxOnTransientLoss(verifyDecision.reason, 0.55)) {
                        return
                    }
                    onLost(verifyDecision.reason)
                    return
                }
            }
        }

        commitTrackingObservation(safe, confidence = 1.0, markNativeAccept = false)
    }

    private fun trackFrameNative(frame: Mat) {
        val nativeCfg = heuristicConfig.nativeGate
        val result = nativeTrackOnFrame(frame)
        if (result == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_track_pending", 0.45)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_track_fail", 0.45)) return
            onLost("native_track_fail")
            return
        }
        val nativeConfidence = result.confidence.toDouble().coerceIn(0.0, 1.0)
        val nativeSimilarity = result.similarity.toDouble().coerceIn(0.0, 1.0)
        val nativeMeasurementConfidence = resolveNativeMeasurementConfidence(result)
        recordNativeConfidenceSample(nativeConfidence)
        recordNativeSimilaritySample(nativeSimilarity)
        when (evaluateNativeConfidence(nativeConfidence)) {
            NativeConfidenceAction.DROP_HARD -> {
                logNativeScoreSample("native_mat", "drop_hard", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_hard", nativeConfidence)) return
                onLost("native_conf_hard")
                return
            }
            NativeConfidenceAction.DROP_SOFT_STREAK -> {
                logNativeScoreSample("native_mat", "drop_soft", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_soft", nativeConfidence)) return
                onLost("native_conf_soft")
                return
            }
            NativeConfidenceAction.HOLD -> {
                logNativeScoreSample("native_mat", "hold", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                suppressOverlayOnUncertainTracking("native_conf_hold", nativeConfidence)
                return
            }
            NativeConfidenceAction.DROP_MIN_CONF -> {
                logNativeScoreSample("native_mat", "drop_min_conf", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                consecutiveTrackerFailures++
                if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                    suppressOverlayOnUncertainTracking("native_conf_pending", nativeConfidence)
                    return
                }
                if (holdCurrentBoxOnTransientLoss("native_conf_low", nativeConfidence)) return
                onLost("native_conf_low")
                return
            }
            NativeConfidenceAction.ACCEPT -> {
                logNativeScoreSample("native_mat", "accept", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
            }
        }

        val tracked = Rect(
            result.x.roundToInt(),
            result.y.roundToInt(),
            result.w.roundToInt(),
            result.h.roundToInt()
        )
        val safe = clampRect(tracked, frame.cols(), frame.rows())
        if (safe == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_track_pending", 0.45)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_invalid_box", nativeConfidence)) return
            onLost("native_invalid_box")
            return
        }

        consecutiveTrackerFailures = 0
        if (shouldDropOnTrackGuard(frame, safe, nativeConfidence)) {
            onLost("track_guard_fail")
            return
        }
        val nativeVerifyInterval = resolveNativeVerifyIntervalFrames()
        if (
            shouldRunNativeOrbVerify() &&
            nativeVerifyInterval > 0 &&
            frameCounter % nativeVerifyInterval.toLong() == 0L
        ) {
            val verifyDecision = maybeVerifyTracking(frame, safe, nativeConfidence)
            if (verifyDecision != null) {
                when (verifyDecision.action) {
                    "realign" -> {
                        verifyDecision.box?.let {
                            initializeTracker(frame, it, verifyDecision.reason)
                            return
                        }
                    }
                    "drop" -> {
                        if (holdCurrentBoxOnTransientLoss(verifyDecision.reason, nativeConfidence)) {
                            return
                        }
                        onLost(verifyDecision.reason)
                        return
                    }
                }
            }
        }

        commitTrackingObservation(safe, confidence = nativeMeasurementConfidence, markNativeAccept = true)
    }

    private fun trackFrameNativeImage(image: ImageProxy, frameW: Int, frameH: Int) {
        val nativeCfg = heuristicConfig.nativeGate
        feedKalmanPriorToNative("pre_track_image")
        val result = NativeTrackerBridge.track(image)
        if (result == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_track_pending", 0.45)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_track_fail", 0.45)) return
            onLost("native_track_fail")
            return
        }
        val nativeConfidence = result.confidence.toDouble().coerceIn(0.0, 1.0)
        val nativeSimilarity = result.similarity.toDouble().coerceIn(0.0, 1.0)
        val nativeMeasurementConfidence = resolveNativeMeasurementConfidence(result)
        recordNativeConfidenceSample(nativeConfidence)
        recordNativeSimilaritySample(nativeSimilarity)
        when (evaluateNativeConfidence(nativeConfidence)) {
            NativeConfidenceAction.DROP_HARD -> {
                logNativeScoreSample("native_img", "drop_hard", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_hard", nativeConfidence)) return
                onLost("native_conf_hard")
                return
            }
            NativeConfidenceAction.DROP_SOFT_STREAK -> {
                logNativeScoreSample("native_img", "drop_soft", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_soft", nativeConfidence)) return
                onLost("native_conf_soft")
                return
            }
            NativeConfidenceAction.HOLD -> {
                logNativeScoreSample("native_img", "hold", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                suppressOverlayOnUncertainTracking("native_conf_hold", nativeConfidence)
                return
            }
            NativeConfidenceAction.DROP_MIN_CONF -> {
                logNativeScoreSample("native_img", "drop_min_conf", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                consecutiveTrackerFailures++
                if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                    suppressOverlayOnUncertainTracking("native_conf_pending", nativeConfidence)
                    return
                }
                if (holdCurrentBoxOnTransientLoss("native_conf_low", nativeConfidence)) return
                onLost("native_conf_low")
                return
            }
            NativeConfidenceAction.ACCEPT -> {
                logNativeScoreSample("native_img", "accept", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
            }
        }

        val tracked = Rect(
            result.x.roundToInt(),
            result.y.roundToInt(),
            result.w.roundToInt(),
            result.h.roundToInt()
        )
        val safe = clampRect(tracked, frameW, frameH)
        if (safe == null) {
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_track_pending", 0.45)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_invalid_box", nativeConfidence)) return
            onLost("native_invalid_box")
            return
        }

        if (shouldDropOnTrackGuard(null, safe, nativeConfidence)) {
            onLost("track_guard_fail")
            return
        }

        val nativeVerifyInterval = resolveNativeVerifyIntervalFrames()
        if (
            shouldRunNativeOrbVerify() &&
            nativeVerifyInterval > 0 &&
            frameCounter % nativeVerifyInterval.toLong() == 0L
        ) {
            val verifyFrame = imageToMat(image)
            try {
                val verifyDecision = maybeVerifyTracking(verifyFrame, safe, nativeConfidence)
                if (verifyDecision != null) {
                    when (verifyDecision.action) {
                        "realign" -> {
                            verifyDecision.box?.let {
                                initializeTracker(verifyFrame, it, verifyDecision.reason, image)
                                return
                            }
                        }
                        "drop" -> {
                            if (holdCurrentBoxOnTransientLoss(verifyDecision.reason, nativeConfidence)) {
                                return
                            }
                            onLost(verifyDecision.reason)
                            return
                        }
                    }
                }
            } finally {
                verifyFrame.release()
            }
        }

        consecutiveTrackerFailures = 0
        commitTrackingObservation(safe, confidence = nativeMeasurementConfidence, markNativeAccept = true)
    }


    private fun evaluateTrackBoxConsistency(current: Rect, previous: Rect?, cfg: TrackGuardConfig): Boolean {
        val prev = previous ?: return true
        val currCenter = rectCenter(current)
        val prevCenter = rectCenter(prev)
        val jump = pointDistance(currCenter, prevCenter)
        val ref = max(min(prev.width, prev.height), min(current.width, current.height)).toDouble().coerceAtLeast(1.0)
        val maxJump = ref * cfg.maxCenterJumpFactor

        val prevArea = (prev.width.toDouble() * prev.height.toDouble()).coerceAtLeast(1.0)
        val currArea = (current.width.toDouble() * current.height.toDouble()).coerceAtLeast(1.0)
        val areaRatio = currArea / prevArea
        val areaOk = areaRatio in cfg.minAreaRatio..cfg.maxAreaRatio
        return jump <= maxJump && areaOk
    }

    private fun resolveNativeMeasurementConfidence(result: NativeTrackerBridge.NativeTrackResult): Double {
        val confidence = result.confidence.toDouble().coerceIn(0.0, 1.0)
        val similarity = result.similarity.toDouble().coerceIn(0.0, 1.0)
        // Dynamic-R should track native model similarity more closely while
        // preserving existing confidence gate behavior.
        return (similarity * 0.70 + confidence * 0.30).coerceIn(0.0, 1.0)
    }

    private fun shouldDropOnTrackGuard(frame: Mat?, current: Rect, confidence: Double): Boolean {
        val guardCfg = heuristicConfig.trackGuard
        val prev = lastMeasuredTrackBox
        val geomOk = evaluateTrackBoxConsistency(current, prev, guardCfg)
        if (!geomOk) {
            trackMismatchStreak = (trackMismatchStreak + 1).coerceAtMost(1000)
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                val jump = if (prev == null) 0.0 else pointDistance(rectCenter(current), rectCenter(prev))
                logDiag(
                    "TRACK",
                    "session=$diagSessionId stage=track_guard reason=geom_fail streak=$trackMismatchStreak conf=${fmt(confidence)} jump=${fmt(jump)}"
                )
            }
        } else {
            trackMismatchStreak = 0
        }

        if (frame != null && frameCounter % guardCfg.appearanceCheckIntervalFrames == 0L) {
            val score = computePatchTemplateCorrelation(frame, current)
            val anchorScore = trackAnchorAppearanceScore
            val anchorMinScore =
                if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
                    val dynamicMin = anchorScore - trackGuardAnchorMaxDrop
                    if (trackGuardAnchorMinScore > -0.999) {
                        max(dynamicMin, trackGuardAnchorMinScore)
                    } else {
                        dynamicMin
                    }
                } else {
                    Double.NEGATIVE_INFINITY
                }
            val minAllowedScore = max(guardCfg.minAppearanceScore, anchorMinScore)
            if (score < minAllowedScore) {
                trackAppearanceLowStreak = (trackAppearanceLowStreak + 1).coerceAtMost(1000)
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    logDiag(
                        "TRACK",
                        "session=$diagSessionId stage=track_guard reason=appearance_fail streak=$trackAppearanceLowStreak conf=${fmt(confidence)} " +
                            "score=${fmt(score)} min=${fmt(minAllowedScore)} anchor=${fmt(anchorScore)}"
                    )
                }
            } else {
                trackAppearanceLowStreak = 0
            }
        }

        return trackMismatchStreak >= guardCfg.dropStreak || trackAppearanceLowStreak >= guardCfg.dropStreak
    }

    private fun evaluateNativeConfidence(confidence: Double): NativeConfidenceAction {
        val nativeCfg = heuristicConfig.nativeGate
        if (nativeFuseWarmupRemaining > 0) {
            nativeFuseWarmupRemaining = (nativeFuseWarmupRemaining - 1).coerceAtLeast(0)
            nativeLowConfidenceStreak = 0
            if (confidence < nativeCfg.fuseSoftConfidence) {
                // Warm-up gate: suppress DROP decisions right after lock but keep box updates flowing.
                metricsSearchLastReason = "native_conf_warmup_bypass"
                return NativeConfidenceAction.ACCEPT
            }
            metricsSearchLastReason = "native_conf_warmup_accept"
            return NativeConfidenceAction.ACCEPT
        }
        if (confidence < nativeCfg.fuseHardConfidence) {
            metricsNativeFuseHardCount++
            metricsSearchLastReason = "native_conf_hard"
            nativeLowConfidenceStreak = 0
            return NativeConfidenceAction.DROP_HARD
        }
        if (confidence < nativeCfg.fuseSoftConfidence) {
            nativeLowConfidenceStreak = (nativeLowConfidenceStreak + 1).coerceAtMost(10_000)
            if (nativeLowConfidenceStreak >= nativeCfg.fuseSoftStreak) {
                metricsNativeFuseSoftCount++
                metricsSearchLastReason = "native_conf_soft"
                nativeLowConfidenceStreak = 0
                return NativeConfidenceAction.DROP_SOFT_STREAK
            }
            if (nativeCfg.holdLastOnSoftReject) {
                metricsNativeLowConfHoldCount++
                metricsSearchLastReason = "native_conf_hold"
                return NativeConfidenceAction.HOLD
            }
        } else {
            nativeLowConfidenceStreak = 0
        }
        if (confidence < nativeCfg.minConfidence) {
            metricsSearchLastReason = "native_conf_low"
            return NativeConfidenceAction.DROP_MIN_CONF
        }
        return NativeConfidenceAction.ACCEPT
    }

    private fun shouldRunNativeOrbVerify(): Boolean {
        return nativeOrbVerifyEnabled || isReplayInput
    }

    private fun resolveNativeVerifyIntervalFrames(): Int {
        val base = heuristicConfig.trackVerify.intervalFrames.coerceAtLeast(1)
        if (isReplayInput && !nativeOrbVerifyEnabled) {
            return min(base, REPLAY_NATIVE_VERIFY_INTERVAL_FRAMES)
        }
        return base
    }

    private fun recordNativeConfidenceSample(confidence: Double) {
        val c = confidence.coerceIn(0.0, 1.0)
        metricsNativeConfSamples++
        metricsNativeConfSum += c
        if (c < metricsNativeConfMin) metricsNativeConfMin = c
        if (c > metricsNativeConfMax) metricsNativeConfMax = c
    }

    private fun recordNativeSimilaritySample(similarity: Double) {
        val s = similarity.coerceIn(0.0, 1.0)
        metricsNativeSimSamples++
        metricsNativeSimSum += s
        if (s < metricsNativeSimMin) metricsNativeSimMin = s
        if (s > metricsNativeSimMax) metricsNativeSimMax = s
    }

    private fun logNativeScoreSample(
        source: String,
        action: String,
        confidence: Double,
        similarity: Double,
        measurement: Double
    ) {
        val logInterval = nativeScoreLogIntervalFrames.coerceAtLeast(1).toLong()
        if (frameCounter % logInterval != 0L) return
        Log.i(
            TAG,
            "EVAL_NATIVE_SCORE src=$source frame=$frameCounter action=$action " +
                "conf=${fmt(confidence)} sim=${fmt(similarity)} meas=${fmt(measurement)} " +
                "tracking=$isTracking"
        )
    }

    private fun holdCurrentBoxOnTransientLoss(reason: String, confidence: Double): Boolean {
        if (!isTracking) return false
        if (lockHoldFramesRemaining <= 0) return false
        val box = lastMeasuredTrackBox ?: lastTrackedBox ?: return false
        val eligible =
            reason.startsWith("verify_") ||
                reason.startsWith("native_conf_") ||
                reason == "native_track_fail" ||
                reason == "native_invalid_box" ||
                reason.startsWith("kcf_")
        if (!eligible) return false

        lockHoldFramesRemaining = (lockHoldFramesRemaining - 1).coerceAtLeast(0)
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        val held = correctKalman(box, confidence.coerceIn(0.0, 1.0), occluded = true)
        lastTrackedBox = held
        dispatchTrackedRect(held)
        updateLatestPrediction(held, confidence.coerceIn(0.20, 0.99), tracking = true)
        feedNativePriorBox(held, "transient_hold")
        metricsSearchLastReason = "loss_grace_hold"
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=LOSS_GRACE reason=$reason remaining=$lockHoldFramesRemaining " +
                    "box=${held.x},${held.y},${held.width}x${held.height}"
            )
        }
        return true
    }

    private fun onLost(reason: String) {
        logDiag("TRACK", "session=$diagSessionId action=lost_enter reason=$reason")
        releaseTracker("lost_$reason", requestGc = true)
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = 0
        lastNativeAcceptMs = 0L
        lastTrackedBox = null
        lastMeasuredTrackBox = null
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackAnchorAppearanceScore = Double.NaN
        metricsLostCount++
        clearFirstLockCandidate("lost")
        latestSearchFrame?.release()
        latestSearchFrame = null
        resetKalman("lost_$reason")
        updateLatestPrediction(null, 0.0, tracking = false)
        Log.w(TAG, "EVAL_EVENT type=LOST reason=$reason lost=$metricsLostCount")
        logDiag("TRACK", "session=$diagSessionId action=lost reason=$reason lost=$metricsLostCount")
        val token = ++overlayResetToken
        if (lostOverlayHoldMs <= 0L) {
            overlayView.post { overlayView.reset() }
        } else {
            overlayView.postDelayed(
                {
                    if (!isTracking && token == overlayResetToken) {
                        overlayView.reset()
                    }
                },
                lostOverlayHoldMs
            )
        }
    }

    private fun initializeTracker(frame: Mat, box: Rect, reason: String, image: ImageProxy? = null) {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return
        trackAnchorAppearanceScore = computePatchTemplateCorrelation(frame, safe).coerceIn(-1.0, 1.0)
        val wasTracking = isTracking
        latestSearchFrame?.release()
        latestSearchFrame = null
        releaseTracker("reinit_$reason", requestGc = false)
        resetKalman("reinit_$reason")

        if (preferredTrackBackend != TrackBackend.KCF) {
            val nativeOk = initializeNativeTracker(frame, safe, preferredTrackBackend, image)
            if (nativeOk) {
                tracker = null
                isTracking = true
                trackingStage = TrackingStage.TRACK
                activeTrackBackend = preferredTrackBackend
                consecutiveTrackerFailures = 0
                trackMismatchStreak = 0
                trackAppearanceLowStreak = 0
                trackVerifyFailStreak = 0
                trackVerifyHardDriftStreak = 0
                searchMissStreak = 0
                searchBudgetCooldownFrames = 0
                nativeLowConfidenceStreak = 0
                nativeFuseWarmupRemaining = heuristicConfig.nativeGate.fuseWarmupFrames
                lockHoldFramesRemaining = lockHoldFrames
                fastFirstLockRemaining = 0
                overlayResetToken++
                commitTrackingObservation(safe, confidence = 1.0, markNativeAccept = true)
                if (!wasTracking) {
                    metricsLockCount++
                    if (metricsFirstLockMs < 0L) {
                        metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
                    }
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=LOCK reason=$reason backend=${activeTrackBackend.name.lowercase(Locale.US)} " +
                            "locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)}"
                    )
                }
                Log.i(TAG, "Native tracker initialized ($reason): ${safe.x},${safe.y},${safe.width}x${safe.height}")
                logDiag(
                    "LOCK",
                    "session=$diagSessionId backend=${activeTrackBackend.name.lowercase(Locale.US)} reason=$reason box=${safe.x},${safe.y},${safe.width}x${safe.height} firstLockSec=${fmt(if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0)}"
                )
                return
            }
            Log.w(TAG, "EVAL_EVENT type=TRACKER_FALLBACK reason=native_init_failed backend=${preferredTrackBackend.name.lowercase(Locale.US)}")
        }

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
        trackingStage = TrackingStage.TRACK
        activeTrackBackend = TrackBackend.KCF
        consecutiveTrackerFailures = 0
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchMissStreak = 0
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = lockHoldFrames
        lastNativeAcceptMs = 0L
        fastFirstLockRemaining = 0
        overlayResetToken++
        commitTrackingObservation(safe, confidence = 1.0, markNativeAccept = false)

        if (!wasTracking) {
            metricsLockCount++
            if (metricsFirstLockMs < 0L) {
                metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
            }
            Log.w(
                TAG,
                "EVAL_EVENT type=LOCK reason=$reason backend=${activeTrackBackend.name.lowercase(Locale.US)} " +
                    "locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)}"
            )
        }

        Log.i(TAG, "KCF initialized ($reason): ${safe.x},${safe.y},${safe.width}x${safe.height}")
        logDiag(
            "LOCK",
            "session=$diagSessionId backend=${activeTrackBackend.name.lowercase(Locale.US)} reason=$reason box=${safe.x},${safe.y},${safe.width}x${safe.height} firstLockSec=${fmt(if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0)}"
        )
    }

    private fun initializeNativeTracker(frame: Mat, box: Rect, backend: TrackBackend, image: ImageProxy? = null): Boolean {
        val nativeBackend = when (backend) {
            TrackBackend.NATIVE_RKNN -> NativeTrackerBridge.Backend.RKNN
            else -> NativeTrackerBridge.Backend.NCNN
        }
        val requestedParam = nativeModelParamPathOverride
        val requestedBin = nativeModelBinPathOverride
        val initEngineOk = NativeTrackerBridge.initializeEngine(nativeBackend, requestedParam, requestedBin)
        if (!initEngineOk || !NativeTrackerBridge.isAvailable()) {
            Log.w(
                TAG,
                "native engine unavailable backend=${nativeBackend.name.lowercase(Locale.US)} " +
                    "param=${summarizePath(requestedParam)} bin=${summarizePath(requestedBin)}"
            )
            return false
        }
        Log.i(
            TAG,
            "native engine ready backend=${NativeTrackerBridge.backendName()} " +
                "requestParam=${summarizePath(requestedParam)} requestBin=${summarizePath(requestedBin)}"
        )
        val nativeBox = android.graphics.Rect(box.x, box.y, box.x + box.width, box.y + box.height)
        if (image != null) {
            return NativeTrackerBridge.initTarget(image, nativeBox)
        }
        val rgb = frameToRgbBytes(frame)
        return NativeTrackerBridge.initTargetGray(rgb, frame.cols(), frame.rows(), nativeBox)
    }

    private fun nativeTrackOnFrame(frame: Mat): NativeTrackerBridge.NativeTrackResult? {
        feedKalmanPriorToNative("pre_track_gray")
        val rgb = frameToRgbBytes(frame)
        return NativeTrackerBridge.trackGray(rgb, frame.cols(), frame.rows())
    }

    private fun frameToRgbBytes(frame: Mat): ByteArray {
        val rgb =
            if (frame.channels() == 3) {
                frame
            } else {
                Imgproc.cvtColor(frame, nativeRgbMat, Imgproc.COLOR_GRAY2RGB)
                nativeRgbMat
            }
        val needed = rgb.cols() * rgb.rows() * 3
        val existing = nativeRgbBuffer
        val buffer =
            if (existing != null && existing.size == needed) {
                existing
            } else {
                ByteArray(needed).also { nativeRgbBuffer = it }
            }
        rgb.get(0, 0, buffer)
        return buffer
    }

    private fun maybeVerifyTracking(
        frame: Mat,
        trackedBox: Rect,
        currentTrackConfidence: Double? = null
    ): TrackVerifyDecision? {
        val verifyCfg = heuristicConfig.trackVerify
        if (verifyCfg.intervalFrames <= 0 || frameCounter % verifyCfg.intervalFrames.toLong() != 0L) {
            return null
        }

        if (activeTrackBackend != TrackBackend.KCF) {
            // NCNN warm-up is sensitive to ORB false negatives; avoid immediate drift drop.
            if (nativeFuseWarmupRemaining > 0) return null
            if ((currentTrackConfidence ?: 0.0) >= verifyCfg.nativeBypassConfidence) {
                trackVerifyFailStreak = 0
                trackVerifyHardDriftStreak = 0
                return null
            }
        }

        val trackedQuality = evaluateBoxMatch(frame, trackedBox)
        val hardDrift = trackedQuality.goodMatches < TRACK_VERIFY_HARD_MIN_GOOD_MATCHES
        val trackedHealthy =
            trackedQuality.goodMatches >= verifyCfg.minGoodMatches &&
                trackedQuality.inlierCount >= verifyCfg.minInliers

        val localRegion = expandRectWithFactor(trackedBox, verifyCfg.localExpandFactor, frame.cols(), frame.rows()) ?: trackedBox
        val localCandidate = findOrbMatchInRoi(frame, localRegion)
        val localHealthy =
            localCandidate != null &&
                localCandidate.goodMatches >= verifyCfg.minGoodMatches &&
                localCandidate.inlierCount >= verifyCfg.minInliers
        if (localCandidate != null && localHealthy) {
            val shift = pointDistance(rectCenter(trackedBox), rectCenter(localCandidate.box))
            val iou = rectIou(trackedBox, localCandidate.box)
            val shouldRealign = shift > verifyCfg.recenterPx || iou < verifyCfg.minIou
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
                            "conf=${fmt(trackedQuality.confidence)} hardStreak=$trackVerifyHardDriftStreak/${verifyCfg.hardDriftTolerance}"
                )
            }
            if (trackVerifyHardDriftStreak >= verifyCfg.hardDriftTolerance) {
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
                        "conf=${fmt(trackedQuality.confidence)} streak=$trackVerifyFailStreak/${verifyCfg.failTolerance}"
                )
            }
            if (trackVerifyFailStreak >= verifyCfg.failTolerance) {
                trackVerifyHardDriftStreak = 0
                return TrackVerifyDecision(
                    action = "drop",
                    box = null,
                    reason = "verify_drift good=${trackedQuality.goodMatches} inliers=${trackedQuality.inlierCount} conf=${fmt(trackedQuality.confidence)}"
                )
            }
        }

        return null
    }

    private fun findOrbMatchInRoi(frame: Mat, roiRect: Rect, loweRatioOverride: Double? = null): OrbMatchCandidate? {
        val safeRoi = clampRect(roiRect, frame.cols(), frame.rows()) ?: return null
        val sub = frame.submat(safeRoi)
        try {
            val local = findOrbMatch(sub, loweRatioOverride = loweRatioOverride, allowSeedAssist = false) ?: return null
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


    private fun computePatchTemplateCorrelation(frame: Mat, box: Rect): Double {
        val template = templateGray ?: return 0.0
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return 0.0
        val roi = frame.submat(safe)
        val roiGray = Mat()
        val roiResized = Mat()
        val response = Mat()
        return try {
            when (roi.channels()) {
                1 -> roi.copyTo(roiGray)
                3 -> Imgproc.cvtColor(roi, roiGray, Imgproc.COLOR_RGB2GRAY)
                4 -> Imgproc.cvtColor(roi, roiGray, Imgproc.COLOR_RGBA2GRAY)
                else -> return 0.0
            }
            Imgproc.resize(roiGray, roiResized, Size(template.cols().toDouble(), template.rows().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.matchTemplate(roiResized, template, response, Imgproc.TM_CCOEFF_NORMED)
            response.get(0, 0)?.getOrNull(0)?.coerceIn(-1.0, 1.0) ?: 0.0
        } catch (_: Throwable) {
            0.0
        } finally {
            response.release()
            roiResized.release()
            roiGray.release()
            roi.release()
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

            configureOrbDetector(max(300, orbMaxFeatures / 2))
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
                val goodMatches = collectGoodMatches(knnMatches, heuristicConfig.orb.loweRatio)
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
        val firstLockCfg = heuristicConfig.firstLock
        val adaptiveCfg = heuristicConfig.firstLockAdaptive
        if (!isCandidateEligibleForTemporal(candidate)) return null
        val now = SystemClock.elapsedRealtime()
        val candidateCenter = Point(
            candidate.box.x + candidate.box.width * 0.5,
            candidate.box.y + candidate.box.height * 0.5
        )
        val appearanceScore = computePatchTemplateCorrelation(frame = latestSearchFrame ?: return null, box = candidate.box)

        val previous = firstLockCandidateBox
        val iouWithPrevious = if (previous != null) rectIou(previous, candidate.box) else 1.0
        val minSide = min(candidate.box.width, candidate.box.height).toDouble()
        val useCenterRule =
            minSide <= FIRST_LOCK_SMALL_BOX_SIDE_PX ||
                (candidate.box.width.toDouble() * candidate.box.height.toDouble()) <= FIRST_LOCK_SMALL_BOX_AREA_PX
        val relaxedSmallTarget =
            useCenterRule &&
                (searchMissStreak >= firstLockCfg.smallRelaxMissStreak || firstLockCandidateFrames > 0) &&
                !candidate.usedHomography &&
                (candidate.fallbackReason?.startsWith("refined_") == true)
        val effectiveStablePx = when {
            !useCenterRule -> firstLockCfg.stablePx
            relaxedSmallTarget -> max(firstLockCfg.stablePx, firstLockCfg.smallStablePx)
            else -> max(firstLockCfg.stablePx, firstLockCfg.smallCenterDriftPx * 2.0)
        }
        val effectiveCenterDriftPx = if (relaxedSmallTarget) {
            firstLockCfg.smallCenterDriftRelaxedPx
        } else {
            firstLockCfg.smallCenterDriftPx
        }
        val adaptiveIouBase = computeAdaptiveFirstLockIou(candidate.box, frameW, frameH)
        val adaptiveIouRelax = when {
            searchMissStreak >= adaptiveCfg.missRelaxHighStreak -> adaptiveCfg.missRelaxHighFactor
            searchMissStreak >= adaptiveCfg.missRelaxMidStreak -> adaptiveCfg.missRelaxMidFactor
            searchMissStreak >= adaptiveCfg.missRelaxLowStreak -> adaptiveCfg.missRelaxLowFactor
            firstLockCandidateFrames >= adaptiveCfg.resetRelaxMinFrames &&
                metricsSearchResetIouCount >= adaptiveCfg.resetRelaxMinIouResets -> adaptiveCfg.resetRelaxFactor
            else -> 1.0
        }
        val iouFloor = when {
            candidate.goodMatches >= adaptiveCfg.iouFloorStrongGood &&
                candidate.inlierCount >= adaptiveCfg.iouFloorStrongInliers -> adaptiveCfg.iouFloorStrong
            candidate.goodMatches >= adaptiveCfg.iouFloorMediumGood &&
                candidate.inlierCount >= adaptiveCfg.iouFloorMediumInliers -> adaptiveCfg.iouFloorMedium
            else -> adaptiveCfg.iouFloorBase
        }
        val adaptiveIou = (adaptiveIouBase * adaptiveIouRelax).coerceAtLeast(iouFloor)
        val centerDrift = pointDistance(firstLockCandidateCenter, candidateCenter)
        val dynamicCenterDriftPx = computeDynamicFirstLockCenterThreshold(
            previous = previous,
            candidate = candidate.box,
            useCenterRule = useCenterRule,
            baseCenterDriftPx = effectiveCenterDriftPx,
            stableDriftPx = effectiveStablePx
        )
        val relaxedSmallIouPass =
            useCenterRule &&
                previous != null &&
                centerDrift <= effectiveStablePx &&
                iouWithPrevious >= firstLockCfg.smallRelaxedIouFloor
        val centerRelaxFactor = when {
            useCenterRule && candidate.goodMatches >= adaptiveCfg.centerRelaxGood &&
                candidate.inlierCount >= adaptiveCfg.centerRelaxInliers -> adaptiveCfg.centerRelaxStrongFactor
            useCenterRule && searchMissStreak >= adaptiveCfg.centerRelaxMissStreak -> adaptiveCfg.centerRelaxMissFactor
            else -> 1.0
        }
        val effectiveCenterThreshold = (dynamicCenterDriftPx * centerRelaxFactor)
            .coerceAtMost(effectiveStablePx * adaptiveCfg.centerRelaxCapFactor)
        val centerRulePass = !useCenterRule || centerDrift <= effectiveCenterThreshold || relaxedSmallIouPass
        val iouPass = useCenterRule || iouWithPrevious >= adaptiveIou
        val confThreshold = when {
            candidate.goodMatches >= adaptiveCfg.confStrongGood &&
                candidate.inlierCount >= adaptiveCfg.confStrongInliers -> adaptiveCfg.confStrongMin
            candidate.goodMatches >= adaptiveCfg.confMediumGood &&
                candidate.inlierCount >= adaptiveCfg.confMediumInliers -> adaptiveCfg.confMediumMin
            useCenterRule -> adaptiveCfg.confSmallMin
            else -> adaptiveCfg.confBaseMin
        }
        val confPass = candidate.confidence >= confThreshold

        val shouldHoldOutlier =
            previous != null &&
                centerDrift > effectiveStablePx &&
                firstLockCandidateFrames > 0 &&
                !isClearlyBetterFirstLockCandidate(candidate)
        if (shouldHoldOutlier) {
            metricsSearchStableOutlierHoldCount++
            firstLockOutlierHoldStreak++
            if (firstLockOutlierHoldStreak < firstLockCfg.outlierHoldMax) {
                metricsSearchLastReason = "stable_outlier_hold"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_STABLE state=hold drift=${fmt(centerDrift)} " +
                            "stableTh=${fmt(effectiveStablePx)} hold=$firstLockOutlierHoldStreak/${firstLockCfg.outlierHoldMax} " +
                            "good=${candidate.goodMatches} inliers=${candidate.inlierCount}"
                    )
                }
                return null
            }
            firstLockOutlierHoldStreak = 0
            metricsSearchLastReason = "stable_outlier_reseed"
        } else {
            firstLockOutlierHoldStreak = 0
        }

        val resetNoSeed = previous == null
        val resetGap = !resetNoSeed && (now - firstLockCandidateLastMs) > firstLockCfg.gapMs
        val resetStableDrift = !resetNoSeed && centerDrift > effectiveStablePx
        val resetCenterRule = !resetNoSeed && useCenterRule && !centerRulePass
        val resetIou = !resetNoSeed && !iouPass
        val resetConf = !resetNoSeed && !confPass
        val needReset = resetNoSeed || resetGap || resetStableDrift || resetCenterRule || resetIou || resetConf

        val candidateScore = computeFirstLockCandidateScore(
            candidate = candidate,
            iouWithPrevious = iouWithPrevious,
            adaptiveIou = adaptiveIou,
            centerDrift = centerDrift,
            stableDriftPx = effectiveStablePx
        )

        if (needReset) {
            metricsSearchStableSeedCount++
            if (resetNoSeed) metricsSearchResetNoSeedCount++
            if (resetGap) metricsSearchResetGapCount++
            if (resetStableDrift) metricsSearchResetStableDriftCount++
            if (resetCenterRule) metricsSearchResetCenterRuleCount++
            if (resetIou) metricsSearchResetIouCount++
            if (resetConf) metricsSearchResetConfidenceCount++
            firstLockCandidateBox = candidate.box
            firstLockCandidateCenter = candidateCenter
            firstLockCandidateFrames = 1
            firstLockCandidateStartMs = now
            firstLockCandidateLastMs = now
            firstLockCandidateBestGood = candidate.goodMatches
            firstLockCandidateBestInliers = candidate.inlierCount
            firstLockCandidateBestScore = candidateScore
            firstLockCandidateAppearanceSum = appearanceScore
            firstLockCandidateAppearanceMin = appearanceScore
            firstLockCandidateAppearanceLast = appearanceScore
            firstLockOutlierHoldStreak = 0
            val reason = when {
                resetNoSeed -> "no_seed"
                resetGap -> "gap"
                resetStableDrift -> "drift"
                resetCenterRule -> "center_fail"
                resetIou -> "iou_fail"
                resetConf -> "conf_fail"
                else -> "reset"
            }
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_STABLE state=seed reason=$reason " +
                        "good=${candidate.goodMatches} inliers=${candidate.inlierCount} conf=${fmt(candidate.confidence)} " +
                        "iou=${fmt(iouWithPrevious)} minIou=${fmt(adaptiveIou)} minIouBase=${fmt(adaptiveIouBase)} iouFloor=${fmt(iouFloor)} center=${fmt(centerDrift)} " +
                        "centerTh=${fmt(effectiveCenterThreshold)} stableTh=${fmt(effectiveStablePx)} confTh=${fmt(confThreshold)} score=${fmt(candidateScore)} app=${fmt(appearanceScore)}"
                )
            }
            return null
        }

        firstLockCandidateFrames++
        metricsSearchStableAccumCount++
        firstLockCandidateLastMs = now
        if (candidateScore >= firstLockCandidateBestScore) {
            firstLockCandidateBox = candidate.box
            firstLockCandidateCenter = candidateCenter
            firstLockCandidateBestScore = candidateScore
        } else {
            firstLockCandidateCenter.x = (firstLockCandidateCenter.x + candidateCenter.x) * 0.5
            firstLockCandidateCenter.y = (firstLockCandidateCenter.y + candidateCenter.y) * 0.5
        }
        firstLockCandidateBestGood = max(firstLockCandidateBestGood, candidate.goodMatches)
        firstLockCandidateBestInliers = max(firstLockCandidateBestInliers, candidate.inlierCount)
        firstLockCandidateAppearanceSum += appearanceScore
        firstLockCandidateAppearanceMin = min(firstLockCandidateAppearanceMin, appearanceScore)
        firstLockCandidateAppearanceLast = appearanceScore

        val stableMs = now - firstLockCandidateStartMs
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=SEARCH_STABLE state=accum frames=$firstLockCandidateFrames stableMs=$stableMs " +
                    "bestGood=$firstLockCandidateBestGood bestInliers=$firstLockCandidateBestInliers score=${fmt(firstLockCandidateBestScore)} " +
                    "appAvg=${fmt(firstLockCandidateAppearanceSum / max(firstLockCandidateFrames, 1))} " +
                    "appMin=${fmt(firstLockCandidateAppearanceMin)} appLast=${fmt(firstLockCandidateAppearanceLast)} " +
                    "iou=${fmt(iouWithPrevious)} minIou=${fmt(adaptiveIou)} center=${fmt(centerDrift)} " +
                    "centerTh=${fmt(effectiveCenterDriftPx)} stableTh=${fmt(effectiveStablePx)}"
            )
        }

        val requiredStableFrames = if (!isReplayInput) max(2, firstLockCfg.stableFrames - 1) else firstLockCfg.stableFrames
        val requiredStableMs = if (!isReplayInput) max(160L, firstLockCfg.stableMs - 80L) else firstLockCfg.stableMs
        val ready = firstLockCandidateFrames >= requiredStableFrames && stableMs >= requiredStableMs
        if (!ready) return null

        val appearanceAvg = firstLockCandidateAppearanceSum / max(firstLockCandidateFrames, 1)
        if (!passesFirstLockAppearanceGate(
                frames = firstLockCandidateFrames,
                avgScore = appearanceAvg,
                minScore = firstLockCandidateAppearanceMin,
                bestGood = firstLockCandidateBestGood,
                bestInliers = firstLockCandidateBestInliers
            )) {
            metricsSearchPromoteRejectCount++
            metricsSearchLastReason = "promoted_appearance_reject"
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=SEARCH_STABLE state=final_reject reason=appearance " +
                        "frames=$firstLockCandidateFrames appAvg=${fmt(appearanceAvg)} appMin=${fmt(firstLockCandidateAppearanceMin)} " +
                        "bestGood=$firstLockCandidateBestGood bestInliers=$firstLockCandidateBestInliers"
                )
            }
            return null
        }

        metricsSearchPromoteCount++
        val promotedBox = firstLockCandidateBox ?: return null
        return OrbMatchCandidate(
            box = promotedBox,
            goodMatches = firstLockCandidateBestGood,
            inlierCount = firstLockCandidateBestInliers,
            confidence = candidate.confidence,
            usedHomography = candidate.usedHomography,
            searchScale = candidate.searchScale,
            fallbackReason = (candidate.fallbackReason ?: "soft_stable"),
            matcherType = candidate.matcherType,
            isStrong = true
        )
    }

    private fun computeFirstLockCandidateScore(
        candidate: OrbMatchCandidate,
        iouWithPrevious: Double,
        adaptiveIou: Double,
        centerDrift: Double,
        stableDriftPx: Double
    ): Double {
        val orbCfg = heuristicConfig.orb
        val iouScore = (iouWithPrevious / max(adaptiveIou, 1e-6)).coerceIn(0.0, 1.2)
        val centerScore = (1.0 - centerDrift / max(stableDriftPx, 1e-6)).coerceIn(0.0, 1.0)
        val matchScore = (candidate.goodMatches.toDouble() / max(orbCfg.minGoodMatches, 1)).coerceIn(0.0, 2.0)
        val confScore = candidate.confidence.coerceIn(0.0, 1.0)
        return iouScore * 0.45 + centerScore * 0.35 + matchScore * 0.10 + confScore * 0.10
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
        firstLockCandidateBestScore = 0.0
        firstLockCandidateAppearanceSum = 0.0
        firstLockCandidateAppearanceMin = 1.0
        firstLockCandidateAppearanceLast = 0.0
        firstLockOutlierHoldStreak = 0
    }

    private fun passesFirstLockAppearanceGate(
        frames: Int,
        avgScore: Double,
        minScore: Double,
        bestGood: Int,
        bestInliers: Int
    ): Boolean {
        if (frames <= 0) return false
        val strong = bestGood >= 24 && bestInliers >= 8
        val minAvg = if (strong) -0.10 else -0.06
        val minSingle = if (strong) -0.42 else -0.32
        return avgScore >= minAvg && minScore >= minSingle
    }

    private fun computeDynamicFirstLockCenterThreshold(
        previous: Rect?,
        candidate: Rect,
        useCenterRule: Boolean,
        baseCenterDriftPx: Double,
        stableDriftPx: Double
    ): Double {
        val firstLockCfg = heuristicConfig.firstLock
        if (!useCenterRule) return baseCenterDriftPx
        val prevSide = previous?.let { min(it.width, it.height).toDouble() } ?: 0.0
        val currSide = min(candidate.width, candidate.height).toDouble()
        val refSide = max(prevSide, currSide).coerceAtLeast(1.0)
        val dynamicByBox = refSide * firstLockCfg.smallDynamicCenterFactor
        return max(baseCenterDriftPx, dynamicByBox).coerceAtMost(stableDriftPx)
    }

    private fun buildFirstLockSeedReason(
        resetNoSeed: Boolean,
        resetGap: Boolean,
        resetStableDrift: Boolean,
        resetCenterRule: Boolean,
        resetIou: Boolean
    ): String {
        val reasons = ArrayList<String>(5)
        if (resetNoSeed) reasons += "init"
        if (resetGap) reasons += "gap"
        if (resetStableDrift) reasons += "drift"
        if (resetCenterRule) reasons += "center"
        if (resetIou) reasons += "iou"
        return if (reasons.isEmpty()) "none" else reasons.joinToString("+")
    }

    private fun isClearlyBetterFirstLockCandidate(candidate: OrbMatchCandidate): Boolean {
        if (candidate.usedHomography) return true
        if (candidate.goodMatches >= firstLockCandidateBestGood + FIRST_LOCK_OUTLIER_GOOD_BOOST) return true
        if (candidate.inlierCount >= firstLockCandidateBestInliers + FIRST_LOCK_OUTLIER_INLIER_BOOST) return true
        return false
    }

    private fun expireFirstLockCandidateIfNeeded() {
        val firstLockCfg = heuristicConfig.firstLock
        if (firstLockCandidateFrames <= 0) return
        val now = SystemClock.elapsedRealtime()
        if (now - firstLockCandidateLastMs > firstLockCfg.gapMs) {
            clearFirstLockCandidate("timeout")
        }
    }

    private fun findOrbMatch(
        frame: Mat,
        loweRatioOverride: Double? = null,
        allowSeedAssist: Boolean = true
    ): OrbMatchCandidate? {
        val firstLockCfg = heuristicConfig.firstLock
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
            val effectiveSearchLongEdge =
                if (templateLibrarySize >= 2) {
                    min(searchMaxLongEdge, searchMultiTemplateMaxLongEdge)
                } else {
                    searchMaxLongEdge
                }
            val searchScale = computeSearchScale(
                frameWidth = frameGray.cols(),
                frameHeight = frameGray.rows(),
                shortEdgeTarget = effectiveSearchShortEdge,
                maxLongEdge = effectiveSearchLongEdge
            )
            searchGray = if (searchScale < 0.999) Mat() else frameGray
            if (searchScale < 0.999) {
                Imgproc.resize(frameGray, searchGray, Size(), searchScale, searchScale, Imgproc.INTER_AREA)
            }
            val searchW = searchGray.cols()
            val searchH = searchGray.rows()

            orbInput = if (orbUseClahe) Mat() else searchGray
            if (orbUseClahe) {
                clahe.apply(searchGray, orbInput)
            }

            val effectiveMaxFeatures = if (searchMissStreak >= searchHighResMissStreak) {
                max(orbMaxFeatures, orbFarBoostFeatures)
            } else {
                orbMaxFeatures
            }
            val finalMaxFeatures =
                if (templateLibrarySize >= 2) min(effectiveMaxFeatures, orbFarBoostMultiTemplateCap) else effectiveMaxFeatures
            configureOrbDetector(finalMaxFeatures, enforceBudgetCap = true)
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
                    detail =
                        "searchShort=$effectiveSearchShortEdge searchLongCap=$effectiveSearchLongEdge " +
                            "searchSize=${searchW}x${searchH} orbFeatures=$finalMaxFeatures"
                )
                return null
            }
            if (frameDescriptors.rows() < 2) return null

            frameDescriptors.convertTo(frameDescriptors32F, CvType.CV_32F)
            val framePoints = frameKeypoints.toArray()
            val searchMat = requireNotNull(searchGray)
            val anchorPoint = if (firstLockCandidateFrames > 0) {
                Point(firstLockCandidateCenter.x, firstLockCandidateCenter.y)
            } else {
                null
            }
            val anchorRadius =
                if (anchorPoint != null) {
                    max(firstLockCfg.stablePx, firstLockCfg.smallStablePx) * FIRST_LOCK_ANCHOR_NEAR_FACTOR
                } else {
                    0.0
                }
            if (allowSeedAssist && firstLockCandidateFrames > 0) {
                val seedBox = firstLockCandidateBox
                if (seedBox != null) {
                    val seedRoi = expandRectWithFactor(
                        seedBox,
                        FIRST_LOCK_SEED_REEVAL_EXPAND_FACTOR,
                        frame.cols(),
                        frame.rows()
                    )
                    if (seedRoi != null) {
                        val localCandidate = findOrbMatchInRoi(frame, seedRoi, loweRatioOverride = loweRatioOverride)
                        if (localCandidate != null && anchorPoint != null) {
                            val localDist = pointDistance(anchorPoint, rectCenter(localCandidate.box))
                            val localAcceptRadius = anchorRadius * FIRST_LOCK_SEED_REEVAL_ACCEPT_FACTOR
                            if (localDist <= localAcceptRadius) {
                                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                                    Log.w(
                                        TAG,
                                        "EVAL_EVENT type=SEARCH_SEED_LOCAL state=use " +
                                            "dist=${fmt(localDist)} radius=${fmt(localAcceptRadius)} " +
                                            "good=${localCandidate.goodMatches} inliers=${localCandidate.inlierCount}"
                                    )
                                }
                                return localCandidate.copy(matcherType = "${localCandidate.matcherType}_seed")
                            }
                        }
                    }
                }
            }

            var bestCandidate: OrbMatchCandidate? = null
            var bestNearCandidate: OrbMatchCandidate? = null
            var bestDiag: SearchLevelResult? = null
            for (level in templatePyramidLevels) {
                val levelResult = findOrbMatchForLevel(
                    frame = frame,
                    searchGray = searchMat,
                    searchScale = searchScale,
                    level = level,
                    framePoints = framePoints,
                    frameDescriptors8U = frameDescriptors,
                    frameDescriptors32F = frameDescriptors32F,
                    loweRatio = loweRatioOverride ?: heuristicConfig.orb.loweRatio
                )
                val candidate = levelResult.candidate
                if (candidate != null && isBetterOrbCandidate(candidate, bestCandidate)) {
                    bestCandidate = candidate
                }
                if (candidate != null && anchorPoint != null) {
                    val distToAnchor = pointDistance(anchorPoint, rectCenter(candidate.box))
                    if (distToAnchor <= anchorRadius && isBetterOrbCandidate(candidate, bestNearCandidate)) {
                        bestNearCandidate = candidate
                    }
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

            if (bestNearCandidate != null) return bestNearCandidate
            if (bestCandidate != null) return bestCandidate

            val diag = bestDiag
            lastSearchDiagReason = diag?.reason ?: "search_no_candidate"
            logSearchDiag(
                reason = lastSearchDiagReason,
                kpFrame = framePoints.size,
                flannGood = diag?.flannGood ?: 0,
                bfGood = diag?.bfGood ?: 0,
                selectedGood = diag?.selectedGood ?: 0,
                selectedInliers = diag?.selectedInliers ?: 0,
                matcher = diag?.matcherType ?: "none",
                detail = (diag?.detail ?: "none") +
                    " searchShort=$effectiveSearchShortEdge searchLongCap=$effectiveSearchLongEdge " +
                    "searchSize=${searchW}x${searchH} orbFeatures=$finalMaxFeatures"
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
        frameDescriptors32F: Mat,
        loweRatio: Double
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
            val cfg = heuristicConfig
            var flannGoodMatches: List<DMatch> = emptyList()
            var bfGoodMatches: List<DMatch> = emptyList()
            var flannPairs: List<KnnPair> = emptyList()
            var bfPairs: List<KnnPair> = emptyList()
            var flannError: String? = null

            if (level.descriptors32F.rows() >= 2 && frameDescriptors32F.rows() >= 2) {
                runCatching {
                    flannMatcher.knnMatch(level.descriptors32F, frameDescriptors32F, knnMatches, 2)
                    flannPairs = extractKnnPairs(knnMatches)
                    flannGoodMatches = collectGoodMatchesFromPairs(flannPairs, loweRatio)
                }.onFailure {
                    flannError = it.javaClass.simpleName ?: "flann_exception"
                }
            }
            clearKnnMatches(knnMatches)

            runCatching {
                bfMatcher.knnMatch(level.descriptors8U, frameDescriptors8U, knnMatches, 2)
                bfPairs = extractKnnPairs(knnMatches)
                bfGoodMatches = collectGoodMatchesFromPairs(bfPairs, loweRatio)
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
            var selectedPairs = flannPairs
            if (bfGoodMatches.size > bestMatches.size) {
                bestMatches = bfGoodMatches
                matcherType = "bf_hamming_fallback"
                selectedPairs = bfPairs
            } else if (bestMatches.isEmpty() && bfGoodMatches.isNotEmpty()) {
                bestMatches = bfGoodMatches
                matcherType = "bf_hamming"
                selectedPairs = bfPairs
            }

            val baseDetail =
                "lvl=${fmt(level.scale)} tplKp=${level.keypointCount} tpl=${level.gray.cols()}x${level.gray.rows()}"
            val softThresholds = computeSoftGateThresholds(searchScale, loweRatio)
            if (bestMatches.size < softThresholds.minGoodMatches) {
                return SearchLevelResult(
                    candidate = null,
                    reason = "few_matches",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = 0,
                    matcherType = matcherType,
                    detail =
                        "$baseDetail needSoft=${softThresholds.minGoodMatches} baseSoft=${cfg.orb.softMinGoodMatches} " +
                            "softRelax=${softThresholds.relaxed} miss=$searchMissStreak " +
                            "ratio=${fmt(loweRatio)} flannErr=${flannError ?: "none"}"
                )
            }

            val srcList = ArrayList<Point>(bestMatches.size)
            val dstList = ArrayList<Point>(bestMatches.size)
            val templatePoints = level.keypoints.toArray()
            fillMatchPointPairs(bestMatches, templatePoints, framePoints, srcList, dstList)
            if (srcList.size < softThresholds.minGoodMatches || dstList.size < softThresholds.minGoodMatches) {
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
                    fallbackReason = "soft_pairs",
                    minGoodRequired = softThresholds.minGoodMatches
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
            if (inlierDstPoints.size < softThresholds.minInliers || inliers < softThresholds.minInliers) {
                val soft = buildSoftCandidate(
                    frame = frame,
                    points = dstList,
                    searchScale = searchScale,
                    goodMatches = bestMatches.size,
                    inliers = inliers,
                    matcherType = matcherType,
                    fallbackReason = "few_inliers",
                    minGoodRequired = softThresholds.minGoodMatches
                )
                return SearchLevelResult(
                    candidate = soft,
                    reason = "few_inliers",
                    flannGood = flannGoodMatches.size,
                    bfGood = bfGoodMatches.size,
                    selectedGood = bestMatches.size,
                    selectedInliers = inliers,
                    matcherType = matcherType,
                    detail =
                        "$baseDetail needSoftInliers=${softThresholds.minInliers} baseSoftInliers=${cfg.orb.softMinInliers} " +
                            "filtered=${inlierDstPoints.size}"
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
                var pointsForFallback = if (inlierDstPoints.isNotEmpty()) inlierDstPoints else dstList
                if (bestMatches.size <= cfg.weakFallback.maxMatches) {
                    pointsForFallback = selectDensestWeakPointSubset(pointsForFallback)
                    val weakStats = measureWeakFallbackStats(pointsForFallback, searchScale)
                    if (weakStats != null) {
                        val weakSpan = max(weakStats.spanX, weakStats.spanY)
                        val spreadRelax =
                            if (
                                searchMissStreak >= cfg.weakFallback.relaxMissStreak &&
                                bestMatches.size >= (cfg.orb.softMinGoodMatches + 1)
                            ) {
                                cfg.weakFallback.relaxFactor
                            } else {
                                1.0
                            }
                        val weakSpanLimit = cfg.weakFallback.maxSpanPx * spreadRelax
                        val weakAreaLimit = cfg.weakFallback.maxAreaPx * spreadRelax * spreadRelax
                        val weakReject = weakSpan > weakSpanLimit || weakStats.area > weakAreaLimit
                        if (weakReject) {
                            var rescued = false
                            var rescueDetail = "rescue=skip"
                            if (cfg.weakFallback.rescueEnabled && selectedPairs.isNotEmpty()) {
                                val tighterRatio = computeTighterRescueRatio(loweRatio)
                                val rescuedMatches = collectGoodMatchesFromPairs(selectedPairs, tighterRatio)
                                if (rescuedMatches.size >= cfg.weakFallback.rescueMinGood) {
                                    val rescueSrc = ArrayList<Point>(rescuedMatches.size)
                                    val rescueDst = ArrayList<Point>(rescuedMatches.size)
                                    fillMatchPointPairs(rescuedMatches, templatePoints, framePoints, rescueSrc, rescueDst)
                                    if (rescueDst.size >= cfg.weakFallback.rescueMinGood) {
                                        val rescueCompactPoints = selectDensestWeakPointSubset(rescueDst)
                                        val rescueStats = measureWeakFallbackStats(rescueCompactPoints, searchScale)
                                        if (rescueStats != null) {
                                            val rescueSpan = max(rescueStats.spanX, rescueStats.spanY)
                                            val rescueReject =
                                                rescueSpan > weakSpanLimit || rescueStats.area > weakAreaLimit
                                            if (!rescueReject) {
                                                pointsForFallback = rescueCompactPoints
                                                bestMatches = rescuedMatches
                                                selectedPairs = emptyList()
                                                matcherType = "${matcherType}_rescue"
                                                rescued = true
                                                rescueDetail =
                                                    "rescue=ok ratio=${fmt(tighterRatio)} good=${rescuedMatches.size} " +
                                                        "span=${fmt(rescueSpan)} area=${fmt(rescueStats.area)}"
                                            } else {
                                                rescueDetail =
                                                    "rescue=reject ratio=${fmt(tighterRatio)} good=${rescuedMatches.size} " +
                                                        "span=${fmt(rescueSpan)} area=${fmt(rescueStats.area)}"
                                            }
                                        } else {
                                            rescueDetail = "rescue=stats_null ratio=${fmt(tighterRatio)}"
                                        }
                                    } else {
                                        rescueDetail = "rescue=pair_small ratio=${fmt(tighterRatio)} dst=${rescueDst.size}"
                                    }
                                } else {
                                    rescueDetail = "rescue=good_small ratio=${fmt(tighterRatio)} good=${rescuedMatches.size}"
                                }
                            }

                            if (!rescued && cfg.weakFallback.coreRescueEnabled) {
                                val corePoints = selectTightCorePoints(pointsForFallback)
                                val coreStats = measureWeakFallbackStats(corePoints, searchScale)
                                if (coreStats != null) {
                                    val coreSpan = max(coreStats.spanX, coreStats.spanY)
                                    val coreReject =
                                        coreSpan > cfg.weakFallback.coreMaxSpanPx || coreStats.area > cfg.weakFallback.coreMaxAreaPx
                                    if (!coreReject) {
                                        pointsForFallback = corePoints
                                        matcherType = "${matcherType}_core"
                                        rescued = true
                                        rescueDetail =
                                            "rescue=core_ok span=${fmt(coreSpan)} area=${fmt(coreStats.area)}"
                                    } else {
                                        rescueDetail =
                                            "rescue=core_reject span=${fmt(coreSpan)} area=${fmt(coreStats.area)}"
                                    }
                                }
                            }

                            if (!rescued) {
                                return SearchLevelResult(
                                    candidate = null,
                                    reason = "weak_fallback_spread",
                                    flannGood = flannGoodMatches.size,
                                    bfGood = bfGoodMatches.size,
                                    selectedGood = bestMatches.size,
                                    selectedInliers = inliers,
                                    matcherType = matcherType,
                                    detail =
                                        "$baseDetail span=${fmt(weakSpan)} area=${fmt(weakStats.area)} " +
                                            "maxSpan=${fmt(weakSpanLimit)} maxArea=${fmt(weakAreaLimit)} " +
                                            rescueDetail
                                )
                            } else if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                                Log.w(
                                    TAG,
                                    "EVAL_EVENT type=SEARCH_RESCUE state=pass matcher=$matcherType good=${bestMatches.size}"
                                )
                            }
                        }
                    }
                }
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
            val strongThresholds = computeSearchThresholds(finalBox, frame.cols(), frame.rows(), searchScale)
            val isStrong =
                bestMatches.size >= strongThresholds.minGoodMatches &&
                    inliers >= strongThresholds.minInliers &&
                    inlierDstPoints.size >= strongThresholds.minInliers

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
                detail =
                    "$baseDetail fallback=${fallbackReason ?: "none"} usedH=$usedHomography " +
                        "strongGood=${strongThresholds.minGoodMatches} strongInliers=${strongThresholds.minInliers} " +
                        "ratio=${fmt(loweRatio)}"
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

    private fun isCandidateLockableForInit(candidate: OrbMatchCandidate, frameW: Int, frameH: Int): Boolean {
        val cfg = heuristicConfig
        if (!candidate.isStrong) return false
        if (candidate.usedHomography) return true
        if (!cfg.firstLock.allowFallbackLock) return false

        val isSmallTarget =
            isSmallTargetCandidate(
                box = candidate.box,
                frameW = frameW,
                frameH = frameH,
                searchScale = candidate.searchScale
            )
        val minGoodFallback: Int
        val minInliersFallback: Int
        val minConfidence: Double
        if (isSmallTarget) {
            minGoodFallback = max(cfg.smallTarget.minGoodMatches, cfg.orb.softMinGoodMatches)
            minInliersFallback = max(cfg.smallTarget.minInliers, cfg.orb.softMinInliers)
            minConfidence =
                if (
                    candidate.goodMatches >= FALLBACK_LOCK_HIGH_GOOD_MATCHES &&
                    candidate.inlierCount >= FALLBACK_LOCK_HIGH_INLIERS
                ) {
                    FALLBACK_LOCK_MIN_CONFIDENCE_HIGH_GOOD
                } else {
                    FALLBACK_LOCK_MIN_CONFIDENCE_SMALL
                }
        } else {
            minGoodFallback =
                (cfg.orb.minGoodMatches - FALLBACK_LOCK_GOOD_RELAX).coerceAtLeast(cfg.orb.softMinGoodMatches + 2)
            minInliersFallback =
                (cfg.orb.minInliers - FALLBACK_LOCK_INLIER_RELAX).coerceAtLeast(cfg.orb.softMinInliers + 1)
            minConfidence = FALLBACK_LOCK_MIN_CONFIDENCE
        }
        return candidate.goodMatches >= minGoodFallback &&
            candidate.inlierCount >= minInliersFallback &&
            candidate.confidence >= minConfidence
    }

    private fun canDirectAutoInit(candidate: OrbMatchCandidate): Boolean {
        // P0: disable direct auto-init; force temporal-confirm path.
        return false
    }

    private fun passesAutoInitVerification(frame: Mat, candidate: OrbMatchCandidate): Boolean {
        val fallbackRefineCfg = heuristicConfig.fallbackRefine
        val verifyCfg = heuristicConfig.autoInitVerify
        val gatePrefix = "session=$diagSessionId stage=auto_verify good=${candidate.goodMatches} inliers=${candidate.inlierCount} conf=${fmt(candidate.confidence)}"
        val appearanceScore = computePatchTemplateCorrelation(frame, candidate.box)
        val strongCandidate =
            candidate.goodMatches >= verifyCfg.strongCandidateMinGood &&
                candidate.inlierCount >= verifyCfg.strongCandidateMinInliers &&
                candidate.confidence >= verifyCfg.strongCandidateMinConfidence
        val strongAppearance = if (!isReplayInput) {
            appearanceScore >= verifyCfg.strongAppearanceMinLive
        } else {
            appearanceScore >= verifyCfg.strongAppearanceMinReplay
        }

        val roi = expandRectWithFactor(candidate.box, verifyCfg.localRoiExpandFactor, frame.cols(), frame.rows())
        val local = roi?.let { findOrbMatchInRoi(frame, it, loweRatioOverride = fallbackRefineCfg.loweRatio) }
        if (local == null) {
            val fallbackPass = !isReplayInput && strongCandidate && strongAppearance
            logDiag(
                "LOCK_GATE",
                "$gatePrefix pass=$fallbackPass reason=local_missing score=${fmt(appearanceScore)}"
            )
            if (!fallbackPass) return false
        } else {
            val iou = rectIou(candidate.box, local.box)
            val centerDist = pointDistance(rectCenter(candidate.box), rectCenter(local.box))
            val side = min(candidate.box.width, candidate.box.height).toDouble().coerceAtLeast(1.0)
            val centerFactor = if (!isReplayInput) verifyCfg.localCenterFactorLive else verifyCfg.localCenterFactorReplay
            val minIou = if (!isReplayInput) verifyCfg.localMinIouLive else verifyCfg.localMinIouReplay
            val minLocalConf = if (!isReplayInput) {
                max(candidate.confidence * verifyCfg.localMinConfScaleLive, verifyCfg.localMinConfFloorLive)
            } else {
                max(candidate.confidence * verifyCfg.localMinConfScaleReplay, verifyCfg.localMinConfFloorReplay)
            }
            val centerOk = centerDist <= side * centerFactor
            val iouOk = iou >= minIou
            val confOk = local.confidence >= minLocalConf
            val geomPass = centerOk && iouOk && confOk
            val geomOverride = !isReplayInput && strongCandidate && strongAppearance
            if (!geomPass && !geomOverride) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=geom iou=${fmt(iou)} center=${fmt(centerDist)} centerTh=${fmt(side * centerFactor)} localConf=${fmt(local.confidence)}"
                )
                return false
            }
            if (!geomPass && geomOverride) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=true reason=geom_override iou=${fmt(iou)} center=${fmt(centerDist)} localConf=${fmt(local.confidence)} score=${fmt(appearanceScore)}"
                )
            }
        }

        var minAppearanceScore = when {
            candidate.goodMatches >= verifyCfg.appearanceMinStrongGood &&
                candidate.inlierCount >= verifyCfg.appearanceMinStrongInliers -> verifyCfg.appearanceMinStrong
            candidate.goodMatches >= verifyCfg.appearanceMinMediumGood &&
                candidate.inlierCount >= verifyCfg.appearanceMinMediumInliers -> verifyCfg.appearanceMinMedium
            else -> verifyCfg.appearanceMinBase
        }
        if (!isReplayInput) {
            minAppearanceScore += verifyCfg.appearanceLiveBias
        }
        val appearanceOk = appearanceScore >= minAppearanceScore
        if (!appearanceOk && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=VERIFY_GATE state=reject reason=appearance " +
                    "score=${fmt(appearanceScore)}/${fmt(minAppearanceScore)} " +
                    "candGood=${candidate.goodMatches} candInliers=${candidate.inlierCount}"
            )
        }
        logDiag(
            "LOCK_GATE",
            "$gatePrefix pass=$appearanceOk reason=appearance score=${fmt(appearanceScore)} min=${fmt(minAppearanceScore)}"
        )
        return appearanceOk
    }

    private fun shouldAutoInitWithoutManual(candidate: OrbMatchCandidate, frameW: Int, frameH: Int): Boolean {
        if (!isCandidateLockableForInit(candidate, frameW, frameH)) return false
        val inFastWindow = !isTracking && fastFirstLockRemaining > 0
        val minGood = if (inFastWindow) FAST_FIRST_LOCK_MIN_GOOD_MATCHES else AUTO_INIT_MIN_GOOD_MATCHES
        val minInliers = if (inFastWindow) FAST_FIRST_LOCK_MIN_INLIERS else AUTO_INIT_MIN_INLIERS
        val minConf = if (inFastWindow) FAST_FIRST_LOCK_MIN_CONFIDENCE else AUTO_INIT_MIN_CONFIDENCE
        if (candidate.goodMatches < minGood) return false
        if (candidate.inlierCount < minInliers) return false
        return candidate.confidence >= minConf
    }

    private fun isCandidateEligibleForTemporal(candidate: OrbMatchCandidate): Boolean {
        val cfg = heuristicConfig
        val temporalCfg = cfg.temporalGate
        var temporalMinGoodMatches = computeTemporalMinGoodMatches(candidate)
        if (!isReplayInput) {
            temporalMinGoodMatches = (temporalMinGoodMatches - 1).coerceAtLeast(max(cfg.orb.softMinGoodMatches, 3))
        }
        if (candidate.goodMatches < temporalMinGoodMatches) return false

        val minInliersBase = if (candidate.usedHomography) {
            cfg.orb.softMinInliers
        } else {
            max(cfg.orb.softMinInliers, cfg.fallbackRefine.minInliers - 1)
        }
        val minInliers = if (!isReplayInput && !candidate.usedHomography) {
            (minInliersBase - 1).coerceAtLeast(cfg.orb.softMinInliers)
        } else {
            minInliersBase
        }
        if (candidate.inlierCount < minInliers) return false

        if (!candidate.usedHomography) {
            val isSmallCandidate =
                min(candidate.box.width, candidate.box.height).toDouble() <= FIRST_LOCK_SMALL_BOX_SIDE_PX ||
                    candidate.box.width.toDouble() * candidate.box.height.toDouble() <= FIRST_LOCK_SMALL_BOX_AREA_PX
            val isRefinedFallback = candidate.fallbackReason?.startsWith("refined_") == true
            var minTemporalConfidence = when {
                candidate.goodMatches >= temporalCfg.highGoodMatches &&
                    candidate.inlierCount >= temporalCfg.highInliers -> temporalCfg.minConfidenceHighGood
                candidate.goodMatches >= temporalCfg.mediumGoodMatches &&
                    candidate.inlierCount >= cfg.orb.softMinInliers -> temporalCfg.minConfidenceMedium
                isSmallCandidate && isRefinedFallback -> min(temporalCfg.minConfidenceSmallRefined, cfg.fallbackRefine.minConfidence)
                else -> min(temporalCfg.minConfidenceBase, cfg.fallbackRefine.minConfidence)
            }
            if (!isReplayInput) {
                minTemporalConfidence = (minTemporalConfidence - temporalCfg.liveConfidenceRelax)
                    .coerceAtLeast(temporalCfg.liveConfidenceFloor)
            }
            if (candidate.confidence < minTemporalConfidence) return false
        }
        return true
    }

    private fun maybeRefineFallbackCandidate(frame: Mat, candidate: OrbMatchCandidate): OrbMatchCandidate {
        val cfg = heuristicConfig
        if (candidate.usedHomography) return candidate
        val isWeakFallbackCandidate = candidate.goodMatches <= cfg.weakFallback.maxMatches

        val refineRegion =
            expandRectWithFactor(candidate.box, cfg.fallbackRefine.expandFactor, frame.cols(), frame.rows())
                ?: candidate.box
        val localCandidate = findOrbMatchInRoi(frame, refineRegion, loweRatioOverride = cfg.fallbackRefine.loweRatio)
        if (localCandidate == null) {
            if (isWeakFallbackCandidate && cfg.weakFallback.requireRefine) {
                metricsSearchRefineRejectCount++
                metricsSearchLastReason = "refine_miss"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_REFINE state=reject reason=weak_refine_miss " +
                            "good=${candidate.goodMatches} inliers=${candidate.inlierCount} " +
                            "conf=${fmt(candidate.confidence)} box=${candidate.box.width}x${candidate.box.height}"
                    )
                }
                return candidate.copy(
                    inlierCount = 0,
                    confidence = 0.0,
                    isStrong = false,
                    fallbackReason = "weak_refine_miss"
                )
            }
            return candidate
        }

        val merged = if (isBetterOrbCandidate(localCandidate, candidate)) localCandidate else candidate
        val refineMinConfidence =
            if (
                localCandidate.goodMatches >= REFINE_HIGH_GOOD_MATCHES &&
                localCandidate.inlierCount >= REFINE_HIGH_INLIERS
            ) {
                FALLBACK_REFINE_MIN_CONFIDENCE_HIGH_GOOD
            } else {
                cfg.fallbackRefine.minConfidence
            }
        val passesRefine =
            localCandidate.goodMatches >= cfg.fallbackRefine.minGoodMatches &&
                localCandidate.inlierCount >= cfg.fallbackRefine.minInliers &&
                localCandidate.confidence >= refineMinConfidence

        if (!passesRefine) {
            if (isWeakFallbackCandidate && cfg.weakFallback.requireRefine) {
                metricsSearchRefineRejectCount++
                metricsSearchLastReason = "refine_fail"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_REFINE state=reject reason=weak_refine_fail " +
                            "good=${localCandidate.goodMatches} inliers=${localCandidate.inlierCount} " +
                            "conf=${fmt(localCandidate.confidence)} minConf=${fmt(refineMinConfidence)}"
                    )
                }
                return merged.copy(
                    inlierCount = 0,
                    confidence = 0.0,
                    isStrong = false,
                    fallbackReason = "weak_refine_fail"
                )
            }
            return merged
        }

        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=SEARCH_REFINE state=pass " +
                    "good=${localCandidate.goodMatches} inliers=${localCandidate.inlierCount} " +
                    "conf=${fmt(localCandidate.confidence)} h=${localCandidate.usedHomography}"
            )
        }
        metricsSearchRefinePassCount++
        metricsSearchLastReason = "refine_pass"
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
        val firstLockCfg = heuristicConfig.firstLock
        if (frameW <= 0 || frameH <= 0) return firstLockCfg.minIou
        val frameArea = frameW.toDouble() * frameH.toDouble()
        val boxArea = box.width.toDouble() * box.height.toDouble()
        val areaRatio = (boxArea / frameArea).coerceAtLeast(1e-5)
        val normalized = sqrt((areaRatio / FIRST_LOCK_BASE_AREA_RATIO).coerceAtMost(1.0))
        val upper = firstLockCfg.minIou.coerceIn(0.0, 1.0)
        val lower = min(FIRST_LOCK_MIN_IOU_FLOOR, upper)
        return (upper * normalized).coerceIn(lower, upper)
    }

    private fun computeSearchScale(
        frameWidth: Int,
        frameHeight: Int,
        shortEdgeTarget: Int = searchShortEdge,
        maxLongEdge: Int = searchMaxLongEdge
    ): Double {
        val shortEdge = min(frameWidth, frameHeight).toDouble()
        val longEdge = max(frameWidth, frameHeight).toDouble()
        val shortScale =
            if (shortEdge > shortEdgeTarget.toDouble()) shortEdgeTarget.toDouble() / shortEdge else 1.0
        val longScale =
            if (maxLongEdge > 0 && longEdge > maxLongEdge.toDouble()) maxLongEdge.toDouble() / longEdge else 1.0
        return min(1.0, min(shortScale, longScale)).coerceIn(0.10, 1.0)
    }

    private fun computeTighterRescueRatio(baseRatio: Double): Double {
        val weakCfg = heuristicConfig.weakFallback
        val maxRatio = (baseRatio - 0.02).coerceAtLeast(0.55)
        return min(weakCfg.rescueRatio, maxRatio).coerceIn(0.55, 0.90)
    }

    private fun measureWeakFallbackStats(points: List<Point>, searchScale: Double): WeakFallbackStats? {
        if (points.isEmpty()) return null
        var minX = Double.POSITIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY
        var maxX = Double.NEGATIVE_INFINITY
        var maxY = Double.NEGATIVE_INFINITY
        for (p in points) {
            minX = min(minX, p.x)
            minY = min(minY, p.y)
            maxX = max(maxX, p.x)
            maxY = max(maxY, p.y)
        }
        if (!minX.isFinite() || !minY.isFinite() || !maxX.isFinite() || !maxY.isFinite()) return null
        val inv = 1.0 / searchScale.coerceAtLeast(1e-6)
        val spanX = ((maxX - minX).coerceAtLeast(0.0)) * inv
        val spanY = ((maxY - minY).coerceAtLeast(0.0)) * inv
        return WeakFallbackStats(spanX, spanY, spanX * spanY)
    }

    private fun selectDensestWeakPointSubset(points: List<Point>): List<Point> {
        if (points.size <= 3) return points
        val targetSize = max(3, points.size - 1)
        val indexBuf = ArrayList<Int>(targetSize)
        var bestIndices: List<Int>? = null
        var bestArea = Double.POSITIVE_INFINITY

        fun evalCurrent() {
            var minX = Double.POSITIVE_INFINITY
            var minY = Double.POSITIVE_INFINITY
            var maxX = Double.NEGATIVE_INFINITY
            var maxY = Double.NEGATIVE_INFINITY
            for (idx in indexBuf) {
                val p = points[idx]
                minX = min(minX, p.x)
                minY = min(minY, p.y)
                maxX = max(maxX, p.x)
                maxY = max(maxY, p.y)
            }
            val area = ((maxX - minX).coerceAtLeast(0.0)) * ((maxY - minY).coerceAtLeast(0.0))
            if (area < bestArea) {
                bestArea = area
                bestIndices = ArrayList(indexBuf)
            }
        }

        fun dfs(start: Int) {
            if (indexBuf.size == targetSize) {
                evalCurrent()
                return
            }
            val need = targetSize - indexBuf.size
            val maxStart = points.size - need
            for (i in start..maxStart) {
                indexBuf += i
                dfs(i + 1)
                indexBuf.removeAt(indexBuf.lastIndex)
            }
        }

        dfs(0)
        val chosen = bestIndices ?: return points
        if (chosen.size >= points.size) return points
        return chosen.map { points[it] }
    }

    private fun selectTightCorePoints(points: List<Point>): List<Point> {
        if (points.size <= 2) return points
        var bestI = 0
        var bestJ = 1
        var bestDist2 = Double.POSITIVE_INFINITY
        for (i in 0 until points.size - 1) {
            val pi = points[i]
            for (j in i + 1 until points.size) {
                val pj = points[j]
                val dx = pi.x - pj.x
                val dy = pi.y - pj.y
                val dist2 = dx * dx + dy * dy
                if (dist2 < bestDist2) {
                    bestDist2 = dist2
                    bestI = i
                    bestJ = j
                }
            }
        }
        return listOf(points[bestI], points[bestJ])
    }

    private fun computeSearchThresholds(box: Rect, frameW: Int, frameH: Int, searchScale: Double): MatchThresholds {
        val cfg = heuristicConfig
        val isSmallTarget = isSmallTargetCandidate(box, frameW, frameH, searchScale)
        if (!isSmallTarget) {
            return MatchThresholds(cfg.orb.minGoodMatches, cfg.orb.minInliers)
        }

        val relaxedGood = min(cfg.orb.minGoodMatches, cfg.smallTarget.minGoodMatches)
            .coerceAtLeast(cfg.orb.softMinGoodMatches)
        val relaxedInliers = min(cfg.orb.minInliers, cfg.smallTarget.minInliers)
            .coerceAtLeast(cfg.orb.softMinInliers)
        return MatchThresholds(relaxedGood, relaxedInliers)
    }

    private fun isSmallTargetCandidate(box: Rect, frameW: Int, frameH: Int, searchScale: Double): Boolean {
        val cfg = heuristicConfig
        val frameArea = (frameW.toDouble() * frameH.toDouble()).coerceAtLeast(1.0)
        val boxArea = (box.width.toDouble() * box.height.toDouble()).coerceAtLeast(1.0)
        val areaRatio = boxArea / frameArea
        return areaRatio <= cfg.smallTarget.areaRatio ||
            searchScale <= cfg.smallTarget.scaleThreshold ||
            min(box.width, box.height).toDouble() <= FIRST_LOCK_SMALL_BOX_SIDE_PX
    }

    private fun clearKnnMatches(knnMatches: MutableList<MatOfDMatch>) {
        for (pair in knnMatches) {
            pair.release()
        }
        knnMatches.clear()
    }

    private fun extractKnnPairs(knnMatches: List<MatOfDMatch>): List<KnnPair> {
        val out = ArrayList<KnnPair>(knnMatches.size)
        for (pair in knnMatches) {
            val m = pair.toArray()
            if (m.size < 2) continue
            out += KnnPair(first = m[0], second = m[1])
        }
        return out
    }

    private fun buildSoftCandidate(
        frame: Mat,
        points: List<Point>,
        searchScale: Double,
        goodMatches: Int,
        inliers: Int,
        matcherType: String,
        fallbackReason: String,
        minGoodRequired: Int
    ): OrbMatchCandidate? {
        if (points.size < minGoodRequired) return null
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

    private fun computeSoftGateThresholds(searchScale: Double, loweRatio: Double): SoftGateThresholds {
        val cfg = heuristicConfig
        val baseGood = cfg.orb.softMinGoodMatches
        val baseInliers = cfg.orb.softMinInliers
        if (!cfg.softRelax.enabled) return SoftGateThresholds(baseGood, baseInliers, relaxed = false)

        val allowRelax =
            searchMissStreak >= cfg.softRelax.missStreak &&
                searchScale <= cfg.softRelax.scaleThreshold &&
                loweRatio <= cfg.softRelax.maxRatio
        if (!allowRelax) return SoftGateThresholds(baseGood, baseInliers, relaxed = false)

        val relaxedGood = cfg.softRelax.minGoodMatches.coerceAtMost(baseGood).coerceAtLeast(3)
        return SoftGateThresholds(relaxedGood, baseInliers, relaxed = relaxedGood < baseGood)
    }

    private fun computeTemporalMinGoodMatches(candidate: OrbMatchCandidate): Int {
        val cfg = heuristicConfig
        val base = cfg.orb.softMinGoodMatches
        if (candidate.goodMatches >= base) return base
        if (!cfg.softRelax.enabled) return base

        val tinyBox =
            min(candidate.box.width, candidate.box.height).toDouble() <= FIRST_LOCK_SMALL_BOX_SIDE_PX ||
                candidate.box.width.toDouble() * candidate.box.height.toDouble() <= FIRST_LOCK_SMALL_BOX_AREA_PX
        val refinedFallback =
            !candidate.usedHomography &&
                (candidate.fallbackReason?.startsWith("refined_") == true)
        val allowRelax =
            searchMissStreak >= cfg.softRelax.missStreak &&
                candidate.searchScale <= cfg.softRelax.scaleThreshold &&
                tinyBox &&
                refinedFallback
        if (!allowRelax) return base
        return cfg.softRelax.minGoodMatches.coerceAtMost(base).coerceAtLeast(3)
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
        val cfg = heuristicConfig
        lastSearchDiagReason = reason
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES != 0L) return
        Log.w(
            TAG,
            "EVAL_EVENT type=SEARCH_DIAG reason=$reason kpTpl=$templateKeypointCount kpFrm=$kpFrame " +
                "good=$selectedGood inliers=$selectedInliers flann=$flannGood bf=$bfGood matcher=$matcher " +
                "minGood=${cfg.orb.minGoodMatches} minInliers=${cfg.orb.minInliers} softGood=${cfg.orb.softMinGoodMatches} " +
                "softInliers=${cfg.orb.softMinInliers} miss=$searchMissStreak stable=$firstLockCandidateFrames/${cfg.firstLock.stableFrames} " +
                "detail=$detail"
        )
    }

    private fun collectGoodMatches(
        knnMatches: List<MatOfDMatch>,
        loweRatio: Double = heuristicConfig.orb.loweRatio
    ): List<DMatch> {
        val out = ArrayList<DMatch>(knnMatches.size)
        for (pair in knnMatches) {
            val m = pair.toArray()
            if (m.size < 2) continue
            if (m[0].distance < loweRatio * m[1].distance) out += m[0]
        }
        return out
    }

    private fun collectGoodMatchesFromPairs(knnPairs: List<KnnPair>, loweRatio: Double): List<DMatch> {
        val out = ArrayList<DMatch>(knnPairs.size)
        for (pair in knnPairs) {
            if (pair.first.distance < loweRatio * pair.second.distance) {
                out += pair.first
            }
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
        val weakCfg = heuristicConfig.weakFallback
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
        val maxSide = max(fallbackMaxBoxSize, fallbackMinBoxSize)
        val weakMinSide = min(fallbackMinBoxSize, WEAK_FALLBACK_MIN_BOX_SIZE)
        val minSide = if (points.size <= weakCfg.maxMatches) weakMinSide else fallbackMinBoxSize
        return ClusterEstimate(center, sideOriginal.coerceIn(minSide, maxSide))
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
        if (!isTracking && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag("OVERLAY", "session=$diagSessionId action=draw_while_not_tracking box=${box.x},${box.y},${box.width}x${box.height}")
        }
        val rect = RectF(
            box.x * scaleX + offsetX,
            box.y * scaleY + offsetY,
            (box.x + box.width) * scaleX + offsetX,
            (box.y + box.height) * scaleY + offsetY
        )
        overlayView.post { overlayView.updateTrackedObject(rect) }
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag(
                "OVERLAY",
                "session=$diagSessionId action=draw tracking=$isTracking box=${box.x},${box.y},${box.width}x${box.height} scale=${fmt(scaleX.toDouble())},${fmt(scaleY.toDouble())} offset=${fmt(offsetX.toDouble())},${fmt(offsetY.toDouble())}"
            )
            if (kalmanConfig.enabled) {
                val k = latestKalmanPrediction
                if (k != null) {
                    logDiag(
                        "KALMAN",
                        "session=$diagSessionId action=predict box=${k.x},${k.y},${k.width}x${k.height} measured=${box.x},${box.y},${box.width}x${box.height}"
                    )
                }
            }
        }
    }
    @Synchronized
    private fun updateLatestPrediction(box: Rect?, confidence: Double, tracking: Boolean) {
        latestPredictionFrame = frameCounter
        latestPredictionTracking = tracking
        latestPredictionConfidence = confidence
        latestPredictionBox = if (box == null) null else Rect(box.x, box.y, box.width, box.height)
    }

    private fun suppressOverlayOnUncertainTracking(reason: String, confidence: Double) {
        if (kalmanConfig.enabled && kalmanConfig.usePredictedHold) {
            val predicted = latestKalmanPrediction
            if (predicted != null && predicted.width >= MIN_BOX_DIM && predicted.height >= MIN_BOX_DIM) {
                metricsKalmanPredHoldCount++
                metricsSearchLastReason = "kalman_pred_hold"
                lastTrackedBox = predicted
                dispatchTrackedRect(predicted)
                updateLatestPrediction(predicted, confidence.coerceIn(0.0, 1.0), tracking = true)
                feedNativePriorBox(predicted, "hold_predict")
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    logDiag(
                        "KALMAN",
                        "session=$diagSessionId action=hold_predict reason=$reason box=${predicted.x},${predicted.y},${predicted.width}x${predicted.height}"
                    )
                }
                return
            }
        }
        overlayView.post { overlayView.reset() }
        updateLatestPrediction(null, confidence.coerceIn(0.0, 1.0), tracking = false)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag("OVERLAY", "session=$diagSessionId action=suppress reason=$reason conf=${fmt(confidence)}")
        }
    }

    private fun updateScaleFactors(frameWidth: Int, frameHeight: Int) {
        if (overlayView.width == 0 || overlayView.height == 0) return
        val sx = overlayView.width.toFloat() / frameWidth.toFloat()
        val sy = overlayView.height.toFloat() / frameHeight.toFloat()
        val fitScale = min(sx, sy)
        scaleX = fitScale
        scaleY = fitScale
        offsetX = (overlayView.width.toFloat() - frameWidth * fitScale) * 0.5f
        offsetY = (overlayView.height.toFloat() - frameHeight * fitScale) * 0.5f
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
            val nativeConfAvg = if (metricsNativeConfSamples > 0L) metricsNativeConfSum / metricsNativeConfSamples else -1.0
            val nativeConfMin = if (metricsNativeConfSamples > 0L) metricsNativeConfMin else -1.0
            val nativeConfMax = if (metricsNativeConfSamples > 0L) metricsNativeConfMax else -1.0
            val nativeSimAvg = if (metricsNativeSimSamples > 0L) metricsNativeSimSum / metricsNativeSimSamples else -1.0
            val nativeSimMin = if (metricsNativeSimSamples > 0L) metricsNativeSimMin else -1.0
            val nativeSimMax = if (metricsNativeSimSamples > 0L) metricsNativeSimMax else -1.0
            Log.w(
                TAG,
                "EVAL_PERF mode=${trackerMode.name.lowercase(Locale.US)} frames=$metricsFrames " +
                    "backendPref=${preferredTrackBackend.name.lowercase(Locale.US)} " +
                    "backendActive=${activeTrackBackend.name.lowercase(Locale.US)} " +
                    "avgFrameMs=${fmt(avgMs)} trackRatio=${fmt(trackRatio)} locks=$metricsLockCount " +
                    "lost=$metricsLostCount firstLockSec=${fmt(firstLockSec)} " +
                    "nativeFuseSoft=$metricsNativeFuseSoftCount nativeFuseHard=$metricsNativeFuseHardCount " +
                    "nativeHold=$metricsNativeLowConfHoldCount kalmanPredHold=$metricsKalmanPredHoldCount nativeLowStreak=$nativeLowConfidenceStreak " +
                    "nativeWarmup=$nativeFuseWarmupRemaining " +
                    "nativeConfAvg=${fmt(nativeConfAvg)} nativeConfMin=${fmt(nativeConfMin)} nativeConfMax=${fmt(nativeConfMax)} " +
                    "nativeSimAvg=${fmt(nativeSimAvg)} nativeSimMin=${fmt(nativeSimMin)} nativeSimMax=${fmt(nativeSimMax)} " +
                    "searchCand=$metricsSearchCandidateCount searchMiss=$metricsSearchMissCount " +
                    "tplSkip=$metricsSearchTemplateSkipCount skipStride=$metricsSearchStrideSkipCount " +
                    "skipBudget=$metricsSearchBudgetSkipCount budgetTrips=$metricsSearchBudgetTripCount " +
                    "budgetCooldown=$searchBudgetCooldownFrames tempRej=$metricsSearchTemporalRejectCount " +
                    "promRej=$metricsSearchPromoteRejectCount refinePass=$metricsSearchRefinePassCount " +
                    "refineRej=$metricsSearchRefineRejectCount stableSeed=$metricsSearchStableSeedCount " +
                    "stableAccum=$metricsSearchStableAccumCount promote=$metricsSearchPromoteCount " +
                    "stableHold=$metricsSearchStableOutlierHoldCount tempHold=$metricsSearchTemporalHoldCount " +
                    "resetInit=$metricsSearchResetNoSeedCount resetGap=$metricsSearchResetGapCount " +
                    "resetDrift=$metricsSearchResetStableDriftCount resetCenter=$metricsSearchResetCenterRuleCount " +
                    "resetIou=$metricsSearchResetIouCount " +
                    "resetConf=$metricsSearchResetConfidenceCount " +
                    "iouFail=$metricsSearchResetIouCount centerFail=$metricsSearchResetCenterRuleCount confFail=$metricsSearchResetConfidenceCount " +
                    "lastReason=$metricsSearchLastReason"
            )
        }

        if (
            !isTracking &&
            searchOverBudgetMs > 0L &&
            searchOverBudgetSkipFrames > 0 &&
            processMs >= searchOverBudgetMs
        ) {
            searchBudgetCooldownFrames = max(searchBudgetCooldownFrames, searchOverBudgetSkipFrames)
            metricsSearchBudgetTripCount++
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

    private fun logicalFrameSize(image: ImageProxy): Pair<Int, Int> {
        return when (image.imageInfo.rotationDegrees) {
            90, 270 -> Pair(image.height, image.width)
            else -> Pair(image.width, image.height)
        }
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

    private fun logDiag(type: String, payload: String) {
        Log.i(TAG, "DIAG_${type.uppercase(Locale.US)} $payload")
    }

    companion object {
        private const val TAG = "Tracker"

        private const val MIN_BOX_DIM = 16
        private const val MIN_TEMPLATE_LEVEL_DIM = 20
        private const val MIN_TEMPLATE_LEVEL_KEYPOINTS = 10
        private const val MIN_TEMPLATE_USABLE_KEYPOINTS = 50
        private const val MAX_ACTIVE_TEMPLATE_LEVELS = 30
        private const val MAX_TEMPLATE_DIM = 480.0
        private val TEMPLATE_PYRAMID_SCALES = doubleArrayOf(1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09, 0.075)

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
        private const val FIRST_LOCK_OUTLIER_GOOD_BOOST = 6
        private const val FIRST_LOCK_OUTLIER_INLIER_BOOST = 2
        private const val FIRST_LOCK_ANCHOR_NEAR_FACTOR = 1.6
        private const val FIRST_LOCK_SEED_REEVAL_EXPAND_FACTOR = 2.0
        private const val FIRST_LOCK_SEED_REEVAL_ACCEPT_FACTOR = 1.25
        private const val FIRST_LOCK_BASE_AREA_RATIO = 0.015
        private const val FIRST_LOCK_MIN_IOU_FLOOR = 0.55
        private const val FALLBACK_LOCK_GOOD_RELAX = 2
        private const val FALLBACK_LOCK_INLIER_RELAX = 2
        private const val FALLBACK_LOCK_MIN_CONFIDENCE = 0.38
        private const val FALLBACK_LOCK_MIN_CONFIDENCE_SMALL = 0.28
        private const val FALLBACK_LOCK_MIN_CONFIDENCE_HIGH_GOOD = 0.12
        private const val FALLBACK_LOCK_HIGH_GOOD_MATCHES = 20
        private const val FALLBACK_LOCK_HIGH_INLIERS = 6
        private const val AUTO_INIT_MIN_GOOD_MATCHES = 12
        private const val AUTO_INIT_MIN_INLIERS = 4
        private const val AUTO_INIT_MIN_CONFIDENCE = 0.22
        private const val REFINE_HIGH_GOOD_MATCHES = 20
        private const val REFINE_HIGH_INLIERS = 6
        private const val FALLBACK_REFINE_MIN_CONFIDENCE_HIGH_GOOD = 0.12
        private const val TEMPORAL_HIGH_GOOD_MATCHES = 24
        private const val TEMPORAL_HIGH_INLIERS = 6
        private const val TEMPORAL_MEDIUM_GOOD_MATCHES = 12
        private const val TEMPORAL_MIN_CONFIDENCE_HIGH_GOOD = 0.12
        private const val TEMPORAL_MIN_CONFIDENCE_MEDIUM = 0.22

        private const val DEFAULT_ORB_FEATURES = 900
        private const val DEFAULT_ORB_FEATURE_HARD_CAP = 700
        private const val DEFAULT_ORB_SCALE_FACTOR = 1.20
        private const val DEFAULT_ORB_N_LEVELS = 8
        private const val DEFAULT_ORB_FAST_THRESHOLD = 20
        private const val DEFAULT_ORB_LOWE_RATIO = 0.75
        private const val DEFAULT_ORB_MIN_GOOD_MATCHES = 10
        private const val DEFAULT_ORB_MIN_INLIERS = 6
        private const val DEFAULT_ORB_SOFT_MIN_GOOD_MATCHES = 4
        private const val DEFAULT_ORB_SOFT_MIN_INLIERS = 3
        private const val DEFAULT_ORB_RANSAC_THRESHOLD = 4.0
        private const val DEFAULT_ORB_USE_CLAHE = true
        private const val DEFAULT_CLAHE_CLIP_LIMIT = 2.0
        private const val DEFAULT_CLAHE_TILE_SIZE = 8.0
        private const val DEFAULT_ORB_FAR_BOOST_FEATURES = 1600

        private const val DEFAULT_SEARCH_SHORT_EDGE = 600
        private const val DEFAULT_SEARCH_STRIDE_FRAMES = 1
        private const val DEFAULT_SEARCH_OVER_BUDGET_MS = 60L
        private const val DEFAULT_SEARCH_OVER_BUDGET_SKIP_FRAMES = 0
        private const val DEFAULT_SEARCH_HIGH_RES_MISS_STREAK = 8
        private const val DEFAULT_SEARCH_HIGH_RES_SHORT_EDGE = 860
        private const val DEFAULT_SEARCH_ULTRA_HIGH_RES_MISS_STREAK = 24
        private const val DEFAULT_SEARCH_ULTRA_HIGH_RES_SHORT_EDGE = 1080
        private const val DEFAULT_SEARCH_MAX_LONG_EDGE = 640
        private const val DEFAULT_SEARCH_MULTI_TEMPLATE_MAX_LONG_EDGE = 520
        private const val DEFAULT_INIT_BOX_SIZE = 100
        private const val DEFAULT_FALLBACK_MIN_BOX_SIZE = 48
        private const val DEFAULT_FALLBACK_MAX_BOX_SIZE = 180
        private const val DEFAULT_HOMOGRAPHY_MAX_DISTORTION = 0.20
        private const val DEFAULT_HOMOGRAPHY_SCALE_MIN = 0.60
        private const val DEFAULT_HOMOGRAPHY_SCALE_MAX = 1.80
        private const val DEFAULT_HOMOGRAPHY_MIN_JACOBIAN_DET = 1e-5
        private const val DEFAULT_TEMPLATE_MIN_TEXTURE_SCORE = 15.0

        private const val DEFAULT_ORB_FAR_BOOST_MULTI_TEMPLATE_CAP = 1200

        private const val DEFAULT_FIRST_LOCK_STABLE_FRAMES = 3
        private const val DEFAULT_FIRST_LOCK_STABLE_MS = 200L
        private const val DEFAULT_FIRST_LOCK_STABLE_PX = 54.0
        private const val DEFAULT_FIRST_LOCK_MIN_IOU = 0.40
        private const val DEFAULT_FIRST_LOCK_GAP_MS = 500L
        private const val DEFAULT_FIRST_LOCK_SMALL_CENTER_DRIFT_PX = 20.0
        private const val DEFAULT_FIRST_LOCK_SMALL_CENTER_DRIFT_RELAXED_PX = 34.0
        private const val DEFAULT_FIRST_LOCK_SMALL_STABLE_PX = 84.0
        private const val DEFAULT_FIRST_LOCK_SMALL_RELAX_MISS_STREAK = 8
        private const val DEFAULT_FIRST_LOCK_SMALL_DYNAMIC_CENTER_FACTOR = 1.2
        private const val DEFAULT_FIRST_LOCK_SMALL_RELAXED_IOU_FLOOR = 0.00
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_HIGH_STREAK = 30
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_HIGH_FACTOR = 0.72
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_MID_STREAK = 15
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_MID_FACTOR = 0.80
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_LOW_STREAK = 8
        private const val DEFAULT_FIRST_LOCK_MISS_RELAX_LOW_FACTOR = 0.90
        private const val DEFAULT_FIRST_LOCK_RESET_RELAX_MIN_FRAMES = 2
        private const val DEFAULT_FIRST_LOCK_RESET_RELAX_MIN_IOU_RESETS = 3
        private const val DEFAULT_FIRST_LOCK_RESET_RELAX_FACTOR = 0.92
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG_GOOD = 24
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG_INLIERS = 10
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_STRONG = 0.50
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM_GOOD = 20
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM_INLIERS = 8
        private const val DEFAULT_FIRST_LOCK_IOU_FLOOR_MEDIUM = 0.53
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_GOOD = 20
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_INLIERS = 8
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_STRONG_FACTOR = 1.25
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_MISS_STREAK = 10
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_MISS_FACTOR = 1.15
        private const val DEFAULT_FIRST_LOCK_CENTER_RELAX_CAP_FACTOR = 1.2
        private const val DEFAULT_FIRST_LOCK_CONF_STRONG_GOOD = 24
        private const val DEFAULT_FIRST_LOCK_CONF_STRONG_INLIERS = 8
        private const val DEFAULT_FIRST_LOCK_CONF_STRONG_MIN = 0.22
        private const val DEFAULT_FIRST_LOCK_CONF_MEDIUM_GOOD = 16
        private const val DEFAULT_FIRST_LOCK_CONF_MEDIUM_INLIERS = 6
        private const val DEFAULT_FIRST_LOCK_CONF_MEDIUM_MIN = 0.24
        private const val DEFAULT_FIRST_LOCK_CONF_SMALL_MIN = 0.30
        private const val DEFAULT_FIRST_LOCK_CONF_BASE_MIN = 0.03
        private const val DEFAULT_FIRST_LOCK_HOLD_ON_TEMPORAL_REJECT = true
        private const val DEFAULT_FIRST_LOCK_OUTLIER_HOLD_MAX = 2
        private const val DEFAULT_ALLOW_FALLBACK_LOCK = true
        private const val DEFAULT_FALLBACK_REFINE_EXPAND_FACTOR = 1.8
        private const val DEFAULT_FALLBACK_REFINE_MIN_GOOD_MATCHES = 8
        private const val DEFAULT_FALLBACK_REFINE_MIN_INLIERS = 4
        private const val DEFAULT_FALLBACK_REFINE_MIN_CONFIDENCE = 0.40
        private const val DEFAULT_FALLBACK_REFINE_LOWE_RATIO = 0.84
        private const val DEFAULT_SMALL_TARGET_AREA_RATIO = 0.05
        private const val DEFAULT_SMALL_TARGET_MIN_GOOD_MATCHES = 4
        private const val DEFAULT_SMALL_TARGET_MIN_INLIERS = 3
        private const val DEFAULT_SMALL_TARGET_SCALE_THRESHOLD = 0.55
        private const val DEFAULT_WEAK_FALLBACK_MAX_MATCHES = 6
        private const val DEFAULT_WEAK_FALLBACK_MAX_SPAN_PX = 96.0
        private const val DEFAULT_WEAK_FALLBACK_MAX_AREA_PX = 7_200.0
        private const val DEFAULT_WEAK_FALLBACK_RELAX_MISS_STREAK = 8
        private const val DEFAULT_WEAK_FALLBACK_RELAX_FACTOR = 1.35
        private const val DEFAULT_WEAK_FALLBACK_REQUIRE_REFINE = true
        private const val DEFAULT_WEAK_FALLBACK_RESCUE_ENABLED = true
        private const val DEFAULT_WEAK_FALLBACK_RESCUE_RATIO = 0.66
        private const val DEFAULT_WEAK_FALLBACK_RESCUE_MIN_GOOD = 4
        private const val DEFAULT_WEAK_FALLBACK_CORE_RESCUE_ENABLED = true
        private const val DEFAULT_WEAK_FALLBACK_CORE_MAX_SPAN_PX = 72.0
        private const val DEFAULT_WEAK_FALLBACK_CORE_MAX_AREA_PX = 3_200.0
        private const val DEFAULT_SOFT_RELAX_ENABLED = true
        private const val DEFAULT_SOFT_RELAX_MISS_STREAK = 10
        private const val DEFAULT_SOFT_RELAX_MIN_GOOD_MATCHES = 3
        private const val DEFAULT_SOFT_RELAX_SCALE_THRESHOLD = 0.75
        private const val DEFAULT_SOFT_RELAX_MAX_RATIO = 0.70
        private const val WEAK_FALLBACK_MIN_BOX_SIZE = 40

        private const val DEFAULT_KCF_MAX_FAIL_STREAK = 3
        private const val DEFAULT_NATIVE_MAX_FAIL_STREAK = 6
        private const val DEFAULT_NATIVE_MIN_CONFIDENCE = 0.15
        private const val DEFAULT_NATIVE_FUSE_SOFT_CONFIDENCE = 0.20
        private const val DEFAULT_NATIVE_FUSE_HARD_CONFIDENCE = 0.10
        private const val DEFAULT_NATIVE_FUSE_SOFT_STREAK = 3
        private const val DEFAULT_NATIVE_FUSE_WARMUP_FRAMES = 12
        private const val DEFAULT_NATIVE_HOLD_LAST_ON_SOFT_REJECT = true
        private const val DEFAULT_NATIVE_SCORE_LOG_INTERVAL_FRAMES = 20
        private const val DEFAULT_NATIVE_ORB_VERIFY_ENABLED = false
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_GOOD = 16
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_INLIERS = 6
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_CONFIDENCE = 0.36
        private const val DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_LIVE = 0.16
        private const val DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_REPLAY = 0.10
        private const val DEFAULT_AUTO_VERIFY_LOCAL_ROI_EXPAND_FACTOR = 1.6
        private const val DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_LIVE = 0.75
        private const val DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_REPLAY = 1.80
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_LIVE = 0.40
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_REPLAY = 0.00
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_LIVE = 0.72
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_REPLAY = 0.75
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_LIVE = 0.24
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_REPLAY = 0.24
        private const val DEFAULT_AUTO_VERIFY_APPEAR_STRONG_GOOD = 24
        private const val DEFAULT_AUTO_VERIFY_APPEAR_STRONG_INLIERS = 10
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_STRONG = -0.06
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_GOOD = 20
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_INLIERS = 8
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_MEDIUM = -0.25
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_BASE = -0.18
        private const val DEFAULT_AUTO_VERIFY_APPEAR_LIVE_BIAS = 0.06
        private const val DEFAULT_KALMAN_ENABLED = false
        private const val DEFAULT_KALMAN_PROCESS_NOISE = 0.10
        private const val DEFAULT_KALMAN_MEASUREMENT_NOISE = 0.24
        private const val DEFAULT_KALMAN_MAX_PREDICT_MS = 280L
        private const val DEFAULT_KALMAN_USE_PREDICTED_HOLD = false
        private const val DEFAULT_KALMAN_DYNAMIC_MEASUREMENT_NOISE = true
        private const val DEFAULT_KALMAN_HIGH_CONFIDENCE_THRESHOLD = 0.80
        private const val DEFAULT_KALMAN_LOW_CONFIDENCE_THRESHOLD = 0.20
        private const val DEFAULT_KALMAN_HIGH_CONFIDENCE_NOISE_SCALE = 0.25
        private const val DEFAULT_KALMAN_LOW_CONFIDENCE_NOISE_SCALE = 6.0
        private const val DEFAULT_KALMAN_OCCLUSION_NOISE_SCALE = 10.0
        private const val DEFAULT_KALMAN_FEED_NATIVE_PRIOR = true
        private const val DEFAULT_KALMAN_PRETRACK_PRIOR_FEED = false
        private const val DEFAULT_KALMAN_PRIOR_ONLY_ON_UNCERTAIN = true
        private const val DEFAULT_KALMAN_PRIOR_MIN_IOU = 0.25
        private const val DEFAULT_KALMAN_PRIOR_STALE_MS = 100L
        private const val DEFAULT_LOCK_HOLD_FRAMES = 0
        private const val DEFAULT_LOST_OVERLAY_HOLD_MS = 120L
        private const val DEFAULT_FAST_FIRST_LOCK_FRAMES = 240
        private const val FAST_FIRST_LOCK_MIN_GOOD_MATCHES = 12
        private const val FAST_FIRST_LOCK_MIN_INLIERS = 4
        private const val FAST_FIRST_LOCK_MIN_CONFIDENCE = 0.24
        private const val DEFAULT_FORCE_TRACKER_GC_ON_DROP = true
        private const val TRACKER_GC_MIN_INTERVAL_MS = 1_200L
        private const val DEFAULT_NATIVE_MODEL_PARAM_PATH = "/data/local/tmp/nanotrack.param"
        private const val DEFAULT_NATIVE_MODEL_BIN_PATH = "/data/local/tmp/nanotrack.bin"

        private const val DEFAULT_TRACK_VERIFY_INTERVAL_FRAMES = 30
        private const val DEFAULT_TRACK_VERIFY_LOCAL_EXPAND_FACTOR = 2.2
        private const val DEFAULT_TRACK_VERIFY_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_MIN_INLIERS = 3
        private const val DEFAULT_TRACK_VERIFY_FAIL_TOLERANCE = 3
        private const val DEFAULT_TRACK_VERIFY_RECENTER_PX = 48.0
        private const val DEFAULT_TRACK_VERIFY_MIN_IOU = 0.35
        private const val DEFAULT_TRACK_VERIFY_SWITCH_CONFIDENCE_MARGIN = 0.15
        private const val TRACK_VERIFY_HARD_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_HARD_DRIFT_TOLERANCE = 3
        private const val DEFAULT_TRACK_VERIFY_NATIVE_BYPASS_CONFIDENCE = 0.75
        private const val DEFAULT_TRACK_GUARD_MAX_CENTER_JUMP_FACTOR = 1.00
        private const val DEFAULT_TRACK_GUARD_MIN_AREA_RATIO = 0.35
        private const val DEFAULT_TRACK_GUARD_MAX_AREA_RATIO = 2.00
        private const val DEFAULT_TRACK_GUARD_DROP_STREAK = 1
        private const val DEFAULT_TRACK_GUARD_APPEARANCE_CHECK_INTERVAL = 4L
        private const val DEFAULT_TRACK_GUARD_MIN_APPEARANCE_SCORE = -0.12
        private const val DEFAULT_TRACK_GUARD_ANCHOR_ENABLED = true
        private const val DEFAULT_TRACK_GUARD_ANCHOR_MAX_DROP = 0.24
        private const val DEFAULT_TRACK_GUARD_ANCHOR_MIN_SCORE = -1.0
        private const val DEFAULT_TEMPORAL_MIN_CONFIDENCE_SMALL_REFINED = 0.24
        private const val DEFAULT_TEMPORAL_MIN_CONFIDENCE_BASE = 0.32
        private const val DEFAULT_TEMPORAL_LIVE_CONFIDENCE_RELAX = 0.03
        private const val DEFAULT_TEMPORAL_LIVE_CONFIDENCE_FLOOR = 0.10
        private const val REPLAY_NATIVE_VERIFY_INTERVAL_FRAMES = 10

        private const val SEARCH_DIAG_INTERVAL_FRAMES = 20
        private const val PERF_LOG_INTERVAL_FRAMES = 10
        private const val SUMMARY_LOG_INTERVAL_FRAMES = 30
    }
}











































































