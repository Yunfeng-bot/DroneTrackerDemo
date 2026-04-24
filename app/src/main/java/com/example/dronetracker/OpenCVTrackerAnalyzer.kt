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
import kotlin.math.exp
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

    private enum class CenterRoiLevel {
        L1,
        L2,
        L3
    }

    private enum class DescendOffsetState {
        L1,
        L2,
        L3,
        TRACKING,
        LOST,
        FAIL
    }

    private enum class PromotedRelockTier {
        NONE,
        NEAR,
        FAR
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

    private data class SearchAnchorHint(
        val point: Point,
        val radius: Double,
        val source: String
    )

    private data class SpatialPrior(
        val box: Rect,
        val source: String
    )

    private data class SpatialScore(
        val score: Double,
        val distance2: Double,
        val source: String
    )

    private data class DescendExplosionGuardDecision(
        val box: Rect,
        val active: Boolean,
        val areaRatio: Double
    )

    private data class LostPriorGeometry(
        val prior: Rect,
        val framesAgo: Long,
        val iou: Double,
        val centerDist: Double,
        val side: Double,
        val near: Boolean
    )

    private data class OrbCandidateDumpRow(
        val candidate: OrbMatchCandidate,
        val appearanceScore: Double,
        val spatialScore: Double?,
        val spatialDistance2: Double?,
        val fusionScore: Double,
        val tier: PromotedRelockTier,
        val source: String
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
        val localMissingMinConfLive: Double,
        val localMissingMinConfReplay: Double,
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
    private val liveFrameCacheLock = Any()
    @Volatile
    private var lastLiveFrameGray: Mat? = null
    @Volatile
    private var lastLiveFrameRgb: Mat? = null
    @Volatile
    private var lastLiveFrameTsMs: Long = 0L
    @Volatile
    private var manualRoiActive = false
    @Volatile
    private var manualRoiSessionActive = false
    @Volatile
    private var manualRoiFrameTsMs: Long = 0L
    @Volatile
    private var manualRoiPatchKp = -1
    @Volatile
    private var manualRoiPatchTexture = Double.NaN
    @Volatile
    private var manualRoiBboxClamped = false
    @Volatile
    private var manualRoiInitPath = "none"
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
    private var currentFrameWidth = 0
    private var currentFrameHeight = 0
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
    private var metricsFirstLockReplayMs = -1L
    private var metricsFirstLockFrame = -1L
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
    private var currentReplayPtsMs = -1L
    private var replayTargetAppearMs = -1L
    private var centerRoiSearchEnabled = DEFAULT_CENTER_ROI_SEARCH_ENABLED
    private var centerRoiGpsReady = DEFAULT_CENTER_ROI_GPS_READY
    private var centerRoiL1Range = DEFAULT_CENTER_ROI_L1_RANGE
    private var centerRoiL2Range = DEFAULT_CENTER_ROI_L2_RANGE
    private var centerRoiL1TimeoutMs = DEFAULT_CENTER_ROI_L1_TIMEOUT_MS
    private var centerRoiL2TimeoutMs = DEFAULT_CENTER_ROI_L2_TIMEOUT_MS
    private var centerRoiL3TimeoutMs = DEFAULT_CENTER_ROI_L3_TIMEOUT_MS
    private var descendExplosionGuardEnabled = DEFAULT_DESCEND_EXPLOSION_GUARD_ENABLED
    private var descendExplosionAreaRatio = DEFAULT_DESCEND_EXPLOSION_AREA_RATIO
    private var descendExplosionReleaseRatio = DEFAULT_DESCEND_EXPLOSION_RELEASE_RATIO
    private var descendExplosionCoreSizePx = DEFAULT_DESCEND_EXPLOSION_CORE_SIZE_PX
    private var centerRoiLevel = CenterRoiLevel.L1
    private var centerRoiLevelStartMs = 0L
    private var centerRoiSessionStartMs = 0L
    private var centerRoiTimeoutRaised = false
    private var centerRoiLastScope = "full"
    private var centerRoiL1ElapsedMs = 0L
    private var centerRoiL2ElapsedMs = 0L
    private var centerRoiL3ElapsedMs = 0L
    private var centerRoiFailEmittedThisSession = false
    private var centerRoiFailLatched = false
    private var descendLastState: DescendOffsetState? = null
    private var descendLastEmitMs = 0L
    private var descendCacheX = Double.NaN
    private var descendCacheY = Double.NaN
    private var descendCacheConf = Double.NaN
    private var descendLostActive = false
    private var descendLostStartMs = 0L
    private var descendExplosionGuardActive = false
    private var descendExplosionLastAreaRatio = 0.0

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
    private var templatePoseAugmentEnabled = DEFAULT_TEMPLATE_POSE_AUGMENT_ENABLED
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
    private var nativeGateUseMeasurement = DEFAULT_NATIVE_GATE_USE_MEASUREMENT
    private var nativeGapPassthrough = DEFAULT_NATIVE_GAP_PASSTHROUGH
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
    private var trackGuardAccelGraceFrames = DEFAULT_TRACK_GUARD_ACCEL_GRACE_FRAMES
    private var trackGuardHardMismatch = false
    private var trackAccelGraceFramesRemaining = 0
    private var trackAnchorAppearanceScore = Double.NaN
    private var lastLostBox: Rect? = null
    private var lastLostFrameId = -1L
    private var smallTargetNativeVerifyIntervalFrames = DEFAULT_SMALL_TARGET_NATIVE_VERIFY_INTERVAL_FRAMES
    private var smallTargetNativeVerifyAreaScale = DEFAULT_SMALL_TARGET_NATIVE_VERIFY_AREA_SCALE
    private var smallTargetAnchorDropScale = DEFAULT_SMALL_TARGET_ANCHOR_DROP_SCALE

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
    private var autoVerifyLocalMissingMinConfLive = DEFAULT_AUTO_VERIFY_LOCAL_MISSING_MIN_CONF_LIVE
    private var autoVerifyLocalMissingMinConfReplay = DEFAULT_AUTO_VERIFY_LOCAL_MISSING_MIN_CONF_REPLAY
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
    private var autoVerifyRejectFlipReplay = DEFAULT_AUTO_VERIFY_REJECT_FLIP_REPLAY
    private var autoVerifyRelockMinConfLive = DEFAULT_AUTO_VERIFY_RELOCK_MIN_CONF_LIVE
    private var autoVerifyRelockMinConfReplay = DEFAULT_AUTO_VERIFY_RELOCK_MIN_CONF_REPLAY
    private var autoVerifyLostPriorMaxFramesLive = DEFAULT_AUTO_VERIFY_LOST_PRIOR_MAX_FRAMES_LIVE
    private var autoVerifyLostPriorMaxFramesReplay = DEFAULT_AUTO_VERIFY_LOST_PRIOR_MAX_FRAMES_REPLAY
    private var autoVerifyLostPriorCenterFactorLive = DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_FACTOR_LIVE
    private var autoVerifyLostPriorCenterFactorReplay = DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_FACTOR_REPLAY
    private var autoVerifyLostPriorMinIouLive = DEFAULT_AUTO_VERIFY_LOST_PRIOR_MIN_IOU_LIVE
    private var autoVerifyLostPriorMinIouReplay = DEFAULT_AUTO_VERIFY_LOST_PRIOR_MIN_IOU_REPLAY
    private var autoVerifyLostPriorCenterBoostFrames = DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_BOOST_FRAMES
    private var autoVerifyLostPriorCenterBoostCap = DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_BOOST_CAP
    private var autoVerifyLostAnchorNearFactorLive = DEFAULT_AUTO_VERIFY_LOST_ANCHOR_NEAR_FACTOR_LIVE
    private var autoVerifyLostAnchorNearFactorReplay = DEFAULT_AUTO_VERIFY_LOST_ANCHOR_NEAR_FACTOR_REPLAY
    private var autoVerifyLostFarRecoverMinGood = DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_GOOD
    private var autoVerifyLostFarRecoverMinInliers = DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_INLIERS
    private var autoVerifyLostFarRecoverMinConfLive = DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_CONF_LIVE
    private var autoVerifyLostFarRecoverMinConfReplay = DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_CONF_REPLAY
    private var autoVerifyLostFarRecoverMinAppearance = DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_APPEARANCE
    private var autoVerifyFirstLockAppearanceMinLive = DEFAULT_AUTO_VERIFY_FIRST_LOCK_APPEAR_MIN_LIVE
    private var autoVerifyFirstLockAppearanceMinReplay = DEFAULT_AUTO_VERIFY_FIRST_LOCK_APPEAR_MIN_REPLAY
    private var autoVerifyFirstLockMinInliersReplay = DEFAULT_AUTO_VERIFY_FIRST_LOCK_MIN_INLIERS_REPLAY
    private var autoVerifyFirstLockRequireLocalLive = DEFAULT_AUTO_VERIFY_FIRST_LOCK_REQUIRE_LOCAL_LIVE
    private var autoVerifyFirstLockRequireLocalReplay = DEFAULT_AUTO_VERIFY_FIRST_LOCK_REQUIRE_LOCAL_REPLAY
    private var autoVerifyFirstLockCenterGuardReplay = DEFAULT_AUTO_VERIFY_FIRST_LOCK_CENTER_GUARD_REPLAY
    private var autoVerifyFirstLockCenterFactorReplay = DEFAULT_AUTO_VERIFY_FIRST_LOCK_CENTER_FACTOR_REPLAY
    private var spatialGateEnabled = DEFAULT_SPATIAL_GATE_ENABLED
    private var spatialGateWeight = DEFAULT_SPATIAL_GATE_WEIGHT
    private var spatialGateCenterSigmaFactor = DEFAULT_SPATIAL_GATE_CENTER_SIGMA_FACTOR
    private var spatialGateSizeSigmaFactor = DEFAULT_SPATIAL_GATE_SIZE_SIGMA_FACTOR
    private var spatialGateRelockMinScoreLive = DEFAULT_SPATIAL_GATE_RELOCK_MIN_SCORE_LIVE
    private var spatialGateRelockMinScoreReplay = DEFAULT_SPATIAL_GATE_RELOCK_MIN_SCORE_REPLAY
    private var spatialGateRelockBypassConf = DEFAULT_SPATIAL_GATE_RELOCK_BYPASS_CONF
    private var spatialGateRelockBypassAppearance = DEFAULT_SPATIAL_GATE_RELOCK_BYPASS_APPEARANCE
    private var autoVerifyRelockSpatialRelaxStreakLive = AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_LIVE
    private var autoVerifyRelockSpatialRelaxStreakReplay = AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_REPLAY
    private var autoVerifyRelockSpatialRelaxMinScore = AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_SCORE
    private var autoVerifyRelockSpatialRelaxMinAppearance = AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_APPEARANCE
    private var candDumpEnable = false
    private var candDumpWindowOnly = true
    private var candDumpStartSec = 20.0
    private var candDumpEndSec = 28.0
    private var candDumpTopK = 5
    private var candDumpExpectedXMin = 0.30
    private var candDumpExpectedXMax = 0.50
    private var s2SuppressKcfFallbackEnabled = DEFAULT_S2_SUPPRESS_KCF_FALLBACK_ENABLED
    private var s3PromotedNearGateEnabled = DEFAULT_S3_PROMOTED_NEAR_GATE_ENABLED
    private var s3PromotedFarGateEnabled = DEFAULT_S3_PROMOTED_FAR_GATE_ENABLED
    private var s3PromotedNearAnchorVetoEnabled = DEFAULT_S3_PROMOTED_NEAR_ANCHOR_VETO_ENABLED
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
    private var relockSpatialRejectStreak = 0
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
                "center_roi_search_enable", "mvp_center_roi_search_enable" ->
                    parseBoolean(value)?.let {
                        centerRoiSearchEnabled = it
                        if (!it) {
                            resetCenterRoiSearchState("override_disable")
                        }
                    }
                "center_roi_gps_ready", "mvp_gps_ready" ->
                    parseBoolean(value)?.let { setCenterRoiGpsReady(it, "override") }
                "center_roi_l1_range", "mvp_roi_l1_range" ->
                    value.toDoubleOrNull()?.let { centerRoiL1Range = it.coerceIn(0.05, 0.45) }
                "center_roi_l2_range", "mvp_roi_l2_range" ->
                    value.toDoubleOrNull()?.let { centerRoiL2Range = it.coerceIn(0.10, 0.49) }
                "center_roi_l1_timeout_ms", "mvp_roi_l1_timeout_ms" ->
                    value.toLongOrNull()?.let { centerRoiL1TimeoutMs = it.coerceIn(100L, 5_000L) }
                "center_roi_l2_timeout_ms", "mvp_roi_l2_timeout_ms" ->
                    value.toLongOrNull()?.let { centerRoiL2TimeoutMs = it.coerceIn(100L, 5_000L) }
                "center_roi_l3_timeout_ms", "mvp_roi_l3_timeout_ms" ->
                    value.toLongOrNull()?.let { centerRoiL3TimeoutMs = it.coerceIn(200L, 10_000L) }
                "descend_explosion_guard", "mvp_descend_explosion_guard" ->
                    parseBoolean(value)?.let { descendExplosionGuardEnabled = it }
                "descend_explosion_area_ratio", "mvp_descend_explosion_area_ratio" ->
                    value.toDoubleOrNull()?.let { descendExplosionAreaRatio = it.coerceIn(0.001, 0.95) }
                "descend_explosion_release_ratio", "mvp_descend_explosion_release_ratio" ->
                    value.toDoubleOrNull()?.let { descendExplosionReleaseRatio = it.coerceIn(0.001, 0.90) }
                "descend_explosion_core_size", "mvp_descend_explosion_core_size" ->
                    value.toIntOrNull()?.let { descendExplosionCoreSizePx = it.coerceIn(MIN_BOX_DIM, 1600) }
                "init_box" -> value.toIntOrNull()?.let { initBoxSize = it.coerceIn(48, 1600) }
                "fallback_box_min" -> value.toIntOrNull()?.let { fallbackMinBoxSize = it.coerceIn(32, 240) }
                "fallback_box_max" -> value.toIntOrNull()?.let { fallbackMaxBoxSize = it.coerceIn(64, 1600) }
                "homography_distortion" -> value.toDoubleOrNull()?.let { homographyMaxDistortion = it.coerceIn(0.05, 0.60) }
                "homography_scale_min" -> value.toDoubleOrNull()?.let { homographyScaleMin = it.coerceIn(0.40, 2.0) }
                "homography_scale_max" -> value.toDoubleOrNull()?.let { homographyScaleMax = it.coerceIn(0.60, 3.0) }
                "homography_min_det" -> value.toDoubleOrNull()?.let { homographyMinJacobianDet = it.coerceIn(1e-7, 1e-2) }
                "template_min_texture" -> value.toDoubleOrNull()?.let { templateMinTextureScore = it.coerceIn(0.0, 200.0) }
                "template_pose_aug", "template_rotation_aug" -> parseBoolean(value)?.let { templatePoseAugmentEnabled = it }
                "orb_clahe" -> parseBoolean(value)?.let { orbUseClahe = it }
                "orb_far_features" -> value.toIntOrNull()?.let { orbFarBoostFeatures = it.coerceIn(700, 2800) }
                "orb_far_features_multi_cap" -> value.toIntOrNull()?.let { orbFarBoostMultiTemplateCap = it.coerceIn(600, 2600) }
                "first_lock_stable_frames" -> value.toIntOrNull()?.let { firstLockStableFrames = it.coerceIn(1, 90) }
                "first_lock_stable_ms" -> value.toLongOrNull()?.let { firstLockStableMs = it.coerceIn(120L, 3_500L) }
                "first_lock_stable_px" -> value.toDoubleOrNull()?.let { firstLockStablePx = it.coerceIn(8.0, 240.0) }
                "first_lock_min_iou" -> value.toDoubleOrNull()?.let { firstLockMinIou = it.coerceIn(0.30, 0.98) }
                "first_lock_gap_ms" -> value.toLongOrNull()?.let { firstLockGapMs = it.coerceIn(80L, 2_500L) }
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
                "auto_verify_local_missing_min_conf_live" -> value.toDoubleOrNull()?.let { autoVerifyLocalMissingMinConfLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_local_missing_min_conf_replay" -> value.toDoubleOrNull()?.let { autoVerifyLocalMissingMinConfReplay = it.coerceIn(0.0, 1.0) }
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
                "auto_verify_reject_flip_replay" -> parseBoolean(value)?.let { autoVerifyRejectFlipReplay = it }
                "auto_verify_relock_min_conf_live" -> value.toDoubleOrNull()?.let { autoVerifyRelockMinConfLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_relock_min_conf_replay" -> value.toDoubleOrNull()?.let { autoVerifyRelockMinConfReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_lost_prior_max_frames_live" -> value.toLongOrNull()?.let { autoVerifyLostPriorMaxFramesLive = it.coerceIn(1L, 600L) }
                "auto_verify_lost_prior_max_frames_replay" -> value.toLongOrNull()?.let { autoVerifyLostPriorMaxFramesReplay = it.coerceIn(1L, 900L) }
                "auto_verify_lost_prior_center_live" -> value.toDoubleOrNull()?.let { autoVerifyLostPriorCenterFactorLive = it.coerceIn(0.5, 8.0) }
                "auto_verify_lost_prior_center_replay" -> value.toDoubleOrNull()?.let { autoVerifyLostPriorCenterFactorReplay = it.coerceIn(0.5, 8.0) }
                "auto_verify_lost_prior_min_iou_live" -> value.toDoubleOrNull()?.let { autoVerifyLostPriorMinIouLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_lost_prior_min_iou_replay" -> value.toDoubleOrNull()?.let { autoVerifyLostPriorMinIouReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_lost_prior_center_boost_frames" -> value.toLongOrNull()?.let { autoVerifyLostPriorCenterBoostFrames = it.coerceIn(1L, 1200L) }
                "auto_verify_lost_prior_center_boost_cap" -> value.toDoubleOrNull()?.let { autoVerifyLostPriorCenterBoostCap = it.coerceIn(0.0, 4.0) }
                "auto_verify_lost_anchor_near_live" -> value.toDoubleOrNull()?.let { autoVerifyLostAnchorNearFactorLive = it.coerceIn(0.5, 10.0) }
                "auto_verify_lost_anchor_near_replay" -> value.toDoubleOrNull()?.let { autoVerifyLostAnchorNearFactorReplay = it.coerceIn(0.5, 10.0) }
                "auto_verify_lost_far_recover_min_good" -> value.toIntOrNull()?.let { autoVerifyLostFarRecoverMinGood = it.coerceIn(4, 120) }
                "auto_verify_lost_far_recover_min_inliers" -> value.toIntOrNull()?.let { autoVerifyLostFarRecoverMinInliers = it.coerceIn(3, 80) }
                "auto_verify_lost_far_recover_min_conf_live" -> value.toDoubleOrNull()?.let { autoVerifyLostFarRecoverMinConfLive = it.coerceIn(0.0, 1.0) }
                "auto_verify_lost_far_recover_min_conf_replay" -> value.toDoubleOrNull()?.let { autoVerifyLostFarRecoverMinConfReplay = it.coerceIn(0.0, 1.0) }
                "auto_verify_lost_far_recover_min_appearance" -> value.toDoubleOrNull()?.let { autoVerifyLostFarRecoverMinAppearance = it.coerceIn(-1.0, 1.0) }
                "auto_verify_first_lock_appear_live" ->
                    value.toDoubleOrNull()?.let { autoVerifyFirstLockAppearanceMinLive = it.coerceIn(-1.0, 1.0) }
                "auto_verify_first_lock_appear_replay" ->
                    value.toDoubleOrNull()?.let { autoVerifyFirstLockAppearanceMinReplay = it.coerceIn(-1.0, 1.0) }
                "auto_verify_first_lock_min_inliers_replay" ->
                    value.toIntOrNull()?.let { autoVerifyFirstLockMinInliersReplay = it.coerceIn(3, 20) }
                "auto_verify_first_lock_require_local_live" ->
                    parseBoolean(value)?.let { autoVerifyFirstLockRequireLocalLive = it }
                "auto_verify_first_lock_require_local_replay" ->
                    parseBoolean(value)?.let { autoVerifyFirstLockRequireLocalReplay = it }
                "auto_verify_first_lock_center_guard_replay" ->
                    parseBoolean(value)?.let { autoVerifyFirstLockCenterGuardReplay = it }
                "auto_verify_first_lock_center_factor_replay" ->
                    value.toDoubleOrNull()?.let { autoVerifyFirstLockCenterFactorReplay = it.coerceIn(0.05, 0.45) }
                "spatial_gate_enable", "spatial_gate_enabled", "mahal_gate_enable", "mahal_enabled" ->
                    parseBoolean(value)?.let { spatialGateEnabled = it }
                "spatial_gate_weight", "mahal_gate_weight", "mahal_weight" ->
                    value.toDoubleOrNull()?.let { spatialGateWeight = it.coerceIn(0.0, 0.90) }
                "spatial_gate_center_sigma", "mahal_center_sigma" ->
                    value.toDoubleOrNull()?.let { spatialGateCenterSigmaFactor = it.coerceIn(0.20, 4.0) }
                "spatial_gate_size_sigma", "mahal_size_sigma" ->
                    value.toDoubleOrNull()?.let { spatialGateSizeSigmaFactor = it.coerceIn(0.20, 4.0) }
                "spatial_gate_relock_min_live", "mahal_relock_min_live" ->
                    value.toDoubleOrNull()?.let { spatialGateRelockMinScoreLive = it.coerceIn(0.0, 1.0) }
                "spatial_gate_relock_min_replay", "mahal_relock_min_replay" ->
                    value.toDoubleOrNull()?.let { spatialGateRelockMinScoreReplay = it.coerceIn(0.0, 1.0) }
                "spatial_gate_relock_bypass_conf", "mahal_relock_bypass_conf" ->
                    value.toDoubleOrNull()?.let { spatialGateRelockBypassConf = it.coerceIn(0.0, 1.0) }
                "spatial_gate_relock_bypass_appearance", "mahal_relock_bypass_appearance" ->
                    value.toDoubleOrNull()?.let { spatialGateRelockBypassAppearance = it.coerceIn(-1.0, 1.0) }
                "auto_verify_relock_spatial_relax_streak_live" ->
                    value.toIntOrNull()?.let { autoVerifyRelockSpatialRelaxStreakLive = it.coerceIn(1, 256) }
                "auto_verify_relock_spatial_relax_streak_replay" ->
                    value.toIntOrNull()?.let { autoVerifyRelockSpatialRelaxStreakReplay = it.coerceIn(1, 512) }
                "auto_verify_relock_spatial_relax_min_score" ->
                    value.toDoubleOrNull()?.let { autoVerifyRelockSpatialRelaxMinScore = it.coerceIn(0.0, 1.0) }
                "auto_verify_relock_spatial_relax_min_appearance" ->
                    value.toDoubleOrNull()?.let { autoVerifyRelockSpatialRelaxMinAppearance = it.coerceIn(-1.0, 1.0) }
                "cand_dump_enable" ->
                    parseBoolean(value)?.let { candDumpEnable = it }
                "cand_dump_window_only" ->
                    parseBoolean(value)?.let { candDumpWindowOnly = it }
                "cand_dump_start_sec" ->
                    value.toDoubleOrNull()?.let { candDumpStartSec = it.coerceIn(0.0, 1200.0) }
                "cand_dump_end_sec" ->
                    value.toDoubleOrNull()?.let { candDumpEndSec = it.coerceIn(0.0, 1200.0) }
                "cand_dump_top_k" ->
                    value.toIntOrNull()?.let { candDumpTopK = it.coerceIn(1, 12) }
                "cand_dump_expected_x_min" ->
                    value.toDoubleOrNull()?.let { candDumpExpectedXMin = it.coerceIn(0.0, 1.0) }
                "cand_dump_expected_x_max" ->
                    value.toDoubleOrNull()?.let { candDumpExpectedXMax = it.coerceIn(0.0, 1.0) }
                "s2_suppress_kcf_fallback", "native_init_retry_no_kcf" ->
                    parseBoolean(value)?.let { s2SuppressKcfFallbackEnabled = it }
                "s3_promoted_near_gate", "s3_near_gate" ->
                    parseBoolean(value)?.let { s3PromotedNearGateEnabled = it }
                "s3_promoted_far_gate", "s3_far_gate" ->
                    parseBoolean(value)?.let { s3PromotedFarGateEnabled = it }
                "s3_promoted_near_anchor_veto", "s3_near_anchor_veto" ->
                    parseBoolean(value)?.let { s3PromotedNearAnchorVetoEnabled = it }
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
                "track_guard_accel_grace", "track_guard_accel_grace_frames" ->
                    value.toIntOrNull()?.let { trackGuardAccelGraceFrames = it.coerceIn(0, 30) }
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
                "small_target_verify_interval", "small_target_native_verify_interval" ->
                    value.toIntOrNull()?.let { smallTargetNativeVerifyIntervalFrames = it.coerceIn(1, 30) }
                "small_target_verify_area_scale", "small_target_native_verify_area_scale" ->
                    value.toDoubleOrNull()?.let { smallTargetNativeVerifyAreaScale = it.coerceIn(0.20, 4.0) }
                "small_target_anchor_drop_scale" ->
                    value.toDoubleOrNull()?.let { smallTargetAnchorDropScale = it.coerceIn(0.30, 1.0) }
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
                "native_gate_use_measurement", "native_use_measurement_gate", "native_gate_use_meas" ->
                    parseBoolean(value)?.let { nativeGateUseMeasurement = it }
                "native_gap_passthrough", "gap_passthrough" ->
                    parseBoolean(value)?.let { nativeGapPassthrough = it }
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
        smallTargetNativeVerifyIntervalFrames = smallTargetNativeVerifyIntervalFrames.coerceIn(1, 30)
        smallTargetNativeVerifyAreaScale = smallTargetNativeVerifyAreaScale.coerceIn(0.20, 4.0)
        smallTargetAnchorDropScale = smallTargetAnchorDropScale.coerceIn(0.30, 1.0)
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
        autoVerifyLocalMissingMinConfLive = autoVerifyLocalMissingMinConfLive.coerceIn(0.0, 1.0)
        autoVerifyLocalMissingMinConfReplay = autoVerifyLocalMissingMinConfReplay.coerceIn(0.0, 1.0)
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
        if (autoVerifyAppearanceMinStrong < autoVerifyAppearanceMinMedium) {
            autoVerifyAppearanceMinStrong = autoVerifyAppearanceMinMedium
        }
        if (autoVerifyAppearanceMinMedium < autoVerifyAppearanceMinBase) {
            autoVerifyAppearanceMinMedium = autoVerifyAppearanceMinBase
        }
        autoVerifyRelockMinConfLive = autoVerifyRelockMinConfLive.coerceIn(0.0, 1.0)
        autoVerifyRelockMinConfReplay = autoVerifyRelockMinConfReplay.coerceIn(0.0, 1.0)
        autoVerifyLostPriorMaxFramesLive = autoVerifyLostPriorMaxFramesLive.coerceIn(1L, 600L)
        autoVerifyLostPriorMaxFramesReplay = autoVerifyLostPriorMaxFramesReplay.coerceIn(1L, 900L)
        autoVerifyLostPriorCenterFactorLive = autoVerifyLostPriorCenterFactorLive.coerceIn(0.5, 8.0)
        autoVerifyLostPriorCenterFactorReplay = autoVerifyLostPriorCenterFactorReplay.coerceIn(0.5, 8.0)
        autoVerifyLostPriorMinIouLive = autoVerifyLostPriorMinIouLive.coerceIn(0.0, 1.0)
        autoVerifyLostPriorMinIouReplay = autoVerifyLostPriorMinIouReplay.coerceIn(0.0, 1.0)
        autoVerifyLostPriorCenterBoostFrames = autoVerifyLostPriorCenterBoostFrames.coerceIn(1L, 1200L)
        autoVerifyLostPriorCenterBoostCap = autoVerifyLostPriorCenterBoostCap.coerceIn(0.0, 4.0)
        autoVerifyLostAnchorNearFactorLive = autoVerifyLostAnchorNearFactorLive.coerceIn(0.5, 10.0)
        autoVerifyLostAnchorNearFactorReplay = autoVerifyLostAnchorNearFactorReplay.coerceIn(0.5, 10.0)
        autoVerifyLostFarRecoverMinGood = autoVerifyLostFarRecoverMinGood.coerceIn(4, 120)
        autoVerifyLostFarRecoverMinInliers = autoVerifyLostFarRecoverMinInliers.coerceIn(3, 80)
        autoVerifyLostFarRecoverMinConfLive = autoVerifyLostFarRecoverMinConfLive.coerceIn(0.0, 1.0)
        autoVerifyLostFarRecoverMinConfReplay = autoVerifyLostFarRecoverMinConfReplay.coerceIn(0.0, 1.0)
        autoVerifyLostFarRecoverMinAppearance = autoVerifyLostFarRecoverMinAppearance.coerceIn(-1.0, 1.0)
        autoVerifyFirstLockAppearanceMinLive = autoVerifyFirstLockAppearanceMinLive.coerceIn(-1.0, 1.0)
        autoVerifyFirstLockAppearanceMinReplay = autoVerifyFirstLockAppearanceMinReplay.coerceIn(-1.0, 1.0)
        autoVerifyFirstLockMinInliersReplay = autoVerifyFirstLockMinInliersReplay.coerceIn(3, 20)
        autoVerifyFirstLockCenterFactorReplay = autoVerifyFirstLockCenterFactorReplay.coerceIn(0.05, 0.45)
        spatialGateWeight = spatialGateWeight.coerceIn(0.0, 0.90)
        spatialGateCenterSigmaFactor = spatialGateCenterSigmaFactor.coerceIn(0.20, 4.0)
        spatialGateSizeSigmaFactor = spatialGateSizeSigmaFactor.coerceIn(0.20, 4.0)
        spatialGateRelockMinScoreLive = spatialGateRelockMinScoreLive.coerceIn(0.0, 1.0)
        spatialGateRelockMinScoreReplay = spatialGateRelockMinScoreReplay.coerceIn(0.0, 1.0)
        spatialGateRelockBypassConf = spatialGateRelockBypassConf.coerceIn(0.0, 1.0)
        spatialGateRelockBypassAppearance = spatialGateRelockBypassAppearance.coerceIn(-1.0, 1.0)
        autoVerifyRelockSpatialRelaxStreakLive = autoVerifyRelockSpatialRelaxStreakLive.coerceIn(1, 256)
        autoVerifyRelockSpatialRelaxStreakReplay = autoVerifyRelockSpatialRelaxStreakReplay.coerceIn(1, 512)
        autoVerifyRelockSpatialRelaxMinScore = autoVerifyRelockSpatialRelaxMinScore.coerceIn(0.0, 1.0)
        autoVerifyRelockSpatialRelaxMinAppearance = autoVerifyRelockSpatialRelaxMinAppearance.coerceIn(-1.0, 1.0)
        candDumpStartSec = candDumpStartSec.coerceIn(0.0, 1200.0)
        candDumpEndSec = candDumpEndSec.coerceIn(0.0, 1200.0)
        if (candDumpEndSec < candDumpStartSec) {
            val tmp = candDumpStartSec
            candDumpStartSec = candDumpEndSec
            candDumpEndSec = tmp
        }
        candDumpTopK = candDumpTopK.coerceIn(1, 12)
        candDumpExpectedXMin = candDumpExpectedXMin.coerceIn(0.0, 1.0)
        candDumpExpectedXMax = candDumpExpectedXMax.coerceIn(0.0, 1.0)
        if (candDumpExpectedXMax < candDumpExpectedXMin) {
            val tmp = candDumpExpectedXMin
            candDumpExpectedXMin = candDumpExpectedXMax
            candDumpExpectedXMax = tmp
        }
        centerRoiL1Range = centerRoiL1Range.coerceIn(0.05, 0.45)
        centerRoiL2Range = centerRoiL2Range.coerceIn(0.10, 0.49)
        if (centerRoiL2Range <= centerRoiL1Range) {
            centerRoiL2Range = (centerRoiL1Range + 0.05).coerceAtMost(0.49)
        }
        centerRoiL1TimeoutMs = centerRoiL1TimeoutMs.coerceIn(100L, 5_000L)
        centerRoiL2TimeoutMs = centerRoiL2TimeoutMs.coerceIn(100L, 5_000L)
        centerRoiL3TimeoutMs = centerRoiL3TimeoutMs.coerceIn(200L, 10_000L)
        descendExplosionAreaRatio = descendExplosionAreaRatio.coerceIn(0.001, 0.95)
        descendExplosionReleaseRatio = descendExplosionReleaseRatio.coerceIn(0.001, 0.90)
        if (descendExplosionReleaseRatio >= descendExplosionAreaRatio) {
            descendExplosionReleaseRatio = (descendExplosionAreaRatio - 0.05).coerceAtLeast(0.05)
        }
        descendExplosionCoreSizePx = descendExplosionCoreSizePx.coerceIn(MIN_BOX_DIM, 1600)
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
        if (!centerRoiSearchEnabled || !centerRoiGpsReady) {
            resetCenterRoiSearchState("override_sync")
        }

        configureOrbDetector(orbMaxFeatures)
        refreshHeuristicConfig()
        if (shouldRefreshTemplate && templateSourceGrays.isNotEmpty()) {
            rebuildTemplatePyramid(templateSourceGrays)
        }
        logEffectiveParams("override")
    }

    fun setReplayTargetAppearSec(seconds: Double) {
        replayTargetAppearMs =
            if (seconds > 0.0) {
                (seconds * 1000.0).toLong().coerceAtLeast(0L)
            } else {
                -1L
            }
    }

    fun isCenterRoiGpsReady(): Boolean = centerRoiGpsReady

    fun setCenterRoiGpsReady(ready: Boolean, reason: String = "external") {
        if (centerRoiGpsReady == ready && !(ready && centerRoiFailLatched)) return
        centerRoiGpsReady = ready
        centerRoiFailLatched = false
        centerRoiFailEmittedThisSession = false
        resetCenterRoiSearchState(if (ready) "gps_ready" else "gps_not_ready")
        resetDescendOffsetState(if (ready) "gps_ready" else "gps_not_ready")
        resetDescendExplosionState(if (ready) "gps_ready" else "gps_not_ready")
        Log.w(
            TAG,
            "EVAL_EVENT type=ROI_SEARCH state=gps_ready ready=$ready reason=$reason enabled=$centerRoiSearchEnabled"
        )
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
                localMissingMinConfLive = autoVerifyLocalMissingMinConfLive,
                localMissingMinConfReplay = autoVerifyLocalMissingMinConfReplay,
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
                "hDet=${fmt(homographyMinJacobianDet)} tplTextureMin=${fmt(templateMinTextureScore)} tplPoseAug=$templatePoseAugmentEnabled clahe=$orbUseClahe " +
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
                "autoVerifyMissingConf=${fmt(cfg.autoInitVerify.localMissingMinConfLive)}/${fmt(cfg.autoInitVerify.localMissingMinConfReplay)} " +
                "autoVerifyLocal=${fmt(cfg.autoInitVerify.localCenterFactorLive)}/${fmt(cfg.autoInitVerify.localCenterFactorReplay)}@${fmt(cfg.autoInitVerify.localMinIouLive)}/${fmt(cfg.autoInitVerify.localMinIouReplay)} " +
                "autoVerifyAppear=${fmt(cfg.autoInitVerify.appearanceMinStrong)}/${fmt(cfg.autoInitVerify.appearanceMinMedium)}/${fmt(cfg.autoInitVerify.appearanceMinBase)}+${fmt(cfg.autoInitVerify.appearanceLiveBias)} " +
                "autoVerifyRejectFlipReplay=$autoVerifyRejectFlipReplay " +
                "autoVerifyRelockConf=${fmt(autoVerifyRelockMinConfLive)}/${fmt(autoVerifyRelockMinConfReplay)} " +
                "autoVerifyLostPriorFrames=${autoVerifyLostPriorMaxFramesLive}/${autoVerifyLostPriorMaxFramesReplay} " +
                "autoVerifyLostPriorCenter=${fmt(autoVerifyLostPriorCenterFactorLive)}/${fmt(autoVerifyLostPriorCenterFactorReplay)} " +
                "autoVerifyLostPriorIou=${fmt(autoVerifyLostPriorMinIouLive)}/${fmt(autoVerifyLostPriorMinIouReplay)} " +
                "autoVerifyLostAnchorNear=${fmt(autoVerifyLostAnchorNearFactorLive)}/${fmt(autoVerifyLostAnchorNearFactorReplay)} " +
                "autoVerifyLostFarRecover=${autoVerifyLostFarRecoverMinGood}/${autoVerifyLostFarRecoverMinInliers}/" +
                "${fmt(autoVerifyLostFarRecoverMinConfLive)}/${fmt(autoVerifyLostFarRecoverMinConfReplay)}/" +
                "${fmt(autoVerifyLostFarRecoverMinAppearance)} " +
                "autoVerifyFirstLockAppear=${fmt(autoVerifyFirstLockAppearanceMinLive)}/${fmt(autoVerifyFirstLockAppearanceMinReplay)} " +
                "autoVerifyFirstLockMinInliersReplay=$autoVerifyFirstLockMinInliersReplay " +
                "autoVerifyFirstLockRequireLocal=$autoVerifyFirstLockRequireLocalLive/$autoVerifyFirstLockRequireLocalReplay " +
                "autoVerifyFirstLockCenterReplay=$autoVerifyFirstLockCenterGuardReplay/${fmt(autoVerifyFirstLockCenterFactorReplay)} " +
                "spatialGate=$spatialGateEnabled/${fmt(spatialGateWeight)} " +
                "spatialSigma=${fmt(spatialGateCenterSigmaFactor)}/${fmt(spatialGateSizeSigmaFactor)} " +
                "spatialRelock=${fmt(spatialGateRelockMinScoreLive)}/${fmt(spatialGateRelockMinScoreReplay)} " +
                "spatialBypass=${fmt(spatialGateRelockBypassConf)}/${fmt(spatialGateRelockBypassAppearance)} " +
                "spatialRelax=${autoVerifyRelockSpatialRelaxStreakLive}/${autoVerifyRelockSpatialRelaxStreakReplay}/${fmt(autoVerifyRelockSpatialRelaxMinScore)}/${fmt(autoVerifyRelockSpatialRelaxMinAppearance)} " +
                "centerRoi=$centerRoiSearchEnabled/$centerRoiGpsReady level=${centerRoiLevel.name.lowercase(Locale.US)} " +
                "centerRoiRange=${fmt(centerRoiL1Range)}/${fmt(centerRoiL2Range)} " +
                "centerRoiTimeoutMs=${centerRoiL1TimeoutMs}/${centerRoiL2TimeoutMs}/${centerRoiL3TimeoutMs} " +
                "descendExplosion=$descendExplosionGuardEnabled/${fmt(descendExplosionAreaRatio)}/${fmt(descendExplosionReleaseRatio)}/$descendExplosionCoreSizePx " +
                "candDump=$candDumpEnable/$candDumpWindowOnly ${fmt(candDumpStartSec)}-${fmt(candDumpEndSec)} topK=$candDumpTopK " +
                "candDumpExpectedX=${fmt(candDumpExpectedXMin)}:${fmt(candDumpExpectedXMax)} " +
                "s2NoKcfFallback=$s2SuppressKcfFallbackEnabled " +
                "s3NearGate=$s3PromotedNearGateEnabled s3FarGate=$s3PromotedFarGateEnabled s3NearAnchorVeto=$s3PromotedNearAnchorVetoEnabled " +
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
                "trackGuardAccelGrace=$trackGuardAccelGraceFrames " +
                "trackGuardAppearInt=${cfg.trackGuard.appearanceCheckIntervalFrames} " +
                "trackGuardAppearMin=${fmt(cfg.trackGuard.minAppearanceScore)} " +
                "trackGuardAnchor=$trackGuardAnchorEnabled/${fmt(trackGuardAnchorMaxDrop)}/${fmt(trackGuardAnchorMinScore)} " +
                "smallTargetVerifyInt=$smallTargetNativeVerifyIntervalFrames smallTargetVerifyAreaScale=${fmt(smallTargetNativeVerifyAreaScale)} smallTargetAnchorDropScale=${fmt(smallTargetAnchorDropScale)} " +
                "trackBackend=${preferredTrackBackend.name.lowercase(Locale.US)} " +
                "kcfFail=$kcfMaxFailStreak nativeFail=${cfg.nativeGate.maxFailStreak} nativeMinConf=${fmt(cfg.nativeGate.minConfidence)} " +
                "nativeFuseSoft=${fmt(cfg.nativeGate.fuseSoftConfidence)} nativeFuseHard=${fmt(cfg.nativeGate.fuseHardConfidence)} " +
                "nativeFuseStreak=${cfg.nativeGate.fuseSoftStreak} nativeFuseWarmup=${cfg.nativeGate.fuseWarmupFrames} " +
                "nativeHoldLast=${cfg.nativeGate.holdLastOnSoftReject} nativeGateUseMeasurement=$nativeGateUseMeasurement " +
                "nativeGapPassthrough=$nativeGapPassthrough nativeOrbVerify=$nativeOrbVerifyEnabled nativeScoreLogInt=$nativeScoreLogIntervalFrames " +
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
        logDiag("MANUAL_ROI_LIFECYCLE", "session=$diagSessionId trigger=setInitialTarget_enter manualActive=$manualRoiActive")
        val sx = if (scaleX > 0f) scaleX else 1f
        val sy = if (scaleY > 0f) scaleY else 1f
        val mapped = Rect(
            ((viewRect.left - offsetX) / sx).toInt(),
            ((viewRect.top - offsetY) / sy).toInt(),
            (viewRect.width() / sx).toInt(),
            (viewRect.height() / sy).toInt()
        )
        val frameW = currentFrameWidth
        val frameH = currentFrameHeight
        val mappedSafe = clampRect(mapped, frameW, frameH)
        val clamped = mappedSafe != null && (
            mappedSafe.x != mapped.x ||
                mappedSafe.y != mapped.y ||
                mappedSafe.width != mapped.width ||
                mappedSafe.height != mapped.height
            )
        val target = mappedSafe ?: mapped
        manualRoiBboxClamped = clamped
        Log.d(
            TAG,
            "manual target selected: overlay=${viewRect.width().toInt()}x${viewRect.height().toInt()} view=${viewWidth}x${viewHeight} " +
                "mapped=${mapped.x},${mapped.y},${mapped.width}x${mapped.height} " +
                "safe=${target.x},${target.y},${target.width}x${target.height} frame=${frameW}x${frameH} clamped=$clamped"
        )
        val refreshOk = refreshManualTemplateFromLiveFrame(target, viewWidth, viewHeight, clamped)
        if (!refreshOk) {
            pendingInitBox = null
            manualRoiInitPath = "fallback_disk"
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=fallback_forbidden init_path=$manualRoiInitPath " +
                    "bbox_clamped=$clamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)} " +
                    "box=${target.x},${target.y},${target.width}x${target.height}"
            )
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI state=clear reason=fallback_forbidden init_path=$manualRoiInitPath " +
                    "bbox_clamped=$clamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)}"
            )
            clearManualRoiState("manual_init_failed")
            return
        }
        pendingInitBox = target
    }

    fun setTemplateImage(bitmap: Bitmap): Boolean = setTemplateImages(listOf(bitmap), source = "disk")

    private fun refreshManualTemplateFromLiveFrame(
        requestedBox: Rect,
        viewWidth: Int,
        viewHeight: Int,
        bboxClamped: Boolean
    ): Boolean {
        val frameGray: Mat
        val frameRgb: Mat
        val frameTs: Long
        synchronized(liveFrameCacheLock) {
            val cachedGray = lastLiveFrameGray
            val cachedRgb = lastLiveFrameRgb
            if (cachedGray == null || cachedGray.empty()) {
                Log.w(TAG, "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=no_live_frame detail=gray")
                return false
            }
            if (cachedRgb == null || cachedRgb.empty()) {
                Log.w(TAG, "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=no_live_frame detail=rgb")
                return false
            }
            frameGray = cachedGray.clone()
            frameRgb = cachedRgb.clone()
            frameTs = lastLiveFrameTsMs
        }
        try {
            val safe = clampRect(requestedBox, frameGray.cols(), frameGray.rows())
            if (safe == null) {
                Log.w(TAG, "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=box_oob")
                return false
            }
            val effectiveClamped =
                bboxClamped ||
                    safe.x != requestedBox.x ||
                    safe.y != requestedBox.y ||
                    safe.width != requestedBox.width ||
                    safe.height != requestedBox.height
            if (safe.width < 32 || safe.height < 32) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=box_small " +
                        "box=${safe.x},${safe.y},${safe.width}x${safe.height}"
                )
                return false
            }

            val patch = frameGray.submat(safe)
            val patchRgba = Mat()
            val patchBitmap: Bitmap
            try {
                when (patch.channels()) {
                    1 -> Imgproc.cvtColor(patch, patchRgba, Imgproc.COLOR_GRAY2RGBA)
                    3 -> Imgproc.cvtColor(patch, patchRgba, Imgproc.COLOR_RGB2RGBA)
                    4 -> patch.copyTo(patchRgba)
                    else -> {
                        Log.w(TAG, "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=patch_channel_unsupported ch=${patch.channels()}")
                        return false
                    }
                }
                patchBitmap = Bitmap.createBitmap(patchRgba.cols(), patchRgba.rows(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(patchRgba, patchBitmap)
            } catch (t: Throwable) {
                Log.w(TAG, "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=patch_bitmap_error", t)
                return false
            } finally {
                patch.release()
                patchRgba.release()
            }

            val templateReady =
                try {
                    setTemplateImages(listOf(patchBitmap), source = "manual_live")
                } finally {
                    if (!patchBitmap.isRecycled) patchBitmap.recycle()
                }
            manualRoiPatchKp = templateKeypointCount
            manualRoiPatchTexture = templateTextureScore
            if (!templateReady) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=template_rebuild_failed init_path=live " +
                        "bbox_clamped=$effectiveClamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)}"
                )
                return false
            }

            val nativeReady = warmupNativeManualInit(frameRgb, safe)
            if (!nativeReady) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=native_init_failed init_path=live " +
                        "bbox_clamped=$effectiveClamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)}"
                )
                return false
            }

            manualRoiActive = true
            manualRoiSessionActive = true
            manualRoiFrameTsMs = frameTs
            manualRoiBboxClamped = effectiveClamped
            manualRoiInitPath = "live"
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI_INIT_OK tsMs=$frameTs init_path=$manualRoiInitPath " +
                    "bbox_clamped=$effectiveClamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)} " +
                    "box=${safe.x},${safe.y},${safe.width}x${safe.height} view=${viewWidth}x${viewHeight}"
            )
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI state=active tsMs=$frameTs init_path=$manualRoiInitPath " +
                    "bbox_clamped=$effectiveClamped patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)} " +
                    "box=${safe.x},${safe.y},${safe.width}x${safe.height}"
            )
            return true
        } finally {
            frameGray.release()
            frameRgb.release()
        }
    }

    private fun warmupNativeManualInit(frameRgb: Mat, box: Rect): Boolean {
        val nativeBackend = when (preferredTrackBackend) {
            TrackBackend.NATIVE_RKNN -> NativeTrackerBridge.Backend.RKNN
            else -> NativeTrackerBridge.Backend.NCNN
        }
        val requestedParam = nativeModelParamPathOverride
        val requestedBin = nativeModelBinPathOverride
        val initEngineOk = NativeTrackerBridge.initializeEngine(nativeBackend, requestedParam, requestedBin)
        if (!initEngineOk || !NativeTrackerBridge.isAvailable()) {
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI_INIT_FAIL reason=native_engine_unavailable " +
                    "backend=${nativeBackend.name.lowercase(Locale.US)} " +
                    "param=${summarizePath(requestedParam)} bin=${summarizePath(requestedBin)}"
            )
            return false
        }
        val nativeBox = android.graphics.Rect(box.x, box.y, box.x + box.width, box.y + box.height)
        val bytes = ByteArray(frameRgb.cols() * frameRgb.rows() * 3)
        frameRgb.get(0, 0, bytes)
        return NativeTrackerBridge.initTargetGray(bytes, frameRgb.cols(), frameRgb.rows(), nativeBox)
    }

    fun setTemplateImages(bitmaps: List<Bitmap>, source: String = "unknown"): Boolean {
        if (bitmaps.isEmpty()) {
            clearTemplateSources()
            clearTemplateFeatures()
            templateGray?.release()
            templateGray = null
            lastTemplateReadyState = false
            return false
        }

        val pendingSources = ArrayList<Mat>(bitmaps.size)
        val enablePoseAugment = templatePoseAugmentEnabled && bitmaps.size == 1
        try {
            for (bitmap in bitmaps) {
                val rgba = Mat()
                val gray = Mat()
                try {
                    Utils.bitmapToMat(bitmap, rgba)
                    Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
                    val normalized = normalizeTemplateSize(gray)
                    try {
                        pendingSources.addAll(buildPoseAugmentedTemplates(normalized, enablePoseAugment))
                    } finally {
                        normalized.release()
                    }
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
            if (enablePoseAugment && templateLibrarySize > 1) {
                logDiag("TEMPLATE", "session=$diagSessionId action=pose_augment variants=$templateLibrarySize")
            }

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
                    "EVAL_EVENT type=TEMPLATE_WEAK source=$source kp=$templateKeypointCount texture=${fmt(templateTextureScore)} " +
                        "textureMin=${fmt(templateMinTextureScore)} templates=$templateLibrarySize"
                )
            }
            lastTemplateReadyState = ok
            clearFirstLockCandidate("template_changed")
            searchMissStreak = 0

            resetTracking(logSummary = false, trigger = "template_changed")
            pendingInitBox = null
            val first = templateSourceGrays.firstOrNull()
            val firstW = first?.cols() ?: 0
            val firstH = first?.rows() ?: 0
            Log.i(TAG, "template loaded: ${templateLibrarySize} templates first=${firstW}x${firstH}")
            Log.w(
                TAG,
                "EVAL_EVENT type=TEMPLATE_READY source=$source templates=$templateLibrarySize width=$firstW height=$firstH " +
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

    fun resetTracking(logSummary: Boolean = true, trigger: String = "unknown") {
        val resolvedTrigger =
            if (trigger != "unknown") {
                trigger
            } else {
                val caller = Throwable().stackTrace.getOrNull(1)
                if (caller != null) "${caller.className}.${caller.methodName}" else "unknown_caller"
            }
        if (logSummary) {
            logEvalSummary("reset")
        }
        logDiag("MANUAL_ROI_LIFECYCLE", "session=$diagSessionId trigger=resetTracking:$resolvedTrigger")
        logDiag("TRACK", "session=$diagSessionId action=reset_tracking trigger=$resolvedTrigger")
        releaseTracker("reset", requestGc = true, callerTrigger = "resetTracking:$resolvedTrigger")
        pendingInitBox = null
        lastTrackedBox = null
        lastMeasuredTrackBox = null
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackAccelGraceFramesRemaining = 0
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        trackGuardHardMismatch = false
        searchMissStreak = 0
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = 0
        fastFirstLockRemaining = fastFirstLockFrames
        lastNativeAcceptMs = 0L
        relockSpatialRejectStreak = 0
        trackAnchorAppearanceScore = Double.NaN
        lastLostBox = null
        lastLostFrameId = -1L
        centerRoiFailEmittedThisSession = false
        centerRoiFailLatched = false
        resetCenterRoiSearchState("reset_tracking")
        resetDescendOffsetState("reset_tracking")
        resetDescendExplosionState("reset_tracking")
        clearManualRoiState("reset_tracking")
        clearLiveFrameCache()
        overlayResetToken++
        clearFirstLockCandidate("reset")
        latestSearchFrame?.release()
        latestSearchFrame = null
        resetKalman("reset_tracking")
        updateLatestPrediction(null, 0.0, tracking = false)
        overlayView.post {
            clearManualRoiState("overlay_reset_reset_tracking")
            overlayView.reset()
        }
    }

    private fun releaseTracker(reason: String, requestGc: Boolean, callerTrigger: String = "unknown") {
        logDiag(
            "MANUAL_ROI_LIFECYCLE",
            "session=$diagSessionId trigger=releaseTracker:$callerTrigger reason=$reason manualActive=$manualRoiActive"
        )
        logDiag(
            "TRACK",
            "session=$diagSessionId action=release reason=$reason requestGc=$requestGc " +
                "trigger=$callerTrigger isTracking=$isTracking backend=${activeTrackBackend.name.lowercase(Locale.US)}"
        )
        val hadTracker = tracker != null
        val hadNativeTracking = isTracking && activeTrackBackend != TrackBackend.KCF
        tracker = null
        if (hadNativeTracking) {
            NativeTrackerBridge.reset()
            val keepManualRoi =
                reason.startsWith("reinit_manual") ||
                    reason.startsWith("reinit_manual_roi")
            if (!keepManualRoi) {
                clearManualRoiState("native_reset_$reason")
            }
        }
        isTracking = false
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackAccelGraceFramesRemaining = 0
        activeTrackBackend = TrackBackend.KCF
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lastNativeAcceptMs = 0L
        resetDescendExplosionState("tracker_release")
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

    @Synchronized
    fun beginEvalSession(reason: String = "manual") {
        metricsSessionStartMs = SystemClock.elapsedRealtime()
        diagSessionId = metricsSessionStartMs
        frameCounter = 0L
        currentReplayPtsMs = -1L

        metricsFrames = 0L
        metricsTotalProcessMs = 0L
        metricsTrackingFrames = 0L
        metricsSearchingFrames = 0L
        metricsLockCount = 0
        metricsLostCount = 0
        metricsFirstLockMs = -1L
        metricsFirstLockReplayMs = -1L
        metricsFirstLockFrame = -1L
        metricsCurrentTrackingStreak = 0
        metricsMaxTrackingStreak = 0
        metricsSearchCandidateCount = 0
        metricsSearchMissCount = 0
        metricsSearchTemplateSkipCount = 0
        metricsSearchStrideSkipCount = 0
        metricsSearchBudgetSkipCount = 0
        metricsSearchBudgetTripCount = 0
        metricsSearchTemporalRejectCount = 0
        metricsSearchPromoteRejectCount = 0
        metricsSearchRefinePassCount = 0
        metricsSearchRefineRejectCount = 0
        metricsSearchStableSeedCount = 0
        metricsSearchStableAccumCount = 0
        metricsSearchPromoteCount = 0
        metricsSearchStableOutlierHoldCount = 0
        metricsSearchTemporalHoldCount = 0
        metricsSearchResetNoSeedCount = 0
        metricsSearchResetGapCount = 0
        metricsSearchResetStableDriftCount = 0
        metricsSearchResetCenterRuleCount = 0
        metricsSearchResetIouCount = 0
        metricsSearchResetConfidenceCount = 0
        metricsNativeFuseSoftCount = 0
        relockSpatialRejectStreak = 0
        metricsNativeFuseHardCount = 0
        metricsNativeLowConfHoldCount = 0
        metricsKalmanPredHoldCount = 0
        metricsNativeConfSamples = 0L
        metricsNativeConfSum = 0.0
        metricsNativeConfMin = 1.0
        metricsNativeConfMax = 0.0
        metricsNativeSimSamples = 0L
        metricsNativeSimSum = 0.0
        metricsNativeSimMin = 1.0
        metricsNativeSimMax = 0.0
        metricsSearchLastReason = "none"
        centerRoiFailEmittedThisSession = false
        centerRoiFailLatched = false
        clearManualRoiState("session_start")
        clearLiveFrameCache()
        resetCenterRoiSearchState("session_start")
        resetDescendOffsetState("session_start")
        resetDescendExplosionState("session_start")
        Log.w(TAG, "EVAL_EVENT type=SESSION_START reason=$reason session=$diagSessionId")
    }

    fun logEvalSummary(reason: String = "manual") {
        val elapsed = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(1L)
        val avgMs = if (metricsFrames > 0) metricsTotalProcessMs.toDouble() / metricsFrames else 0.0
        val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
        val firstLockReplaySec = if (metricsFirstLockReplayMs >= 0L) metricsFirstLockReplayMs.toDouble() / 1000.0 else -1.0
        val replayPtsSec = if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0
        val trackingRatio = if (metricsFrames > 0) metricsTrackingFrames.toDouble() / metricsFrames else 0.0
        Log.w(
            TAG,
            String.format(
                Locale.US,
                "EVAL_SUMMARY reason=%s mode=%s elapsedMs=%d replayPtsSec=%.3f frames=%d avgFrameMs=%.2f locks=%d lost=%d firstLockSec=%.3f firstLockReplaySec=%.3f firstLockFrame=%d trackRatio=%.3f maxTrackStreak=%d searchSkipStride=%d searchSkipBudget=%d budgetTrips=%d nativeFuseSoft=%d nativeFuseHard=%d nativeHold=%d kalmanPredHold=%d",
                reason,
                trackerMode.name.lowercase(Locale.US),
                elapsed,
                replayPtsSec,
                metricsFrames,
                avgMs,
                metricsLockCount,
                metricsLostCount,
                firstLockSec,
                firstLockReplaySec,
                metricsFirstLockFrame,
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
            currentReplayPtsMs = -1L
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

    fun analyzeReplayFrame(frame: Mat, replayPtsMs: Long = -1L) {
        val startMs = SystemClock.elapsedRealtime()
        try {
            isReplayInput = true
            currentReplayPtsMs = replayPtsMs
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
        cacheLiveFrame(frame)
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

    private fun isCenterRoiSearchActive(): Boolean {
        return centerRoiSearchEnabled && centerRoiGpsReady && !centerRoiFailLatched && !isManualRoiSessionActive()
    }

    private fun resolveDescendSearchState(): DescendOffsetState {
        return when (centerRoiLevel) {
            CenterRoiLevel.L1 -> DescendOffsetState.L1
            CenterRoiLevel.L2 -> DescendOffsetState.L2
            CenterRoiLevel.L3 -> DescendOffsetState.L3
        }
    }

    private fun cacheLiveFrame(frame: Mat) {
        val grayClone = Mat()
        if (frame.channels() == 1) {
            frame.copyTo(grayClone)
        } else {
            Imgproc.cvtColor(frame, grayClone, Imgproc.COLOR_RGB2GRAY)
        }
        val rgbClone = Mat()
        if (frame.channels() == 3) {
            frame.copyTo(rgbClone)
        } else {
            Imgproc.cvtColor(frame, rgbClone, Imgproc.COLOR_GRAY2RGB)
        }
        var oldGray: Mat? = null
        var oldRgb: Mat? = null
        synchronized(liveFrameCacheLock) {
            oldGray = lastLiveFrameGray
            oldRgb = lastLiveFrameRgb
            lastLiveFrameGray = grayClone
            lastLiveFrameRgb = rgbClone
            lastLiveFrameTsMs = currentTimelineMs()
        }
        oldGray?.release()
        oldRgb?.release()
    }

    private fun clearLiveFrameCache() {
        var oldGray: Mat? = null
        var oldRgb: Mat? = null
        synchronized(liveFrameCacheLock) {
            oldGray = lastLiveFrameGray
            oldRgb = lastLiveFrameRgb
            lastLiveFrameGray = null
            lastLiveFrameRgb = null
            lastLiveFrameTsMs = 0L
        }
        oldGray?.release()
        oldRgb?.release()
    }

    private fun clearManualRoiState(reason: String) {
        val hadActive = manualRoiActive || manualRoiFrameTsMs > 0L
        if (hadActive) {
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI state=clear reason=$reason tsMs=$manualRoiFrameTsMs " +
                    "init_path=$manualRoiInitPath bbox_clamped=$manualRoiBboxClamped " +
                    "patch_kp=$manualRoiPatchKp patch_texture=${fmt(manualRoiPatchTexture)}"
            )
        }
        manualRoiActive = false
        manualRoiFrameTsMs = 0L
        val clearSession =
            reason == "manual_init_failed" ||
                reason == "session_start" ||
                reason == "reset_tracking"
        if (clearSession && manualRoiSessionActive) {
            Log.w(
                TAG,
                "EVAL_EVENT type=MANUAL_ROI state=session_clear reason=$reason"
            )
            manualRoiSessionActive = false
        }
        manualRoiPatchKp = -1
        manualRoiPatchTexture = Double.NaN
        manualRoiBboxClamped = false
        manualRoiInitPath = "none"
    }

    private fun isManualRoiSessionActive(): Boolean = manualRoiActive || manualRoiSessionActive

    private fun resolveDescendTimestampSec(): Double {
        return currentTimelineMs().toDouble() / 1000.0
    }

    private fun currentTimelineMs(): Long {
        if (isReplayInput && currentReplayPtsMs >= 0L) {
            return currentReplayPtsMs
        }
        return SystemClock.elapsedRealtime()
    }

    private fun resolveDescendOffset(box: Rect): Pair<Double, Double>? {
        val frameW = currentFrameWidth
        val frameH = currentFrameHeight
        if (frameW <= 1 || frameH <= 1) return null
        val halfW = frameW * 0.5
        val halfH = frameH * 0.5
        val centerX = box.x + box.width * 0.5
        val centerY = box.y + box.height * 0.5
        val normX = (centerX - halfW) / halfW
        val normY = (centerY - halfH) / halfH
        return normX to normY
    }

    private fun resetDescendExplosionState(reason: String) {
        if (descendExplosionGuardActive && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(TAG, "EVAL_EVENT type=DESCEND_GUARD state=reset reason=$reason session=$diagSessionId")
        }
        descendExplosionGuardActive = false
        descendExplosionLastAreaRatio = 0.0
    }

    private fun applyDescendExplosionGuard(
        box: Rect,
        frameW: Int,
        frameH: Int,
        source: String
    ): DescendExplosionGuardDecision {
        val frameArea = (frameW.toDouble() * frameH.toDouble()).coerceAtLeast(1.0)
        val areaRatio =
            ((box.width.toDouble().coerceAtLeast(1.0) * box.height.toDouble().coerceAtLeast(1.0)) / frameArea)
                .coerceIn(0.0, 1.0)

        if (!descendExplosionGuardEnabled || !centerRoiGpsReady) {
            if (descendExplosionGuardActive && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=DESCEND_GUARD state=release reason=disabled src=$source " +
                        "areaRatio=${fmt(areaRatio)} threshold=${fmt(descendExplosionAreaRatio)}"
                )
            }
            descendExplosionGuardActive = false
            descendExplosionLastAreaRatio = areaRatio
            return DescendExplosionGuardDecision(box, false, areaRatio)
        }

        val engageThreshold = descendExplosionAreaRatio.coerceIn(0.001, 0.95)
        val releaseThreshold = descendExplosionReleaseRatio.coerceIn(0.001, engageThreshold - 1e-3)
        val shouldEngage =
            areaRatio >= engageThreshold ||
                (descendExplosionGuardActive && areaRatio >= releaseThreshold)
        if (!shouldEngage) {
            if (descendExplosionGuardActive && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=DESCEND_GUARD state=release reason=ratio_low src=$source " +
                        "areaRatio=${fmt(areaRatio)} release=${fmt(releaseThreshold)}"
                )
            }
            descendExplosionGuardActive = false
            descendExplosionLastAreaRatio = areaRatio
            return DescendExplosionGuardDecision(box, false, areaRatio)
        }

        val coreSide = min(descendExplosionCoreSizePx, min(frameW, frameH)).coerceAtLeast(MIN_BOX_DIM)
        val coreCenter = Point(frameW * 0.5, frameH * 0.5)
        val coreRect = clampRect(buildCenteredSquare(coreCenter, coreSide), frameW, frameH) ?: box
        val transition = if (descendExplosionGuardActive) "hold" else "engage"
        if (
            !descendExplosionGuardActive ||
            frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L ||
            abs(areaRatio - descendExplosionLastAreaRatio) >= 0.05
        ) {
            Log.w(
                TAG,
                "EVAL_EVENT type=DESCEND_GUARD state=$transition src=$source " +
                    "areaRatio=${fmt(areaRatio)} threshold=${fmt(engageThreshold)} release=${fmt(releaseThreshold)} " +
                    "raw=${box.x},${box.y},${box.width}x${box.height} " +
                    "core=${coreRect.x},${coreRect.y},${coreRect.width}x${coreRect.height}"
            )
        }
        descendExplosionGuardActive = true
        descendExplosionLastAreaRatio = areaRatio
        return DescendExplosionGuardDecision(coreRect, true, areaRatio)
    }

    private fun clampDescendValue(
        value: Double,
        minValue: Double,
        maxValue: Double,
        field: String,
        state: DescendOffsetState
    ): Double {
        if (!value.isFinite()) return Double.NaN
        val clamped = value.coerceIn(minValue, maxValue)
        if (clamped != value && frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=DESCEND_CLAMP field=$field state=${state.name} raw=${fmt(value)} clamped=${fmt(clamped)}"
            )
        }
        return clamped
    }

    private fun shouldEmitDescendOffset(state: DescendOffsetState, nowMs: Long, force: Boolean): Boolean {
        if (force) return true
        val changed = descendLastState != state
        return when (state) {
            DescendOffsetState.TRACKING -> true
            DescendOffsetState.L1, DescendOffsetState.L2, DescendOffsetState.L3 ->
                changed || frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L
            DescendOffsetState.LOST ->
                changed || nowMs - descendLastEmitMs >= DESCEND_HEARTBEAT_MS
            DescendOffsetState.FAIL ->
                changed
        }
    }

    private fun emitDescendOffset(
        state: DescendOffsetState,
        box: Rect? = null,
        confidence: Double? = null,
        force: Boolean = false
    ): Boolean {
        if (!centerRoiGpsReady) return false
        val nowMs = currentTimelineMs()
        if (!shouldEmitDescendOffset(state, nowMs, force)) return false

        var x = Double.NaN
        var y = Double.NaN
        var conf = Double.NaN

        when (state) {
            DescendOffsetState.TRACKING -> {
                val current = box ?: return false
                val offset = resolveDescendOffset(current) ?: return false
                x = offset.first
                y = offset.second
                conf = (confidence ?: latestPredictionConfidence).coerceIn(0.0, 1.0)
            }
            DescendOffsetState.LOST -> {
                x = descendCacheX
                y = descendCacheY
                conf = descendCacheConf
            }
            else -> Unit
        }

        val xOut = clampDescendValue(x, -1.0, 1.0, "x", state)
        val yOut = clampDescendValue(y, -1.0, 1.0, "y", state)
        val confOut = clampDescendValue(conf, 0.0, 1.0, "conf", state)
        if (state == DescendOffsetState.TRACKING) {
            descendCacheX = xOut
            descendCacheY = yOut
            descendCacheConf = confOut
        }

        Log.w(
            TAG,
            "EVAL_EVENT type=DESCEND_OFFSET x=${fmt(xOut)} y=${fmt(yOut)} conf=${fmt(confOut)} " +
                "state=${state.name} t=${fmt(resolveDescendTimestampSec().coerceAtLeast(0.0))} session=$diagSessionId"
        )
        descendLastState = state
        descendLastEmitMs = nowMs
        return true
    }

    private fun emitDescendOffsetPerFrame() {
        if (!centerRoiGpsReady) return
        if (centerRoiFailLatched) {
            emitDescendOffset(DescendOffsetState.FAIL)
            return
        }
        if (isTracking) {
            descendLostActive = false
            val trackBox = latestPredictionBox ?: lastTrackedBox ?: lastMeasuredTrackBox
            emitDescendOffset(
                state = DescendOffsetState.TRACKING,
                box = trackBox,
                confidence = latestPredictionConfidence
            )
            return
        }

        if (descendLostActive) {
            val lostElapsed = (currentTimelineMs() - descendLostStartMs).coerceAtLeast(0L)
            if (lostElapsed <= DESCEND_LOST_BUFFER_MS) {
                emitDescendOffset(DescendOffsetState.LOST)
                return
            }
            descendLostActive = false
        }

        if (isCenterRoiSearchActive()) {
            emitDescendOffset(resolveDescendSearchState())
        }
    }

    private fun resetDescendOffsetState(reason: String) {
        descendLastState = null
        descendLastEmitMs = 0L
        descendCacheX = Double.NaN
        descendCacheY = Double.NaN
        descendCacheConf = Double.NaN
        descendLostActive = false
        descendLostStartMs = 0L
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(TAG, "EVAL_EVENT type=DESCEND_STATE state=reset reason=$reason session=$diagSessionId")
        }
    }

    private fun resetCenterRoiSearchState(reason: String) {
        centerRoiLevel = CenterRoiLevel.L1
        centerRoiLevelStartMs = 0L
        centerRoiSessionStartMs = 0L
        centerRoiTimeoutRaised = false
        centerRoiLastScope = "full"
        centerRoiL1ElapsedMs = 0L
        centerRoiL2ElapsedMs = 0L
        centerRoiL3ElapsedMs = 0L
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            Log.w(
                TAG,
                "EVAL_EVENT type=ROI_SEARCH state=reset reason=$reason enabled=$centerRoiSearchEnabled gpsReady=$centerRoiGpsReady"
            )
        }
    }

    private fun maybeTriggerCenterRoiFail(levelElapsedMs: Long, totalElapsedMs: Long) {
        if (centerRoiFailEmittedThisSession) return
        if (lockHoldFramesRemaining > 0) return
        centerRoiL3ElapsedMs = max(centerRoiL3ElapsedMs, levelElapsedMs.coerceAtLeast(0L))
        centerRoiFailEmittedThisSession = true
        centerRoiFailLatched = true
        metricsSearchLastReason = "center_roi_fail_timeout"
        emitDescendOffset(DescendOffsetState.FAIL, force = true)
        Log.w(
            TAG,
            "EVAL_EVENT type=ROI_SEARCH state=fail reason=l3_timeout " +
                "l1ms=$centerRoiL1ElapsedMs l2ms=$centerRoiL2ElapsedMs l3ms=$centerRoiL3ElapsedMs totalElapsedMs=$totalElapsedMs"
        )
        resetCenterRoiSearchState("fail_timeout")
    }

    private fun maybeAdvanceCenterRoiLevel(nowMs: Long) {
        if (!isCenterRoiSearchActive()) return
        if (centerRoiSessionStartMs <= 0L) {
            centerRoiSessionStartMs = nowMs
            centerRoiLevelStartMs = nowMs
            centerRoiLevel = CenterRoiLevel.L1
            centerRoiTimeoutRaised = false
            centerRoiL1ElapsedMs = 0L
            centerRoiL2ElapsedMs = 0L
            centerRoiL3ElapsedMs = 0L
            Log.w(
                TAG,
                "EVAL_EVENT type=ROI_SEARCH state=start level=l1 l1Range=${fmt(centerRoiL1Range)} l2Range=${fmt(centerRoiL2Range)} " +
                    "timeoutMs=${centerRoiL1TimeoutMs}/${centerRoiL2TimeoutMs}/${centerRoiL3TimeoutMs}"
            )
            return
        }
        val levelElapsed = (nowMs - centerRoiLevelStartMs).coerceAtLeast(0L)
        when (centerRoiLevel) {
            CenterRoiLevel.L1 -> {
                if (levelElapsed >= centerRoiL1TimeoutMs) {
                    centerRoiL1ElapsedMs = levelElapsed
                    centerRoiLevel = CenterRoiLevel.L2
                    centerRoiLevelStartMs = nowMs
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=ROI_SEARCH state=escalate from=l1 to=l2 elapsedMs=$levelElapsed"
                    )
                }
            }
            CenterRoiLevel.L2 -> {
                if (levelElapsed >= centerRoiL2TimeoutMs) {
                    centerRoiL2ElapsedMs = levelElapsed
                    centerRoiLevel = CenterRoiLevel.L3
                    centerRoiLevelStartMs = nowMs
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=ROI_SEARCH state=escalate from=l2 to=l3 elapsedMs=$levelElapsed"
                    )
                }
            }
            CenterRoiLevel.L3 -> {
                centerRoiL3ElapsedMs = max(centerRoiL3ElapsedMs, levelElapsed)
                if (!centerRoiTimeoutRaised && levelElapsed >= centerRoiL3TimeoutMs) {
                    centerRoiTimeoutRaised = true
                    val totalElapsed = (nowMs - centerRoiSessionStartMs).coerceAtLeast(0L)
                    metricsSearchLastReason = "center_roi_timeout"
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=ROI_SEARCH state=timeout level=l3 levelElapsedMs=$levelElapsed totalElapsedMs=$totalElapsed"
                    )
                    maybeTriggerCenterRoiFail(levelElapsedMs = levelElapsed, totalElapsedMs = totalElapsed)
                }
            }
        }
    }

    private fun buildCenterRoiRect(frameW: Int, frameH: Int, centerRange: Double): Rect? {
        val safeRange = centerRange.coerceIn(0.05, 0.49)
        val left = (frameW * (0.5 - safeRange)).roundToInt()
        val top = (frameH * (0.5 - safeRange)).roundToInt()
        val width = (frameW * safeRange * 2.0).roundToInt().coerceAtLeast(MIN_BOX_DIM)
        val height = (frameH * safeRange * 2.0).roundToInt().coerceAtLeast(MIN_BOX_DIM)
        return clampRect(Rect(left, top, width, height), frameW, frameH)
    }

    private fun findOrbMatchForAcquire(frame: Mat): OrbMatchCandidate? {
        centerRoiLastScope = "full"
        if (!isCenterRoiSearchActive()) {
            return findOrbMatch(frame)
        }
        val timelineNowMs = currentTimelineMs()
        maybeAdvanceCenterRoiLevel(timelineNowMs)
        return when (centerRoiLevel) {
            CenterRoiLevel.L1 -> {
                centerRoiLastScope = "l1"
                val roi = buildCenterRoiRect(frame.cols(), frame.rows(), centerRoiL1Range)
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L && roi != null) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=ROI_SEARCH state=scan level=l1 roi=${roi.x},${roi.y},${roi.width}x${roi.height}"
                    )
                }
                roi?.let { findOrbMatchInRoi(frame, it) }
            }
            CenterRoiLevel.L2 -> {
                centerRoiLastScope = "l2"
                val roi = buildCenterRoiRect(frame.cols(), frame.rows(), centerRoiL2Range)
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L && roi != null) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=ROI_SEARCH state=scan level=l2 roi=${roi.x},${roi.y},${roi.width}x${roi.height}"
                    )
                }
                roi?.let { findOrbMatchInRoi(frame, it) }
            }
            CenterRoiLevel.L3 -> {
                centerRoiLastScope = "l3"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(TAG, "EVAL_EVENT type=ROI_SEARCH state=scan level=l3 roi=full")
                }
                findOrbMatch(frame)
            }
        }
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

        val candidate = findOrbMatchForAcquire(frame)
        if (candidate == null) {
            metricsSearchMissCount++
            metricsSearchLastReason =
                if (isCenterRoiSearchActive()) {
                    "roi_${centerRoiLastScope}_${lastSearchDiagReason}"
                } else {
                    lastSearchDiagReason
                }
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
        metricsSearchLastReason =
            if (isCenterRoiSearchActive()) {
                "candidate_roi_${centerRoiLastScope}"
            } else {
                "candidate"
            }
        if (isCenterRoiSearchActive()) {
            Log.w(
                TAG,
                "EVAL_EVENT type=ROI_SEARCH state=hit level=$centerRoiLastScope " +
                    "box=${candidate.box.x},${candidate.box.y},${candidate.box.width}x${candidate.box.height} " +
                    "good=${candidate.goodMatches} inliers=${candidate.inlierCount} conf=${fmt(candidate.confidence)}"
            )
        }
        val confirmed = maybeRefineFallbackCandidate(frame, candidate)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag(
                "SEARCH",
                "session=$diagSessionId stage=candidate good=${confirmed.goodMatches} inliers=${confirmed.inlierCount} conf=${fmt(confirmed.confidence)} box=${confirmed.box.x},${confirmed.box.y},${confirmed.box.width}x${confirmed.box.height} strong=${confirmed.isStrong} matcher=${confirmed.matcherType} fallback=${confirmed.fallbackReason ?: "none"}"
            )
        }
        updateLatestPrediction(confirmed.box, confirmed.confidence, tracking = false)

        if (isManualRoiSessionActive() && isCandidateLockableForInit(confirmed, frame.cols(), frame.rows())) {
            searchMissStreak = 0
            clearFirstLockCandidate("manual_roi_direct")
            metricsSearchLastReason = "manual_roi_direct_lock"
            initializeTracker(
                frame,
                confirmed.box,
                "manual_roi_direct good=${confirmed.goodMatches} inliers=${confirmed.inlierCount} " +
                    "conf=${fmt(confirmed.confidence)} h=${confirmed.usedHomography} ds=${fmt(confirmed.searchScale)} " +
                    "fallback=${confirmed.fallbackReason ?: "none"} matcher=${confirmed.matcherType}",
                image
            )
            return
        }

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
            val s3Enabled = s3PromotedNearGateEnabled || s3PromotedFarGateEnabled
            val promotedTier =
                if (s3Enabled) classifyPromotedRelockTier(frame, promoted) else PromotedRelockTier.NONE
            if (
                s3PromotedFarGateEnabled &&
                promotedTier == PromotedRelockTier.FAR &&
                !passesPromotedFarRelockGate(frame, promoted)
            ) {
                metricsSearchPromoteRejectCount++
                metricsSearchLastReason = "promoted_far_gate_reject"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_STABLE state=final_reject reason=promoted_far_gate_reject " +
                            "good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
                    )
                }
                searchMissStreak = (searchMissStreak + 1).coerceAtMost(50_000)
                clearFirstLockCandidate("promoted_far_gate_reject")
                return
            }
            var promotedVerified = passesAutoInitVerification(frame, promoted)
            if (!promotedVerified && s3PromotedNearGateEnabled && promotedTier == PromotedRelockTier.NEAR) {
                promotedVerified = passesPromotedNearRelockOverride(frame, promoted)
            }
            if (!promotedVerified) {
                logDiag(
                    "LOCK_GATE",
                    "session=$diagSessionId stage=promoted_verify pass=false tier=${promotedTier.name.lowercase(Locale.US)} " +
                        "good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
                )
                metricsSearchPromoteRejectCount++
                metricsSearchLastReason = "promoted_verify_reject"
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_STABLE state=final_reject reason=promoted_verify_reject tier=${promotedTier.name.lowercase(Locale.US)} " +
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
                "session=$diagSessionId stage=promote pass=true tier=${promotedTier.name.lowercase(Locale.US)} " +
                    "good=${promoted.goodMatches} inliers=${promoted.inlierCount} conf=${fmt(promoted.confidence)}"
            )
            initializeTracker(
                frame,
                promoted.box,
                "orb_temporal_confirm good=${promoted.goodMatches} inliers=${promoted.inlierCount} " +
                    "conf=${fmt(promoted.confidence)} h=${promoted.usedHomography} ds=${fmt(promoted.searchScale)} " +
                    "fallback=${promoted.fallbackReason ?: "none"} matcher=${promoted.matcherType} " +
                    "tier=${promotedTier.name.lowercase(Locale.US)}",
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
        val guardDecision = applyDescendExplosionGuard(safe, frame.cols(), frame.rows(), "kcf")
        val guardedBox = guardDecision.box
        if (!guardDecision.active) {
            val verifyDecision = maybeVerifyTracking(frame, guardedBox)
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
        }

        commitTrackingObservation(guardedBox, confidence = 1.0, markNativeAccept = false)
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
        val nativeGateConfidence = if (nativeGateUseMeasurement) nativeMeasurementConfidence else nativeConfidence
        recordNativeConfidenceSample(nativeConfidence)
        recordNativeSimilaritySample(nativeSimilarity)
        when (evaluateNativeConfidence(nativeGateConfidence)) {
            NativeConfidenceAction.DROP_HARD -> {
                logNativeScoreSample("native_mat", "drop_hard", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_hard", nativeGateConfidence)) return
                onLost("native_conf_hard")
                return
            }
            NativeConfidenceAction.DROP_SOFT_STREAK -> {
                logNativeScoreSample("native_mat", "drop_soft", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_soft", nativeGateConfidence)) return
                onLost("native_conf_soft")
                return
            }
            NativeConfidenceAction.HOLD -> {
                logNativeScoreSample("native_mat", "hold", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                suppressOverlayOnUncertainTracking("native_conf_hold", nativeGateConfidence)
                return
            }
            NativeConfidenceAction.DROP_MIN_CONF -> {
                logNativeScoreSample("native_mat", "drop_min_conf", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                consecutiveTrackerFailures++
                if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                    suppressOverlayOnUncertainTracking("native_conf_pending", nativeGateConfidence)
                    return
                }
                if (holdCurrentBoxOnTransientLoss("native_conf_low", nativeGateConfidence)) return
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
            if (holdCurrentBoxOnTransientLoss("native_invalid_box", nativeGateConfidence)) return
            onLost("native_invalid_box")
            return
        }

        val guardDecision = applyDescendExplosionGuard(safe, frame.cols(), frame.rows(), "native_mat")
        val guardedBox = guardDecision.box

        if (!guardDecision.active && !passesNativeSpatialGate(guardedBox, nativeGateConfidence, source = "native_mat")) {
            metricsSearchLastReason = "native_spatial_gate_fail"
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_spatial_gate_fail", nativeGateConfidence)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_spatial_gate_fail", nativeGateConfidence)) return
            onLost("native_spatial_gate_fail")
            return
        }

        consecutiveTrackerFailures = 0
        if (!guardDecision.active && shouldDropOnTrackGuard(frame, guardedBox, nativeGateConfidence)) {
            if (holdCurrentBoxOnTransientLoss("track_guard_fail", nativeGateConfidence)) return
            onLost("track_guard_fail")
            return
        }
        val nativeVerifyInterval = resolveNativeVerifyIntervalFrames(
            current = guardedBox,
            frameW = frame.cols(),
            frameH = frame.rows()
        )
        if (
            !guardDecision.active &&
            shouldRunNativeOrbVerify() &&
            nativeVerifyInterval > 0 &&
            frameCounter % nativeVerifyInterval.toLong() == 0L
        ) {
            val verifyDecision = maybeVerifyTracking(frame, guardedBox, nativeGateConfidence)
            if (verifyDecision != null) {
                when (verifyDecision.action) {
                    "realign" -> {
                        verifyDecision.box?.let {
                            initializeTracker(frame, it, verifyDecision.reason)
                            return
                        }
                    }
                    "drop" -> {
                        if (holdCurrentBoxOnTransientLoss(verifyDecision.reason, nativeGateConfidence)) {
                            return
                        }
                        onLost(verifyDecision.reason)
                        return
                    }
                }
            }
        }

        commitTrackingObservation(guardedBox, confidence = nativeMeasurementConfidence, markNativeAccept = true)
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
        val nativeGateConfidence = if (nativeGateUseMeasurement) nativeMeasurementConfidence else nativeConfidence
        recordNativeConfidenceSample(nativeConfidence)
        recordNativeSimilaritySample(nativeSimilarity)
        when (evaluateNativeConfidence(nativeGateConfidence)) {
            NativeConfidenceAction.DROP_HARD -> {
                logNativeScoreSample("native_img", "drop_hard", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_hard", nativeGateConfidence)) return
                onLost("native_conf_hard")
                return
            }
            NativeConfidenceAction.DROP_SOFT_STREAK -> {
                logNativeScoreSample("native_img", "drop_soft", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                if (holdCurrentBoxOnTransientLoss("native_conf_soft", nativeGateConfidence)) return
                onLost("native_conf_soft")
                return
            }
            NativeConfidenceAction.HOLD -> {
                logNativeScoreSample("native_img", "hold", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                suppressOverlayOnUncertainTracking("native_conf_hold", nativeGateConfidence)
                return
            }
            NativeConfidenceAction.DROP_MIN_CONF -> {
                logNativeScoreSample("native_img", "drop_min_conf", nativeConfidence, nativeSimilarity, nativeMeasurementConfidence)
                consecutiveTrackerFailures++
                if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                    suppressOverlayOnUncertainTracking("native_conf_pending", nativeGateConfidence)
                    return
                }
                if (holdCurrentBoxOnTransientLoss("native_conf_low", nativeGateConfidence)) return
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
            if (holdCurrentBoxOnTransientLoss("native_invalid_box", nativeGateConfidence)) return
            onLost("native_invalid_box")
            return
        }

        val guardDecision = applyDescendExplosionGuard(safe, frameW, frameH, "native_img")
        val guardedBox = guardDecision.box

        if (!guardDecision.active && !passesNativeSpatialGate(guardedBox, nativeGateConfidence, source = "native_img")) {
            metricsSearchLastReason = "native_spatial_gate_fail"
            consecutiveTrackerFailures++
            if (consecutiveTrackerFailures < nativeCfg.maxFailStreak) {
                suppressOverlayOnUncertainTracking("native_spatial_gate_fail", nativeGateConfidence)
                return
            }
            if (holdCurrentBoxOnTransientLoss("native_spatial_gate_fail", nativeGateConfidence)) return
            onLost("native_spatial_gate_fail")
            return
        }

        if (!guardDecision.active && shouldDropOnTrackGuard(null, guardedBox, nativeGateConfidence)) {
            if (holdCurrentBoxOnTransientLoss("track_guard_fail", nativeGateConfidence)) return
            onLost("track_guard_fail")
            return
        }

        val nativeVerifyInterval = resolveNativeVerifyIntervalFrames(
            current = guardedBox,
            frameW = frameW,
            frameH = frameH
        )
        if (
            !guardDecision.active &&
            shouldRunNativeOrbVerify() &&
            nativeVerifyInterval > 0 &&
            frameCounter % nativeVerifyInterval.toLong() == 0L
        ) {
            val verifyFrame = imageToMat(image)
            try {
                val verifyDecision = maybeVerifyTracking(verifyFrame, guardedBox, nativeGateConfidence)
                if (verifyDecision != null) {
                    when (verifyDecision.action) {
                        "realign" -> {
                            verifyDecision.box?.let {
                                initializeTracker(verifyFrame, it, verifyDecision.reason, image)
                                return
                            }
                        }
                        "drop" -> {
                            if (holdCurrentBoxOnTransientLoss(verifyDecision.reason, nativeGateConfidence)) {
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
        commitTrackingObservation(guardedBox, confidence = nativeMeasurementConfidence, markNativeAccept = true)
    }


    private fun evaluateTrackBoxConsistency(
        current: Rect,
        previous: Rect?,
        cfg: TrackGuardConfig,
        confidence: Double
    ): Boolean {
        val prev = previous ?: return true
        val currCenter = rectCenter(current)
        val prevCenter = rectCenter(prev)
        val jump = pointDistance(currCenter, prevCenter)
        val ref = max(min(prev.width, prev.height), min(current.width, current.height)).toDouble().coerceAtLeast(1.0)
        val accelRelaxEligible =
            activeTrackBackend != TrackBackend.KCF &&
                confidence >= max(trackVerifyNativeBypassConfidence, TRACK_GUARD_ACCEL_CONF_MIN) &&
                trackMismatchStreak <= 1 &&
                trackAppearanceLowStreak == 0
        val jumpFactor =
            if (accelRelaxEligible) {
                cfg.maxCenterJumpFactor * TRACK_GUARD_ACCEL_JUMP_BOOST
            } else {
                cfg.maxCenterJumpFactor
            }
        val maxJump = ref * jumpFactor

        val prevArea = (prev.width.toDouble() * prev.height.toDouble()).coerceAtLeast(1.0)
        val currArea = (current.width.toDouble() * current.height.toDouble()).coerceAtLeast(1.0)
        val areaRatio = currArea / prevArea
        val minAreaRatio =
            if (accelRelaxEligible) {
                (cfg.minAreaRatio * TRACK_GUARD_ACCEL_AREA_MIN_SCALE).coerceAtLeast(0.10)
            } else {
                cfg.minAreaRatio
            }
        val maxAreaRatio =
            if (accelRelaxEligible) {
                (cfg.maxAreaRatio * TRACK_GUARD_ACCEL_AREA_MAX_SCALE).coerceAtMost(12.0)
            } else {
                cfg.maxAreaRatio
            }
        val areaOk = areaRatio in minAreaRatio..maxAreaRatio
        return jump <= maxJump && areaOk
    }

    private fun resolveNativeMeasurementConfidence(result: NativeTrackerBridge.NativeTrackResult): Double {
        val confidence = result.confidence.toDouble().coerceIn(0.0, 1.0)
        val similarity = result.similarity.toDouble().coerceIn(0.0, 1.0)
        // Dynamic-R should track native model similarity more closely while
        // preserving existing confidence gate behavior.
        return (similarity * 0.70 + confidence * 0.30).coerceIn(0.0, 1.0)
    }

    private fun passesNativeSpatialGate(current: Rect, confidence: Double, source: String): Boolean {
        if (!spatialGateEnabled) return true

        val kalmanPrior = latestKalmanPrediction
        val priorBox = kalmanPrior ?: lastMeasuredTrackBox ?: return true
        val prior = SpatialPrior(priorBox, if (kalmanPrior != null) "kalman" else "last_measured")
        val spatial = evaluateSpatialGate(current, prior) ?: return true
        val appearanceScore = confidence.coerceIn(0.0, 1.0)
        val fusionScore = computeDeepSortFusionScore(appearanceScore, spatial.score)
        val minScore =
            if (!isReplayInput) {
                spatialGateRelockMinScoreLive
            } else {
                spatialGateRelockMinScoreReplay
            }
        val mahalMax =
            if (!isReplayInput) {
                NATIVE_SPATIAL_MAHAL_MAX_LIVE
            } else {
                NATIVE_SPATIAL_MAHAL_MAX_REPLAY
            }
        val fusionMin =
            if (!isReplayInput) {
                NATIVE_SPATIAL_FUSION_MIN_LIVE
            } else {
                NATIVE_SPATIAL_FUSION_MIN_REPLAY
            }
        val bypassFusion =
            appearanceScore >= spatialGateRelockBypassConf &&
                appearanceScore >= spatialGateRelockBypassAppearance
        val passMahal = spatial.distance2 <= mahalMax
        val passSpatial = spatial.score >= minScore
        val passFusion = fusionScore >= fusionMin
        val pass = passMahal && (passSpatial || passFusion || bypassFusion)
        if (!pass) {
            Log.w(
                TAG,
                "EVAL_EVENT type=NATIVE_SPATIAL_GATE state=reject src=$source " +
                    "score=${fmt(spatial.score)} min=${fmt(minScore)} d2=${fmt(spatial.distance2)} " +
                    "mahalMax=${fmt(mahalMax)} fusion=${fmt(fusionScore)} fusionMin=${fmt(fusionMin)} " +
                    "conf=${fmt(confidence)} bypass=$bypassFusion prior=${spatial.source} " +
                    "box=${current.x},${current.y},${current.width}x${current.height}"
            )
        }
        return pass
    }

    private fun computeDeepSortFusionScore(appearanceScore: Double, spatialScore: Double): Double {
        val spatialWeight = spatialGateWeight.coerceIn(0.05, 0.90)
        val alpha = (1.0 - spatialWeight).coerceIn(0.10, 0.95)
        return (alpha * appearanceScore + (1.0 - alpha) * spatialScore).coerceIn(0.0, 1.0)
    }

    private fun isNativeGapPassthroughActive(): Boolean {
        return nativeGapPassthrough && activeTrackBackend != TrackBackend.KCF
    }

    private fun shouldDropOnTrackGuard(frame: Mat?, current: Rect, confidence: Double): Boolean {
        if (isNativeGapPassthroughActive()) return false
        trackGuardHardMismatch = false
        val guardCfg = heuristicConfig.trackGuard
        val prev = lastMeasuredTrackBox
        val smallTargetForGuard =
            frame != null &&
                isSmallTargetForExtraVerify(current = current, frameW = frame.cols(), frameH = frame.rows())
        val prevCenter = prev?.let { rectCenter(it) }
        val currCenter = rectCenter(current)
        val jumpDistance = if (prevCenter == null) 0.0 else pointDistance(currCenter, prevCenter)
        val refSide =
            if (prev == null) {
                1.0
            } else {
                max(min(prev.width, prev.height), min(current.width, current.height)).toDouble().coerceAtLeast(1.0)
            }
        val guardMaxJump = refSide * guardCfg.maxCenterJumpFactor
        val prevArea = prev?.let { (it.width.toDouble() * it.height.toDouble()).coerceAtLeast(1.0) } ?: 1.0
        val currArea = (current.width.toDouble() * current.height.toDouble()).coerceAtLeast(1.0)
        val areaRatio = currArea / prevArea
        val geomOk = evaluateTrackBoxConsistency(current, prev, guardCfg, confidence)
        val accelSpikeCandidate =
            !geomOk &&
                !smallTargetForGuard &&
                prev != null &&
                activeTrackBackend != TrackBackend.KCF &&
                confidence >= max(trackVerifyNativeBypassConfidence, TRACK_GUARD_ACCEL_CONF_MIN) &&
                trackAppearanceLowStreak == 0 &&
                !trackGuardHardMismatch &&
                jumpDistance > guardMaxJump &&
                jumpDistance <= guardMaxJump * TRACK_GUARD_ACCEL_SPIKE_JUMP_RATIO &&
                areaRatio in
                ((guardCfg.minAreaRatio * TRACK_GUARD_ACCEL_SPIKE_AREA_MIN_SCALE).coerceAtLeast(0.08))..
                ((guardCfg.maxAreaRatio * TRACK_GUARD_ACCEL_SPIKE_AREA_MAX_SCALE).coerceAtMost(16.0))
        if (!geomOk) {
            if (accelSpikeCandidate && trackGuardAccelGraceFrames > 0) {
                trackAccelGraceFramesRemaining = max(trackAccelGraceFramesRemaining, trackGuardAccelGraceFrames)
                trackMismatchStreak = (trackMismatchStreak - 1).coerceAtLeast(0)
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    logDiag(
                        "TRACK",
                        "session=$diagSessionId stage=track_guard reason=accel_spike_grace " +
                            "conf=${fmt(confidence)} jump=${fmt(jumpDistance)} maxJump=${fmt(guardMaxJump)} " +
                            "area=${fmt(areaRatio)} grace=$trackAccelGraceFramesRemaining"
                    )
                }
            } else {
                trackMismatchStreak = (trackMismatchStreak + 1).coerceAtMost(1000)
            }
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                logDiag(
                    "TRACK",
                    "session=$diagSessionId stage=track_guard reason=geom_fail streak=$trackMismatchStreak conf=${fmt(confidence)} jump=${fmt(jumpDistance)}"
                )
            }
        } else {
            trackMismatchStreak = 0
            trackAccelGraceFramesRemaining = 0
        }

        val shouldCheckAppearance =
            frame != null &&
                (smallTargetForGuard || frameCounter % guardCfg.appearanceCheckIntervalFrames == 0L)
        if (shouldCheckAppearance) {
            val guardFrame = frame ?: return false
            val score = computePatchTemplateCorrelation(guardFrame, current)
            val anchorScore = trackAnchorAppearanceScore
            val anchorMinScore =
                if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
                    val anchorDrop =
                        if (smallTargetForGuard) {
                            (trackGuardAnchorMaxDrop * smallTargetAnchorDropScale).coerceAtLeast(0.05)
                        } else {
                            trackGuardAnchorMaxDrop
                        }
                    val dynamicMin = anchorScore - anchorDrop
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
                val hardMismatch =
                    anchorScore.isFinite() &&
                        score <= anchorMinScore - TRACK_GUARD_HARD_APPEAR_MARGIN &&
                        confidence >= TRACK_GUARD_HARD_APPEAR_CONF_MIN
                if (hardMismatch) {
                    trackGuardHardMismatch = true
                }
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    logDiag(
                        "TRACK",
                        "session=$diagSessionId stage=track_guard reason=appearance_fail streak=$trackAppearanceLowStreak conf=${fmt(confidence)} " +
                            "score=${fmt(score)} min=${fmt(minAllowedScore)} anchor=${fmt(anchorScore)} smallTarget=$smallTargetForGuard " +
                            "hardMismatch=$hardMismatch"
                    )
                }
            } else {
                trackAppearanceLowStreak = 0
            }
        }

        var requiredDropStreak = guardCfg.dropStreak
        if (activeTrackBackend != TrackBackend.KCF && confidence >= trackVerifyNativeBypassConfidence) {
            requiredDropStreak = max(requiredDropStreak, 2)
        }
        if (
            activeTrackBackend != TrackBackend.KCF &&
            confidence >= max(trackVerifyNativeBypassConfidence, TRACK_GUARD_ACCEL_CONF_MIN) &&
            trackAppearanceLowStreak == 0
        ) {
            requiredDropStreak = max(requiredDropStreak, TRACK_GUARD_ACCEL_DROP_STREAK_MIN)
        }
        var requiredAppearanceDropStreak = requiredDropStreak
        // Small-target identity is more fragile: when appearance keeps failing,
        // drop faster to avoid latching onto adjacent look-alike objects.
        if (smallTargetForGuard && trackGuardAnchorEnabled) {
            requiredAppearanceDropStreak = min(requiredDropStreak, 2)
        }
        if (trackGuardHardMismatch) {
            requiredAppearanceDropStreak = 1
        }
        if (
            trackAccelGraceFramesRemaining > 0 &&
            trackAppearanceLowStreak == 0 &&
            !trackGuardHardMismatch &&
            !smallTargetForGuard
        ) {
            trackAccelGraceFramesRemaining = (trackAccelGraceFramesRemaining - 1).coerceAtLeast(0)
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=TRACK_GUARD_GRACE remain=$trackAccelGraceFramesRemaining " +
                        "streak=$trackMismatchStreak conf=${fmt(confidence)}"
                )
            }
            return false
        }
        val geomDrop = trackMismatchStreak >= requiredDropStreak
        val appearanceDrop = trackAppearanceLowStreak >= requiredAppearanceDropStreak
        if (geomDrop || appearanceDrop) {
            val subtype =
                when {
                    trackGuardHardMismatch -> "appearance_hard"
                    geomDrop && appearanceDrop -> "geom_appearance"
                    appearanceDrop -> "appearance"
                    else -> "geom"
                }
            Log.w(
                TAG,
                "EVAL_EVENT type=TRACK_GUARD_REJECT reason=$subtype " +
                    "conf=${fmt(confidence)} jump=${fmt(jumpDistance)} maxJump=${fmt(guardMaxJump)} " +
                    "area=${fmt(areaRatio)} areaMin=${fmt(guardCfg.minAreaRatio)} areaMax=${fmt(guardCfg.maxAreaRatio)} " +
                    "geomStreak=$trackMismatchStreak/$requiredDropStreak " +
                    "appStreak=$trackAppearanceLowStreak/$requiredAppearanceDropStreak " +
                    "hard=$trackGuardHardMismatch smallTarget=$smallTargetForGuard"
            )
        }
        return geomDrop || appearanceDrop
    }

    private fun evaluateNativeConfidence(confidence: Double): NativeConfidenceAction {
        if (isNativeGapPassthroughActive()) {
            metricsSearchLastReason = "native_gap_passthrough"
            nativeLowConfidenceStreak = 0
            return NativeConfidenceAction.ACCEPT
        }
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
        if (isNativeGapPassthroughActive()) return false
        return nativeOrbVerifyEnabled || isReplayInput
    }

    private fun resolveNativeVerifyIntervalFrames(current: Rect? = null, frameW: Int = 0, frameH: Int = 0): Int {
        val base = heuristicConfig.trackVerify.intervalFrames.coerceAtLeast(1)
        var interval = base
        if (isReplayInput && !nativeOrbVerifyEnabled) {
            interval = min(interval, REPLAY_NATIVE_VERIFY_INTERVAL_FRAMES)
        }
        if (current != null && isSmallTargetForExtraVerify(current, frameW, frameH)) {
            interval = min(interval, smallTargetNativeVerifyIntervalFrames.coerceAtLeast(1))
        }
        return interval
    }

    private fun isSmallTargetForExtraVerify(current: Rect, frameW: Int, frameH: Int): Boolean {
        if (frameW <= 0 || frameH <= 0) return false
        val frameArea = (frameW.toDouble() * frameH.toDouble()).coerceAtLeast(1.0)
        val boxArea = (current.width.toDouble() * current.height.toDouble()).coerceAtLeast(1.0)
        val areaRatio = boxArea / frameArea
        val threshold = (smallTargetAreaRatio * smallTargetNativeVerifyAreaScale).coerceIn(0.005, 0.25)
        return areaRatio <= threshold
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
        if (
            isManualRoiSessionActive() &&
            (reason == "track_guard_fail" || reason.startsWith("native_conf_"))
        ) {
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=LOSS_GRACE state=skip reason=manual_roi_release gate=$reason"
                )
            }
            return false
        }
        if (
            reason == "track_guard_fail" &&
            (
                !isReplayInput ||
                    trackGuardHardMismatch ||
                    trackAppearanceLowStreak >= 2 ||
                    trackMismatchStreak >= 3
                )
        ) {
            if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                Log.w(
                    TAG,
                    "EVAL_EVENT type=LOSS_GRACE state=skip reason=track_guard_release " +
                        "hard=$trackGuardHardMismatch appStreak=$trackAppearanceLowStreak geomStreak=$trackMismatchStreak"
                )
            }
            return false
        }
        val box = lastMeasuredTrackBox ?: lastTrackedBox ?: return false
        val eligible =
            reason.startsWith("verify_") ||
                reason.startsWith("native_conf_") ||
                reason == "native_track_fail" ||
                reason == "native_invalid_box" ||
                reason == "track_guard_fail" ||
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
        val priorLost = lastMeasuredTrackBox ?: lastTrackedBox
        releaseTracker("lost_$reason", requestGc = true, callerTrigger = "onLost:$reason")
        consecutiveTrackerFailures = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        trackGuardHardMismatch = false
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = 0
        trackAccelGraceFramesRemaining = 0
        relockSpatialRejectStreak = 0
        val fastReacquireBoost =
            when {
                reason.contains("track_guard") || reason.contains("native_conf") -> {
                    max(48, fastFirstLockFrames)
                }
                else -> max(32, fastFirstLockFrames / 2)
            }
        fastFirstLockRemaining = max(fastFirstLockRemaining, fastReacquireBoost.coerceIn(0, 480))
        lastNativeAcceptMs = 0L
        lastLostBox = priorLost?.let { Rect(it.x, it.y, it.width, it.height) }
        lastLostFrameId = frameCounter
        lastTrackedBox = null
        lastMeasuredTrackBox = null
        trackingStage = TrackingStage.ACQUIRE
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackGuardHardMismatch = false
        metricsLostCount++
        clearFirstLockCandidate("lost")
        resetCenterRoiSearchState("lost")
        latestSearchFrame?.release()
        latestSearchFrame = null
        resetKalman("lost_$reason")
        updateLatestPrediction(null, 0.0, tracking = false)
        descendLostActive = true
        descendLostStartMs = currentTimelineMs()
        Log.w(
            TAG,
                "EVAL_EVENT type=LOST reason=$reason lost=$metricsLostCount " +
                "backend=${activeTrackBackend.name.lowercase(Locale.US)} " +
                "replayPtsSec=${fmt(if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0)} " +
                "lastBox=${priorLost?.let { "${it.x},${it.y},${it.width}x${it.height}" } ?: "none"}"
        )
        logDiag("TRACK", "session=$diagSessionId action=lost reason=$reason lost=$metricsLostCount")
        val token = ++overlayResetToken
        if (lostOverlayHoldMs <= 0L) {
            overlayView.post {
                clearManualRoiState("overlay_reset_lost")
                overlayView.reset()
            }
        } else {
            overlayView.postDelayed(
                {
                    if (!isTracking && token == overlayResetToken) {
                        clearManualRoiState("overlay_reset_lost_delay")
                        overlayView.reset()
                    }
                },
                lostOverlayHoldMs
            )
        }
    }

    private fun initializeTracker(frame: Mat, box: Rect, reason: String, image: ImageProxy? = null) {
        val safe = clampRect(box, frame.cols(), frame.rows()) ?: return
        val currentAnchorScore = computePatchTemplateCorrelation(frame, safe).coerceIn(-1.0, 1.0)
        val prevAnchorScore = trackAnchorAppearanceScore
        trackAnchorAppearanceScore =
            if (!trackGuardAnchorEnabled || !prevAnchorScore.isFinite()) {
                currentAnchorScore
            } else {
                // Keep anchor identity stable across re-locks: only adapt when the
                // new patch is still close to the existing anchor.
                val allowUpdateMin = prevAnchorScore - trackGuardAnchorMaxDrop
                if (currentAnchorScore >= allowUpdateMin) {
                    (prevAnchorScore * 0.90 + currentAnchorScore * 0.10).coerceIn(-1.0, 1.0)
                } else {
                    prevAnchorScore
                }
            }
        val wasTracking = isTracking
        latestSearchFrame?.release()
        latestSearchFrame = null
        releaseTracker("reinit_$reason", requestGc = false, callerTrigger = "reinit:$reason")
        resetKalman("reinit_$reason")

        if (preferredTrackBackend != TrackBackend.KCF) {
            val nativeOk = initializeNativeTracker(frame, safe, preferredTrackBackend, image)
            if (nativeOk) {
                tracker = null
                isTracking = true
                trackingStage = TrackingStage.TRACK
                activeTrackBackend = preferredTrackBackend
                relockSpatialRejectStreak = 0
                consecutiveTrackerFailures = 0
                trackMismatchStreak = 0
                trackAppearanceLowStreak = 0
                trackAccelGraceFramesRemaining = 0
                trackVerifyFailStreak = 0
                trackVerifyHardDriftStreak = 0
                searchMissStreak = 0
                searchBudgetCooldownFrames = 0
                nativeLowConfidenceStreak = 0
                nativeFuseWarmupRemaining = heuristicConfig.nativeGate.fuseWarmupFrames
                lockHoldFramesRemaining = lockHoldFrames
                fastFirstLockRemaining = 0
                resetCenterRoiSearchState("lock")
                descendLostActive = false
                overlayResetToken++
                commitTrackingObservation(safe, confidence = 1.0, markNativeAccept = true)
                if (!wasTracking) {
                    metricsLockCount++
                    if (metricsFirstLockMs < 0L) {
                        metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
                        metricsFirstLockFrame = frameCounter
                        if (isReplayInput && currentReplayPtsMs >= 0L) {
                            metricsFirstLockReplayMs = currentReplayPtsMs
                        }
                    }
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=LOCK reason=$reason backend=${activeTrackBackend.name.lowercase(Locale.US)} " +
                            "locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)} " +
                            "firstLockReplaySec=${fmt(if (metricsFirstLockReplayMs >= 0L) metricsFirstLockReplayMs.toDouble() / 1000.0 else -1.0)} " +
                            "firstLockFrame=$metricsFirstLockFrame " +
                            "replayPtsSec=${fmt(if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0)} " +
                            "box=${safe.x},${safe.y},${safe.width}x${safe.height}"
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
            if (shouldSuppressKcfFallback(reason)) {
                val framesAgo = framesSinceLastLost()
                metricsSearchLastReason = "native_init_retry_no_kcf"
                Log.w(
                    TAG,
                    "EVAL_EVENT type=TRACKER_FALLBACK reason=native_init_retry_no_kcf " +
                        "backend=${preferredTrackBackend.name.lowercase(Locale.US)} " +
                        "lostFrames=$framesAgo initReason=$reason"
                )
                return
            }
        }

        val freshTracker = runCatching {
            val kcf = TrackerKCF.create()
            kcf.init(frame, safe)
            kcf
        }.getOrElse { err ->
            Log.e(TAG, "KCF init failed", err)
            releaseTracker("reinit_failed", requestGc = true, callerTrigger = "reinit_failed:$reason")
            return
        }

        tracker = freshTracker
        isTracking = true
        trackingStage = TrackingStage.TRACK
        activeTrackBackend = TrackBackend.KCF
        relockSpatialRejectStreak = 0
        consecutiveTrackerFailures = 0
        trackMismatchStreak = 0
        trackAppearanceLowStreak = 0
        trackAccelGraceFramesRemaining = 0
        trackVerifyFailStreak = 0
        trackVerifyHardDriftStreak = 0
        searchMissStreak = 0
        searchBudgetCooldownFrames = 0
        nativeLowConfidenceStreak = 0
        nativeFuseWarmupRemaining = 0
        lockHoldFramesRemaining = lockHoldFrames
        lastNativeAcceptMs = 0L
        fastFirstLockRemaining = 0
        resetCenterRoiSearchState("lock")
        descendLostActive = false
        overlayResetToken++
        commitTrackingObservation(safe, confidence = 1.0, markNativeAccept = false)

        if (!wasTracking) {
            metricsLockCount++
            if (metricsFirstLockMs < 0L) {
                metricsFirstLockMs = (SystemClock.elapsedRealtime() - metricsSessionStartMs).coerceAtLeast(0L)
                metricsFirstLockFrame = frameCounter
                if (isReplayInput && currentReplayPtsMs >= 0L) {
                    metricsFirstLockReplayMs = currentReplayPtsMs
                }
            }
            Log.w(
                TAG,
                "EVAL_EVENT type=LOCK reason=$reason backend=${activeTrackBackend.name.lowercase(Locale.US)} " +
                    "locks=$metricsLockCount firstLockSec=${fmt(metricsFirstLockMs.toDouble() / 1000.0)} " +
                    "firstLockReplaySec=${fmt(if (metricsFirstLockReplayMs >= 0L) metricsFirstLockReplayMs.toDouble() / 1000.0 else -1.0)} " +
                    "firstLockFrame=$metricsFirstLockFrame " +
                    "replayPtsSec=${fmt(if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0)} " +
                    "box=${safe.x},${safe.y},${safe.width}x${safe.height}"
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
        if (isNativeGapPassthroughActive()) {
            trackVerifyFailStreak = 0
            trackVerifyHardDriftStreak = 0
            return null
        }
        val verifyCfg = heuristicConfig.trackVerify
        if (verifyCfg.intervalFrames <= 0 || frameCounter % verifyCfg.intervalFrames.toLong() != 0L) {
            return null
        }

        val trackedAppearance = computePatchTemplateCorrelation(frame, trackedBox)
        val anchorScore = trackAnchorAppearanceScore
        val bypassBlockedByAppearance =
            anchorScore.isFinite() &&
                trackedAppearance < anchorScore - TRACK_VERIFY_BYPASS_ANCHOR_MARGIN
        val currentConfidence = (currentTrackConfidence ?: 0.0).coerceIn(0.0, 1.0)
        val prevMeasured = lastMeasuredTrackBox
        val motionJump =
            if (prevMeasured == null) {
                0.0
            } else {
                pointDistance(rectCenter(trackedBox), rectCenter(prevMeasured))
            }
        val motionRef =
            if (prevMeasured == null) {
                1.0
            } else {
                max(min(prevMeasured.width, prevMeasured.height), min(trackedBox.width, trackedBox.height))
                    .toDouble()
                    .coerceAtLeast(1.0)
            }
        val accelVerifyRelaxed =
            activeTrackBackend != TrackBackend.KCF &&
                !bypassBlockedByAppearance &&
                trackAppearanceLowStreak == 0 &&
                !trackGuardHardMismatch &&
                currentConfidence >= (verifyCfg.nativeBypassConfidence - TRACK_VERIFY_ACCEL_CONF_RELAX).coerceAtLeast(0.0) &&
                (
                    motionJump >= motionRef * TRACK_VERIFY_ACCEL_MOTION_RATIO ||
                        motionJump >= verifyCfg.recenterPx
                    )
        val effectiveRecenterPx =
            if (accelVerifyRelaxed) {
                verifyCfg.recenterPx * TRACK_VERIFY_ACCEL_RECENTER_BOOST
            } else {
                verifyCfg.recenterPx
            }
        val effectiveMinIou =
            if (accelVerifyRelaxed) {
                (verifyCfg.minIou - TRACK_VERIFY_ACCEL_MIN_IOU_RELAX).coerceAtLeast(0.05)
            } else {
                verifyCfg.minIou
            }
        val effectiveFailTolerance =
            if (accelVerifyRelaxed) {
                (verifyCfg.failTolerance + TRACK_VERIFY_ACCEL_FAIL_TOL_BONUS).coerceAtMost(12)
            } else {
                verifyCfg.failTolerance
            }
        val effectiveHardTolerance =
            if (accelVerifyRelaxed) {
                (verifyCfg.hardDriftTolerance + TRACK_VERIFY_ACCEL_HARD_TOL_BONUS).coerceAtMost(8)
            } else {
                verifyCfg.hardDriftTolerance
            }

        if (activeTrackBackend != TrackBackend.KCF) {
            // NCNN warm-up is sensitive to ORB false negatives; avoid immediate drift drop.
            if (nativeFuseWarmupRemaining > 0) return null
            if (currentConfidence >= verifyCfg.nativeBypassConfidence && !bypassBlockedByAppearance) {
                trackVerifyFailStreak = 0
                trackVerifyHardDriftStreak = 0
                return null
            }
            if (accelVerifyRelaxed) {
                trackVerifyFailStreak = 0
                trackVerifyHardDriftStreak = 0
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=TRACK_VERIFY state=accel_bypass jump=${fmt(motionJump)} ref=${fmt(motionRef)} " +
                            "conf=${fmt(currentConfidence)} recenter=${fmt(effectiveRecenterPx)} iou=${fmt(effectiveMinIou)}"
                    )
                }
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
            val shouldRealign = shift > effectiveRecenterPx || iou < effectiveMinIou
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

        val shouldTryGlobalRealign =
            !localHealthy &&
                (
                    hardDrift ||
                        bypassBlockedByAppearance ||
                        trackAppearanceLowStreak >= 2 ||
                        trackGuardHardMismatch
                    )
        if (shouldTryGlobalRealign) {
            val globalCandidate = findOrbMatch(frame, allowSeedAssist = false)
            if (globalCandidate != null) {
                val globalStrong =
                    globalCandidate.goodMatches >= TRACK_VERIFY_GLOBAL_REALIGN_MIN_GOOD &&
                        globalCandidate.inlierCount >= TRACK_VERIFY_GLOBAL_REALIGN_MIN_INLIERS
                if (globalStrong && passesAutoInitVerification(frame, globalCandidate)) {
                    val shift = pointDistance(rectCenter(trackedBox), rectCenter(globalCandidate.box))
                    val iou = rectIou(trackedBox, globalCandidate.box)
                    if (shift > effectiveRecenterPx || iou < effectiveMinIou) {
                        trackVerifyFailStreak = 0
                        trackVerifyHardDriftStreak = 0
                        return TrackVerifyDecision(
                            action = "realign",
                            box = globalCandidate.box,
                            reason = "verify_global_realign shift=${fmt(shift)} iou=${fmt(iou)} " +
                                "good=${globalCandidate.goodMatches} inliers=${globalCandidate.inlierCount}"
                        )
                    }
                }
            }
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
                            "conf=${fmt(trackedQuality.confidence)} hardStreak=$trackVerifyHardDriftStreak/$effectiveHardTolerance"
                )
            }
            if (trackVerifyHardDriftStreak >= effectiveHardTolerance) {
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
                        "conf=${fmt(trackedQuality.confidence)} streak=$trackVerifyFailStreak/$effectiveFailTolerance"
                )
            }
            if (trackVerifyFailStreak >= effectiveFailTolerance) {
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

        var requiredStableFrames = if (!isReplayInput) max(2, firstLockCfg.stableFrames - 1) else firstLockCfg.stableFrames
        var requiredStableMs = if (!isReplayInput) max(160L, firstLockCfg.stableMs - 80L) else firstLockCfg.stableMs
        if (isStartupLockWindow()) {
            requiredStableFrames = (requiredStableFrames - STARTUP_STABLE_FRAMES_RELAX).coerceAtLeast(2)
            requiredStableMs = (requiredStableMs - STARTUP_STABLE_MS_RELAX).coerceAtLeast(120L)
        }
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
        var minAvg = if (strong) -0.10 else -0.06
        var minSingle = if (strong) -0.42 else -0.32
        if (isStartupLockWindow()) {
            minAvg -= STARTUP_APPEAR_AVG_RELAX
            minSingle -= STARTUP_APPEAR_SINGLE_RELAX
        }
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

    private fun resolveSearchAnchorHint(frameW: Int, frameH: Int): SearchAnchorHint? {
        val firstLockCfg = heuristicConfig.firstLock
        if (firstLockCandidateFrames > 0) {
            val radius = max(firstLockCfg.stablePx, firstLockCfg.smallStablePx) * FIRST_LOCK_ANCHOR_NEAR_FACTOR
            return SearchAnchorHint(
                point = Point(firstLockCandidateCenter.x, firstLockCandidateCenter.y),
                radius = radius,
                source = "first_lock"
            )
        }

        val lost = lastLostBox ?: return null
        val lostFramesAgo = if (lastLostFrameId >= 0L) frameCounter - lastLostFrameId else Long.MAX_VALUE
        val maxFrames = if (!isReplayInput) autoVerifyLostPriorMaxFramesLive else autoVerifyLostPriorMaxFramesReplay
        if (lostFramesAgo !in 1..maxFrames) return null
        val prior = clampRect(lost, frameW, frameH) ?: return null
        val side = lostPriorReferenceSide(prior)
        val baseNearFactor =
            if (!isReplayInput) autoVerifyLostAnchorNearFactorLive else autoVerifyLostAnchorNearFactorReplay
        val boostFrames = autoVerifyLostPriorCenterBoostFrames.coerceAtLeast(1L).toDouble()
        val boost = (lostFramesAgo.toDouble() / boostFrames).coerceIn(0.0, autoVerifyLostPriorCenterBoostCap)
        return SearchAnchorHint(
            point = rectCenter(prior),
            radius = side * (baseNearFactor + boost),
            source = "lost_prior"
        )
    }

    private fun resolveSpatialPriorForCandidateGate(): SpatialPrior? {
        if (!spatialGateEnabled) return null
        latestKalmanPrediction?.let { return SpatialPrior(it, "kalman") }

        val lost = lastLostBox
        val lostFramesAgo = if (lastLostFrameId >= 0L) frameCounter - lastLostFrameId else Long.MAX_VALUE
        val maxFrames = if (!isReplayInput) autoVerifyLostPriorMaxFramesLive else autoVerifyLostPriorMaxFramesReplay
        if (lost != null && lostFramesAgo in 1..maxFrames) {
            return SpatialPrior(lost, "lost_prior")
        }

        lastMeasuredTrackBox?.let { return SpatialPrior(it, "last_measured") }
        lastTrackedBox?.let { return SpatialPrior(it, "last_tracked") }
        return null
    }

    private fun evaluateSpatialGate(candidate: Rect, prior: SpatialPrior?): SpatialScore? {
        if (!spatialGateEnabled || prior == null) return null
        val priorBox = prior.box
        val center = rectCenter(candidate)
        val priorCenter = rectCenter(priorBox)

        var refW = priorBox.width.toDouble().coerceAtLeast(MIN_BOX_DIM.toDouble())
        var refH = priorBox.height.toDouble().coerceAtLeast(MIN_BOX_DIM.toDouble())
        if (prior.source == "lost_prior") {
            val sideFloor = lostPriorReferenceSide(priorBox)
            refW = max(refW, sideFloor)
            refH = max(refH, sideFloor)
        }
        val sigmaX = (refW * spatialGateCenterSigmaFactor).coerceAtLeast(8.0)
        val sigmaY = (refH * spatialGateCenterSigmaFactor).coerceAtLeast(8.0)
        val sigmaW = (refW * spatialGateSizeSigmaFactor).coerceAtLeast(6.0)
        val sigmaH = (refH * spatialGateSizeSigmaFactor).coerceAtLeast(6.0)

        val dx = (center.x - priorCenter.x) / sigmaX
        val dy = (center.y - priorCenter.y) / sigmaY
        val dw = (candidate.width.toDouble() - priorBox.width.toDouble()) / sigmaW
        val dh = (candidate.height.toDouble() - priorBox.height.toDouble()) / sigmaH
        val distance2 = dx * dx + dy * dy + dw * dw + dh * dh
        return SpatialScore(
            score = exp(-0.5 * distance2).coerceIn(0.0, 1.0),
            distance2 = distance2,
            source = prior.source
        )
    }

    private fun computeOrbAppearanceScore(candidate: OrbMatchCandidate): Double {
        val orbCfg = heuristicConfig.orb
        val conf = candidate.confidence.coerceIn(0.0, 1.0)
        val inlierNorm = (
            candidate.inlierCount.toDouble() /
                max(orbCfg.softMinInliers, 1).toDouble()
            ).coerceIn(0.0, 3.0) / 3.0
        val goodNorm = (
            candidate.goodMatches.toDouble() /
                max(orbCfg.softMinGoodMatches, 1).toDouble()
            ).coerceIn(0.0, 3.0) / 3.0
        var score = conf * 0.55 + inlierNorm * 0.30 + goodNorm * 0.15
        if (candidate.usedHomography) {
            score += 0.03
        }
        return score.coerceIn(0.0, 1.0)
    }

    private fun computeOrbCandidateFinalScore(candidate: OrbMatchCandidate, prior: SpatialPrior?): Double {
        val appearanceScore = computeOrbAppearanceScore(candidate)
        val spatialScore = evaluateSpatialGate(candidate.box, prior)?.score ?: return appearanceScore
        val spatialWeight = spatialGateWeight.coerceIn(0.0, 0.90)
        return ((1.0 - spatialWeight) * appearanceScore + spatialWeight * spatialScore).coerceIn(0.0, 1.0)
    }

    private fun shouldEmitCandidateDump(): Boolean {
        if (!candDumpEnable) return false
        if (!isReplayInput) return false
        val replayMs = currentReplayPtsMs
        if (replayMs < 0L) return false
        val replaySec = replayMs.toDouble() / 1000.0
        val inWindow = replaySec in candDumpStartSec..candDumpEndSec
        if (!inWindow) return false
        if (candDumpWindowOnly && replayTargetAppearMs < 0L) return false
        return true
    }

    private fun collectCandidateDumpRow(
        frame: Mat,
        candidate: OrbMatchCandidate,
        prior: SpatialPrior?
    ): OrbCandidateDumpRow {
        val appearance = computeOrbAppearanceScore(candidate)
        val spatial = evaluateSpatialGate(candidate.box, prior)
        val spatialWeight = spatialGateWeight.coerceIn(0.0, 0.90)
        val fusion =
            if (spatial != null) {
                ((1.0 - spatialWeight) * appearance + spatialWeight * spatial.score).coerceIn(0.0, 1.0)
            } else {
                appearance
            }
        val tier = classifyPromotedRelockTier(frame, candidate)
        return OrbCandidateDumpRow(
            candidate = candidate,
            appearanceScore = appearance,
            spatialScore = spatial?.score,
            spatialDistance2 = spatial?.distance2,
            fusionScore = fusion,
            tier = tier,
            source = spatial?.source ?: (prior?.source ?: "none")
        )
    }

    private fun emitCandidateDump(rows: List<OrbCandidateDumpRow>, frameW: Int) {
        if (rows.isEmpty()) return
        val replaySec = if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0
        val topRows =
            rows
                .sortedWith(
                    compareByDescending<OrbCandidateDumpRow> { it.fusionScore }
                        .thenByDescending { it.candidate.inlierCount }
                        .thenByDescending { it.candidate.goodMatches }
                )
                .take(candDumpTopK.coerceAtLeast(1))
        val expectedX =
            if (frameW > 0) {
                "${fmt(frameW * candDumpExpectedXMin)},${fmt(frameW * candDumpExpectedXMax)}"
            } else {
                "na"
            }
        topRows.forEachIndexed { rank, row ->
            val box = row.candidate.box
            val cx = box.x + box.width * 0.5
            val cy = box.y + box.height * 0.5
            Log.w(
                TAG,
                "EVAL_EVENT type=CAND_DUMP replayPtsSec=${fmt(replaySec)} session=$diagSessionId " +
                    "src=${row.source} rank=$rank cx=${fmt(cx)} cy=${fmt(cy)} " +
                    "w=${box.width} h=${box.height} appearance=${fmt(row.appearanceScore)} " +
                    "spatial=${fmt(row.spatialScore ?: -1.0)} d2=${fmt(row.spatialDistance2 ?: -1.0)} " +
                    "fusion=${fmt(row.fusionScore)} conf=${fmt(row.candidate.confidence)} " +
                    "good=${row.candidate.goodMatches} inliers=${row.candidate.inlierCount} " +
                    "homo=${if (row.candidate.usedHomography) 1 else 0} " +
                    "tier=${row.tier.name.lowercase(Locale.US)} expectedX=$expectedX"
            )
        }
    }

    private fun findOrbMatch(
        frame: Mat,
        loweRatioOverride: Double? = null,
        allowSeedAssist: Boolean = true
    ): OrbMatchCandidate? {
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
            val anchorHint = resolveSearchAnchorHint(frame.cols(), frame.rows())
            val anchorPoint = anchorHint?.point
            val anchorRadius = anchorHint?.radius ?: 0.0
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
            val dumpEnabled = shouldEmitCandidateDump()
            val candidateDumpRows: MutableList<OrbCandidateDumpRow> =
                if (dumpEnabled) ArrayList<OrbCandidateDumpRow>(templatePyramidLevels.size) else ArrayList(0)
            val candidatePrior = if (dumpEnabled) resolveSpatialPriorForCandidateGate() else null
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
                if (dumpEnabled && candidate != null) {
                    candidateDumpRows += collectCandidateDumpRow(frame, candidate, candidatePrior)
                }
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

            if (dumpEnabled && candidateDumpRows.isNotEmpty()) {
                emitCandidateDump(candidateDumpRows, frame.cols())
            }

            if (bestNearCandidate != null) {
                if (
                    anchorHint != null &&
                    anchorHint.source == "lost_prior" &&
                    frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L
                ) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=SEARCH_ANCHOR_BIAS source=${anchorHint.source} " +
                            "radius=${fmt(anchorHint.radius)} box=${bestNearCandidate.box.x},${bestNearCandidate.box.y}," +
                            "${bestNearCandidate.box.width}x${bestNearCandidate.box.height}"
                    )
                }
                return bestNearCandidate
            }
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
        val prior = resolveSpatialPriorForCandidateGate()
        if (prior != null) {
            val candidateScore = computeOrbCandidateFinalScore(candidate, prior)
            val currentScore = computeOrbCandidateFinalScore(current, prior)
            if (abs(candidateScore - currentScore) > 1e-4) return candidateScore > currentScore
        }
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
        if (!isStartupLockWindow()) return false
        if (candidate.usedHomography) return true
        return candidate.goodMatches >= STARTUP_DIRECT_INIT_MIN_GOOD &&
            candidate.inlierCount >= STARTUP_DIRECT_INIT_MIN_INLIERS &&
            candidate.confidence >= STARTUP_DIRECT_INIT_MIN_CONF
    }

    private fun framesSinceLastLost(): Long {
        if (lastLostFrameId < 0L) return Long.MAX_VALUE
        return (frameCounter - lastLostFrameId).coerceAtLeast(0L)
    }

    private fun lostPriorReferenceSide(prior: Rect): Double {
        val rawSide = min(prior.width, prior.height).toDouble().coerceAtLeast(1.0)
        val minSide =
            if (!isReplayInput) AUTO_VERIFY_LOST_PRIOR_MIN_SIDE_LIVE else AUTO_VERIFY_LOST_PRIOR_MIN_SIDE_REPLAY
        return rawSide.coerceAtLeast(minSide)
    }

    private fun resolveLostPriorGeometry(frame: Mat, candidate: OrbMatchCandidate): LostPriorGeometry? {
        if (metricsLockCount <= 0) return null
        val lost = lastLostBox ?: return null
        val maxFrames = if (!isReplayInput) autoVerifyLostPriorMaxFramesLive else autoVerifyLostPriorMaxFramesReplay
        val lostFramesAgo = framesSinceLastLost()
        if (lostFramesAgo !in 1..maxFrames) return null
        val prior = clampRect(lost, frame.cols(), frame.rows()) ?: return null
        val iou = rectIou(candidate.box, prior)
        val centerDist = pointDistance(rectCenter(candidate.box), rectCenter(prior))
        val side = lostPriorReferenceSide(prior)
        val near =
            iou >= PROMOTED_RELOCK_NEAR_MIN_IOU ||
                centerDist <= side * PROMOTED_RELOCK_NEAR_CENTER_FACTOR
        return LostPriorGeometry(
            prior = prior,
            framesAgo = lostFramesAgo,
            iou = iou,
            centerDist = centerDist,
            side = side,
            near = near
        )
    }

    private fun classifyPromotedRelockTier(frame: Mat, candidate: OrbMatchCandidate): PromotedRelockTier {
        val geom = resolveLostPriorGeometry(frame, candidate) ?: return PromotedRelockTier.NONE
        return if (geom.near) PromotedRelockTier.NEAR else PromotedRelockTier.FAR
    }

    private fun passesPromotedFarRelockGate(frame: Mat, candidate: OrbMatchCandidate): Boolean {
        val geom = resolveLostPriorGeometry(frame, candidate) ?: return true
        if (geom.near) return true
        val appearance = computePatchTemplateCorrelation(frame, candidate.box)
        val anchorScore = trackAnchorAppearanceScore
        val minConfBase =
            if (!isReplayInput) autoVerifyLostFarRecoverMinConfLive else autoVerifyLostFarRecoverMinConfReplay
        val minConf = (minConfBase + PROMOTED_RELOCK_FAR_EXTRA_CONF).coerceAtMost(1.0)
        val anchorMin =
            if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
                anchorScore - PROMOTED_RELOCK_FAR_ANCHOR_MAX_DROP
            } else {
                autoVerifyLostFarRecoverMinAppearance
            }
        val minAppearance = max(autoVerifyLostFarRecoverMinAppearance, anchorMin)
        val pass =
            candidate.goodMatches >= autoVerifyLostFarRecoverMinGood &&
                candidate.inlierCount >= autoVerifyLostFarRecoverMinInliers &&
                candidate.confidence >= minConf &&
                appearance >= minAppearance
        logDiag(
            "LOCK_GATE",
            "session=$diagSessionId stage=promoted_far_gate pass=$pass " +
                "good=${candidate.goodMatches}/${autoVerifyLostFarRecoverMinGood} " +
                "inliers=${candidate.inlierCount}/${autoVerifyLostFarRecoverMinInliers} " +
                "conf=${fmt(candidate.confidence)}/${fmt(minConf)} " +
                "app=${fmt(appearance)}/${fmt(minAppearance)} " +
                "iou=${fmt(geom.iou)} center=${fmt(geom.centerDist)}"
        )
        return pass
    }

    private fun passesPromotedNearRelockOverride(frame: Mat, candidate: OrbMatchCandidate): Boolean {
        val geom = resolveLostPriorGeometry(frame, candidate) ?: return false
        if (!geom.near) return false
        val appearance = computePatchTemplateCorrelation(frame, candidate.box)
        val anchorScore = trackAnchorAppearanceScore
        val minConfBase =
            if (!isReplayInput) autoVerifyRelockMinConfLive else autoVerifyRelockMinConfReplay
        val minConf = (minConfBase - PROMOTED_RELOCK_NEAR_CONF_RELAX).coerceAtLeast(0.0)
        val minGood = max(heuristicConfig.orb.softMinGoodMatches + 2, 8)
        val minInliers = max(heuristicConfig.orb.softMinInliers + 1, 4)
        val enforceAnchorVeto =
            s3PromotedNearAnchorVetoEnabled &&
                trackGuardAnchorEnabled &&
                anchorScore.isFinite()
        val anchorOk =
            !enforceAnchorVeto ||
                appearance >= anchorScore - PROMOTED_RELOCK_NEAR_ANCHOR_MAX_DROP
        val pass =
            candidate.goodMatches >= minGood &&
                candidate.inlierCount >= minInliers &&
                candidate.confidence >= minConf &&
                anchorOk
        logDiag(
            "LOCK_GATE",
            "session=$diagSessionId stage=promoted_near_override pass=$pass " +
                "good=${candidate.goodMatches}/$minGood inliers=${candidate.inlierCount}/$minInliers " +
                "conf=${fmt(candidate.confidence)}/${fmt(minConf)} " +
                "app=${fmt(appearance)} anchor=${fmt(anchorScore)} anchorVeto=$enforceAnchorVeto anchorOk=$anchorOk " +
                "iou=${fmt(geom.iou)} center=${fmt(geom.centerDist)}"
        )
        return pass
    }

    private fun shouldSuppressKcfFallback(reason: String): Boolean {
        if (!s2SuppressKcfFallbackEnabled) return false
        if (!isReplayInput) return false
        if (preferredTrackBackend == TrackBackend.KCF) return false
        if (metricsLockCount <= 0) return false
        val framesAgo = framesSinceLastLost()
        if (framesAgo !in 1..AUTO_VERIFY_RELOCK_RECENT_LOST_WINDOW_FRAMES_REPLAY) return false
        val normalized = reason.lowercase(Locale.US)
        return normalized.startsWith("orb_") ||
            normalized.startsWith("verify_") ||
            normalized.contains("promoted")
    }

    private fun passesAutoInitVerification(frame: Mat, candidate: OrbMatchCandidate): Boolean {
        val fallbackRefineCfg = heuristicConfig.fallbackRefine
        val verifyCfg = heuristicConfig.autoInitVerify
        val gatePrefix = "session=$diagSessionId stage=auto_verify good=${candidate.goodMatches} inliers=${candidate.inlierCount} conf=${fmt(candidate.confidence)}"
        val fallbackReason = candidate.fallbackReason?.lowercase(Locale.US)
        if (isReplayInput && autoVerifyRejectFlipReplay && fallbackReason?.contains("flip") == true) {
            logDiag(
                "LOCK_GATE",
                "$gatePrefix pass=false reason=flip_reject fallback=${candidate.fallbackReason}"
            )
            return false
        }
        val isRelock = metricsLockCount > 0
        val isFirstLock = metricsLockCount == 0
        if (
            isFirstLock &&
            isReplayInput &&
            replayTargetAppearMs > 0L &&
            currentReplayPtsMs >= 0L &&
            currentReplayPtsMs + REPLAY_TARGET_APPEAR_GRACE_MS < replayTargetAppearMs
        ) {
            logDiag(
                "LOCK_GATE",
                "$gatePrefix pass=false reason=before_target_appear replayPtsSec=${fmt(currentReplayPtsMs.toDouble() / 1000.0)} targetSec=${fmt(replayTargetAppearMs.toDouble() / 1000.0)}"
            )
            return false
        }
        val appearanceScore = computePatchTemplateCorrelation(frame, candidate.box)
        val anchorScore = trackAnchorAppearanceScore
        val lostPrior = lastLostBox
        val lostFramesAgo = if (lastLostFrameId >= 0L) frameCounter - lastLostFrameId else Long.MAX_VALUE
        val fastRelockWindow =
            if (!isReplayInput) AUTO_VERIFY_FAST_RELOCK_WINDOW_FRAMES_LIVE else AUTO_VERIFY_FAST_RELOCK_WINDOW_FRAMES_REPLAY
        val fastRelockNearLostPrior =
            run {
                if (lostPrior == null || lostFramesAgo !in 1..fastRelockWindow) return@run false
                val prior = clampRect(lostPrior, frame.cols(), frame.rows()) ?: return@run false
                val priorIou = rectIou(candidate.box, prior)
                val priorCenterDist = pointDistance(rectCenter(candidate.box), rectCenter(prior))
                val side = lostPriorReferenceSide(prior)
                val nearPrior =
                    priorCenterDist <= side * AUTO_VERIFY_FAST_RELOCK_CENTER_FACTOR ||
                        priorIou >= AUTO_VERIFY_FAST_RELOCK_MIN_IOU
                val anchorOk =
                    !trackGuardAnchorEnabled ||
                        !anchorScore.isFinite() ||
                        appearanceScore >= anchorScore - AUTO_VERIFY_FAST_RELOCK_ANCHOR_MAX_DROP
                nearPrior && anchorOk
            }
        if (isFirstLock) {
            if (isReplayInput && candidate.inlierCount < autoVerifyFirstLockMinInliersReplay) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=first_lock_inliers inliers=${candidate.inlierCount} min=$autoVerifyFirstLockMinInliersReplay"
                )
                return false
            }
            val firstLockMinAppear =
                if (!isReplayInput) autoVerifyFirstLockAppearanceMinLive else autoVerifyFirstLockAppearanceMinReplay
            if (appearanceScore < firstLockMinAppear) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=first_lock_appearance score=${fmt(appearanceScore)} min=${fmt(firstLockMinAppear)}"
                )
                return false
            }
            if (isReplayInput && autoVerifyFirstLockCenterGuardReplay) {
                val center = rectCenter(candidate.box)
                val frameW = frame.cols().toDouble().coerceAtLeast(1.0)
                val frameH = frame.rows().toDouble().coerceAtLeast(1.0)
                val dxNorm = abs(center.x / frameW - 0.5)
                val dyNorm = abs(center.y / frameH - 0.5)
                if (dxNorm > autoVerifyFirstLockCenterFactorReplay || dyNorm > autoVerifyFirstLockCenterFactorReplay) {
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=first_lock_center dx=${fmt(dxNorm)} dy=${fmt(dyNorm)} " +
                            "th=${fmt(autoVerifyFirstLockCenterFactorReplay)}"
                    )
                    return false
                }
            }
        }
        if (isRelock) {
            val manualStrictRelock = isManualRoiSessionActive()
            val relockMinConfBase =
                if (!isReplayInput) autoVerifyRelockMinConfLive else autoVerifyRelockMinConfReplay
            val relockMinConf =
                if (fastRelockNearLostPrior) {
                    (relockMinConfBase - AUTO_VERIFY_FAST_RELOCK_CONF_RELAX).coerceAtLeast(0.0)
                } else {
                    relockMinConfBase
                }
            if (candidate.confidence < relockMinConf) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=relock_conf conf=${fmt(candidate.confidence)} min=${fmt(relockMinConf)} fastRelock=$fastRelockNearLostPrior"
                )
                return false
            }
            if (manualStrictRelock && MANUAL_ROI_STRICT_RELOCK_ENABLED) {
                val manualRelockMinConf = max(relockMinConf, MANUAL_ROI_RELOCK_MIN_CONFIDENCE)
                if (candidate.confidence < manualRelockMinConf) {
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=manual_relock_conf conf=${fmt(candidate.confidence)} min=${fmt(manualRelockMinConf)}"
                    )
                    return false
                }
                if (appearanceScore < MANUAL_ROI_RELOCK_MIN_APPEARANCE) {
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=manual_relock_appearance score=${fmt(appearanceScore)} min=${fmt(MANUAL_ROI_RELOCK_MIN_APPEARANCE)}"
                    )
                    return false
                }
            }
            if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
                val relockAnchorMaxDrop =
                    if (!isReplayInput) AUTO_VERIFY_RELOCK_ANCHOR_MAX_DROP_LIVE else AUTO_VERIFY_RELOCK_ANCHOR_MAX_DROP_REPLAY
                val relockAnchorMin = anchorScore - relockAnchorMaxDrop
                if (appearanceScore < relockAnchorMin) {
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=relock_anchor score=${fmt(appearanceScore)} min=${fmt(relockAnchorMin)} anchor=${fmt(anchorScore)}"
                    )
                    return false
                }
            }
            val prior = resolveSpatialPriorForCandidateGate()
            val spatial = evaluateSpatialGate(candidate.box, prior)
            if (spatial != null) {
                val relockLostFramesAgo =
                    if (lastLostFrameId >= 0L) frameCounter - lastLostFrameId else Long.MAX_VALUE
                val recentLostWindow =
                    if (!isReplayInput) AUTO_VERIFY_RELOCK_RECENT_LOST_WINDOW_FRAMES_LIVE else AUTO_VERIFY_RELOCK_RECENT_LOST_WINDOW_FRAMES_REPLAY
                val recentLostRelock = relockLostFramesAgo in 1..recentLostWindow
                val minSpatialBase =
                    if (!isReplayInput) spatialGateRelockMinScoreLive else spatialGateRelockMinScoreReplay
                val recentLostSpatialFloorBase =
                    if (!isReplayInput) AUTO_VERIFY_RELOCK_RECENT_LOST_MIN_SPATIAL_LIVE else AUTO_VERIFY_RELOCK_RECENT_LOST_MIN_SPATIAL_REPLAY
                val recentLostSpatialFloor =
                    if (fastRelockNearLostPrior) {
                        minSpatialBase
                    } else {
                        recentLostSpatialFloorBase
                    }
                val minSpatial = if (recentLostRelock) max(minSpatialBase, recentLostSpatialFloor) else minSpatialBase
                val spatialRelaxStreakThreshold =
                    if (!isReplayInput) autoVerifyRelockSpatialRelaxStreakLive else autoVerifyRelockSpatialRelaxStreakReplay
                val relaxLostPriorSpatialGate =
                    recentLostRelock &&
                        spatial.source == "lost_prior" &&
                        (relockSpatialRejectStreak + 1) >= spatialRelaxStreakThreshold
                val effectiveMinSpatial =
                    if (relaxLostPriorSpatialGate) {
                        min(minSpatial, autoVerifyRelockSpatialRelaxMinScore)
                    } else {
                        minSpatial
                    }
                if (relaxLostPriorSpatialGate) {
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=trace reason=spatial_relax_apply score=${fmt(spatial.score)} " +
                            "min=${fmt(minSpatial)} effectiveMin=${fmt(effectiveMinSpatial)} " +
                            "d2=${fmt(spatial.distance2)} src=${spatial.source} " +
                            "spatialStreak=$relockSpatialRejectStreak"
                    )
                }
                val bypassConf =
                    if (recentLostRelock) {
                        (spatialGateRelockBypassConf + AUTO_VERIFY_RELOCK_RECENT_LOST_BYPASS_CONF_BONUS).coerceAtMost(1.0)
                    } else {
                        spatialGateRelockBypassConf
                    }
                val bypassAppearance =
                    if (recentLostRelock) {
                        (spatialGateRelockBypassAppearance + AUTO_VERIFY_RELOCK_RECENT_LOST_BYPASS_APPEARANCE_BONUS).coerceAtMost(1.0)
                    } else {
                        spatialGateRelockBypassAppearance
                    }
                val bypassSpatial =
                    candidate.confidence >= bypassConf &&
                        appearanceScore >= bypassAppearance
                if (relaxLostPriorSpatialGate &&
                    appearanceScore < autoVerifyRelockSpatialRelaxMinAppearance
                ) {
                    relockSpatialRejectStreak = (relockSpatialRejectStreak + 1).coerceAtMost(256)
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=spatial_relax_appearance app=${fmt(appearanceScore)} " +
                            "minApp=${fmt(autoVerifyRelockSpatialRelaxMinAppearance)} score=${fmt(spatial.score)} " +
                            "min=${fmt(effectiveMinSpatial)} d2=${fmt(spatial.distance2)} src=${spatial.source} " +
                            "spatialStreak=$relockSpatialRejectStreak"
                    )
                    return false
                }
                if (spatial.score < effectiveMinSpatial && !bypassSpatial) {
                    relockSpatialRejectStreak = (relockSpatialRejectStreak + 1).coerceAtMost(256)
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=false reason=spatial score=${fmt(spatial.score)} min=${fmt(effectiveMinSpatial)} " +
                            "d2=${fmt(spatial.distance2)} src=${spatial.source} conf=${fmt(candidate.confidence)} " +
                            "app=${fmt(appearanceScore)} bypass=${fmt(bypassConf)}/${fmt(bypassAppearance)} " +
                            "recentLost=$recentLostRelock fastRelock=$fastRelockNearLostPrior " +
                            "spatialStreak=$relockSpatialRejectStreak relax=$relaxLostPriorSpatialGate"
                    )
                    return false
                }
                relockSpatialRejectStreak =
                    if (recentLostRelock && spatial.source == "lost_prior") {
                        (relockSpatialRejectStreak - 1).coerceAtLeast(0)
                    } else {
                        0
                    }
            }
        }
        if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
            val smallCandidate =
                isSmallTargetForExtraVerify(
                    current = candidate.box,
                    frameW = frame.cols(),
                    frameH = frame.rows()
                )
            val anchorDrop =
                if (smallCandidate) {
                    (trackGuardAnchorMaxDrop * smallTargetAnchorDropScale).coerceAtLeast(0.05)
                } else {
                    trackGuardAnchorMaxDrop
                }
            val dynamicAnchorMin = anchorScore - anchorDrop
            val anchorMinScore =
                if (trackGuardAnchorMinScore > -0.999) {
                    max(dynamicAnchorMin, trackGuardAnchorMinScore)
                } else {
                    dynamicAnchorMin
                }
            if (appearanceScore < anchorMinScore) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=anchor score=${fmt(appearanceScore)} min=${fmt(anchorMinScore)} anchor=${fmt(anchorScore)} small=$smallCandidate"
                )
                if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
                    Log.w(
                        TAG,
                        "EVAL_EVENT type=VERIFY_GATE state=reject reason=anchor " +
                            "score=${fmt(appearanceScore)}/${fmt(anchorMinScore)} anchor=${fmt(anchorScore)} small=$smallCandidate " +
                            "candGood=${candidate.goodMatches} candInliers=${candidate.inlierCount}"
                    )
                }
                return false
            }
        }
        val maxLostPriorFrames =
            if (!isReplayInput) autoVerifyLostPriorMaxFramesLive else autoVerifyLostPriorMaxFramesReplay
        if (lostPrior != null && lostFramesAgo in 1..maxLostPriorFrames) {
            val prior = clampRect(lostPrior, frame.cols(), frame.rows())
            if (prior != null) {
                val priorIou = rectIou(candidate.box, prior)
                val priorCenterDist = pointDistance(rectCenter(candidate.box), rectCenter(prior))
                val side = lostPriorReferenceSide(prior)
                val baseCenterFactor =
                    if (!isReplayInput) autoVerifyLostPriorCenterFactorLive else autoVerifyLostPriorCenterFactorReplay
                val boostFrames = autoVerifyLostPriorCenterBoostFrames.coerceAtLeast(1L).toDouble()
                val boost = (lostFramesAgo.toDouble() / boostFrames)
                    .coerceIn(0.0, autoVerifyLostPriorCenterBoostCap)
                val centerLimit = side * (baseCenterFactor + boost)
                val minPriorIou =
                    if (!isReplayInput) autoVerifyLostPriorMinIouLive else autoVerifyLostPriorMinIouReplay
                val geomNear = priorCenterDist <= centerLimit || priorIou >= minPriorIou
                if (!geomNear) {
                    val farRecoverMinConfBase =
                        if (!isReplayInput) autoVerifyLostFarRecoverMinConfLive else autoVerifyLostFarRecoverMinConfReplay
                    val farRecoverMinConf =
                        (
                            farRecoverMinConfBase +
                                if (!isReplayInput) AUTO_VERIFY_LOST_FAR_RECOVER_EXTRA_CONF_LIVE else AUTO_VERIFY_LOST_FAR_RECOVER_EXTRA_CONF_REPLAY
                            ).coerceAtMost(1.0)
                    val farRecoverAnchorMinAppearance =
                        if (trackGuardAnchorEnabled && anchorScore.isFinite()) {
                            anchorScore - AUTO_VERIFY_LOST_FAR_RECOVER_ANCHOR_MAX_DROP
                        } else {
                            autoVerifyLostFarRecoverMinAppearance
                        }
                    val farRecoverMinAppearance = max(autoVerifyLostFarRecoverMinAppearance, farRecoverAnchorMinAppearance)
                    val farRecoverPass =
                        candidate.goodMatches >= autoVerifyLostFarRecoverMinGood &&
                            candidate.inlierCount >= autoVerifyLostFarRecoverMinInliers &&
                            candidate.confidence >= farRecoverMinConf &&
                            appearanceScore >= farRecoverMinAppearance
                    if (!farRecoverPass) {
                        logDiag(
                            "LOCK_GATE",
                            "$gatePrefix pass=false reason=lost_prior iou=${fmt(priorIou)} minIou=${fmt(minPriorIou)} center=${fmt(priorCenterDist)} centerTh=${fmt(centerLimit)} " +
                                "lostFrames=$lostFramesAgo farConf=${fmt(candidate.confidence)}/${fmt(farRecoverMinConf)} " +
                                "farApp=${fmt(appearanceScore)}/${fmt(farRecoverMinAppearance)}"
                        )
                        return false
                    }
                    logDiag(
                        "LOCK_GATE",
                        "$gatePrefix pass=true reason=lost_prior_far_recover iou=${fmt(priorIou)} center=${fmt(priorCenterDist)} " +
                            "good=${candidate.goodMatches} inliers=${candidate.inlierCount} conf=${fmt(candidate.confidence)} " +
                            "app=${fmt(appearanceScore)} farConfMin=${fmt(farRecoverMinConf)} farAppMin=${fmt(farRecoverMinAppearance)}"
                    )
                }
            }
        }
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
            val requireLocalForFirstLock =
                if (!isReplayInput) autoVerifyFirstLockRequireLocalLive else autoVerifyFirstLockRequireLocalReplay
            if (isFirstLock && requireLocalForFirstLock) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=local_missing_first_lock score=${fmt(appearanceScore)}"
                )
                return false
            }
            val relockRequireLocal =
                isRelock &&
                    isReplayInput &&
                    lostPrior != null &&
                    lostFramesAgo in 1..autoVerifyLostPriorMaxFramesReplay
            if (relockRequireLocal || (isRelock && isManualRoiSessionActive() && MANUAL_ROI_STRICT_RELOCK_ENABLED)) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=local_missing_relock score=${fmt(appearanceScore)} conf=${fmt(candidate.confidence)} " +
                        "lostFrames=$lostFramesAgo manualSession=${isManualRoiSessionActive()}"
                )
                return false
            }
            val localMissingMinConf =
                if (!isReplayInput) verifyCfg.localMissingMinConfLive else verifyCfg.localMissingMinConfReplay
            val fallbackPass = strongCandidate && strongAppearance && candidate.confidence >= localMissingMinConf
            logDiag(
                "LOCK_GATE",
                "$gatePrefix pass=$fallbackPass reason=local_missing score=${fmt(appearanceScore)} conf=${fmt(candidate.confidence)} minConf=${fmt(localMissingMinConf)}"
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
            val geomOverride = strongCandidate && strongAppearance
            if (!geomPass && isRelock && isManualRoiSessionActive() && MANUAL_ROI_STRICT_RELOCK_ENABLED) {
                logDiag(
                    "LOCK_GATE",
                    "$gatePrefix pass=false reason=geom_manual_relock iou=${fmt(iou)} center=${fmt(centerDist)} " +
                        "centerTh=${fmt(side * centerFactor)} localConf=${fmt(local.confidence)}"
                )
                return false
            }
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
        val finalMinConf =
            if (!isReplayInput) AUTO_VERIFY_FINAL_MIN_CONF_LIVE else AUTO_VERIFY_FINAL_MIN_CONF_REPLAY
        if (candidate.confidence < finalMinConf) {
            logDiag(
                "LOCK_GATE",
                "$gatePrefix pass=false reason=conf_floor conf=${fmt(candidate.confidence)} min=${fmt(finalMinConf)}"
            )
            return false
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

    private fun isStartupLockWindow(): Boolean {
        return !isTracking && metricsLockCount == 0 && fastFirstLockRemaining > 0
    }

    private fun isCandidateEligibleForTemporal(candidate: OrbMatchCandidate): Boolean {
        val cfg = heuristicConfig
        val temporalCfg = cfg.temporalGate
        val startupWindow = isStartupLockWindow()
        var temporalMinGoodMatches = computeTemporalMinGoodMatches(candidate)
        if (!isReplayInput) {
            temporalMinGoodMatches = (temporalMinGoodMatches - 1).coerceAtLeast(max(cfg.orb.softMinGoodMatches, 3))
        }
        if (startupWindow) {
            temporalMinGoodMatches = (temporalMinGoodMatches - STARTUP_TEMPORAL_MIN_GOOD_RELAX).coerceAtLeast(3)
        }
        if (candidate.goodMatches < temporalMinGoodMatches) return false

        val minInliersBase = if (candidate.usedHomography) {
            cfg.orb.softMinInliers
        } else {
            max(cfg.orb.softMinInliers, cfg.fallbackRefine.minInliers - 1)
        }
        var minInliers = if (!isReplayInput && !candidate.usedHomography) {
            (minInliersBase - 1).coerceAtLeast(cfg.orb.softMinInliers)
        } else {
            minInliersBase
        }
        if (startupWindow) {
            minInliers = (minInliers - STARTUP_TEMPORAL_MIN_INLIERS_RELAX).coerceAtLeast(cfg.orb.softMinInliers)
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
            if (startupWindow) {
                minTemporalConfidence = (minTemporalConfidence - STARTUP_TEMPORAL_CONFIDENCE_RELAX)
                    .coerceAtLeast(STARTUP_TEMPORAL_CONFIDENCE_FLOOR)
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

    private fun currentTemplateAspectForFallback(): Double {
        val template = templateGray ?: return 1.0
        if (template.empty()) return 1.0
        val h = template.rows().coerceAtLeast(1).toDouble()
        val w = template.cols().coerceAtLeast(1).toDouble()
        return (w / h).coerceIn(FALLBACK_TEMPLATE_ASPECT_MIN, FALLBACK_TEMPLATE_ASPECT_MAX)
    }

    private fun normalizeTemplateSize(template: Mat): Mat {
        if (template.empty()) return Mat()
        val out = Mat()
        val w = template.cols().toDouble()
        val h = template.rows().toDouble()
        val maxDim = max(w, h)
        val scale = when {
            maxDim > MAX_TEMPLATE_DIM -> MAX_TEMPLATE_DIM / maxDim
            maxDim < MIN_TEMPLATE_DIM -> MIN_TEMPLATE_DIM / maxDim
            else -> 1.0
        }
        if (abs(scale - 1.0) < 1e-6) {
            template.copyTo(out)
        } else {
            val interpolation = if (scale < 1.0) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR
            Imgproc.resize(template, out, Size(w * scale, h * scale), 0.0, 0.0, interpolation)
        }
        return out
    }

    private fun buildPoseAugmentedTemplates(templateGray: Mat, enablePoseAugment: Boolean): List<Mat> {
        val outputs = ArrayList<Mat>(TEMPLATE_POSE_AUGMENT_DEGREES.size)
        outputs += templateGray.clone()
        if (!enablePoseAugment) return outputs
        for (angleDeg in TEMPLATE_POSE_AUGMENT_DEGREES) {
            if (abs(angleDeg) < 0.1) continue
            val rotated = rotateTemplateByDegrees(templateGray, angleDeg) ?: continue
            outputs += rotated
        }
        return outputs
    }

    private fun rotateTemplateByDegrees(templateGray: Mat, angleDeg: Double): Mat? {
        if (templateGray.empty()) return null
        val center = Point(templateGray.cols() * 0.5, templateGray.rows() * 0.5)
        val rotation = Imgproc.getRotationMatrix2D(center, angleDeg, 1.0)
        val rotated = Mat()
        return try {
            Imgproc.warpAffine(
                templateGray,
                rotated,
                rotation,
                Size(templateGray.cols().toDouble(), templateGray.rows().toDouble()),
                Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE
            )
            normalizeTemplateSize(rotated).also { rotated.release() }
        } catch (_: Throwable) {
            rotated.release()
            null
        } finally {
            rotation.release()
        }
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
        overlayView.post {
            clearManualRoiState("overlay_reset_uncertain")
            overlayView.reset()
        }
        updateLatestPrediction(null, confidence.coerceIn(0.0, 1.0), tracking = false)
        if (frameCounter % SEARCH_DIAG_INTERVAL_FRAMES == 0L) {
            logDiag("OVERLAY", "session=$diagSessionId action=suppress reason=$reason conf=${fmt(confidence)}")
        }
    }

    private fun updateScaleFactors(frameWidth: Int, frameHeight: Int) {
        currentFrameWidth = frameWidth
        currentFrameHeight = frameHeight
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

        emitDescendOffsetPerFrame()

        if (metricsFrames % PERF_LOG_INTERVAL_FRAMES == 0L) {
            val avgMs = metricsTotalProcessMs.toDouble() / metricsFrames
            val trackRatio = metricsTrackingFrames.toDouble() / metricsFrames
            val firstLockSec = if (metricsFirstLockMs >= 0L) metricsFirstLockMs.toDouble() / 1000.0 else -1.0
            val firstLockReplaySec = if (metricsFirstLockReplayMs >= 0L) metricsFirstLockReplayMs.toDouble() / 1000.0 else -1.0
            val replayPtsSec = if (currentReplayPtsMs >= 0L) currentReplayPtsMs.toDouble() / 1000.0 else -1.0
            val perfBox = lastMeasuredTrackBox ?: lastTrackedBox
            val perfBoxToken =
                perfBox?.let { "${it.x},${it.y},${it.width}x${it.height}" } ?: "none"
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
                    "lost=$metricsLostCount replayPtsSec=${fmt(replayPtsSec)} firstLockSec=${fmt(firstLockSec)} firstLockReplaySec=${fmt(firstLockReplaySec)} firstLockFrame=$metricsFirstLockFrame " +
                    "stage=${trackingStage.name.lowercase(Locale.US)} tracking=$isTracking box=$perfBoxToken " +
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
        private const val MIN_TEMPLATE_DIM = 128.0
        private const val MAX_TEMPLATE_DIM = 480.0
        private val TEMPLATE_PYRAMID_SCALES = doubleArrayOf(1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09, 0.075)
        private val TEMPLATE_POSE_AUGMENT_DEGREES = doubleArrayOf(-30.0, -15.0, 0.0, 15.0, 30.0)

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
        private const val DEFAULT_CENTER_ROI_SEARCH_ENABLED = false
        private const val DEFAULT_CENTER_ROI_GPS_READY = false
        private const val DEFAULT_CENTER_ROI_L1_RANGE = 0.20
        private const val DEFAULT_CENTER_ROI_L2_RANGE = 0.40
        private const val DEFAULT_CENTER_ROI_L1_TIMEOUT_MS = 500L
        private const val DEFAULT_CENTER_ROI_L2_TIMEOUT_MS = 1_000L
        private const val DEFAULT_CENTER_ROI_L3_TIMEOUT_MS = 2_000L
        private const val DEFAULT_DESCEND_EXPLOSION_GUARD_ENABLED = true
        private const val DEFAULT_DESCEND_EXPLOSION_AREA_RATIO = 0.40
        private const val DEFAULT_DESCEND_EXPLOSION_RELEASE_RATIO = 0.30
        private const val DEFAULT_DESCEND_EXPLOSION_CORE_SIZE_PX = 400
        private const val DESCEND_HEARTBEAT_MS = 1_000L
        private const val DESCEND_LOST_BUFFER_MS = 1_000L
        private const val DEFAULT_INIT_BOX_SIZE = 100
        private const val DEFAULT_FALLBACK_MIN_BOX_SIZE = 48
        private const val DEFAULT_FALLBACK_MAX_BOX_SIZE = 180
        private const val FALLBACK_TEMPLATE_ASPECT_MIN = 0.50
        private const val FALLBACK_TEMPLATE_ASPECT_MAX = 2.20
        private const val DEFAULT_HOMOGRAPHY_MAX_DISTORTION = 0.20
        private const val DEFAULT_HOMOGRAPHY_SCALE_MIN = 0.60
        private const val DEFAULT_HOMOGRAPHY_SCALE_MAX = 1.80
        private const val DEFAULT_HOMOGRAPHY_MIN_JACOBIAN_DET = 1e-5
        private const val DEFAULT_TEMPLATE_MIN_TEXTURE_SCORE = 15.0
        private const val DEFAULT_TEMPLATE_POSE_AUGMENT_ENABLED = false

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
        private const val DEFAULT_SMALL_TARGET_AREA_RATIO = 0.08
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
        private const val DEFAULT_NATIVE_GATE_USE_MEASUREMENT = true
        private const val DEFAULT_NATIVE_GAP_PASSTHROUGH = false
        private const val DEFAULT_NATIVE_SCORE_LOG_INTERVAL_FRAMES = 20
        private const val DEFAULT_NATIVE_ORB_VERIFY_ENABLED = false
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_GOOD = 16
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_INLIERS = 6
        private const val DEFAULT_AUTO_VERIFY_STRONG_MIN_CONFIDENCE = 0.36
        private const val DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_LIVE = 0.16
        private const val DEFAULT_AUTO_VERIFY_STRONG_APPEARANCE_MIN_REPLAY = 0.10
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MISSING_MIN_CONF_LIVE = 0.38
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MISSING_MIN_CONF_REPLAY = 0.42
        private const val DEFAULT_AUTO_VERIFY_LOCAL_ROI_EXPAND_FACTOR = 1.6
        private const val DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_LIVE = 0.75
        private const val DEFAULT_AUTO_VERIFY_LOCAL_CENTER_FACTOR_REPLAY = 1.80
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_LIVE = 0.40
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_IOU_REPLAY = 0.15
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_LIVE = 0.72
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_SCALE_REPLAY = 0.75
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_LIVE = 0.24
        private const val DEFAULT_AUTO_VERIFY_LOCAL_MIN_CONF_FLOOR_REPLAY = 0.24
        private const val DEFAULT_AUTO_VERIFY_APPEAR_STRONG_GOOD = 24
        private const val DEFAULT_AUTO_VERIFY_APPEAR_STRONG_INLIERS = 10
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_STRONG = 0.0
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_GOOD = 20
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MEDIUM_INLIERS = 8
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_MEDIUM = -0.25
        private const val DEFAULT_AUTO_VERIFY_APPEAR_MIN_BASE = -0.25
        private const val DEFAULT_AUTO_VERIFY_APPEAR_LIVE_BIAS = 0.06
        private const val DEFAULT_AUTO_VERIFY_REJECT_FLIP_REPLAY = true
        private const val DEFAULT_AUTO_VERIFY_RELOCK_MIN_CONF_LIVE = 0.26
        private const val DEFAULT_AUTO_VERIFY_RELOCK_MIN_CONF_REPLAY = 0.34
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
        private const val DEFAULT_LOCK_HOLD_FRAMES = 3
        private const val DEFAULT_LOST_OVERLAY_HOLD_MS = 120L
        private const val DEFAULT_FAST_FIRST_LOCK_FRAMES = 240
        private const val FAST_FIRST_LOCK_MIN_GOOD_MATCHES = 12
        private const val FAST_FIRST_LOCK_MIN_INLIERS = 4
        private const val FAST_FIRST_LOCK_MIN_CONFIDENCE = 0.24
        private const val STARTUP_DIRECT_INIT_MIN_GOOD = 20
        private const val STARTUP_DIRECT_INIT_MIN_INLIERS = 8
        private const val STARTUP_DIRECT_INIT_MIN_CONF = 0.40
        private const val STARTUP_TEMPORAL_MIN_GOOD_RELAX = 1
        private const val STARTUP_TEMPORAL_MIN_INLIERS_RELAX = 1
        private const val STARTUP_TEMPORAL_CONFIDENCE_RELAX = 0.05
        private const val STARTUP_TEMPORAL_CONFIDENCE_FLOOR = 0.08
        private const val STARTUP_STABLE_FRAMES_RELAX = 1
        private const val STARTUP_STABLE_MS_RELAX = 60L
        private const val STARTUP_APPEAR_AVG_RELAX = 0.04
        private const val STARTUP_APPEAR_SINGLE_RELAX = 0.05
        private const val DEFAULT_FORCE_TRACKER_GC_ON_DROP = true
        private const val TRACKER_GC_MIN_INTERVAL_MS = 1_200L
        private const val DEFAULT_NATIVE_MODEL_PARAM_PATH = "/data/local/tmp/nanotrack.param"
        private const val DEFAULT_NATIVE_MODEL_BIN_PATH = "/data/local/tmp/nanotrack.bin"

        private const val DEFAULT_TRACK_VERIFY_INTERVAL_FRAMES = 15
        private const val DEFAULT_TRACK_VERIFY_LOCAL_EXPAND_FACTOR = 2.2
        private const val DEFAULT_TRACK_VERIFY_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_MIN_INLIERS = 3
        private const val DEFAULT_TRACK_VERIFY_FAIL_TOLERANCE = 3
        private const val DEFAULT_TRACK_VERIFY_RECENTER_PX = 48.0
        private const val DEFAULT_TRACK_VERIFY_MIN_IOU = 0.35
        private const val DEFAULT_TRACK_VERIFY_SWITCH_CONFIDENCE_MARGIN = 0.25
        private const val TRACK_VERIFY_HARD_MIN_GOOD_MATCHES = 5
        private const val DEFAULT_TRACK_VERIFY_HARD_DRIFT_TOLERANCE = 3
        private const val DEFAULT_TRACK_VERIFY_NATIVE_BYPASS_CONFIDENCE = 0.72
        private const val DEFAULT_TRACK_GUARD_MAX_CENTER_JUMP_FACTOR = 1.45
        private const val DEFAULT_TRACK_GUARD_MIN_AREA_RATIO = 0.30
        private const val DEFAULT_TRACK_GUARD_MAX_AREA_RATIO = 2.00
        private const val DEFAULT_TRACK_GUARD_DROP_STREAK = 6
        private const val DEFAULT_TRACK_GUARD_APPEARANCE_CHECK_INTERVAL = 2L
        private const val DEFAULT_TRACK_GUARD_MIN_APPEARANCE_SCORE = -0.12
        private const val DEFAULT_TRACK_GUARD_ANCHOR_ENABLED = true
        private const val DEFAULT_TRACK_GUARD_ANCHOR_MAX_DROP = 0.20
        private const val DEFAULT_TRACK_GUARD_ANCHOR_MIN_SCORE = -1.0
        private const val DEFAULT_TRACK_GUARD_ACCEL_GRACE_FRAMES = 5
        private const val TRACK_GUARD_ACCEL_CONF_MIN = 0.72
        private const val TRACK_GUARD_ACCEL_JUMP_BOOST = 1.8
        private const val TRACK_GUARD_ACCEL_AREA_MIN_SCALE = 0.75
        private const val TRACK_GUARD_ACCEL_AREA_MAX_SCALE = 1.25
        private const val TRACK_GUARD_ACCEL_DROP_STREAK_MIN = 8
        private const val TRACK_GUARD_ACCEL_SPIKE_JUMP_RATIO = 1.45
        private const val TRACK_GUARD_ACCEL_SPIKE_AREA_MIN_SCALE = 0.60
        private const val TRACK_GUARD_ACCEL_SPIKE_AREA_MAX_SCALE = 1.40
        private const val TRACK_GUARD_HARD_APPEAR_MARGIN = 0.05
        private const val TRACK_GUARD_HARD_APPEAR_CONF_MIN = 0.45
        private const val TRACK_VERIFY_BYPASS_ANCHOR_MARGIN = 0.08
        private const val TRACK_VERIFY_GLOBAL_REALIGN_MIN_GOOD = 12
        private const val TRACK_VERIFY_GLOBAL_REALIGN_MIN_INLIERS = 5
        private const val TRACK_VERIFY_ACCEL_CONF_RELAX = 0.10
        private const val TRACK_VERIFY_ACCEL_MOTION_RATIO = 1.10
        private const val TRACK_VERIFY_ACCEL_RECENTER_BOOST = 1.60
        private const val TRACK_VERIFY_ACCEL_MIN_IOU_RELAX = 0.10
        private const val TRACK_VERIFY_ACCEL_FAIL_TOL_BONUS = 2
        private const val TRACK_VERIFY_ACCEL_HARD_TOL_BONUS = 1
        private const val DEFAULT_SMALL_TARGET_NATIVE_VERIFY_INTERVAL_FRAMES = 1
        private const val DEFAULT_SMALL_TARGET_NATIVE_VERIFY_AREA_SCALE = 1.0
        private const val DEFAULT_SMALL_TARGET_ANCHOR_DROP_SCALE = 0.40
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_MAX_FRAMES_LIVE = 36L
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_MAX_FRAMES_REPLAY = 240L
        private const val AUTO_VERIFY_LOST_PRIOR_MIN_SIDE_LIVE = 20.0
        private const val AUTO_VERIFY_LOST_PRIOR_MIN_SIDE_REPLAY = 40.0
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_FACTOR_LIVE = 2.4
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_FACTOR_REPLAY = 2.6
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_MIN_IOU_LIVE = 0.05
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_MIN_IOU_REPLAY = 0.02
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_BOOST_FRAMES = 120L
        private const val DEFAULT_AUTO_VERIFY_LOST_PRIOR_CENTER_BOOST_CAP = 0.6
        private const val DEFAULT_AUTO_VERIFY_LOST_ANCHOR_NEAR_FACTOR_LIVE = 2.8
        private const val DEFAULT_AUTO_VERIFY_LOST_ANCHOR_NEAR_FACTOR_REPLAY = 3.2
        private const val DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_GOOD = 16
        private const val DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_INLIERS = 6
        private const val DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_CONF_LIVE = 0.46
        private const val DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_CONF_REPLAY = 0.52
        private const val DEFAULT_AUTO_VERIFY_LOST_FAR_RECOVER_MIN_APPEARANCE = -0.14
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_APPEAR_MIN_LIVE = 0.05
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_APPEAR_MIN_REPLAY = 0.02
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_MIN_INLIERS_REPLAY = 4
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_REQUIRE_LOCAL_LIVE = false
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_REQUIRE_LOCAL_REPLAY = false
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_CENTER_GUARD_REPLAY = false
        private const val DEFAULT_AUTO_VERIFY_FIRST_LOCK_CENTER_FACTOR_REPLAY = 0.22
        private const val DEFAULT_SPATIAL_GATE_ENABLED = true
        private const val DEFAULT_SPATIAL_GATE_WEIGHT = 0.35
        private const val DEFAULT_SPATIAL_GATE_CENTER_SIGMA_FACTOR = 1.25
        private const val DEFAULT_SPATIAL_GATE_SIZE_SIGMA_FACTOR = 0.80
        private const val DEFAULT_SPATIAL_GATE_RELOCK_MIN_SCORE_LIVE = 0.08
        private const val DEFAULT_SPATIAL_GATE_RELOCK_MIN_SCORE_REPLAY = 0.05
        private const val DEFAULT_SPATIAL_GATE_RELOCK_BYPASS_CONF = 0.72
        private const val DEFAULT_SPATIAL_GATE_RELOCK_BYPASS_APPEARANCE = 0.16
        private const val NATIVE_SPATIAL_MAHAL_MAX_LIVE = 9.49
        private const val NATIVE_SPATIAL_MAHAL_MAX_REPLAY = 11.50
        private const val NATIVE_SPATIAL_FUSION_MIN_LIVE = 0.28
        private const val NATIVE_SPATIAL_FUSION_MIN_REPLAY = 0.24
        private const val AUTO_VERIFY_RELOCK_ANCHOR_MAX_DROP_LIVE = 0.14
        private const val AUTO_VERIFY_RELOCK_ANCHOR_MAX_DROP_REPLAY = 0.18
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_WINDOW_FRAMES_LIVE = 48L
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_WINDOW_FRAMES_REPLAY = 180L
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_MIN_SPATIAL_LIVE = 0.14
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_MIN_SPATIAL_REPLAY = 0.08
        private const val AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_LIVE = 4
        private const val AUTO_VERIFY_RELOCK_SPATIAL_RELAX_STREAK_REPLAY = 5
        private const val AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_SCORE = 0.0
        private const val AUTO_VERIFY_RELOCK_SPATIAL_RELAX_MIN_APPEARANCE = -1.0
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_BYPASS_CONF_BONUS = 0.12
        private const val AUTO_VERIFY_RELOCK_RECENT_LOST_BYPASS_APPEARANCE_BONUS = 0.10
        private const val AUTO_VERIFY_LOST_FAR_RECOVER_EXTRA_CONF_LIVE = 0.04
        private const val AUTO_VERIFY_LOST_FAR_RECOVER_EXTRA_CONF_REPLAY = 0.04
        private const val AUTO_VERIFY_LOST_FAR_RECOVER_ANCHOR_MAX_DROP = 0.12
        private const val AUTO_VERIFY_FAST_RELOCK_WINDOW_FRAMES_LIVE = 45L
        private const val AUTO_VERIFY_FAST_RELOCK_WINDOW_FRAMES_REPLAY = 180L
        private const val AUTO_VERIFY_FAST_RELOCK_CENTER_FACTOR = 2.4
        private const val AUTO_VERIFY_FAST_RELOCK_MIN_IOU = 0.01
        private const val AUTO_VERIFY_FAST_RELOCK_ANCHOR_MAX_DROP = 0.20
        private const val AUTO_VERIFY_FAST_RELOCK_CONF_RELAX = 0.12
        private const val AUTO_VERIFY_FAST_RELOCK_SPATIAL_RELAX = 0.08
        private const val PROMOTED_RELOCK_NEAR_CENTER_FACTOR = 2.2
        private const val PROMOTED_RELOCK_NEAR_MIN_IOU = 0.02
        private const val PROMOTED_RELOCK_NEAR_CONF_RELAX = 0.08
        private const val PROMOTED_RELOCK_NEAR_ANCHOR_MAX_DROP = 0.22
        private const val PROMOTED_RELOCK_FAR_EXTRA_CONF = 0.03
        private const val PROMOTED_RELOCK_FAR_ANCHOR_MAX_DROP = 0.14
        private const val DEFAULT_S2_SUPPRESS_KCF_FALLBACK_ENABLED = false
        private const val DEFAULT_S3_PROMOTED_NEAR_GATE_ENABLED = false
        private const val DEFAULT_S3_PROMOTED_FAR_GATE_ENABLED = false
        private const val DEFAULT_S3_PROMOTED_NEAR_ANCHOR_VETO_ENABLED = false
        private const val AUTO_VERIFY_FINAL_MIN_CONF_LIVE = 0.26
        private const val AUTO_VERIFY_FINAL_MIN_CONF_REPLAY = 0.32
        // Phase 1 WIP: strict relock gates (conf/appearance/geom) default OFF -
        // awaiting distribution data from T4.3 probe before activating.
        private const val MANUAL_ROI_STRICT_RELOCK_ENABLED = false
        private const val MANUAL_ROI_RELOCK_MIN_CONFIDENCE = 0.35
        private const val MANUAL_ROI_RELOCK_MIN_APPEARANCE = 0.05
        private const val DEFAULT_TEMPORAL_MIN_CONFIDENCE_SMALL_REFINED = 0.24
        private const val DEFAULT_TEMPORAL_MIN_CONFIDENCE_BASE = 0.32
        private const val DEFAULT_TEMPORAL_LIVE_CONFIDENCE_RELAX = 0.03
        private const val DEFAULT_TEMPORAL_LIVE_CONFIDENCE_FLOOR = 0.10
        private const val REPLAY_TARGET_APPEAR_GRACE_MS = 120L
        private const val REPLAY_NATIVE_VERIFY_INTERVAL_FRAMES = 10

        private const val SEARCH_DIAG_INTERVAL_FRAMES = 20
        private const val PERF_LOG_INTERVAL_FRAMES = 10
        private const val SUMMARY_LOG_INTERVAL_FRAMES = 30
    }
}











































































