package com.example.dronetracker

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import com.canhub.cropper.CropImageContract
import com.canhub.cropper.CropImageContractOptions
import com.canhub.cropper.CropImageOptions
import com.canhub.cropper.CropImageView
import com.example.dronetracker.nativebridge.NativeTrackerBridge
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var replayFrameView: ImageView
    private lateinit var overlayView: TrackingOverlayView
    private lateinit var btnSelectImage: Button
    private lateinit var btnReset: Button
    private lateinit var btnGpsReady: Button

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var trackerAnalyzer: OpenCVTrackerAnalyzer
    private lateinit var requiredPermissions: Array<String>

    private var replayRunner: VideoReplayRunner? = null
    private var isReplayMode = false
    private var replayTargetPath: String? = null
    private var replayTargetPaths: List<String> = emptyList()
    private var replayVideoPath: String? = null
    private var replayLoopEnabled = true
    private var replayStartSec = 0f
    private var replayTargetAppearSec = 0f
    private var replayCatchupEnabled = false
    private var replayFpsOverride = 0f
    private var lastReplayBitmap: Bitmap? = null
    private var gpsReady = false
    private var gpsReadyReceiverRegistered = false
    private val isDebuggable: Boolean by lazy(LazyThreadSafetyMode.NONE) {
        (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }

    private val gpsReadyReceiver =
        object : BroadcastReceiver() {
            override fun onReceive(context: android.content.Context?, intent: Intent?) {
                if (intent?.action != ACTION_GPS_READY) return
                if (!isDebuggable) {
                    Log.w(TAG, "Ignore GPS_READY broadcast in non-debug build")
                    return
                }
                if (!intent.hasExtra(EXTRA_GPS_READY)) {
                    Log.w(TAG, "Ignore GPS_READY broadcast without ready extra")
                    return
                }
                val ready = intent.getBooleanExtra(EXTRA_GPS_READY, false)
                val reason = intent.getStringExtra(EXTRA_GPS_REASON)?.takeIf { it.isNotBlank() }
                    ?: DEFAULT_GPS_READY_REASON
                applyGpsReady(ready, reason, "broadcast")
            }
        }

    private val cropImageLauncher = registerForActivityResult(CropImageContract()) { result ->
        if (!result.isSuccessful) {
            return@registerForActivityResult
        }

        val uri = result.uriContent ?: return@registerForActivityResult
        runCatching {
            decodeBitmapFromUriWithExif(uri)
        }.onSuccess { bitmap ->
            if (bitmap == null) {
                Toast.makeText(this, getString(R.string.toast_template_load_failed), Toast.LENGTH_SHORT).show()
                return@onSuccess
            }
            val quality = evaluateTemplateQuality(bitmap)
            if (!quality.pass) {
                Toast.makeText(this, buildTemplateQualityMessage(quality), Toast.LENGTH_LONG).show()
                if (!bitmap.isRecycled) bitmap.recycle()
                return@onSuccess
            }
            val ok = try {
                trackerAnalyzer.setTemplateImage(bitmap)
            } finally {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
            if (ok) {
                Toast.makeText(this, getString(R.string.toast_template_loaded), Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "\u6A21\u677F\u7EB9\u7406\u8FC7\u5F31\uFF0C\u8BF7\u91CD\u65B0\u63D0\u4F9B\u76EE\u6807\u6E05\u6670\u7167\u7247", Toast.LENGTH_LONG).show()
            }
        }.onFailure { e ->
            Log.e(TAG, "Failed to load cropped image", e)
            Toast.makeText(this, getString(R.string.toast_template_load_failed), Toast.LENGTH_SHORT).show()
        }
    }

    private val selectImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri ?: return@registerForActivityResult
        cropImageLauncher.launch(
            CropImageContractOptions(
                uri = uri,
                cropImageOptions = CropImageOptions(
                    guidelines = CropImageView.Guidelines.ON,
                    fixAspectRatio = false,
                    minCropResultWidth = CROP_MIN_RESULT_SIDE,
                    minCropResultHeight = CROP_MIN_RESULT_SIDE,
                    maxCropResultWidth = CROP_OUTPUT_MAX_SIDE,
                    maxCropResultHeight = CROP_OUTPUT_MAX_SIDE
                )
            )
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        viewFinder.scaleType = PreviewView.ScaleType.FIT_CENTER
        replayFrameView = findViewById(R.id.replayFrameView)
        overlayView = findViewById(R.id.overlayView)
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnReset = findViewById(R.id.btnReset)
        btnGpsReady = findViewById(R.id.btnGpsReady)
        findViewById<ImageButton>(R.id.btnClose).setOnClickListener { finish() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, getString(R.string.toast_opencv_failed), Toast.LENGTH_LONG).show()
            Log.e(TAG, "OpenCV init failed")
        } else {
            Log.i(TAG, "OpenCV runtime ready for SEARCH/verify pipeline")
        }

        trackerAnalyzer = OpenCVTrackerAnalyzer(overlayView)
        val ncnnBackbonePaths = stageNcnnModelsFromAssets()
        if (ncnnBackbonePaths != null) {
            NativeTrackerBridge.setDefaultModelPaths(ncnnBackbonePaths.first, ncnnBackbonePaths.second)
            Log.i(
                TAG,
                "native model assets staged backboneParam=${ncnnBackbonePaths.first} backboneBin=${ncnnBackbonePaths.second}"
            )
        } else {
            Log.w(TAG, "native model assets staging failed, fallback to external model paths")
        }
        NativeTrackerBridge.initializeEngine()
        val backend = NativeTrackerBridge.backendName()
        Log.i(TAG, "native bridge backend=$backend available=${NativeTrackerBridge.isAvailable()}")
        Toast.makeText(
            this,
            "SEARCH=OpenCV, TRACK=$backend",
            Toast.LENGTH_SHORT
        ).show()
        applyIntentConfig(intent)

        overlayView.onBoxSelectedListener = { rect ->
            trackerAnalyzer.setInitialTarget(rect, overlayView.width, overlayView.height)
        }

        btnSelectImage.setOnClickListener {
            Toast.makeText(this, TEMPLATE_ORIENTATION_HINT_MESSAGE, Toast.LENGTH_LONG).show()
            selectImageLauncher.launch("image/*")
        }
        btnReset.setOnClickListener {
            trackerAnalyzer.logEvalSummary("reset_button")
            trackerAnalyzer.resetTracking(logSummary = false)
        }
        btnGpsReady.setOnClickListener {
            if (!isDebuggable) return@setOnClickListener
            applyGpsReady(!gpsReady, "debug_ui", "debug_ui")
        }

        refreshUiByMode()
        registerGpsReadyReceiverIfNeeded()
        syncGpsReadyFromAnalyzer()

        if (allPermissionsGranted()) {
            startInputPipeline()
        } else {
            ActivityCompat.requestPermissions(
                this,
                requiredPermissions,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)

        applyIntentConfig(intent)
        syncGpsReadyFromAnalyzer()
        refreshUiByMode()
        trackerAnalyzer.resetTracking(logSummary = false)

        if (allPermissionsGranted()) {
            startInputPipeline()
        } else {
            ActivityCompat.requestPermissions(
                this,
                requiredPermissions,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    private fun applyIntentConfig(incomingIntent: Intent?) {
        trackerAnalyzer.setTrackerMode(incomingIntent?.getStringExtra(EXTRA_TRACKER_MODE))
        trackerAnalyzer.applyRuntimeOverrides(incomingIntent?.getStringExtra(EXTRA_EVAL_PARAMS))
        isReplayMode = incomingIntent?.getBooleanExtra(EXTRA_EVAL_USE_REPLAY, false) == true
        replayTargetPath = incomingIntent?.getStringExtra(EXTRA_EVAL_TARGET_PATH)
        replayTargetPaths = parsePathList(incomingIntent?.getStringExtra(EXTRA_EVAL_TARGET_PATHS))
        replayVideoPath = incomingIntent?.getStringExtra(EXTRA_EVAL_VIDEO_PATH)
        replayLoopEnabled = incomingIntent?.getBooleanExtra(EXTRA_EVAL_LOOP, true) ?: true
        replayStartSec = (incomingIntent?.getFloatExtra(EXTRA_EVAL_REPLAY_START_SEC, 0f) ?: 0f).coerceAtLeast(0f)
        replayTargetAppearSec = (incomingIntent?.getFloatExtra(EXTRA_EVAL_TARGET_APPEAR_SEC, 0f) ?: 0f).coerceAtLeast(0f)
        replayCatchupEnabled = incomingIntent?.getBooleanExtra(EXTRA_EVAL_REPLAY_CATCHUP, false) ?: false
        replayFpsOverride = (incomingIntent?.getFloatExtra(EXTRA_EVAL_REPLAY_FPS, 0f) ?: 0f).coerceAtLeast(0f)
        trackerAnalyzer.setReplayTargetAppearSec(if (isReplayMode) replayTargetAppearSec.toDouble() else 0.0)
        requiredPermissions = buildRequiredPermissions()

        Log.i(TAG, "tracker mode=${trackerAnalyzer.currentTrackerMode()}")
        Log.i(
            TAG,
            "replay mode=$isReplayMode target=$replayTargetPath targets=${replayTargetPaths.size} " +
                "video=$replayVideoPath loop=$replayLoopEnabled startSec=${"%.2f".format(replayStartSec)} " +
                "targetAppearSec=${"%.2f".format(replayTargetAppearSec)} " +
                "catchup=$replayCatchupEnabled replayFps=${"%.2f".format(replayFpsOverride)}"
        )
    }

    private fun parsePathList(raw: String?): List<String> {
        if (raw.isNullOrBlank()) return emptyList()
        return raw.split(';', ',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .distinct()
    }

    private fun refreshUiByMode() {
        if (isReplayMode) {
            btnSelectImage.isEnabled = false
            btnSelectImage.alpha = 0.5f
        } else {
            btnSelectImage.isEnabled = true
            btnSelectImage.alpha = 1.0f
        }
        if (isDebuggable) {
            btnGpsReady.visibility = View.VISIBLE
            btnGpsReady.isEnabled = true
            btnGpsReady.alpha = 1.0f
        } else {
            btnGpsReady.visibility = View.GONE
            btnGpsReady.isEnabled = false
            btnGpsReady.alpha = 0.0f
        }
        updateGpsReadyButton()
    }

    private fun syncGpsReadyFromAnalyzer() {
        gpsReady = trackerAnalyzer.isCenterRoiGpsReady()
        updateGpsReadyButton()
    }

    private fun updateGpsReadyButton() {
        if (!::btnGpsReady.isInitialized) return
        val text = if (gpsReady) "GPS: ON" else "GPS: OFF"
        btnGpsReady.text = text
    }

    private fun applyGpsReady(ready: Boolean, reason: String, source: String) {
        gpsReady = ready
        trackerAnalyzer.setCenterRoiGpsReady(ready, reason)
        updateGpsReadyButton()
        Log.w(
            TRACKER_TAG,
            "EVAL_EVENT type=GPS_READY ready=$ready source=$source reason=$reason debug=$isDebuggable"
        )
    }

    private fun registerGpsReadyReceiverIfNeeded() {
        if (!isDebuggable || gpsReadyReceiverRegistered) return
        ContextCompat.registerReceiver(
            this,
            gpsReadyReceiver,
            IntentFilter(ACTION_GPS_READY),
            ContextCompat.RECEIVER_EXPORTED
        )
        gpsReadyReceiverRegistered = true
    }

    private fun unregisterGpsReadyReceiverIfNeeded() {
        if (!gpsReadyReceiverRegistered) return
        runCatching { unregisterReceiver(gpsReadyReceiver) }
        gpsReadyReceiverRegistered = false
    }

    private fun startInputPipeline() {
        if (isReplayMode) {
            startReplay()
        } else {
            startCamera()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setImageQueueDepth(2)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, trackerAnalyzer)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startReplay() {
        val baseTarget = replayTargetPath?.takeIf { it.isNotBlank() } ?: DEFAULT_REPLAY_TARGET_PATH
        val targets = LinkedHashSet<String>()
        if (replayTargetPaths.isNotEmpty()) {
            targets.addAll(replayTargetPaths)
            // Auto-include common multi-view templates only when multi-template mode is explicitly requested.
            val baseFile = File(baseTarget)
            val parent = baseFile.parentFile
            if (parent != null && parent.exists()) {
                listOf("T1.jpg", "T2.jpg", "target_side.jpg", "target_alt.jpg").forEach { name ->
                    val extra = File(parent, name)
                    if (extra.exists()) targets += extra.absolutePath
                }
            }
        } else {
            targets += baseTarget
        }

        if (!loadTemplatesFromPaths(targets.toList())) {
            Toast.makeText(this, getString(R.string.toast_replay_target_missing), Toast.LENGTH_LONG).show()
            Log.e(TAG, "Replay target is missing or unreadable: ${targets.joinToString(";")}")
            Log.w(
                TRACKER_TAG,
                "EVAL_EVENT type=REPLAY_INPUT state=target_missing targets=${targets.joinToString(";")}"
            )
            return
        }

        val videoPath = replayVideoPath?.takeIf { it.isNotBlank() } ?: DEFAULT_REPLAY_VIDEO_PATH
        val replayStartMs = (replayStartSec * 1000f).toLong().coerceAtLeast(0L)
        val videoFile = File(videoPath)
        if (!videoFile.exists()) {
            Toast.makeText(this, getString(R.string.toast_replay_video_missing), Toast.LENGTH_LONG).show()
            Log.e(TAG, "Replay video does not exist: $videoPath")
            Log.w(TRACKER_TAG, "EVAL_EVENT type=REPLAY_INPUT state=video_missing path=$videoPath")
            return
        }
        Log.w(
            TRACKER_TAG,
            "EVAL_EVENT type=REPLAY_INPUT state=ready video=$videoPath size=${videoFile.length()} " +
                "targets=${targets.size} loop=$replayLoopEnabled startSec=${"%.2f".format(replayStartSec)} " +
                "targetAppearSec=${"%.2f".format(replayTargetAppearSec)} " +
                "catchup=$replayCatchupEnabled replayFps=${"%.2f".format(replayFpsOverride)}"
        )

        viewFinder.visibility = View.INVISIBLE
        replayFrameView.visibility = View.VISIBLE

        trackerAnalyzer.beginEvalSession("replay_start")
        replayRunner?.stop()
        replayRunner = VideoReplayRunner(
            videoPath = videoPath,
            startOffsetMs = replayStartMs,
            loop = replayLoopEnabled,
            enableCatchup = replayCatchupEnabled,
            fpsOverride = replayFpsOverride.toDouble(),
            onFrame = { bitmap -> renderReplayFrame(bitmap) },
            onMatFrame = { mat, ptsMs -> trackerAnalyzer.analyzeReplayFrame(mat, ptsMs) },
            onError = { message, throwable ->
                Log.e(TAG, message, throwable)
                runOnUiThread {
                    Toast.makeText(this, getString(R.string.toast_replay_failed), Toast.LENGTH_LONG).show()
                }
            }
        )
        cameraExecutor.execute(replayRunner)
        Toast.makeText(this, getString(R.string.toast_replay_started), Toast.LENGTH_SHORT).show()
    }

    private fun renderReplayFrame(bitmap: Bitmap) {
        replayFrameView.post {
            val old = lastReplayBitmap
            replayFrameView.setImageBitmap(bitmap)
            lastReplayBitmap = bitmap
            if (old != null && !old.isRecycled) {
                old.recycle()
            }
        }
    }

    private fun loadTemplateFromPath(path: String): Boolean {
        return loadTemplatesFromPaths(listOf(path))
    }

    private fun loadTemplatesFromPaths(paths: List<String>): Boolean {
        val bitmaps = ArrayList<Bitmap>(paths.size)
        var skippedBadTemplate = false
        try {
            for (path in paths) {
                val file = File(path)
                if (!file.exists()) continue
                val bitmap = decodeBitmapFromPathWithExif(path) ?: continue
                val quality = evaluateTemplateQuality(bitmap)
                if (!quality.pass) {
                    skippedBadTemplate = true
                    Log.w(
                        TAG,
                        "Skip bad template: $path size=${bitmap.width}x${bitmap.height} " +
                            "normalized=${quality.normalizedWidth}x${quality.normalizedHeight} " +
                            "pixels=${quality.usablePixels} kp=${quality.orbKeypoints} reason=${quality.reason}"
                    )
                    if (!bitmap.isRecycled) bitmap.recycle()
                    continue
                }
                bitmaps += bitmap
            }
            if (bitmaps.isEmpty()) {
                if (skippedBadTemplate) {
                    Toast.makeText(this, TEMPLATE_QUALITY_REJECT_MESSAGE, Toast.LENGTH_LONG).show()
                }
                return false
            }

            val ready = trackerAnalyzer.setTemplateImages(bitmaps, source = "disk")
            if (!ready) {
                Toast.makeText(this, "\u6A21\u677F\u7EB9\u7406\u8FC7\u5F31\uFF0C\u8BF7\u91CD\u65B0\u63D0\u4F9B\u76EE\u6807\u6E05\u6670\u7167\u7247", Toast.LENGTH_LONG).show()
            }
            return ready
        } finally {
            // release java-side pixel refs after OpenCV copies them into Mats
            for (bitmap in bitmaps) {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
    }

    private fun decodeBitmapFromUriWithExif(uri: Uri): Bitmap? {
        val decoded = contentResolver.openInputStream(uri)?.use { stream ->
            BitmapFactory.decodeStream(stream)
        } ?: return null
        val orientation = readExifOrientationFromUri(uri)
        return normalizeBitmapWithExif(decoded, orientation)
    }

    private fun decodeBitmapFromPathWithExif(path: String): Bitmap? {
        val decoded = BitmapFactory.decodeFile(path) ?: return null
        val orientation = readExifOrientationFromPath(path)
        return normalizeBitmapWithExif(decoded, orientation)
    }

    private fun readExifOrientationFromUri(uri: Uri): Int {
        return runCatching {
            contentResolver.openInputStream(uri)?.use { stream ->
                ExifInterface(stream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        }.getOrElse { ExifInterface.ORIENTATION_NORMAL }
    }

    private fun readExifOrientationFromPath(path: String): Int {
        return runCatching {
            ExifInterface(path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
        }.getOrElse { ExifInterface.ORIENTATION_NORMAL }
    }

    private fun normalizeBitmapWithExif(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(-90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            else -> return bitmap
        }
        return runCatching {
            val normalized = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (normalized != bitmap && !bitmap.isRecycled) {
                bitmap.recycle()
            }
            normalized
        }.getOrElse { bitmap }
    }

    private fun evaluateTemplateQuality(bitmap: Bitmap): TemplateQuality {
        val minSide = minOf(bitmap.width, bitmap.height)
        if (minSide < MIN_TEMPLATE_INPUT_SIDE) {
            return TemplateQuality(
                pass = false,
                reason = "short_side",
                normalizedWidth = bitmap.width,
                normalizedHeight = bitmap.height,
                usablePixels = bitmap.width * bitmap.height,
                orbKeypoints = 0
            )
        }

        val (normalizedWidth, normalizedHeight) = normalizeTemplateDimensions(bitmap.width, bitmap.height)
        val usablePixels = normalizedWidth * normalizedHeight
        if (usablePixels < MIN_TEMPLATE_USABLE_PIXELS) {
            return TemplateQuality(
                pass = false,
                reason = "usable_pixels",
                normalizedWidth = normalizedWidth,
                normalizedHeight = normalizedHeight,
                usablePixels = usablePixels,
                orbKeypoints = 0
            )
        }

        val normalizedBitmap =
            if (normalizedWidth == bitmap.width && normalizedHeight == bitmap.height) {
                bitmap
            } else {
                Bitmap.createScaledBitmap(bitmap, normalizedWidth, normalizedHeight, true)
            }
        val keypoints = countOrbKeypoints(normalizedBitmap)
        if (normalizedBitmap !== bitmap && !normalizedBitmap.isRecycled) {
            normalizedBitmap.recycle()
        }

        if (keypoints < MIN_TEMPLATE_ORB_KEYPOINTS) {
            return TemplateQuality(
                pass = false,
                reason = "orb_keypoints",
                normalizedWidth = normalizedWidth,
                normalizedHeight = normalizedHeight,
                usablePixels = usablePixels,
                orbKeypoints = keypoints
            )
        }

        return TemplateQuality(
            pass = true,
            reason = "ok",
            normalizedWidth = normalizedWidth,
            normalizedHeight = normalizedHeight,
            usablePixels = usablePixels,
            orbKeypoints = keypoints
        )
    }

    private fun normalizeTemplateDimensions(width: Int, height: Int): Pair<Int, Int> {
        val w = width.toDouble()
        val h = height.toDouble()
        val maxDim = maxOf(w, h).coerceAtLeast(1.0)
        val scale = when {
            maxDim > MAX_TEMPLATE_INPUT_SIDE -> MAX_TEMPLATE_INPUT_SIDE / maxDim
            maxDim < MIN_TEMPLATE_INPUT_SIDE -> MIN_TEMPLATE_INPUT_SIDE / maxDim
            else -> 1.0
        }
        val outW = maxOf(1, (w * scale).toInt())
        val outH = maxOf(1, (h * scale).toInt())
        return outW to outH
    }

    private fun countOrbKeypoints(bitmap: Bitmap): Int {
        val rgba = Mat()
        val gray = Mat()
        val keypoints = MatOfKeyPoint()
        return try {
            Utils.bitmapToMat(bitmap, rgba)
            if (rgba.channels() == 4) {
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            } else {
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGB2GRAY)
            }
            val orb = ORB.create(QUALITY_CHECK_ORB_FEATURES)
            orb.detect(gray, keypoints)
            keypoints.toArray().size
        } catch (e: Exception) {
            Log.w(TAG, "evaluateTemplateQuality ORB detect failed", e)
            0
        } finally {
            keypoints.release()
            gray.release()
            rgba.release()
        }
    }

    private fun buildTemplateQualityMessage(quality: TemplateQuality): String {
        return when (quality.reason) {
            "short_side" ->
                "图片短边不足 ${MIN_TEMPLATE_INPUT_SIDE}px，请放大目标后重试"
            "usable_pixels" ->
                "模板有效像素不足（${quality.usablePixels} < ${MIN_TEMPLATE_USABLE_PIXELS}），请换更近更清晰目标图"
            "orb_keypoints" ->
                "模板特征点不足（${quality.orbKeypoints} < ${MIN_TEMPLATE_ORB_KEYPOINTS}），请选纹理更清晰且方向一致的目标图"
            else ->
                TEMPLATE_QUALITY_REJECT_MESSAGE
        }
    }

    private fun buildRequiredPermissions(): Array<String> {
        val permissions = mutableListOf<String>()
        if (isReplayMode) {
            if (Build.VERSION.SDK_INT >= 33) {
                permissions += Manifest.permission.READ_MEDIA_IMAGES
                permissions += Manifest.permission.READ_MEDIA_VIDEO
            } else {
                permissions += Manifest.permission.READ_EXTERNAL_STORAGE
            }
        } else {
            permissions += Manifest.permission.CAMERA
        }
        return permissions.toTypedArray()
    }

    private fun allPermissionsGranted(): Boolean {
        return requiredPermissions.all {
            ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun stageNcnnModelsFromAssets(): Pair<String, String>? {
        return runCatching {
            val modelDir = File(filesDir, "ncnn_models")
            if (!modelDir.exists()) {
                modelDir.mkdirs()
            }

            val modelFiles = listOf(
                ASSET_NCNN_BACKBONE_PARAM,
                ASSET_NCNN_BACKBONE_BIN,
                ASSET_NCNN_HEAD_PARAM,
                ASSET_NCNN_HEAD_BIN
            )
            for (assetName in modelFiles) {
                val target = File(modelDir, assetName)
                assets.open(assetName).use { input ->
                    target.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }

            File(modelDir, ASSET_NCNN_BACKBONE_PARAM).absolutePath to
                File(modelDir, ASSET_NCNN_BACKBONE_BIN).absolutePath
        }.onFailure { error ->
            Log.e(TAG, "stageNcnnModelsFromAssets failed", error)
        }.getOrNull()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST_CODE_PERMISSIONS) {
            return
        }

        if (allPermissionsGranted()) {
            startInputPipeline()
        } else {
            Toast.makeText(this, getString(R.string.toast_camera_permission_required), Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    override fun onDestroy() {
        unregisterGpsReadyReceiverIfNeeded()
        trackerAnalyzer.logEvalSummary("activity_destroy")
        replayRunner?.stop()
        replayRunner = null
        lastReplayBitmap?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }
        lastReplayBitmap = null
        NativeTrackerBridge.release()
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val TRACKER_TAG = "Tracker"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val EXTRA_TRACKER_MODE = "tracker_mode"
        private const val EXTRA_EVAL_USE_REPLAY = "eval_use_replay"
        private const val EXTRA_EVAL_TARGET_PATH = "eval_target_path"
        private const val EXTRA_EVAL_TARGET_PATHS = "eval_target_paths"
        private const val EXTRA_EVAL_VIDEO_PATH = "eval_video_path"
        private const val EXTRA_EVAL_REPLAY_START_SEC = "eval_replay_start_sec"
        private const val EXTRA_EVAL_TARGET_APPEAR_SEC = "eval_target_appear_sec"
        private const val EXTRA_EVAL_REPLAY_CATCHUP = "eval_replay_catchup"
        private const val EXTRA_EVAL_REPLAY_FPS = "eval_replay_fps"
        private const val EXTRA_EVAL_LOOP = "eval_loop"
        private const val EXTRA_EVAL_PARAMS = "eval_params"
        private const val ACTION_GPS_READY = "com.example.dronetracker.GPS_READY"
        private const val EXTRA_GPS_READY = "ready"
        private const val EXTRA_GPS_REASON = "reason"
        private const val DEFAULT_GPS_READY_REASON = "adb_broadcast"
        private const val MIN_TEMPLATE_INPUT_SIDE = 128
        private const val MAX_TEMPLATE_INPUT_SIDE = 480.0
        private const val MIN_TEMPLATE_USABLE_PIXELS = 50_000
        private const val MIN_TEMPLATE_ORB_KEYPOINTS = 50
        private const val QUALITY_CHECK_ORB_FEATURES = 900
        private const val CROP_MIN_RESULT_SIDE = 256
        private const val CROP_OUTPUT_MAX_SIDE = 640
        private const val TEMPLATE_TOO_SMALL_MESSAGE = "\u56FE\u7247\u592A\u5C0F\uFF0C\u8BF7\u9009\u62E9\u66F4\u6E05\u6670\u7684\u76EE\u6807\u7167\u7247"
        private const val TEMPLATE_QUALITY_REJECT_MESSAGE = "模板质量不足，请选择清晰目标图并保持目标方向与视频首帧一致"
        private const val TEMPLATE_ORIENTATION_HINT_MESSAGE = "请尽量让目标方向与首帧一致（建议朝上，偏差不超过 ±30°）"
        private const val DEFAULT_REPLAY_TARGET_PATH = "/sdcard/Download/Video_Search/target0417_s640.jpg"
        private const val DEFAULT_REPLAY_VIDEO_PATH = "/sdcard/Download/Video_Search/scene_20260417.mp4"
        private const val ASSET_NCNN_BACKBONE_PARAM = "nanotrack_backbone_sim-opt.param"
        private const val ASSET_NCNN_BACKBONE_BIN = "nanotrack_backbone_sim-opt.bin"
        private const val ASSET_NCNN_HEAD_PARAM = "nanotrack_head_sim-opt.param"
        private const val ASSET_NCNN_HEAD_BIN = "nanotrack_head_sim-opt.bin"
    }

    private data class TemplateQuality(
        val pass: Boolean,
        val reason: String,
        val normalizedWidth: Int,
        val normalizedHeight: Int,
        val usablePixels: Int,
        val orbKeypoints: Int
    )
}


