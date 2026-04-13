package com.example.dronetracker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import com.canhub.cropper.CropImageContract
import com.canhub.cropper.CropImageContractOptions
import com.canhub.cropper.CropImageOptions
import com.canhub.cropper.CropImageView
import com.example.dronetracker.nativebridge.NativeTrackerBridge
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var replayFrameView: ImageView
    private lateinit var overlayView: TrackingOverlayView
    private lateinit var btnSelectImage: Button
    private lateinit var btnReset: Button

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var trackerAnalyzer: OpenCVTrackerAnalyzer
    private lateinit var requiredPermissions: Array<String>

    private var replayRunner: VideoReplayRunner? = null
    private var isReplayMode = false
    private var replayTargetPath: String? = null
    private var replayTargetPaths: List<String> = emptyList()
    private var replayVideoPath: String? = null
    private var lastReplayBitmap: Bitmap? = null

    private val cropImageLauncher = registerForActivityResult(CropImageContract()) { result ->
        if (!result.isSuccessful) {
            return@registerForActivityResult
        }

        val uri = result.uriContent ?: return@registerForActivityResult
        runCatching {
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream)
            }
        }.onSuccess { bitmap ->
            if (bitmap == null) {
                Toast.makeText(this, getString(R.string.toast_template_load_failed), Toast.LENGTH_SHORT).show()
                return@onSuccess
            }
            val ok = trackerAnalyzer.setTemplateImage(bitmap)
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
                    fixAspectRatio = false
                )
            )
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        replayFrameView = findViewById(R.id.replayFrameView)
        overlayView = findViewById(R.id.overlayView)
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnReset = findViewById(R.id.btnReset)
        findViewById<ImageButton>(R.id.btnClose).setOnClickListener { finish() }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, getString(R.string.toast_opencv_failed), Toast.LENGTH_LONG).show()
            Log.e(TAG, "OpenCV init failed")
        } else {
            Log.i(TAG, "OpenCV runtime ready for SEARCH/verify pipeline")
        }

        trackerAnalyzer = OpenCVTrackerAnalyzer(overlayView)
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
            selectImageLauncher.launch("image/*")
        }
        btnReset.setOnClickListener {
            trackerAnalyzer.logEvalSummary("reset_button")
            trackerAnalyzer.resetTracking(logSummary = false)
        }

        refreshUiByMode()

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
        requiredPermissions = buildRequiredPermissions()

        Log.i(TAG, "tracker mode=${trackerAnalyzer.currentTrackerMode()}")
        Log.i(TAG, "replay mode=$isReplayMode target=$replayTargetPath targets=${replayTargetPaths.size} video=$replayVideoPath")
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
            return
        }

        val videoPath = replayVideoPath?.takeIf { it.isNotBlank() } ?: DEFAULT_REPLAY_VIDEO_PATH
        if (!File(videoPath).exists()) {
            Toast.makeText(this, getString(R.string.toast_replay_video_missing), Toast.LENGTH_LONG).show()
            Log.e(TAG, "Replay video does not exist: $videoPath")
            return
        }

        viewFinder.visibility = View.INVISIBLE
        replayFrameView.visibility = View.VISIBLE

        replayRunner?.stop()
        replayRunner = VideoReplayRunner(
            videoPath = videoPath,
            loop = true,
            onFrame = { bitmap -> renderReplayFrame(bitmap) },
            onMatFrame = { mat -> trackerAnalyzer.analyzeReplayFrame(mat) },
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
        try {
            for (path in paths) {
                val file = File(path)
                if (!file.exists()) continue
                val bitmap = BitmapFactory.decodeFile(path) ?: continue
                bitmaps += bitmap
            }
            if (bitmaps.isEmpty()) return false

            val ready = trackerAnalyzer.setTemplateImages(bitmaps)
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
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val EXTRA_TRACKER_MODE = "tracker_mode"
        private const val EXTRA_EVAL_USE_REPLAY = "eval_use_replay"
        private const val EXTRA_EVAL_TARGET_PATH = "eval_target_path"
        private const val EXTRA_EVAL_TARGET_PATHS = "eval_target_paths"
        private const val EXTRA_EVAL_VIDEO_PATH = "eval_video_path"
        private const val EXTRA_EVAL_PARAMS = "eval_params"
        private const val DEFAULT_REPLAY_TARGET_PATH = "/sdcard/Download/Video_Search/target.jpg"
        private const val DEFAULT_REPLAY_VIDEO_PATH = "/sdcard/Download/Video_Search/scene.mp4"
    }
}

