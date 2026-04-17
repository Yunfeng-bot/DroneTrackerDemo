package com.example.dronetracker

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.dronetracker.nativebridge.NativeTrackerBridge
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.LinkedHashSet
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class EvaluationActivity : AppCompatActivity() {

    private lateinit var replayFrameView: ImageView
    private lateinit var overlayView: TrackingOverlayView
    private lateinit var statusView: TextView

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var trackerAnalyzer: OpenCVTrackerAnalyzer
    private lateinit var requiredPermissions: Array<String>

    private var replayRunner: VideoReplayRunner? = null
    private var csvLogger: EvalCsvLogger? = null
    private var csvOutputFile: File? = null
    private var lastReplayBitmap: Bitmap? = null
    private var frameSeq = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_evaluation)

        replayFrameView = findViewById(R.id.evalReplayFrameView)
        overlayView = findViewById(R.id.evalOverlayView)
        statusView = findViewById(R.id.evalStatusView)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV init failed", Toast.LENGTH_LONG).show()
            Log.e(TAG, "OpenCV init failed")
            finish()
            return
        }

        val ncnnBackbonePaths = stageNcnnModelsFromAssets()
        if (ncnnBackbonePaths != null) {
            NativeTrackerBridge.setDefaultModelPaths(ncnnBackbonePaths.first, ncnnBackbonePaths.second)
            Log.i(
                TAG,
                "native model assets staged backboneParam=${ncnnBackbonePaths.first} backboneBin=${ncnnBackbonePaths.second}"
            )
        } else {
            Log.w(TAG, "native model assets staging failed in EvaluationActivity")
        }

        trackerAnalyzer = OpenCVTrackerAnalyzer(overlayView)
        trackerAnalyzer.setTrackerMode(intent.getStringExtra(EXTRA_TRACKER_MODE))
        trackerAnalyzer.applyRuntimeOverrides(intent.getStringExtra(EXTRA_EVAL_PARAMS))

        requiredPermissions = buildRequiredPermissions()

        if (allPermissionsGranted()) {
            startReplayEvaluation()
        } else {
            ActivityCompat.requestPermissions(this, requiredPermissions, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun startReplayEvaluation() {
        val targetPaths = resolveTargetPaths()
        if (!loadTemplatesFromPaths(targetPaths)) {
            Toast.makeText(this, "Template image is missing or unreadable", Toast.LENGTH_LONG).show()
            statusView.text = "Template unavailable"
            return
        }

        val videoPath = intent.getStringExtra(EXTRA_EVAL_VIDEO_PATH)
            ?.takeIf { it.isNotBlank() }
            ?: DEFAULT_EVAL_VIDEO_PATH
        if (!File(videoPath).exists()) {
            Toast.makeText(this, "Replay video does not exist: $videoPath", Toast.LENGTH_LONG).show()
            statusView.text = "Replay video missing"
            return
        }

        val csvPathFromIntent = intent.getStringExtra(EXTRA_EVAL_CSV_PATH)?.trim().orEmpty()
        csvOutputFile = prepareCsvFile(csvPathFromIntent)
        csvLogger = EvalCsvLogger(csvOutputFile!!)
        frameSeq = 0L

        val loop = intent.getBooleanExtra(EXTRA_EVAL_LOOP, false)
        statusView.text = "Running replay benchmark..."
        Log.i(TAG, "Evaluation start video=$videoPath csv=${csvOutputFile?.absolutePath} loop=$loop")

        replayRunner?.stop()
        replayRunner = VideoReplayRunner(
            videoPath = videoPath,
            loop = loop,
            previewEveryNFrames = 2,
            onFrame = { bitmap -> renderReplayFrame(bitmap) },
            onMatFrame = { mat -> analyzeAndLogFrame(mat) },
            onError = { message, throwable ->
                Log.e(TAG, message, throwable)
                runOnUiThread {
                    statusView.text = "Replay failed"
                    Toast.makeText(this, "Replay failed", Toast.LENGTH_LONG).show()
                }
            }
        )
        cameraExecutor.execute(replayRunner)
    }

    private fun analyzeAndLogFrame(mat: Mat) {
        val startNs = SystemClock.elapsedRealtimeNanos()
        trackerAnalyzer.analyzeReplayFrame(mat)
        val latencyMs = (SystemClock.elapsedRealtimeNanos() - startNs).toDouble() / 1_000_000.0
        val prediction = trackerAnalyzer.latestPredictionSnapshot()

        val frameId = frameSeq++
        csvLogger?.append(frameId, latencyMs, prediction)
        if (frameId % CSV_FLUSH_INTERVAL == 0L) {
            csvLogger?.flush()
            writeEvalSummaryFile()
        }

        if (frameId % STATUS_UPDATE_INTERVAL == 0L) {
            val text = String.format(
                Locale.US,
                "frames=%d latency=%.2fms pred=(%d,%d,%d,%d) conf=%.3f tracking=%s\nCSV=%s",
                frameId,
                latencyMs,
                prediction.x,
                prediction.y,
                prediction.width,
                prediction.height,
                prediction.confidence,
                prediction.tracking,
                csvOutputFile?.absolutePath ?: "n/a"
            )
            statusView.post { statusView.text = text }
        }
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

    private fun resolveTargetPaths(): List<String> {
        val set = LinkedHashSet<String>()
        val singlePath = intent.getStringExtra(EXTRA_EVAL_TARGET_PATH)
            ?.takeIf { it.isNotBlank() }
            ?: DEFAULT_EVAL_TARGET_PATH
        set += singlePath

        val listRaw = intent.getStringExtra(EXTRA_EVAL_TARGET_PATHS)
        if (!listRaw.isNullOrBlank()) {
            listRaw.split(';', ',')
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .forEach { set += it }
        }
        return set.toList()
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
                Toast.makeText(this, "Template texture is too weak", Toast.LENGTH_LONG).show()
            }
            return ready
        } finally {
            for (bitmap in bitmaps) {
                if (!bitmap.isRecycled) bitmap.recycle()
            }
        }
    }

    private fun prepareCsvFile(explicitPath: String): File {
        val outFile = if (explicitPath.isNotBlank()) {
            File(explicitPath)
        } else {
            val dir = File(getExternalFilesDir(null), "eval")
            if (!dir.exists()) {
                dir.mkdirs()
            }
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            File(dir, "eval_${stamp}.csv")
        }
        outFile.parentFile?.mkdirs()
        return outFile
    }

    private fun buildRequiredPermissions(): Array<String> {
        val permissions = mutableListOf<String>()
        if (Build.VERSION.SDK_INT >= 33) {
            permissions += Manifest.permission.READ_MEDIA_IMAGES
            permissions += Manifest.permission.READ_MEDIA_VIDEO
        } else {
            permissions += Manifest.permission.READ_EXTERNAL_STORAGE
        }
        return permissions.toTypedArray()
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
            startReplayEvaluation()
        } else {
            Toast.makeText(this, "Storage permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onDestroy() {
        replayRunner?.stop()
        replayRunner = null

        csvLogger?.runCatching { close() }
        csvLogger = null

        trackerAnalyzer.logEvalSummary("evaluation_destroy")
        writeEvalSummaryFile()

        lastReplayBitmap?.let {
            if (!it.isRecycled) {
                it.recycle()
            }
        }
        lastReplayBitmap = null

        cameraExecutor.shutdown()
        super.onDestroy()
    }

    private fun writeEvalSummaryFile() {
        val csv = csvOutputFile ?: return
        val metrics = trackerAnalyzer.evalMetricsSnapshot()
        val summary = resolveSummaryFile(csv)
        val text = buildString {
            appendLine("csv=${csv.absolutePath}")
            appendLine("frames=${metrics.frames}")
            appendLine("locks=${metrics.locks}")
            appendLine("lost=${metrics.lost}")
            appendLine(String.format(Locale.US, "trackRatio=%.4f", metrics.trackRatio))
            appendLine(String.format(Locale.US, "avgFrameMs=%.4f", metrics.avgFrameMs))
            appendLine(String.format(Locale.US, "firstLockSec=%.4f", metrics.firstLockSec))
            appendLine("kalmanPredHold=${metrics.kalmanPredHold}")
            appendLine(String.format(Locale.US, "kalmanPredHoldRatio=%.4f", metrics.kalmanPredHoldRatio))
        }
        try {
            summary.writeText(text, Charsets.UTF_8)
            Log.i(TAG, "Evaluation summary saved: ${summary.absolutePath}")
        } catch (e: IOException) {
            // Fallback to app-private eval dir if external custom path is not writable.
            val fallbackDir = File(getExternalFilesDir(null), "eval").apply { mkdirs() }
            val fallback = File(fallbackDir, "${csv.nameWithoutExtension}.summary.txt")
            runCatching { fallback.writeText(text, Charsets.UTF_8) }
                .onSuccess { Log.i(TAG, "Evaluation summary saved (fallback): ${fallback.absolutePath}") }
                .onFailure { err -> Log.e(TAG, "writeEvalSummaryFile failed: ${summary.absolutePath}", err) }
        }
    }

    private fun resolveSummaryFile(csv: File): File {
        val parent = csv.parentFile
        if (parent != null && (parent.exists() || parent.mkdirs()) && parent.canWrite()) {
            return File(parent, "${csv.nameWithoutExtension}.summary.txt")
        }
        val fallbackDir = File(getExternalFilesDir(null), "eval").apply { mkdirs() }
        return File(fallbackDir, "${csv.nameWithoutExtension}.summary.txt")
    }

    companion object {
        private const val TAG = "EvaluationActivity"
        private const val REQUEST_CODE_PERMISSIONS = 11

        private const val STATUS_UPDATE_INTERVAL = 10L
        private const val CSV_FLUSH_INTERVAL = 15L

        private const val EXTRA_TRACKER_MODE = "tracker_mode"
        private const val EXTRA_EVAL_PARAMS = "eval_params"
        private const val EXTRA_EVAL_VIDEO_PATH = "eval_video_path"
        private const val EXTRA_EVAL_TARGET_PATH = "eval_target_path"
        private const val EXTRA_EVAL_TARGET_PATHS = "eval_target_paths"
        private const val EXTRA_EVAL_CSV_PATH = "eval_csv_path"
        private const val EXTRA_EVAL_LOOP = "eval_loop"

        private const val DEFAULT_EVAL_VIDEO_PATH = "/sdcard/test.mp4"
        private const val DEFAULT_EVAL_TARGET_PATH = "/sdcard/Download/Video_Search/target.jpg"
        private const val ASSET_NCNN_BACKBONE_PARAM = "nanotrack_backbone_sim-opt.param"
        private const val ASSET_NCNN_BACKBONE_BIN = "nanotrack_backbone_sim-opt.bin"
        private const val ASSET_NCNN_HEAD_PARAM = "nanotrack_head_sim-opt.param"
        private const val ASSET_NCNN_HEAD_BIN = "nanotrack_head_sim-opt.bin"
    }
}

