package com.example.dronetracker.nativebridge

import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy

object NativeTrackerBridge {

    enum class Backend(val value: Int) {
        NCNN(0),
        RKNN(1)
    }

    data class NativeTrackResult(
        val x: Float,
        val y: Float,
        val w: Float,
        val h: Float,
        val confidence: Float,
        val similarity: Float
    )

    private const val TAG = "NativeTrackerBridge"

    @Volatile
    private var libraryLoaded = false

    @Volatile
    private var loadFailure: Throwable? = null

    @Volatile
    private var defaultModelParamPath: String? = null

    @Volatile
    private var defaultModelBinPath: String? = null

    init {
        runCatching {
            System.loadLibrary("dronetracker_native")
            libraryLoaded = true
        }.onFailure { error ->
            loadFailure = error
            Log.w(TAG, "native library unavailable: ${error.message}")
        }
    }

    fun initializeEngine(
        backend: Backend = Backend.NCNN,
        modelParamPath: String? = null,
        modelBinPath: String? = null
    ): Boolean {
        if (!libraryLoaded) return false
        val resolvedParam = modelParamPath?.takeIf { it.isNotBlank() }
            ?: if (backend == Backend.NCNN) defaultModelParamPath else null
        val resolvedBin = modelBinPath?.takeIf { it.isNotBlank() }
            ?: if (backend == Backend.NCNN) defaultModelBinPath else null
        return runCatching {
            nativeInitEngine(backend.value, resolvedParam, resolvedBin)
        }.getOrElse {
            Log.e(TAG, "initializeEngine failed", it)
            false
        }
    }

    fun setDefaultModelPaths(modelParamPath: String?, modelBinPath: String?) {
        defaultModelParamPath = modelParamPath?.takeIf { it.isNotBlank() }
        defaultModelBinPath = modelBinPath?.takeIf { it.isNotBlank() }
    }

    fun initTarget(image: ImageProxy, bbox: Rect): Boolean {
        if (!libraryLoaded) return false
        if (image.planes.size < 3) return false

        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]

        return runCatching {
            nativeInitTarget(
                y.buffer,
                y.rowStride,
                y.pixelStride,
                u.buffer,
                u.rowStride,
                u.pixelStride,
                v.buffer,
                v.rowStride,
                v.pixelStride,
                image.width,
                image.height,
                image.imageInfo.rotationDegrees,
                bbox.left.toFloat(),
                bbox.top.toFloat(),
                bbox.width().toFloat(),
                bbox.height().toFloat()
            )
        }.getOrElse {
            Log.e(TAG, "initTarget failed", it)
            false
        }
    }

    fun track(image: ImageProxy): NativeTrackResult? {
        if (!libraryLoaded) return null
        if (image.planes.size < 3) return null

        val y = image.planes[0]
        val u = image.planes[1]
        val v = image.planes[2]

        val raw = runCatching {
            nativeTrack(
                y.buffer,
                y.rowStride,
                y.pixelStride,
                u.buffer,
                u.rowStride,
                u.pixelStride,
                v.buffer,
                v.rowStride,
                v.pixelStride,
                image.width,
                image.height,
                image.imageInfo.rotationDegrees
            )
        }.getOrElse {
            Log.e(TAG, "track failed", it)
            null
        } ?: return null

        if (raw.size < 5) return null
        val similarity = if (raw.size >= 6) raw[5] else raw[4]
        return NativeTrackResult(raw[0], raw[1], raw[2], raw[3], raw[4], similarity)
    }

    fun setPriorBbox(bbox: Rect): Boolean {
        if (!libraryLoaded) return false
        return runCatching {
            nativeSetPriorBbox(
                bbox.left.toFloat(),
                bbox.top.toFloat(),
                bbox.width().toFloat(),
                bbox.height().toFloat()
            )
        }.getOrElse {
            Log.e(TAG, "setPriorBbox failed", it)
            false
        }
    }

    fun initTargetGray(gray: ByteArray, width: Int, height: Int, bbox: Rect): Boolean {
        if (!libraryLoaded) return false
        if (gray.size < width * height || width <= 0 || height <= 0) return false
        return runCatching {
            nativeInitTargetGray(
                gray,
                width,
                height,
                bbox.left.toFloat(),
                bbox.top.toFloat(),
                bbox.width().toFloat(),
                bbox.height().toFloat()
            )
        }.getOrElse {
            Log.e(TAG, "initTargetGray failed", it)
            false
        }
    }

    fun trackGray(gray: ByteArray, width: Int, height: Int): NativeTrackResult? {
        if (!libraryLoaded) return null
        if (gray.size < width * height || width <= 0 || height <= 0) return null
        val raw = runCatching {
            nativeTrackGray(gray, width, height)
        }.getOrElse {
            Log.e(TAG, "trackGray failed", it)
            null
        } ?: return null

        if (raw.size < 5) return null
        val similarity = if (raw.size >= 6) raw[5] else raw[4]
        return NativeTrackResult(raw[0], raw[1], raw[2], raw[3], raw[4], similarity)
    }

    fun reset() {
        if (!libraryLoaded) return
        runCatching { nativeReset() }
    }

    fun release() {
        if (!libraryLoaded) return
        runCatching { nativeRelease() }
    }

    fun backendName(): String {
        if (!libraryLoaded) return "unloaded"
        return runCatching { nativeBackendName() }.getOrDefault("unknown")
    }

    fun isAvailable(): Boolean {
        if (!libraryLoaded) return false
        return runCatching { nativeIsAvailable() }.getOrDefault(false)
    }

    fun loadErrorMessage(): String? = loadFailure?.message

    private external fun nativeInitEngine(backend: Int, modelParamPath: String?, modelBinPath: String?): Boolean

    private external fun nativeInitTarget(
        yBuffer: java.nio.ByteBuffer,
        yRowStride: Int,
        yPixelStride: Int,
        uBuffer: java.nio.ByteBuffer,
        uRowStride: Int,
        uPixelStride: Int,
        vBuffer: java.nio.ByteBuffer,
        vRowStride: Int,
        vPixelStride: Int,
        width: Int,
        height: Int,
        rotation: Int,
        x: Float,
        y: Float,
        w: Float,
        h: Float
    ): Boolean

    private external fun nativeTrack(
        yBuffer: java.nio.ByteBuffer,
        yRowStride: Int,
        yPixelStride: Int,
        uBuffer: java.nio.ByteBuffer,
        uRowStride: Int,
        uPixelStride: Int,
        vBuffer: java.nio.ByteBuffer,
        vRowStride: Int,
        vPixelStride: Int,
        width: Int,
        height: Int,
        rotation: Int
    ): FloatArray?

    private external fun nativeSetPriorBbox(
        x: Float,
        y: Float,
        w: Float,
        h: Float
    ): Boolean

    private external fun nativeInitTargetGray(
        grayBuffer: ByteArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float,
        w: Float,
        h: Float
    ): Boolean

    private external fun nativeTrackGray(
        grayBuffer: ByteArray,
        width: Int,
        height: Int
    ): FloatArray?

    private external fun nativeReset()

    private external fun nativeRelease()

    private external fun nativeBackendName(): String

    private external fun nativeIsAvailable(): Boolean
}
