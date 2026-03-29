package com.example.dronetracker

import android.graphics.ImageFormat
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect2d
import org.opencv.imgproc.Imgproc
import org.opencv.tracking.TrackerKCF
import java.nio.ByteBuffer

class OpenCVTrackerAnalyzer(
    private val overlayView: TrackingOverlayView
) : ImageAnalysis.Analyzer {

    private var tracker: TrackerKCF? = null
    private var initBox: Rect2d? = null
    private var isTracking = false
    
    // Ratios to map between Camera resolution and View resolution
    private var scaleX = 1f
    private var scaleY = 1f

    // Call this when the user draws a box on the View
    fun setInitialTarget(viewRect: RectF, viewWidth: Int, viewHeight: Int) {
        // We will initialize the tracker when the next frame arrives based on this box
        initBox = Rect2d(
            (viewRect.left / scaleX).toDouble(),
            (viewRect.top / scaleY).toDouble(),
            (viewRect.width() / scaleX).toDouble(),
            (viewRect.height() / scaleY).toDouble()
        )
    }

    fun resetTracking() {
        tracker?.clear()
        tracker = null
        isTracking = false
        initBox = null
        overlayView.reset()
    }

    override fun analyze(image: ImageProxy) {
        // 1. Calculate scaling factors
        // Note: CameraX image dimensions usually represent rotated images, 
        // so image.width and image.height are dependent on device rotation.
        // Assuming portrait mode for simplicity in this MVP.
        if (image.imageInfo.rotationDegrees == 90 || image.imageInfo.rotationDegrees == 270) {
            scaleX = overlayView.width.toFloat() / image.height.toFloat()
            scaleY = overlayView.height.toFloat() / image.width.toFloat()
        } else {
            scaleX = overlayView.width.toFloat() / image.width.toFloat()
            scaleY = overlayView.height.toFloat() / image.height.toFloat()
        }

        // 2. Convert ImageProxy (YUV_420_888) to OpenCV Mat (RGB)
        val rgbMat = imageToMat(image)

        // 3. Tracking Logic
        if (!isTracking && initBox != null) {
            // First time: Initialize the tracker with the frame and the bounding box
            tracker = TrackerKCF.create()
            
            // Safety check: ensure box is within image bounds
            val box = initBox!!
            if (box.x >= 0 && box.y >= 0 && box.x + box.width <= rgbMat.cols() && box.y + box.height <= rgbMat.rows()) {
                tracker?.init(rgbMat, box)
                isTracking = true
            } else {
                Log.e("Tracker", "Initial box is out of bounds")
                initBox = null
            }
        } else if (isTracking && tracker != null) {
            // Subsequent frames: Update the tracker
            val outRect = Rect2d()
            val success = tracker!!.update(rgbMat, outRect)

            if (success) {
                // Map the tracking rect back to the screen (View) coordinates
                val viewRect = RectF(
                    (outRect.x * scaleX).toFloat(),
                    (outRect.y * scaleY).toFloat(),
                    ((outRect.x + outRect.width) * scaleX).toFloat(),
                    ((outRect.y + outRect.height) * scaleY).toFloat()
                )
                overlayView.updateTrackedObject(viewRect)
            } else {
                Log.w("Tracker", "Tracking lost!")
                // Optionally handle tracking lost (e.g., change box color to red, or auto-reset)
            }
        }

        rgbMat.release()
        image.close()
    }

    /**
     * Helper to convert CameraX ImageProxy (YUV) to OpenCV Mat (RGB)
     * For 100% accurate YUV mapping, using a script or rs is better, but this works for demo tracking.
     */
    private fun imageToMat(image: ImageProxy): Mat {
        require(image.format == ImageFormat.YUV_420_888) { "Invalid image format" }

        val yPlane = image.planes[0].buffer
        val uPlane = image.planes[1].buffer
        val vPlane = image.planes[2].buffer

        val ySize = yPlane.remaining()
        val uSize = uPlane.remaining()
        val vSize = vPlane.remaining()

        val data = ByteArray(ySize + ySize / 2)

        yPlane.get(data, 0, ySize)

        // NV21 interleaving
        var i = ySize
        var uvStride = image.planes[1].rowStride
        var uvPixelStride = image.planes[1].pixelStride
        var rowEnd = ySize
        for (row in 0 until Math.floorDiv(image.height, 2)) {
            var vPos = vPlane.position()
            var uPos = uPlane.position()
            for (col in 0 until Math.floorDiv(image.width, 2)) {
                data[i++] = vPlane.get(vPos)
                data[i++] = uPlane.get(uPos)
                vPos += uvPixelStride
                uPos += uvPixelStride
            }
            vPlane.position(vPlane.position() + uvStride)
            uPlane.position(uPlane.position() + uvStride)
        }

        val yuvMat = Mat(image.height + image.height / 2, image.width, CvType.CV_8UC1)
        yuvMat.put(0, 0, data)

        val rgbMat = Mat()
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21, 3)

        // Rotate if necessary to match the device orientation (default Portrait)
        if (image.imageInfo.rotationDegrees == 90) {
            Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_CLOCKWISE)
        } else if (image.imageInfo.rotationDegrees == 270) {
            Core.rotate(rgbMat, rgbMat, Core.ROTATE_90_COUNTERCLOCKWISE)
        }

        yuvMat.release()
        return rgbMat
    }
}
