package com.example.dronetracker

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.tracking.TrackerKCF
import org.opencv.android.Utils
import java.nio.ByteBuffer

class OpenCVTrackerAnalyzer(
    private val overlayView: TrackingOverlayView
) : ImageAnalysis.Analyzer {

    private var tracker: TrackerKCF? = null
    private var initBox: Rect? = null
    private var isTracking = false
    private var templateMat: Mat? = null
    
    // Ratios to map between Camera resolution and View resolution
    private var scaleX = 1f
    private var scaleY = 1f

    // Call this when the user draws a box on the View
    fun setInitialTarget(viewRect: RectF, viewWidth: Int, viewHeight: Int) {
        // We will initialize the tracker when the next frame arrives based on this box
        initBox = Rect(
            (viewRect.left / scaleX).toInt(),
            (viewRect.top / scaleY).toInt(),
            (viewRect.width() / scaleX).toInt(),
            (viewRect.height() / scaleY).toInt()
        )
    }

    fun setTemplateImage(bitmap: Bitmap) {
        val tempMat = Mat()
        Utils.bitmapToMat(bitmap, tempMat)
        // Convert RGBA to RGB to match the camera frame colorspace
        Imgproc.cvtColor(tempMat, tempMat, Imgproc.COLOR_RGBA2RGB)
        
        // Normalize template size to max 150px to prevent huge gallery images from breaking the matcher
        val maxDim = 150.0
        val tempCols = tempMat.cols().toDouble()
        val tempRows = tempMat.rows().toDouble()
        if (tempCols > maxDim || tempRows > maxDim) {
            val scale = maxDim / java.lang.Math.max(tempCols, tempRows)
            Imgproc.resize(tempMat, tempMat, org.opencv.core.Size(tempCols * scale, tempRows * scale))
        }
        
        templateMat = tempMat
        resetTracking() // Reset tracking state to start hunting for the template
        initBox = null // clear manual box
        Log.d("Tracker", "Template Loaded! Size: ${tempMat.cols()} x ${tempMat.rows()}")
    }

    fun resetTracking() {
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
        if (!isTracking) {
            if (initBox != null) {
                // First time: Initialize the tracker with the frame and the bounding box from touch
                tracker = TrackerKCF.create()
                val box = initBox!!
                if (box.x >= 0 && box.y >= 0 && box.x + box.width <= rgbMat.cols() && box.y + box.height <= rgbMat.rows()) {
                    tracker?.init(rgbMat, box)
                    isTracking = true
                    templateMat = null // Clear template so it doesn't conflict
                } else {
                    Log.e("Tracker", "Initial box is out of bounds")
                    initBox = null
                }
            } else if (templateMat != null) {
                // Auto-detection via Multi-Scale Template Matching
                val template = templateMat!!
                var bestScale = 1.0
                var maxScore = -1.0
                var bestLoc: org.opencv.core.Point? = null
                
                // Coarse-to-Fine Search: downsample the search frame by 2.0 to scan fast, then HD verify to prevent false positives.
                val downsampleFactor = 2.0
                val smallMat = Mat()
                Imgproc.resize(rgbMat, smallMat, org.opencv.core.Size(rgbMat.cols() / downsampleFactor, rgbMat.rows() / downsampleFactor))
                
                val scales = doubleArrayOf(0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
                
                var bestCoarseScore = -1.0
                var bestCoarseLoc: org.opencv.core.Point? = null
                var bestCoarseScale = 1.0
                
                for (scale in scales) {
                    val targetColsOriginal = template.cols() * scale
                    val targetRowsOriginal = template.rows() * scale
                    
                    val targetColsSmall = (targetColsOriginal / downsampleFactor).toInt()
                    val targetRowsSmall = (targetRowsOriginal / downsampleFactor).toInt()
                    
                    if (targetColsSmall <= smallMat.cols() && targetRowsSmall <= smallMat.rows() && targetColsSmall > 5 && targetRowsSmall > 5) {
                        val scaledTemplate = Mat()
                        Imgproc.resize(template, scaledTemplate, org.opencv.core.Size(targetColsSmall.toDouble(), targetRowsSmall.toDouble()))
                        val result = Mat()
                        Imgproc.matchTemplate(smallMat, scaledTemplate, result, Imgproc.TM_CCOEFF_NORMED)
                        val minMaxLoc = Core.minMaxLoc(result)
                        
                        if (minMaxLoc.maxVal > bestCoarseScore) {
                            bestCoarseScore = minMaxLoc.maxVal
                            bestCoarseLoc = minMaxLoc.maxLoc
                            bestCoarseScale = scale
                        }
                        result.release()
                        scaledTemplate.release()
                    }
                }
                smallMat.release()
                
                if (bestCoarseScore >= 0.50 && bestCoarseLoc != null) {
                    // Double Verification: Run HD template match on the exact spot to eliminate False Positives
                    val finalWidth = (template.cols() * bestCoarseScale).toInt()
                    val finalHeight = (template.rows() * bestCoarseScale).toInt()
                    val origX = bestCoarseLoc.x * downsampleFactor
                    val origY = bestCoarseLoc.y * downsampleFactor
                    
                    val padding = 20.0
                    val roiX = Math.max(0.0, origX - padding).toInt()
                    val roiY = Math.max(0.0, origY - padding).toInt()
                    val roiW = Math.min(rgbMat.cols() - roiX.toDouble(), finalWidth + padding * 2).toInt()
                    val roiH = Math.min(rgbMat.rows() - roiY.toDouble(), finalHeight + padding * 2).toInt()
                    
                    if (roiW >= finalWidth && roiH >= finalHeight) {
                        val hdTemplate = Mat()
                        Imgproc.resize(template, hdTemplate, org.opencv.core.Size(finalWidth.toDouble(), finalHeight.toDouble()))
                        val roi = rgbMat.submat(Rect(roiX, roiY, roiW, roiH))
                        val verifyResult = Mat()
                        Imgproc.matchTemplate(roi, hdTemplate, verifyResult, Imgproc.TM_CCOEFF_NORMED)
                        val verifyMinMax = Core.minMaxLoc(verifyResult)
                        
                        if (verifyMinMax.maxVal >= 0.60) {
                            maxScore = verifyMinMax.maxVal
                            bestScale = bestCoarseScale
                            bestLoc = org.opencv.core.Point(roiX + verifyMinMax.maxLoc.x, roiY + verifyMinMax.maxLoc.y)
                        } else {
                            Log.w("Tracker", "False Positive Rejected! Coarse: $bestCoarseScore, HD: ${verifyMinMax.maxVal}")
                        }
                        hdTemplate.release()
                        roi.release()
                        verifyResult.release()
                    }
                }
                
                Log.d("Tracker", "Searching... Max score: $maxScore at scale: $bestScale")
                
                if (maxScore >= 0.60) { // Lowered slightly down to 0.60 since resizing interpolations introduce minor blurring
                    val matchLoc = bestLoc!!
                    val finalWidth = (template.cols() * bestScale).toInt()
                    val finalHeight = (template.rows() * bestScale).toInt()
                    Log.d("Tracker", "★★★ Match Found! Score: $maxScore. Scale: $bestScale. Starting KCF Tracker... ★★★")
                    val box = Rect(matchLoc.x.toInt(), matchLoc.y.toInt(), finalWidth, finalHeight)
                    
                    // Initialize tracker
                    tracker = TrackerKCF.create()
                    tracker?.init(rgbMat, box)
                    isTracking = true
                    
                    // Pass to UI
                    val viewRect = RectF(
                        (box.x * scaleX).toFloat(),
                        (box.y * scaleY).toFloat(),
                        ((box.x + box.width) * scaleX).toFloat(),
                        ((box.y + box.height) * scaleY).toFloat()
                    )
                    overlayView.post {
                        overlayView.updateTrackedObject(viewRect)
                    }
                }
            }
        } else if (isTracking && tracker != null) {
            // Subsequent frames: Update the tracker
            val outRect = Rect()
            val success = tracker!!.update(rgbMat, outRect)

            if (success) {
                // Map the tracking rect back to the screen (View) coordinates
                Log.d("Tracker", "KCF Tracking Lock: ${outRect.x}, ${outRect.y} [${outRect.width}x${outRect.height}]")
                val viewRect = RectF(
                    (outRect.x * scaleX).toFloat(),
                    (outRect.y * scaleY).toFloat(),
                    ((outRect.x + outRect.width) * scaleX).toFloat(),
                    ((outRect.y + outRect.height) * scaleY).toFloat()
                )
                overlayView.updateTrackedObject(viewRect)
            } else {
                Log.w("Tracker", "Tracking lost! Target went out of frame. Reverting to Hunting Mode...")
                // Auto-recovery: Fall back to template matching!
                tracker = null
                isTracking = false
                initBox = null
                // Clear the UI box
                overlayView.post {
                    overlayView.reset()
                }
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
        val uvStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride
        val vStart = vPlane.position()
        val uStart = uPlane.position()
        for (row in 0 until Math.floorDiv(image.height, 2)) {
            var vPos = vStart + row * uvStride
            var uPos = uStart + row * uvStride
            for (col in 0 until Math.floorDiv(image.width, 2)) {
                data[i++] = vPlane.get(vPos)
                data[i++] = uPlane.get(uPos)
                vPos += uvPixelStride
                uPos += uvPixelStride
            }
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
