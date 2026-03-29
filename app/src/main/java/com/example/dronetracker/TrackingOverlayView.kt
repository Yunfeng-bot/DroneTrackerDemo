package com.example.dronetracker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

class TrackingOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var isDrawing = false
    private var startX = 0f
    private var startY = 0f
    private var currentX = 0f
    private var currentY = 0f

    // The rect drawn by the user
    private var userDrawnRect: RectF? = null
    // The rect continuously updated by the tracking algorithm
    private var trackedRect: RectF? = null

    var onBoxSelectedListener: ((RectF) -> Unit)? = null

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val trackingPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // If we are already tracking something, don"t allow drawing a new box until reset
        if (trackedRect != null) return false

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                isDrawing = true
                startX = event.x
                startY = event.y
                currentX = event.x
                currentY = event.y
                userDrawnRect = null
                invalidate()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (isDrawing) {
                    currentX = event.x
                    currentY = event.y
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP -> {
                if (isDrawing) {
                    isDrawing = false
                    val left = minOf(startX, currentX)
                    val right = maxOf(startX, currentX)
                    val top = minOf(startY, currentY)
                    val bottom = maxOf(startY, currentY)

                    // Only consider it a box if it"s large enough (avoid accidental clicks)
                    if (right - left > 50 && bottom - top > 50) {
                        val rect = RectF(left, top, right, bottom)
                        userDrawnRect = rect
                        onBoxSelectedListener?.invoke(rect)
                    } else {
                        userDrawnRect = null
                    }
                    invalidate()
                }
            }
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Draw the user"s selection box while dragging
        if (isDrawing) {
            val left = minOf(startX, currentX)
            val right = maxOf(startX, currentX)
            val top = minOf(startY, currentY)
            val bottom = maxOf(startY, currentY)
            canvas.drawRect(left, top, right, bottom, boxPaint)
        } 
        // Draw the finalized user box (before tracking starts)
        else if (userDrawnRect != null && trackedRect == null) {
            canvas.drawRect(userDrawnRect!!, boxPaint)
        }

        // Draw the tracking box defined by the algorithm
        trackedRect?.let {
            canvas.drawRect(it, trackingPaint)
        }
    }

    fun updateTrackedObject(rect: RectF) {
        trackedRect = rect
        // We can clear the user drawn rect since we assume algorithm took over
        userDrawnRect = null 
        postInvalidate()
    }

    fun reset() {
        isDrawing = false
        userDrawnRect = null
        trackedRect = null
        postInvalidate()
    }
}
