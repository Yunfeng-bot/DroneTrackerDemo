package com.example.dronetracker

import java.io.Closeable
import java.io.File
import java.io.FileWriter
import java.io.Writer
import java.util.Locale

class EvalCsvLogger(file: File) : Closeable {

    private val writer: Writer = FileWriter(file, false)

    init {
        writer.write("frame_id,latency_ms,predicted_x,predicted_y,predicted_w,predicted_h,confidence_score\n")
        writer.flush()
    }

    @Synchronized
    fun append(frameId: Long, latencyMs: Double, prediction: OpenCVTrackerAnalyzer.PredictionSnapshot) {
        writer.write(
            String.format(
                Locale.US,
                "%d,%.3f,%d,%d,%d,%d,%.4f\n",
                frameId,
                latencyMs,
                prediction.x,
                prediction.y,
                prediction.width,
                prediction.height,
                prediction.confidence
            )
        )
    }

    @Synchronized
    fun flush() {
        writer.flush()
    }

    @Synchronized
    override fun close() {
        writer.flush()
        writer.close()
    }
}
