package com.example.dronetracker

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object Phase1TrialLogger {

    private const val LOG_DIR_NAME = "phase1_logs"
    private const val MAX_LOG_FILES = 12

    @Volatile
    private var activeLogFile: File? = null

    @Synchronized
    fun startSession(context: Context, reason: String): File {
        val file = File(logDir(context), "phase1_${timestampForFile()}.log")
        activeLogFile = file
        appendLine(file, "SESSION_START reason=$reason")
        pruneOldLogs(context)
        return file
    }

    @Synchronized
    fun append(context: Context, source: String, message: String) {
        val file = ensureActiveLogFile(context)
        appendLine(file, "[$source] $message")
    }

    @Synchronized
    fun latestLogFile(context: Context): File? {
        val active = activeLogFile
        if (active != null && active.exists()) return active
        return logDir(context)
            .listFiles { file -> file.isFile && file.extension.equals("log", ignoreCase = true) }
            ?.maxByOrNull { it.lastModified() }
    }

    private fun ensureActiveLogFile(context: Context): File {
        val active = activeLogFile
        if (active != null) return active
        return startSession(context, "implicit")
    }

    private fun appendLine(file: File, body: String) {
        file.parentFile?.mkdirs()
        file.appendText("${timestampForLine()} $body\n", Charsets.UTF_8)
    }

    private fun logDir(context: Context): File {
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        return File(baseDir, LOG_DIR_NAME).apply { mkdirs() }
    }

    private fun pruneOldLogs(context: Context) {
        val files = logDir(context)
            .listFiles { file -> file.isFile && file.extension.equals("log", ignoreCase = true) }
            ?.sortedByDescending { it.lastModified() }
            ?: return
        files.drop(MAX_LOG_FILES).forEach { runCatching { it.delete() } }
    }

    private fun timestampForFile(): String =
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())

    private fun timestampForLine(): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
}
