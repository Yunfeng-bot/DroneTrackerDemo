#include <jni.h>

#include <string>

#include <android/log.h>

#include "tracker/NanoTrackerEngine.h"
#include "tracker/TrackerTypes.h"

namespace {
constexpr const char* kTag = "NativeTracker";

using dronetracker::FrameBuffer;
using dronetracker::NanoTrackerEngine;
using dronetracker::TrackResult;
using dronetracker::TrackerBackend;
using dronetracker::TrackerBbox;

std::string jStringToStdString(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return "";
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

bool fillFrameBuffer(
    JNIEnv* env,
    jobject yBuffer,
    jint yRowStride,
    jint yPixelStride,
    jobject uBuffer,
    jint uRowStride,
    jint uPixelStride,
    jobject vBuffer,
    jint vRowStride,
    jint vPixelStride,
    jint width,
    jint height,
    jint rotation,
    FrameBuffer* out) {
    if (env == nullptr || out == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    if (yBuffer == nullptr || uBuffer == nullptr || vBuffer == nullptr) {
        return false;
    }

    auto* yPlane = static_cast<const uint8_t*>(env->GetDirectBufferAddress(yBuffer));
    auto* uPlane = static_cast<const uint8_t*>(env->GetDirectBufferAddress(uBuffer));
    auto* vPlane = static_cast<const uint8_t*>(env->GetDirectBufferAddress(vBuffer));
    if (yPlane == nullptr || uPlane == nullptr || vPlane == nullptr) {
        return false;
    }

    out->yPlane = yPlane;
    out->uPlane = uPlane;
    out->vPlane = vPlane;
    out->width = width;
    out->height = height;
    out->rotation = rotation;
    out->yRowStride = yRowStride;
    out->uRowStride = uRowStride;
    out->vRowStride = vRowStride;
    out->yPixelStride = yPixelStride;
    out->uPixelStride = uPixelStride;
    out->vPixelStride = vPixelStride;
    return true;
}

bool fillGrayFrameBuffer(
    const uint8_t* grayPtr,
    jint bufferLength,
    jint width,
    jint height,
    FrameBuffer* out) {
    if (out == nullptr || grayPtr == nullptr || width <= 0 || height <= 0 || bufferLength <= 0) {
        return false;
    }
    const jint grayBytes = width * height;
    const bool packedRgb = bufferLength >= grayBytes * 3;
    out->yPlane = grayPtr;
    out->uPlane = nullptr;
    out->vPlane = nullptr;
    out->width = width;
    out->height = height;
    out->rotation = 0;
    out->yRowStride = packedRgb ? width * 3 : width;
    out->uRowStride = 0;
    out->vRowStride = 0;
    out->yPixelStride = packedRgb ? 3 : 1;
    out->uPixelStride = 0;
    out->vPixelStride = 0;
    return true;
}

jfloatArray buildTrackResultArray(JNIEnv* env, const TrackResult& result) {
    jfloatArray out = env->NewFloatArray(5);
    if (out == nullptr) {
        return nullptr;
    }
    const jfloat values[5] = {
        result.bbox.x,
        result.bbox.y,
        result.bbox.w,
        result.bbox.h,
        result.confidence,
    };
    env->SetFloatArrayRegion(out, 0, 5, values);
    return out;
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeInitEngine(
    JNIEnv* env,
    jobject /*thiz*/,
    jint backend,
    jstring modelParamPath,
    jstring modelBinPath) {
    const std::string paramPath = jStringToStdString(env, modelParamPath);
    const std::string binPath = jStringToStdString(env, modelBinPath);

    const auto backendType = backend == 1 ? TrackerBackend::kRknn : TrackerBackend::kNcnn;
    const bool ok = NanoTrackerEngine::instance().init(backendType, paramPath, binPath);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeInitTarget(
    JNIEnv* env,
    jobject /*thiz*/,
    jobject yBuffer,
    jint yRowStride,
    jint yPixelStride,
    jobject uBuffer,
    jint uRowStride,
    jint uPixelStride,
    jobject vBuffer,
    jint vRowStride,
    jint vPixelStride,
    jint width,
    jint height,
    jint rotation,
    jfloat x,
    jfloat y,
    jfloat w,
    jfloat h) {
    FrameBuffer frame;
    if (!fillFrameBuffer(
            env,
            yBuffer,
            yRowStride,
            yPixelStride,
            uBuffer,
            uRowStride,
            uPixelStride,
            vBuffer,
            vRowStride,
            vPixelStride,
            width,
            height,
            rotation,
            &frame)) {
        return JNI_FALSE;
    }

    TrackerBbox bbox;
    bbox.x = x;
    bbox.y = y;
    bbox.w = w;
    bbox.h = h;
    const bool ok = NanoTrackerEngine::instance().initTarget(frame, bbox);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeTrack(
    JNIEnv* env,
    jobject /*thiz*/,
    jobject yBuffer,
    jint yRowStride,
    jint yPixelStride,
    jobject uBuffer,
    jint uRowStride,
    jint uPixelStride,
    jobject vBuffer,
    jint vRowStride,
    jint vPixelStride,
    jint width,
    jint height,
    jint rotation) {
    FrameBuffer frame;
    if (!fillFrameBuffer(
            env,
            yBuffer,
            yRowStride,
            yPixelStride,
            uBuffer,
            uRowStride,
            uPixelStride,
            vBuffer,
            vRowStride,
            vPixelStride,
            width,
            height,
            rotation,
            &frame)) {
        return nullptr;
    }

    TrackResult result;
    const bool ok = NanoTrackerEngine::instance().track(frame, &result);
    if (!ok || !result.ok) {
        return nullptr;
    }
    return buildTrackResultArray(env, result);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeInitTargetGray(
    JNIEnv* env,
    jobject /*thiz*/,
    jbyteArray grayBuffer,
    jint width,
    jint height,
    jfloat x,
    jfloat y,
    jfloat w,
    jfloat h) {
    if (grayBuffer == nullptr || width <= 0 || height <= 0) {
        return JNI_FALSE;
    }

    const jsize length = env->GetArrayLength(grayBuffer);
    const jsize expected = width * height;
    if (length < expected) {
        return JNI_FALSE;
    }

    jbyte* grayPtr = env->GetByteArrayElements(grayBuffer, nullptr);
    if (grayPtr == nullptr) {
        return JNI_FALSE;
    }

    FrameBuffer frame;
    const bool frameOk = fillGrayFrameBuffer(reinterpret_cast<const uint8_t*>(grayPtr), length, width, height, &frame);
    if (!frameOk) {
        env->ReleaseByteArrayElements(grayBuffer, grayPtr, JNI_ABORT);
        return JNI_FALSE;
    }

    TrackerBbox bbox;
    bbox.x = x;
    bbox.y = y;
    bbox.w = w;
    bbox.h = h;
    const bool ok = NanoTrackerEngine::instance().initTarget(frame, bbox);
    env->ReleaseByteArrayElements(grayBuffer, grayPtr, JNI_ABORT);
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeTrackGray(
    JNIEnv* env,
    jobject /*thiz*/,
    jbyteArray grayBuffer,
    jint width,
    jint height) {
    if (grayBuffer == nullptr || width <= 0 || height <= 0) {
        return nullptr;
    }

    const jsize length = env->GetArrayLength(grayBuffer);
    const jsize expected = width * height;
    if (length < expected) {
        return nullptr;
    }

    jbyte* grayPtr = env->GetByteArrayElements(grayBuffer, nullptr);
    if (grayPtr == nullptr) {
        return nullptr;
    }

    FrameBuffer frame;
    const bool frameOk = fillGrayFrameBuffer(reinterpret_cast<const uint8_t*>(grayPtr), length, width, height, &frame);
    if (!frameOk) {
        env->ReleaseByteArrayElements(grayBuffer, grayPtr, JNI_ABORT);
        return nullptr;
    }

    TrackResult result;
    const bool ok = NanoTrackerEngine::instance().track(frame, &result);
    env->ReleaseByteArrayElements(grayBuffer, grayPtr, JNI_ABORT);
    if (!ok || !result.ok) {
        return nullptr;
    }
    return buildTrackResultArray(env, result);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeReset(
    JNIEnv* /*env*/,
    jobject /*thiz*/) {
    NanoTrackerEngine::instance().reset();
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeRelease(
    JNIEnv* /*env*/,
    jobject /*thiz*/) {
    NanoTrackerEngine::instance().release();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeBackendName(
    JNIEnv* env,
    jobject /*thiz*/) {
    const std::string backend = NanoTrackerEngine::instance().backendName();
    return env->NewStringUTF(backend.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_dronetracker_nativebridge_NativeTrackerBridge_nativeIsAvailable(
    JNIEnv* /*env*/,
    jobject /*thiz*/) {
    return NanoTrackerEngine::instance().available() ? JNI_TRUE : JNI_FALSE;
}
