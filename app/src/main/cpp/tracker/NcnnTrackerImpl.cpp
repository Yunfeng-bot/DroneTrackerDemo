#include "tracker/NcnnTrackerImpl.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <utility>

namespace {
constexpr const char* kTag = "NativeTracker";

inline float clampFloat(float value, float lo, float hi) {
    return std::max(lo, std::min(hi, value));
}

inline float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

inline float cosineToUnit(float cosine) {
    return clampFloat((cosine + 1.0f) * 0.5f, 0.0f, 1.0f);
}

inline float changeRatio(float v) {
    if (v <= 1e-6f) {
        return 1.0f;
    }
    return std::max(v, 1.0f / v);
}

inline float szWhFun(float w, float h) {
    const float pad = (w + h) * 0.5f;
    return std::sqrt(std::max(1e-6f, (w + pad) * (h + pad)));
}

inline void computeLogicalSize(const dronetracker::FrameBuffer& frame, int* outWidth, int* outHeight) {
    const bool swapAxes = frame.rotation == 90 || frame.rotation == 270;
    if (outWidth != nullptr) {
        *outWidth = std::max(1, swapAxes ? frame.height : frame.width);
    }
    if (outHeight != nullptr) {
        *outHeight = std::max(1, swapAxes ? frame.width : frame.height);
    }
}

inline void logicalToRaw(const dronetracker::FrameBuffer& frame, float logicalX, float logicalY, float* rawX, float* rawY) {
    const float w = static_cast<float>(std::max(1, frame.width));
    const float h = static_cast<float>(std::max(1, frame.height));
    switch (frame.rotation) {
    case 90:
        if (rawX != nullptr) *rawX = logicalY;
        if (rawY != nullptr) *rawY = (h - 1.0f) - logicalX;
        break;
    case 180:
        if (rawX != nullptr) *rawX = (w - 1.0f) - logicalX;
        if (rawY != nullptr) *rawY = (h - 1.0f) - logicalY;
        break;
    case 270:
        if (rawX != nullptr) *rawX = (w - 1.0f) - logicalY;
        if (rawY != nullptr) *rawY = logicalX;
        break;
    default:
        if (rawX != nullptr) *rawX = logicalX;
        if (rawY != nullptr) *rawY = logicalY;
        break;
    }
}

inline float sampleLumaLogical(const dronetracker::FrameBuffer& frame, float logicalX, float logicalY) {
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const float lx = clampFloat(logicalX, 0.0f, static_cast<float>(logicalW - 1));
    const float ly = clampFloat(logicalY, 0.0f, static_cast<float>(logicalH - 1));

    float rawX = 0.0f;
    float rawY = 0.0f;
    logicalToRaw(frame, lx, ly, &rawX, &rawY);

    const int width = std::max(1, frame.width);
    const int height = std::max(1, frame.height);
    const int x0 = static_cast<int>(std::floor(clampFloat(rawX, 0.0f, static_cast<float>(width - 1))));
    const int y0 = static_cast<int>(std::floor(clampFloat(rawY, 0.0f, static_cast<float>(height - 1))));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);

    const float tx = clampFloat(rawX - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = clampFloat(rawY - static_cast<float>(y0), 0.0f, 1.0f);

    const int rowStride = std::max(1, frame.yRowStride);
    const int pixelStride = std::max(1, frame.yPixelStride);
    const uint8_t* row0 = frame.yPlane + y0 * rowStride;
    const uint8_t* row1 = frame.yPlane + y1 * rowStride;

    const float p00 = static_cast<float>(row0[x0 * pixelStride]);
    const float p10 = static_cast<float>(row0[x1 * pixelStride]);
    const float p01 = static_cast<float>(row1[x0 * pixelStride]);
    const float p11 = static_cast<float>(row1[x1 * pixelStride]);

    const float top = p00 + (p10 - p00) * tx;
    const float bottom = p01 + (p11 - p01) * tx;
    return top + (bottom - top) * ty;
}

inline void samplePackedRgbLogical(
    const dronetracker::FrameBuffer& frame,
    float logicalX,
    float logicalY,
    float* outR,
    float* outG,
    float* outB) {
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const float lx = clampFloat(logicalX, 0.0f, static_cast<float>(logicalW - 1));
    const float ly = clampFloat(logicalY, 0.0f, static_cast<float>(logicalH - 1));

    float rawX = 0.0f;
    float rawY = 0.0f;
    logicalToRaw(frame, lx, ly, &rawX, &rawY);

    const int width = std::max(1, frame.width);
    const int height = std::max(1, frame.height);
    const int x0 = static_cast<int>(std::floor(clampFloat(rawX, 0.0f, static_cast<float>(width - 1))));
    const int y0 = static_cast<int>(std::floor(clampFloat(rawY, 0.0f, static_cast<float>(height - 1))));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);

    const float tx = clampFloat(rawX - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = clampFloat(rawY - static_cast<float>(y0), 0.0f, 1.0f);

    const int rowStride = std::max(1, frame.yRowStride);
    const int pixelStride = std::max(3, frame.yPixelStride);
    const uint8_t* row0 = frame.yPlane + y0 * rowStride;
    const uint8_t* row1 = frame.yPlane + y1 * rowStride;

    const int p00 = x0 * pixelStride;
    const int p10 = x1 * pixelStride;
    const int p01 = x0 * pixelStride;
    const int p11 = x1 * pixelStride;

    const float r00 = static_cast<float>(row0[p00 + 0]);
    const float g00 = static_cast<float>(row0[p00 + 1]);
    const float b00 = static_cast<float>(row0[p00 + 2]);
    const float r10 = static_cast<float>(row0[p10 + 0]);
    const float g10 = static_cast<float>(row0[p10 + 1]);
    const float b10 = static_cast<float>(row0[p10 + 2]);
    const float r01 = static_cast<float>(row1[p01 + 0]);
    const float g01 = static_cast<float>(row1[p01 + 1]);
    const float b01 = static_cast<float>(row1[p01 + 2]);
    const float r11 = static_cast<float>(row1[p11 + 0]);
    const float g11 = static_cast<float>(row1[p11 + 1]);
    const float b11 = static_cast<float>(row1[p11 + 2]);

    const float rTop = r00 + (r10 - r00) * tx;
    const float gTop = g00 + (g10 - g00) * tx;
    const float bTop = b00 + (b10 - b00) * tx;
    const float rBottom = r01 + (r11 - r01) * tx;
    const float gBottom = g01 + (g11 - g01) * tx;
    const float bBottom = b01 + (b11 - b01) * tx;

    *outR = clampFloat(rTop + (rBottom - rTop) * ty, 0.0f, 255.0f);
    *outG = clampFloat(gTop + (gBottom - gTop) * ty, 0.0f, 255.0f);
    *outB = clampFloat(bTop + (bBottom - bTop) * ty, 0.0f, 255.0f);
}

inline void sampleRgbLogical(const dronetracker::FrameBuffer& frame, float logicalX, float logicalY, float* outR, float* outG, float* outB) {
    const float y = sampleLumaLogical(frame, logicalX, logicalY);
    if (frame.uPlane == nullptr || frame.vPlane == nullptr || outR == nullptr || outG == nullptr || outB == nullptr) {
        if (outR != nullptr && outG != nullptr && outB != nullptr && frame.yPlane != nullptr && frame.yPixelStride >= 3) {
            samplePackedRgbLogical(frame, logicalX, logicalY, outR, outG, outB);
            return;
        }
        const float gray = clampFloat(y, 0.0f, 255.0f);
        if (outR != nullptr) *outR = gray;
        if (outG != nullptr) *outG = gray;
        if (outB != nullptr) *outB = gray;
        return;
    }

    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const float lx = clampFloat(logicalX, 0.0f, static_cast<float>(logicalW - 1));
    const float ly = clampFloat(logicalY, 0.0f, static_cast<float>(logicalH - 1));

    float rawX = 0.0f;
    float rawY = 0.0f;
    logicalToRaw(frame, lx, ly, &rawX, &rawY);

    const int uvX = std::max(0, static_cast<int>(std::floor(rawX * 0.5f)));
    const int uvY = std::max(0, static_cast<int>(std::floor(rawY * 0.5f)));
    const int uStride = std::max(1, frame.uRowStride);
    const int vStride = std::max(1, frame.vRowStride);
    const int uPix = std::max(1, frame.uPixelStride);
    const int vPix = std::max(1, frame.vPixelStride);

    const uint8_t* uRow = frame.uPlane + uvY * uStride;
    const uint8_t* vRow = frame.vPlane + uvY * vStride;
    const float u = static_cast<float>(uRow[uvX * uPix]) - 128.0f;
    const float v = static_cast<float>(vRow[uvX * vPix]) - 128.0f;

    const float r = clampFloat(y + 1.402f * v, 0.0f, 255.0f);
    const float g = clampFloat(y - 0.344136f * u - 0.714136f * v, 0.0f, 255.0f);
    const float b = clampFloat(y + 1.772f * u, 0.0f, 255.0f);
    *outR = r;
    *outG = g;
    *outB = b;
}

inline float sampleLumaRaw(const dronetracker::FrameBuffer& frame, float rawX, float rawY) {
    const int width = std::max(1, frame.width);
    const int height = std::max(1, frame.height);
    const int rowStride = std::max(1, frame.yRowStride);
    const int pixelStride = std::max(1, frame.yPixelStride);

    const float x = clampFloat(rawX, 0.0f, static_cast<float>(width - 1));
    const float y = clampFloat(rawY, 0.0f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const uint8_t* row0 = frame.yPlane + y0 * rowStride;
    const uint8_t* row1 = frame.yPlane + y1 * rowStride;
    const float p00 = static_cast<float>(row0[x0 * pixelStride]);
    const float p10 = static_cast<float>(row0[x1 * pixelStride]);
    const float p01 = static_cast<float>(row1[x0 * pixelStride]);
    const float p11 = static_cast<float>(row1[x1 * pixelStride]);
    const float top = p00 + (p10 - p00) * tx;
    const float bottom = p01 + (p11 - p01) * tx;
    return top + (bottom - top) * ty;
}

inline void samplePackedRgbRaw(
    const dronetracker::FrameBuffer& frame,
    float rawX,
    float rawY,
    float* outR,
    float* outG,
    float* outB) {
    const int width = std::max(1, frame.width);
    const int height = std::max(1, frame.height);
    const int rowStride = std::max(1, frame.yRowStride);
    const int pixelStride = std::max(3, frame.yPixelStride);

    const float x = clampFloat(rawX, 0.0f, static_cast<float>(width - 1));
    const float y = clampFloat(rawY, 0.0f, static_cast<float>(height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const uint8_t* row0 = frame.yPlane + y0 * rowStride;
    const uint8_t* row1 = frame.yPlane + y1 * rowStride;
    const int p00 = x0 * pixelStride;
    const int p10 = x1 * pixelStride;
    const int p01 = x0 * pixelStride;
    const int p11 = x1 * pixelStride;

    const float r00 = static_cast<float>(row0[p00 + 0]);
    const float g00 = static_cast<float>(row0[p00 + 1]);
    const float b00 = static_cast<float>(row0[p00 + 2]);
    const float r10 = static_cast<float>(row0[p10 + 0]);
    const float g10 = static_cast<float>(row0[p10 + 1]);
    const float b10 = static_cast<float>(row0[p10 + 2]);
    const float r01 = static_cast<float>(row1[p01 + 0]);
    const float g01 = static_cast<float>(row1[p01 + 1]);
    const float b01 = static_cast<float>(row1[p01 + 2]);
    const float r11 = static_cast<float>(row1[p11 + 0]);
    const float g11 = static_cast<float>(row1[p11 + 1]);
    const float b11 = static_cast<float>(row1[p11 + 2]);

    const float rTop = r00 + (r10 - r00) * tx;
    const float gTop = g00 + (g10 - g00) * tx;
    const float bTop = b00 + (b10 - b00) * tx;
    const float rBottom = r01 + (r11 - r01) * tx;
    const float gBottom = g01 + (g11 - g01) * tx;
    const float bBottom = b01 + (b11 - b01) * tx;

    *outR = clampFloat(rTop + (rBottom - rTop) * ty, 0.0f, 255.0f);
    *outG = clampFloat(gTop + (gBottom - gTop) * ty, 0.0f, 255.0f);
    *outB = clampFloat(bTop + (bBottom - bTop) * ty, 0.0f, 255.0f);
}

inline void sampleRgbRaw(const dronetracker::FrameBuffer& frame, float rawX, float rawY, float* outR, float* outG, float* outB) {
    if (outR == nullptr || outG == nullptr || outB == nullptr) {
        return;
    }
    if (frame.uPlane == nullptr || frame.vPlane == nullptr) {
        if (frame.yPlane != nullptr && frame.yPixelStride >= 3) {
            samplePackedRgbRaw(frame, rawX, rawY, outR, outG, outB);
            return;
        }
        const float gray = clampFloat(sampleLumaRaw(frame, rawX, rawY), 0.0f, 255.0f);
        *outR = gray;
        *outG = gray;
        *outB = gray;
        return;
    }

    const float y = sampleLumaRaw(frame, rawX, rawY);
    const int uvX = std::max(0, static_cast<int>(std::floor(rawX * 0.5f)));
    const int uvY = std::max(0, static_cast<int>(std::floor(rawY * 0.5f)));
    const int uStride = std::max(1, frame.uRowStride);
    const int vStride = std::max(1, frame.vRowStride);
    const int uPix = std::max(1, frame.uPixelStride);
    const int vPix = std::max(1, frame.vPixelStride);
    const uint8_t* uRow = frame.uPlane + uvY * uStride;
    const uint8_t* vRow = frame.vPlane + uvY * vStride;
    const float u = static_cast<float>(uRow[uvX * uPix]) - 128.0f;
    const float v = static_cast<float>(vRow[uvX * vPix]) - 128.0f;

    *outR = clampFloat(y + 1.402f * v, 0.0f, 255.0f);
    *outG = clampFloat(y - 0.344136f * u - 0.714136f * v, 0.0f, 255.0f);
    *outB = clampFloat(y + 1.772f * u, 0.0f, 255.0f);
}
} // namespace

namespace dronetracker {

#if defined(DRONETRACKER_HAVE_NCNN) && DRONETRACKER_HAVE_NCNN

bool NcnnTrackerImpl::loadModel(const std::string& modelParamPath, const std::string& modelBinPath) {
    modelParamPath_ = modelParamPath;
    modelBinPath_ = modelBinPath;
    useDualNetPipeline_ = false;
    headParamPath_.clear();
    headBinPath_.clear();
    backboneFeatureBlob_.clear();

    std::string resolvedParam;
    std::string resolvedBin;
    if (!resolveModelPaths(&resolvedParam, &resolvedBin)) {
        __android_log_print(
            ANDROID_LOG_ERROR,
            kTag,
            "backend=ncnn loadModel failed (model files not found) param=%s bin=%s",
            modelParamPath_.c_str(),
            modelBinPath_.c_str());
        modelReady_ = false;
        return false;
    }

    std::string resolvedHeadParam;
    std::string resolvedHeadBin;
    if (resolveDualNetHeadPaths(resolvedParam, resolvedBin, &resolvedHeadParam, &resolvedHeadBin)) {
        netBackbone_.clear();
        netHead_.clear();

        netBackbone_.opt.use_vulkan_compute = false;
        netBackbone_.opt.use_fp16_storage = useFp16Storage_;
        netBackbone_.opt.use_fp16_arithmetic = useFp16Arithmetic_;
        netBackbone_.opt.use_fp16_packed = useFp16Storage_;

        netHead_.opt.use_vulkan_compute = false;
        netHead_.opt.use_fp16_storage = useFp16Storage_;
        netHead_.opt.use_fp16_arithmetic = useFp16Arithmetic_;
        netHead_.opt.use_fp16_packed = useFp16Storage_;

        if (netBackbone_.load_param(resolvedParam.c_str()) != 0 ||
            netBackbone_.load_model(resolvedBin.c_str()) != 0 ||
            netHead_.load_param(resolvedHeadParam.c_str()) != 0 ||
            netHead_.load_model(resolvedHeadBin.c_str()) != 0) {
            __android_log_print(
                ANDROID_LOG_ERROR,
                kTag,
                "backend=ncnn dual-net load failed backbone=%s/%s head=%s/%s",
                resolvedParam.c_str(),
                resolvedBin.c_str(),
                resolvedHeadParam.c_str(),
                resolvedHeadBin.c_str());
            __android_log_print(ANDROID_LOG_ERROR, kTag, "DT_NCNN_PIPELINE_DUAL_FAIL");
            modelReady_ = false;
            return false;
        }

        useDualNetPipeline_ = true;
        headParamPath_ = resolvedHeadParam;
        headBinPath_ = resolvedHeadBin;
        {
            std::vector<std::string> backboneOutputs;
            for (const char* name : netBackbone_.output_names()) {
                backboneOutputs.emplace_back(name == nullptr ? "" : name);
            }
            backboneFeatureBlob_ = backboneOutputs.empty() || backboneOutputs[0].empty() ? "output" : backboneOutputs[0];
        }
        modelMode_ = ModelMode::kSiamLike;
        templateInputBlob_ = "input";
        searchInputBlob_ = "input";
        scoreOutputBlob_ = "output1";
        minScoreThreshold_ = 0.20f;
        searchScale_ = 2.2f;
        smoothAlpha_ = 0.55f;
        modelReady_ = true;

        __android_log_print(
            ANDROID_LOG_INFO,
            kTag,
            "backend=ncnn dual-net load ok backbone=%s/%s head=%s/%s featBlob=%s fp16=%d",
            resolvedParam.c_str(),
            resolvedBin.c_str(),
            resolvedHeadParam.c_str(),
            resolvedHeadBin.c_str(),
            backboneFeatureBlob_.c_str(),
            useFp16Arithmetic_ ? 1 : 0);
        __android_log_print(ANDROID_LOG_INFO, kTag, "DT_NCNN_PIPELINE_DUAL_OK");
        return true;
    }

    net_.clear();
    net_.opt.use_vulkan_compute = false;
    net_.opt.use_fp16_storage = useFp16Storage_;
    net_.opt.use_fp16_arithmetic = useFp16Arithmetic_;
    net_.opt.use_fp16_packed = useFp16Storage_;

    if (net_.load_param(resolvedParam.c_str()) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "backend=ncnn load_param failed path=%s", resolvedParam.c_str());
        modelReady_ = false;
        return false;
    }
    if (net_.load_model(resolvedBin.c_str()) != 0) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "backend=ncnn load_model failed path=%s", resolvedBin.c_str());
        modelReady_ = false;
        return false;
    }

    inputBlobNames_.clear();
    outputBlobNames_.clear();
    for (const char* name : net_.input_names()) {
        inputBlobNames_.emplace_back(name == nullptr ? "" : name);
    }
    for (const char* name : net_.output_names()) {
        outputBlobNames_.emplace_back(name == nullptr ? "" : name);
    }
    if (inputBlobNames_.empty() || outputBlobNames_.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "backend=ncnn model has no input/output blobs");
        modelReady_ = false;
        return false;
    }

    templateInputBlob_ = inputBlobNames_[0];
    searchInputBlob_ = inputBlobNames_.size() >= 2 ? inputBlobNames_[1] : inputBlobNames_[0];
    scoreOutputBlob_ = outputBlobNames_[0];

    modelMode_ = inputBlobNames_.size() >= 2 ? ModelMode::kSiamLike : ModelMode::kEmbedding;
    minScoreThreshold_ = modelMode_ == ModelMode::kSiamLike ? 0.52f : 0.55f;
    searchScale_ = modelMode_ == ModelMode::kSiamLike ? 2.2f : 2.0f;
    smoothAlpha_ = 0.55f;

    modelReady_ = true;
    __android_log_print(
        ANDROID_LOG_INFO,
        kTag,
        "backend=ncnn single-net load ok param=%s bin=%s mode=%s input0=%s input1=%s out0=%s fp16=%d",
        resolvedParam.c_str(),
        resolvedBin.c_str(),
        modelMode_ == ModelMode::kSiamLike ? "siam-like" : "embedding",
        templateInputBlob_.c_str(),
        searchInputBlob_.c_str(),
        scoreOutputBlob_.c_str(),
        useFp16Arithmetic_ ? 1 : 0);
    __android_log_print(ANDROID_LOG_INFO, kTag, "DT_NCNN_PIPELINE_SINGLE_OK");
    return true;
}
bool NcnnTrackerImpl::init(const FrameBuffer& frame, const TrackerBbox& bbox) {
    if (!modelReady_ || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const TrackerBbox safe = clampBox(bbox, logicalW, logicalH);
    if (safe.w < 8.0f || safe.h < 8.0f) {
        return false;
    }

    const float cx = safe.x + safe.w * 0.5f;
    const float cy = safe.y + safe.h * 0.5f;

    if (modelMode_ == ModelMode::kEmbedding || useDualNetPipeline_) {
        ncnn::Mat templatePatch;
        float patchW = safe.w;
        float patchH = safe.h;
        if (useDualNetPipeline_) {
            constexpr float kContextAmount = 0.5f;
            const float wcZ = safe.w + kContextAmount * (safe.w + safe.h);
            const float hcZ = safe.h + kContextAmount * (safe.w + safe.h);
            const float sZ = std::sqrt(std::max(1e-6f, wcZ * hcZ));
            patchW = sZ;
            patchH = sZ;
        }
        if (!extractPatchToMat(frame, cx, cy, patchW, patchH, templateInputSize_, &templatePatch)) {
            return false;
        }
        ncnn::Mat feature;
        if (!runEmbeddingFeature(templatePatch, &feature) || feature.empty()) {
            return false;
        }
        templateFeature_ = feature.clone();
        if (!computeGapEmbedding(templateFeature_, &templateGapEmbedding_) || templateGapEmbedding_.empty()) {
            return false;
        }
        templateInputMat_ = ncnn::Mat();
    } else {
        ncnn::Mat templatePatch;
        if (!extractPatchToMat(frame, cx, cy, safe.w, safe.h, templateInputSize_, &templatePatch)) {
            return false;
        }
        templateInputMat_ = templatePatch.clone();
        templateFeature_ = ncnn::Mat();
        templateGapEmbedding_.clear();
    }

    hasTemplate_ = true;
    lastBox_ = safe;
    return true;
}

bool NcnnTrackerImpl::track(const FrameBuffer& frame, TrackResult* outResult) {
    if (!hasTemplate_ || outResult == nullptr || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }
    if (modelMode_ == ModelMode::kEmbedding && templateFeature_.empty()) {
        return false;
    }
    if (modelMode_ == ModelMode::kSiamLike && !useDualNetPipeline_ && templateInputMat_.empty()) {
        return false;
    }
    if (modelMode_ == ModelMode::kSiamLike && useDualNetPipeline_ && templateFeature_.empty()) {
        return false;
    }

    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);

    const float prevCx = lastBox_.x + lastBox_.w * 0.5f;
    const float prevCy = lastBox_.y + lastBox_.h * 0.5f;
    const float prevW = std::max(8.0f, lastBox_.w);
    const float prevH = std::max(8.0f, lastBox_.h);

    if (modelMode_ == ModelMode::kSiamLike && useDualNetPipeline_) {
        constexpr float kContextAmount = 0.5f;
        constexpr float kPenaltyK = 0.148f;
        constexpr float kWindowInfluence = 0.462f;
        constexpr float kSizeLr = 0.390f;
        constexpr float kStride = 16.0f;

        const float wcZ = prevW + kContextAmount * (prevW + prevH);
        const float hcZ = prevH + kContextAmount * (prevW + prevH);
        const float sZ = std::sqrt(std::max(1e-6f, wcZ * hcZ));
        const float scaleZ = static_cast<float>(templateInputSize_) / std::max(1e-6f, sZ);
        const float dSearch = static_cast<float>(searchInputSize_ - templateInputSize_) * 0.5f;
        const float pad = dSearch / std::max(1e-6f, scaleZ);
        const float sX = sZ + 2.0f * pad;

        ncnn::Mat searchPatch;
        if (!extractPatchToMat(frame, prevCx, prevCy, sX, sX, searchInputSize_, &searchPatch)) {
            outResult->ok = false;
            outResult->confidence = 0.0f;
            outResult->similarity = 0.0f;
            outResult->bbox = lastBox_;
            return false;
        }

        ncnn::Mat xf;
        if (!runEmbeddingFeature(searchPatch, &xf) || xf.empty()) {
            outResult->ok = false;
            outResult->confidence = 0.0f;
            outResult->similarity = 0.0f;
            outResult->bbox = lastBox_;
            return false;
        }
        std::vector<float> searchGapEmbedding;
        float gapSimilarity = 0.0f;
        bool hasGapSimilarity = false;
        if (!templateGapEmbedding_.empty() && computeGapEmbedding(xf, &searchGapEmbedding) && !searchGapEmbedding.empty()) {
            gapSimilarity = cosineToUnit(cosineSimilarity(templateGapEmbedding_, searchGapEmbedding));
            hasGapSimilarity = true;
        }

        ncnn::Extractor exHead = netHead_.create_extractor();
        exHead.set_light_mode(true);
        if (exHead.input("input1", templateFeature_) != 0 || exHead.input("input2", xf) != 0) {
            outResult->ok = false;
            outResult->confidence = 0.0f;
            outResult->similarity = 0.0f;
            outResult->bbox = lastBox_;
            return false;
        }

        ncnn::Mat cls;
        ncnn::Mat bbox;
        if (exHead.extract("output1", cls) != 0 || cls.empty() || exHead.extract("output2", bbox) != 0 || bbox.empty() ||
            bbox.c < 4) {
            outResult->ok = false;
            outResult->confidence = 0.0f;
            outResult->similarity = 0.0f;
            outResult->bbox = lastBox_;
            return false;
        }

        const ncnn::Mat scoreMap = cls.c >= 2 ? cls.channel(1) : cls;
        const int rows = scoreMap.h;
        const int cols = scoreMap.w;
        if (rows <= 0 || cols <= 0 || scoreMap.total() != static_cast<size_t>(rows * cols)) {
            outResult->ok = false;
            outResult->confidence = 0.0f;
            outResult->similarity = 0.0f;
            outResult->bbox = lastBox_;
            return false;
        }

        ensureDualHanningCache(rows, cols);

        const float targetWScaled = prevW * scaleZ;
        const float targetHScaled = prevH * scaleZ;
        const float targetSzScaled = szWhFun(targetWScaled, targetHScaled);
        const float targetRatioScaled = targetWScaled / std::max(1e-6f, targetHScaled);

        float bestPScore = -1.0f;
        float bestRawScore = 0.0f;
        float bestPenalty = 1.0f;
        float bestX1 = 0.0f;
        float bestY1 = 0.0f;
        float bestX2 = 0.0f;
        float bestY2 = 0.0f;

        const float* bbox0 = bbox.channel(0);
        const float* bbox1 = bbox.channel(1);
        const float* bbox2 = bbox.channel(2);
        const float* bbox3 = bbox.channel(3);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const int idx = i * cols + j;
                const float score = sigmoid(scoreMap[idx]);

                const float gridX = j * kStride;
                const float gridY = i * kStride;
                const float x1 = gridX - bbox0[idx];
                const float y1 = gridY - bbox1[idx];
                const float x2 = gridX + bbox2[idx];
                const float y2 = gridY + bbox3[idx];
                const float w = std::max(1e-3f, x2 - x1);
                const float h = std::max(1e-3f, y2 - y1);

                const float sC = changeRatio(szWhFun(w, h) / std::max(1e-6f, targetSzScaled));
                const float predRatio = w / std::max(1e-6f, h);
                const float rC = changeRatio(targetRatioScaled / std::max(1e-6f, predRatio));
                const float penalty = std::exp(-1.0f * (sC * rC - 1.0f) * kPenaltyK);
                const float window = dualHanningY_[i] * dualHanningX_[j];
                const float pScore = (penalty * score) * (1.0f - kWindowInfluence) + window * kWindowInfluence;

                if (pScore > bestPScore) {
                    bestPScore = pScore;
                    bestRawScore = score;
                    bestPenalty = penalty;
                    bestX1 = x1;
                    bestY1 = y1;
                    bestX2 = x2;
                    bestY2 = y2;
                }
            }
        }

        const bool recoverByGap = hasGapSimilarity && gapSimilarity >= gapRecoverSimilarityThreshold_;
        if (bestRawScore < minScoreThreshold_ && !recoverByGap) {
            outResult->ok = false;
            outResult->confidence = clampFloat(bestRawScore, 0.0f, 1.0f);
            outResult->similarity = hasGapSimilarity ? gapSimilarity : clampFloat(bestRawScore, 0.0f, 1.0f);
            outResult->bbox = lastBox_;
            return false;
        }

        if (recoverByGap && bestRawScore < minScoreThreshold_) {
            __android_log_print(
                ANDROID_LOG_INFO,
                kTag,
                "backend=ncnn dual recover-by-gap raw=%.3f gap=%.3f th=%.3f",
                bestRawScore,
                gapSimilarity,
                gapRecoverSimilarityThreshold_);
        }

        const float predXs = (bestX1 + bestX2) * 0.5f;
        const float predYs = (bestY1 + bestY2) * 0.5f;
        const float predW = std::max(1e-3f, bestX2 - bestX1) / std::max(1e-6f, scaleZ);
        const float predH = std::max(1e-3f, bestY2 - bestY1) / std::max(1e-6f, scaleZ);
        const float diffXs = (predXs - static_cast<float>(searchInputSize_) * 0.5f) / std::max(1e-6f, scaleZ);
        const float diffYs = (predYs - static_cast<float>(searchInputSize_) * 0.5f) / std::max(1e-6f, scaleZ);
        const float sizeLr = clampFloat(bestPenalty * bestRawScore * kSizeLr, 0.0f, 1.0f);

        const float nextCx = prevCx + diffXs;
        const float nextCy = prevCy + diffYs;
        const float nextW = predW * sizeLr + prevW * (1.0f - sizeLr);
        const float nextH = predH * sizeLr + prevH * (1.0f - sizeLr);

        TrackerBbox decoded{};
        decoded.w = std::max(8.0f, nextW);
        decoded.h = std::max(8.0f, nextH);
        decoded.x = nextCx - decoded.w * 0.5f;
        decoded.y = nextCy - decoded.h * 0.5f;
        decoded = clampBox(decoded, logicalW, logicalH);

        lastBox_ = decoded;
        outResult->ok = true;
        outResult->bbox = decoded;
        const float calibratedConfidence = clampFloat(bestRawScore * 1.35f + 0.05f, 0.0f, 1.0f);
        float fusedConfidence = clampFloat(calibratedConfidence * 0.90f + bestPScore * 0.10f, 0.0f, 1.0f);
        if (recoverByGap) {
            fusedConfidence = std::max(fusedConfidence, gapSimilarity * 0.92f);
        }
        outResult->confidence = fusedConfidence;
        outResult->similarity = hasGapSimilarity ? gapSimilarity : clampFloat(bestRawScore, 0.0f, 1.0f);
        return true;
    }

    const float searchRadiusX = std::max(6.0f, prevW * (searchScale_ - 1.0f) * 0.5f);
    const float searchRadiusY = std::max(6.0f, prevH * (searchScale_ - 1.0f) * 0.5f);
    const float scaleProbes[] = {0.92f, 1.08f};
    const float offsets[] = {-1.0f, 0.0f, 1.0f};

    float bestConfidence = -1.0f;
    float bestWindowedConfidence = -1.0f;
    TrackerBbox bestBox = lastBox_;

    auto evaluateCandidate = [&](float cx, float cy, float boxW, float boxH, float* outConfidence, float* outWindowedConfidence) -> bool {
        ncnn::Mat searchPatch;
        if (!extractPatchToMat(frame, cx, cy, boxW, boxH, searchInputSize_, &searchPatch)) {
            return false;
        }

        float confidence = 0.0f;
        if (modelMode_ == ModelMode::kEmbedding) {
            ncnn::Mat feature;
            if (!runEmbeddingFeature(searchPatch, &feature) || feature.empty()) {
                return false;
            }
            float cosine = cosineSimilarity(templateFeature_, feature);
            if (!templateGapEmbedding_.empty()) {
                std::vector<float> searchGapEmbedding;
                if (computeGapEmbedding(feature, &searchGapEmbedding) && !searchGapEmbedding.empty()) {
                    cosine = cosineSimilarity(templateGapEmbedding_, searchGapEmbedding);
                }
            }
            confidence = cosineToUnit(cosine);
        } else {
            float siamScore = 0.0f;
            const ncnn::Mat& templateRef = useDualNetPipeline_ ? templateFeature_ : templateInputMat_;
            if (!runSiamScore(templateRef, searchPatch, &siamScore)) {
                return false;
            }
            confidence = useDualNetPipeline_
                ? clampFloat(siamScore, 0.0f, 1.0f)
                : clampFloat(sigmoid(siamScore), 0.0f, 1.0f);
        }

        const float windowed = applyCosineWindow(
            confidence,
            cx,
            cy,
            prevCx,
            prevCy,
            searchRadiusX,
            searchRadiusY);
        if (outConfidence != nullptr) {
            *outConfidence = confidence;
        }
        if (outWindowedConfidence != nullptr) {
            *outWindowedConfidence = windowed;
        }
        return true;
    };

    // Stage-1: 2D coarse search (3x3), fixed scale = 1.0
    float coarseBestCx = prevCx;
    float coarseBestCy = prevCy;
    float coarseBestConfidence = -1.0f;
    float coarseBestWindowed = -1.0f;
    for (float oy : offsets) {
        for (float ox : offsets) {
            const float cx = prevCx + ox * searchRadiusX;
            const float cy = prevCy + oy * searchRadiusY;
            float confidence = 0.0f;
            float windowedConfidence = 0.0f;
            if (!evaluateCandidate(cx, cy, prevW, prevH, &confidence, &windowedConfidence)) {
                continue;
            }
            if (windowedConfidence > coarseBestWindowed) {
                coarseBestWindowed = windowedConfidence;
                coarseBestConfidence = confidence;
                coarseBestCx = cx;
                coarseBestCy = cy;
            }
        }
    }
    if (coarseBestWindowed < 0.0f) {
        outResult->ok = false;
        outResult->confidence = 0.0f;
        outResult->similarity = 0.0f;
        outResult->bbox = lastBox_;
        return false;
    }

    // Stage-2: 1D scale probing on coarse best center
    float bestScale = 1.0f;
    float bestScaleConfidence = coarseBestConfidence;
    float bestScaleWindowed = coarseBestWindowed;
    for (float scale : scaleProbes) {
        float confidence = 0.0f;
        float windowedConfidence = 0.0f;
        if (!evaluateCandidate(
                coarseBestCx,
                coarseBestCy,
                prevW * scale,
                prevH * scale,
                &confidence,
                &windowedConfidence)) {
            continue;
        }
        if (windowedConfidence > bestScaleWindowed) {
            bestScaleWindowed = windowedConfidence;
            bestScaleConfidence = confidence;
            bestScale = scale;
        }
    }

    // Seed with the best center/scale pair before local refinement.
    bestWindowedConfidence = bestScaleWindowed;
    bestConfidence = bestScaleConfidence;
    bestBox.x = coarseBestCx - (prevW * bestScale) * 0.5f;
    bestBox.y = coarseBestCy - (prevH * bestScale) * 0.5f;
    bestBox.w = prevW * bestScale;
    bestBox.h = prevH * bestScale;

    // Stage-3: local 2D fine search (3x3) at selected scale.
    const float refineRadiusX = std::max(2.0f, searchRadiusX * 0.35f);
    const float refineRadiusY = std::max(2.0f, searchRadiusY * 0.35f);
    for (float oy : offsets) {
        for (float ox : offsets) {
            const float cx = coarseBestCx + ox * refineRadiusX;
            const float cy = coarseBestCy + oy * refineRadiusY;
            float confidence = 0.0f;
            float windowedConfidence = 0.0f;
            if (!evaluateCandidate(cx, cy, prevW * bestScale, prevH * bestScale, &confidence, &windowedConfidence)) {
                continue;
            }
            if (windowedConfidence > bestWindowedConfidence) {
                bestWindowedConfidence = windowedConfidence;
                bestConfidence = confidence;
                bestBox.x = cx - (prevW * bestScale) * 0.5f;
                bestBox.y = cy - (prevH * bestScale) * 0.5f;
                bestBox.w = prevW * bestScale;
                bestBox.h = prevH * bestScale;
            }
        }
    }

    if (bestWindowedConfidence < minScoreThreshold_) {
        outResult->ok = false;
        outResult->confidence = 0.0f;
        outResult->similarity = 0.0f;
        outResult->bbox = lastBox_;
        return false;
    }

    TrackerBbox smoothed{};
    smoothed.w = lastBox_.w * (1.0f - smoothAlpha_) + bestBox.w * smoothAlpha_;
    smoothed.h = lastBox_.h * (1.0f - smoothAlpha_) + bestBox.h * smoothAlpha_;
    const float bestCxSmooth = bestBox.x + bestBox.w * 0.5f;
    const float bestCySmooth = bestBox.y + bestBox.h * 0.5f;
    const float smoothedCx = prevCx * (1.0f - smoothAlpha_) + bestCxSmooth * smoothAlpha_;
    const float smoothedCy = prevCy * (1.0f - smoothAlpha_) + bestCySmooth * smoothAlpha_;
    smoothed.x = smoothedCx - smoothed.w * 0.5f;
    smoothed.y = smoothedCy - smoothed.h * 0.5f;
    smoothed = clampBox(smoothed, logicalW, logicalH);

    if (modelMode_ == ModelMode::kEmbedding) {
        ncnn::Mat patch;
        if (extractPatchToMat(
                frame,
                smoothed.x + smoothed.w * 0.5f,
                smoothed.y + smoothed.h * 0.5f,
                smoothed.w,
                smoothed.h,
                templateInputSize_,
                &patch)) {
            ncnn::Mat feat;
            if (runEmbeddingFeature(patch, &feat) && !feat.empty()) {
                updateTemplateFeature(feat);
            }
        }
    } else {
        ncnn::Mat patch;
        if (extractPatchToMat(
                frame,
                smoothed.x + smoothed.w * 0.5f,
                smoothed.y + smoothed.h * 0.5f,
                smoothed.w,
                smoothed.h,
                templateInputSize_,
                &patch) &&
            !patch.empty()) {
            if (useDualNetPipeline_) {
                // Keep fixed template feature for dual-net Siam pipeline.
                // Online template update often introduces drift on occlusion.
            } else if (templateInputMat_.empty() || templateInputMat_.total() != patch.total()) {
                templateInputMat_ = patch.clone();
            } else {
                const float beta = clampFloat(templateUpdateRate_, 0.0f, 0.20f);
                const size_t total = patch.total();
                for (size_t i = 0; i < total; ++i) {
                    templateInputMat_[i] = templateInputMat_[i] * (1.0f - beta) + patch[i] * beta;
                }
            }
        }
    }

    lastBox_ = smoothed;
    outResult->ok = true;
    outResult->bbox = lastBox_;
    outResult->confidence = clampFloat(bestConfidence * 0.70f + bestWindowedConfidence * 0.30f, 0.0f, 1.0f);
    outResult->similarity = clampFloat(bestConfidence, 0.0f, 1.0f);
    return true;
}

bool NcnnTrackerImpl::setPrior(const TrackerBbox& bbox) {
    if (!hasTemplate_) {
        return false;
    }
    if (!std::isfinite(bbox.x) || !std::isfinite(bbox.y) ||
        !std::isfinite(bbox.w) || !std::isfinite(bbox.h)) {
        return false;
    }

    TrackerBbox prior = bbox;
    prior.w = std::max(8.0f, prior.w);
    prior.h = std::max(8.0f, prior.h);
    lastBox_ = prior;
    return true;
}

void NcnnTrackerImpl::reset() {
    hasTemplate_ = false;
    lastBox_ = TrackerBbox{};
    templateFeature_ = ncnn::Mat();
    templateGapEmbedding_.clear();
    templateInputMat_ = ncnn::Mat();
}

const char* NcnnTrackerImpl::name() const {
    return "ncnn";
}

bool NcnnTrackerImpl::resolveModelPaths(std::string* outParamPath, std::string* outBinPath) const {
    if (outParamPath == nullptr || outBinPath == nullptr) {
        return false;
    }

    auto tryPair = [&](const std::string& param, const std::string& bin) -> bool {
        if (param.empty() || bin.empty()) return false;
        if (!fileExists(param) || !fileExists(bin)) return false;
        *outParamPath = param;
        *outBinPath = bin;
        return true;
    };

    if (!modelParamPath_.empty() && !modelBinPath_.empty() && tryPair(modelParamPath_, modelBinPath_)) {
        return true;
    }

    if (!modelParamPath_.empty() && modelBinPath_.empty()) {
        const std::string derived = modelParamPath_.substr(0, modelParamPath_.find_last_of('.')) + ".bin";
        if (tryPair(modelParamPath_, derived)) return true;
    }
    if (modelParamPath_.empty() && !modelBinPath_.empty()) {
        const std::string derived = modelBinPath_.substr(0, modelBinPath_.find_last_of('.')) + ".param";
        if (tryPair(derived, modelBinPath_)) return true;
    }

    const std::vector<std::pair<std::string, std::string>> candidates = {
        {"/sdcard/Download/Video_Search/nanotrack.param", "/sdcard/Download/Video_Search/nanotrack.bin"},
        {"/sdcard/Download/Video_Search/nanotrack_backbone_sim-opt.param", "/sdcard/Download/Video_Search/nanotrack_backbone_sim-opt.bin"},
        {"/sdcard/Download/Video_Search/model.param", "/sdcard/Download/Video_Search/model.bin"},
        {"/data/local/tmp/nanotrack.param", "/data/local/tmp/nanotrack.bin"},
    };
    for (const auto& candidate : candidates) {
        if (tryPair(candidate.first, candidate.second)) {
            return true;
        }
    }

    return false;
}

bool NcnnTrackerImpl::resolveDualNetHeadPaths(
    const std::string& backboneParamPath,
    const std::string& backboneBinPath,
    std::string* outHeadParamPath,
    std::string* outHeadBinPath) const {
    if (outHeadParamPath == nullptr || outHeadBinPath == nullptr) {
        return false;
    }

    auto tryPair = [&](const std::string& param, const std::string& bin) -> bool {
        if (param.empty() || bin.empty()) {
            return false;
        }
        if (!fileExists(param) || !fileExists(bin)) {
            return false;
        }
        *outHeadParamPath = param;
        *outHeadBinPath = bin;
        return true;
    };

    const size_t slash = backboneParamPath.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string() : backboneParamPath.substr(0, slash + 1);
    const std::vector<std::pair<std::string, std::string>> candidates = {
        {dir + "nanotrack_head_sim-opt.param", dir + "nanotrack_head_sim-opt.bin"},
        {dir + "nanotrack_head.param", dir + "nanotrack_head.bin"},
        {"/sdcard/Download/Video_Search/nanotrack_head_sim-opt.param", "/sdcard/Download/Video_Search/nanotrack_head_sim-opt.bin"},
        {"/data/local/tmp/nanotrack_head_sim-opt.param", "/data/local/tmp/nanotrack_head_sim-opt.bin"},
    };

    for (const auto& candidate : candidates) {
        if (tryPair(candidate.first, candidate.second)) {
            return true;
        }
    }

    return false;
}

bool NcnnTrackerImpl::fileExists(const std::string& path) const {
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

bool NcnnTrackerImpl::extractPatchToMat(
    const FrameBuffer& frame,
    float centerX,
    float centerY,
    float boxW,
    float boxH,
    int patchSize,
    ncnn::Mat* outMat) const {
    if (outMat == nullptr || patchSize < 8) {
        return false;
    }
    if (boxW < 4.0f || boxH < 4.0f) {
        return false;
    }

    const bool rgbInput = useDualNetPipeline_;
    if (rgbInput) {
        outMat->create(patchSize, patchSize, 3);
    } else {
        outMat->create(patchSize, patchSize, 1);
    }
    if (outMat->empty()) {
        return false;
    }

    const float invSize = 1.0f / static_cast<float>(patchSize);
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const float logicalMaxX = static_cast<float>(std::max(0, logicalW - 1));
    const float logicalMaxY = static_cast<float>(std::max(0, logicalH - 1));
    const float rawMaxX = static_cast<float>(std::max(0, frame.width - 1));
    const float rawMaxY = static_cast<float>(std::max(0, frame.height - 1));
    const float rowStartX = centerX + (((0.5f * invSize) - 0.5f) * boxW);
    const float stepX = boxW * invSize;

    if (rgbInput) {
        float* ch0 = outMat->channel(0);
        float* ch1 = outMat->channel(1);
        float* ch2 = outMat->channel(2);
        size_t idx = 0;
        for (int py = 0; py < patchSize; ++py) {
            const float ry = (static_cast<float>(py) + 0.5f) * invSize - 0.5f;
            const float srcY = clampFloat(centerY + ry * boxH, 0.0f, logicalMaxY);
            float srcX = rowStartX;
            for (int px = 0; px < patchSize; ++px) {
                const float logicalX = clampFloat(srcX, 0.0f, logicalMaxX);
                float rawX = logicalX;
                float rawY = srcY;
                switch (frame.rotation) {
                case 90:
                    rawX = srcY;
                    rawY = rawMaxY - logicalX;
                    break;
                case 180:
                    rawX = rawMaxX - logicalX;
                    rawY = rawMaxY - srcY;
                    break;
                case 270:
                    rawX = rawMaxX - srcY;
                    rawY = logicalX;
                    break;
                default:
                    break;
                }
                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                sampleRgbRaw(frame, rawX, rawY, &r, &g, &b);
                ch0[idx] = r;
                ch1[idx] = g;
                ch2[idx] = b;
                ++idx;
                srcX += stepX;
            }
        }
    } else {
        size_t idx = 0;
        for (int py = 0; py < patchSize; ++py) {
            const float ry = (static_cast<float>(py) + 0.5f) * invSize - 0.5f;
            const float srcY = clampFloat(centerY + ry * boxH, 0.0f, logicalMaxY);
            float srcX = rowStartX;
            for (int px = 0; px < patchSize; ++px) {
                const float logicalX = clampFloat(srcX, 0.0f, logicalMaxX);
                float rawX = logicalX;
                float rawY = srcY;
                switch (frame.rotation) {
                case 90:
                    rawX = srcY;
                    rawY = rawMaxY - logicalX;
                    break;
                case 180:
                    rawX = rawMaxX - logicalX;
                    rawY = rawMaxY - srcY;
                    break;
                case 270:
                    rawX = rawMaxX - srcY;
                    rawY = logicalX;
                    break;
                default:
                    break;
                }
                const float value = sampleLumaRaw(frame, rawX, rawY) / 255.0f;
                (*outMat)[idx++] = value;
                srcX += stepX;
            }
        }
    }
    return true;
}

bool NcnnTrackerImpl::runEmbeddingFeature(const ncnn::Mat& patch, ncnn::Mat* outFeature) const {
    if (outFeature == nullptr || patch.empty()) {
        return false;
    }

    if (useDualNetPipeline_) {
        ncnn::Extractor extractor = netBackbone_.create_extractor();
        extractor.set_light_mode(true);
        if (extractor.input("input", patch) != 0) {
            return false;
        }
        const char* featureBlob = backboneFeatureBlob_.empty() ? "output" : backboneFeatureBlob_.c_str();
        ncnn::Mat feature;
        if (extractor.extract(featureBlob, feature) != 0 || feature.empty()) {
            return false;
        }
        *outFeature = feature;
        return true;
    }

    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);

    if (extractor.input(templateInputBlob_.c_str(), patch) != 0) {
        return false;
    }
    ncnn::Mat feature;
    if (extractor.extract(scoreOutputBlob_.c_str(), feature) != 0 || feature.empty()) {
        return false;
    }
    *outFeature = feature;
    return true;
}

bool NcnnTrackerImpl::computeGapEmbedding(const ncnn::Mat& featureMap, std::vector<float>* outEmbedding) const {
    if (outEmbedding == nullptr || featureMap.empty()) {
        return false;
    }
    outEmbedding->clear();

    if (featureMap.dims <= 2) {
        const size_t total = featureMap.total();
        if (total == 0) {
            return false;
        }
        double sum = 0.0;
        for (size_t i = 0; i < total; ++i) {
            sum += featureMap[i];
        }
        outEmbedding->push_back(static_cast<float>(sum / static_cast<double>(total)));
        return true;
    }

    const int channels = std::max(1, featureMap.c);
    outEmbedding->resize(static_cast<size_t>(channels), 0.0f);
    for (int c = 0; c < channels; ++c) {
        const ncnn::Mat channel = featureMap.channel(c);
        const size_t elems = channel.total();
        if (elems == 0) {
            continue;
        }
        double sum = 0.0;
        for (size_t i = 0; i < elems; ++i) {
            sum += channel[i];
        }
        (*outEmbedding)[static_cast<size_t>(c)] = static_cast<float>(sum / static_cast<double>(elems));
    }
    return !outEmbedding->empty();
}

bool NcnnTrackerImpl::runSiamScore(const ncnn::Mat& templatePatch, const ncnn::Mat& searchPatch, float* outScore) const {
    if (outScore == nullptr || templatePatch.empty() || searchPatch.empty()) {
        return false;
    }

    if (useDualNetPipeline_) {
        ncnn::Mat xf;
        if (!runEmbeddingFeature(searchPatch, &xf) || xf.empty()) {
            return false;
        }

        ncnn::Extractor exHead = netHead_.create_extractor();
        exHead.set_light_mode(true);
        if (exHead.input("input1", templatePatch) != 0) {
            return false;
        }
        if (exHead.input("input2", xf) != 0) {
            return false;
        }

        ncnn::Mat cls;
        if (exHead.extract("output1", cls) != 0 || cls.empty()) {
            return false;
        }

        const ncnn::Mat channel = cls.c >= 2 ? cls.channel(1) : cls;
        const size_t total = channel.total();
        if (total == 0) {
            return false;
        }

        float vmax = 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < total; ++i) {
            const float s = sigmoid(channel[i]);
            vmax = std::max(vmax, s);
            sum += s;
        }
        const float mean = sum / static_cast<float>(total);
        *outScore = clampFloat(vmax * 0.75f + mean * 0.25f, 0.0f, 1.0f);
        return true;
    }

    ncnn::Extractor extractor = net_.create_extractor();
    extractor.set_light_mode(true);

    if (extractor.input(templateInputBlob_.c_str(), templatePatch) != 0) {
        return false;
    }
    if (extractor.input(searchInputBlob_.c_str(), searchPatch) != 0) {
        return false;
    }

    ncnn::Mat out;
    if (extractor.extract(scoreOutputBlob_.c_str(), out) != 0 || out.empty()) {
        return false;
    }
    *outScore = reduceScore(out);
    return true;
}

float NcnnTrackerImpl::cosineSimilarity(const ncnn::Mat& a, const ncnn::Mat& b) const {
    if (a.empty() || b.empty()) {
        return -1.0f;
    }
    const size_t total = std::min(a.total(), b.total());
    if (total == 0) {
        return -1.0f;
    }
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        const float va = a[i];
        const float vb = b[i];
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    const float denom = std::sqrt(std::max(na * nb, 1e-6f));
    return clampFloat(dot / denom, -1.0f, 1.0f);
}

float NcnnTrackerImpl::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.empty() || b.empty()) {
        return -1.0f;
    }
    const size_t total = std::min(a.size(), b.size());
    if (total == 0) {
        return -1.0f;
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < total; ++i) {
        const double va = static_cast<double>(a[i]);
        const double vb = static_cast<double>(b[i]);
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    const double denom = std::sqrt(std::max(na * nb, 1e-12));
    return clampFloat(static_cast<float>(dot / denom), -1.0f, 1.0f);
}

float NcnnTrackerImpl::reduceScore(const ncnn::Mat& out) const {
    if (out.empty() || out.total() == 0) {
        return -1.0f;
    }
    float sum = 0.0f;
    float vmax = out[0];
    const size_t total = out.total();
    for (size_t i = 0; i < total; ++i) {
        const float v = out[i];
        sum += v;
        vmax = std::max(vmax, v);
    }
    const float mean = sum / static_cast<float>(total);
    return vmax * 0.7f + mean * 0.3f;
}

void NcnnTrackerImpl::updateTemplateFeature(const ncnn::Mat& feature) {
    if (feature.empty()) {
        return;
    }
    if (templateFeature_.empty() || templateFeature_.total() != feature.total()) {
        templateFeature_ = feature.clone();
        computeGapEmbedding(templateFeature_, &templateGapEmbedding_);
        return;
    }
    const float beta = clampFloat(templateUpdateRate_, 0.0f, 0.20f);
    const size_t total = feature.total();
    for (size_t i = 0; i < total; ++i) {
        templateFeature_[i] = templateFeature_[i] * (1.0f - beta) + feature[i] * beta;
    }
    computeGapEmbedding(templateFeature_, &templateGapEmbedding_);
}

void NcnnTrackerImpl::ensureDualHanningCache(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        dualHanningY_.clear();
        dualHanningX_.clear();
        dualHanningRows_ = 0;
        dualHanningCols_ = 0;
        return;
    }
    if (dualHanningRows_ == rows && dualHanningCols_ == cols &&
        static_cast<int>(dualHanningY_.size()) == rows &&
        static_cast<int>(dualHanningX_.size()) == cols) {
        return;
    }

    dualHanningY_.assign(rows, 0.0f);
    dualHanningX_.assign(cols, 0.0f);
    for (int i = 0; i < rows; ++i) {
        dualHanningY_[i] = 0.5f - 0.5f * std::cos((2.0f * 3.14159265358979323846f * i) / std::max(1, rows - 1));
    }
    for (int j = 0; j < cols; ++j) {
        dualHanningX_[j] = 0.5f - 0.5f * std::cos((2.0f * 3.14159265358979323846f * j) / std::max(1, cols - 1));
    }
    dualHanningRows_ = rows;
    dualHanningCols_ = cols;
}

float NcnnTrackerImpl::cosineWindowValue(float centerOffset, float radius) const {
    if (radius <= 1e-3f) {
        return 1.0f;
    }
    const float normalized = clampFloat(std::fabs(centerOffset) / radius, 0.0f, 1.0f);
    return 0.5f * (1.0f + std::cos(normalized * 3.14159265358979323846f));
}

float NcnnTrackerImpl::applyCosineWindow(
    float confidence,
    float candidateCx,
    float candidateCy,
    float prevCx,
    float prevCy,
    float searchRadiusX,
    float searchRadiusY) const {
    if (!enableCosineWindow_) {
        return confidence;
    }
    const float wx = cosineWindowValue(candidateCx - prevCx, searchRadiusX);
    const float wy = cosineWindowValue(candidateCy - prevCy, searchRadiusY);
    const float window = wx * wy;
    const float influence = clampFloat(cosineWindowInfluence_, 0.0f, 0.95f);
    return clampFloat(confidence * ((1.0f - influence) + window * influence), 0.0f, 1.0f);
}

#else

bool NcnnTrackerImpl::loadModel(const std::string& modelParamPath, const std::string& modelBinPath) {
    modelParamPath_ = modelParamPath;
    modelBinPath_ = modelBinPath;

    // Fallback implementation when ncnn runtime is not linked.
    modelReady_ = true;

    __android_log_print(
        ANDROID_LOG_INFO,
        kTag,
        "backend=ncnn(no-runtime) loadModel param=%s bin=%s",
        modelParamPath_.c_str(),
        modelBinPath_.c_str());
    __android_log_print(ANDROID_LOG_INFO, kTag, "DT_NCNN_NO_RUNTIME");
    return modelReady_;
}

bool NcnnTrackerImpl::init(const FrameBuffer& frame, const TrackerBbox& bbox) {
    if (!modelReady_ || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);
    const TrackerBbox safe = clampBox(bbox, logicalW, logicalH);
    if (safe.w < 8.0f || safe.h < 8.0f) {
        return false;
    }

    if (!extractNormalizedPatch(
            frame,
            safe.x + safe.w * 0.5f,
            safe.y + safe.h * 0.5f,
            safe.w,
            safe.h,
            &templatePatch_)) {
        return false;
    }

    scratchPatch_.resize(templatePatch_.size(), 0.0f);
    hasTemplate_ = true;
    lastBox_ = safe;
    return true;
}

bool NcnnTrackerImpl::track(const FrameBuffer& frame, TrackResult* outResult) {
    if (!hasTemplate_ || outResult == nullptr || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }
    int logicalW = 1;
    int logicalH = 1;
    computeLogicalSize(frame, &logicalW, &logicalH);

    const float prevCx = lastBox_.x + lastBox_.w * 0.5f;
    const float prevCy = lastBox_.y + lastBox_.h * 0.5f;
    const float prevW = std::max(8.0f, lastBox_.w);
    const float prevH = std::max(8.0f, lastBox_.h);

    const float searchRadiusX = std::max(10.0f, prevW * (searchScale_ - 1.0f));
    const float searchRadiusY = std::max(10.0f, prevH * (searchScale_ - 1.0f));
    const float coarseStep = std::max(2.0f, std::min(prevW, prevH) * 0.20f);
    const float fineStep = std::max(1.0f, coarseStep * 0.5f);
    const float scales[] = {0.92f, 1.0f, 1.08f};

    float bestScore = -1.0f;
    TrackerBbox bestBox = lastBox_;

    auto evaluateCandidate = [&](float cx, float cy, float boxW, float boxH) {
        if (!extractNormalizedPatch(frame, cx, cy, boxW, boxH, &scratchPatch_)) {
            return;
        }
        const float score = scorePatch(scratchPatch_);
        if (score > bestScore) {
            bestScore = score;
            bestBox.x = cx - boxW * 0.5f;
            bestBox.y = cy - boxH * 0.5f;
            bestBox.w = boxW;
            bestBox.h = boxH;
        }
    };

    for (float dy = -searchRadiusY; dy <= searchRadiusY; dy += coarseStep) {
        for (float dx = -searchRadiusX; dx <= searchRadiusX; dx += coarseStep) {
            const float cx = prevCx + dx;
            const float cy = prevCy + dy;
            for (float scale : scales) {
                evaluateCandidate(cx, cy, prevW * scale, prevH * scale);
            }
        }
    }

    const float bestCxCoarse = bestBox.x + bestBox.w * 0.5f;
    const float bestCyCoarse = bestBox.y + bestBox.h * 0.5f;
    for (float dy = -coarseStep; dy <= coarseStep; dy += fineStep) {
        for (float dx = -coarseStep; dx <= coarseStep; dx += fineStep) {
            const float cx = bestCxCoarse + dx;
            const float cy = bestCyCoarse + dy;
            evaluateCandidate(cx, cy, bestBox.w, bestBox.h);
        }
    }

    if (bestScore < minScoreThreshold_) {
        outResult->ok = false;
        outResult->confidence = 0.0f;
        outResult->similarity = 0.0f;
        outResult->bbox = lastBox_;
        return false;
    }

    TrackerBbox smoothed{};
    smoothed.w = lastBox_.w * (1.0f - smoothAlpha_) + bestBox.w * smoothAlpha_;
    smoothed.h = lastBox_.h * (1.0f - smoothAlpha_) + bestBox.h * smoothAlpha_;
    const float bestCx = bestBox.x + bestBox.w * 0.5f;
    const float bestCy = bestBox.y + bestBox.h * 0.5f;
    const float smoothedCx = prevCx * (1.0f - smoothAlpha_) + bestCx * smoothAlpha_;
    const float smoothedCy = prevCy * (1.0f - smoothAlpha_) + bestCy * smoothAlpha_;
    smoothed.x = smoothedCx - smoothed.w * 0.5f;
    smoothed.y = smoothedCy - smoothed.h * 0.5f;
    smoothed = clampBox(smoothed, logicalW, logicalH);

    if (extractNormalizedPatch(
            frame,
            smoothed.x + smoothed.w * 0.5f,
            smoothed.y + smoothed.h * 0.5f,
            smoothed.w,
            smoothed.h,
            &scratchPatch_)) {
        updateTemplate(scratchPatch_);
    }

    lastBox_ = smoothed;
    outResult->ok = true;
    outResult->bbox = lastBox_;
    outResult->confidence = clampFloat((bestScore + 1.0f) * 0.5f, 0.0f, 1.0f);
    outResult->similarity = outResult->confidence;
    return true;
}

bool NcnnTrackerImpl::setPrior(const TrackerBbox& bbox) {
    if (!hasTemplate_) {
        return false;
    }
    if (!std::isfinite(bbox.x) || !std::isfinite(bbox.y) ||
        !std::isfinite(bbox.w) || !std::isfinite(bbox.h)) {
        return false;
    }

    TrackerBbox prior = bbox;
    prior.w = std::max(8.0f, prior.w);
    prior.h = std::max(8.0f, prior.h);
    lastBox_ = prior;
    return true;
}

void NcnnTrackerImpl::reset() {
    hasTemplate_ = false;
    lastBox_ = TrackerBbox{};
    templatePatch_.clear();
    scratchPatch_.clear();
}

const char* NcnnTrackerImpl::name() const {
    return "ncnn-stub";
}

bool NcnnTrackerImpl::extractNormalizedPatch(
    const FrameBuffer& frame,
    float centerX,
    float centerY,
    float boxW,
    float boxH,
    std::vector<float>* outPatch) const {
    if (outPatch == nullptr || patchSize_ <= 4) {
        return false;
    }
    if (boxW < 4.0f || boxH < 4.0f) {
        return false;
    }

    const int total = patchSize_ * patchSize_;
    outPatch->resize(total);

    const float invSize = 1.0f / static_cast<float>(patchSize_);
    float sum = 0.0f;
    for (int py = 0; py < patchSize_; ++py) {
        const float ry = (static_cast<float>(py) + 0.5f) * invSize - 0.5f;
        const float srcY = centerY + ry * boxH;
        for (int px = 0; px < patchSize_; ++px) {
            const float rx = (static_cast<float>(px) + 0.5f) * invSize - 0.5f;
            const float srcX = centerX + rx * boxW;
            const float value = sampleLumaLogical(frame, srcX, srcY);
            (*outPatch)[py * patchSize_ + px] = value;
            sum += value;
        }
    }

    const float mean = sum / static_cast<float>(total);
    float sqSum = 0.0f;
    for (float value : *outPatch) {
        const float d = value - mean;
        sqSum += d * d;
    }
    const float stdDev = std::sqrt(std::max(1e-6f, sqSum / static_cast<float>(total)));
    if (stdDev < 1e-4f) {
        return false;
    }

    const float invStd = 1.0f / stdDev;
    for (float& value : *outPatch) {
        value = (value - mean) * invStd;
    }
    return true;
}

float NcnnTrackerImpl::scorePatch(const std::vector<float>& patch) const {
    if (templatePatch_.empty() || patch.size() != templatePatch_.size()) {
        return -1.0f;
    }

    float dot = 0.0f;
    for (size_t i = 0; i < patch.size(); ++i) {
        dot += templatePatch_[i] * patch[i];
    }
    const float score = dot / static_cast<float>(patch.size());
    return clampFloat(score, -1.0f, 1.0f);
}

void NcnnTrackerImpl::updateTemplate(const std::vector<float>& patch) {
    if (patch.size() != templatePatch_.size()) {
        return;
    }
    const float beta = clampFloat(templateUpdateRate_, 0.0f, 0.20f);
    for (size_t i = 0; i < patch.size(); ++i) {
        templatePatch_[i] = templatePatch_[i] * (1.0f - beta) + patch[i] * beta;
    }
}

#endif

TrackerBbox NcnnTrackerImpl::clampBox(const TrackerBbox& box, int frameW, int frameH) const {
    TrackerBbox out = box;
    out.w = std::max(8.0f, std::min(out.w, static_cast<float>(frameW)));
    out.h = std::max(8.0f, std::min(out.h, static_cast<float>(frameH)));
    out.x = clampFloat(out.x, 0.0f, std::max(0.0f, static_cast<float>(frameW) - out.w));
    out.y = clampFloat(out.y, 0.0f, std::max(0.0f, static_cast<float>(frameH) - out.h));
    return out;
}

} // namespace dronetracker
