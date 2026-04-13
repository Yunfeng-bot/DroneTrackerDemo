#include "tracker/NcnnTrackerImpl.h"

#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace {
constexpr const char* kTag = "NativeTracker";

inline float clampFloat(float value, float lo, float hi) {
    return std::max(lo, std::min(hi, value));
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
}

namespace dronetracker {

bool NcnnTrackerImpl::loadModel(const std::string& modelParamPath, const std::string& modelBinPath) {
    modelParamPath_ = modelParamPath;
    modelBinPath_ = modelBinPath;

    // P0 native tracker core: keep interface compatible with future ncnn net loading.
    // For now we run a lightweight template-search correlation core to validate
    // ORB_SEARCH -> Native_TRACK mixed pipeline end-to-end.
    modelReady_ = true;

    __android_log_print(
        ANDROID_LOG_INFO,
        kTag,
        "backend=ncnn loadModel param=%s bin=%s",
        modelParamPath_.c_str(),
        modelBinPath_.c_str());
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
    return true;
}

void NcnnTrackerImpl::reset() {
    hasTemplate_ = false;
    lastBox_ = TrackerBbox{};
    templatePatch_.clear();
    scratchPatch_.clear();
}

const char* NcnnTrackerImpl::name() const {
    return "ncnn";
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

TrackerBbox NcnnTrackerImpl::clampBox(const TrackerBbox& box, int frameW, int frameH) const {
    TrackerBbox out = box;
    out.w = std::max(8.0f, std::min(out.w, static_cast<float>(frameW)));
    out.h = std::max(8.0f, std::min(out.h, static_cast<float>(frameH)));
    out.x = clampFloat(out.x, 0.0f, std::max(0.0f, static_cast<float>(frameW) - out.w));
    out.y = clampFloat(out.y, 0.0f, std::max(0.0f, static_cast<float>(frameH) - out.h));
    return out;
}

} // namespace dronetracker
