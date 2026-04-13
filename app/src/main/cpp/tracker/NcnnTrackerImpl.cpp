#include "tracker/NcnnTrackerImpl.h"

#include <android/log.h>

namespace {
constexpr const char* kTag = "NativeTracker";
}

namespace dronetracker {

bool NcnnTrackerImpl::loadModel(const std::string& modelParamPath, const std::string& modelBinPath) {
    modelParamPath_ = modelParamPath;
    modelBinPath_ = modelBinPath;

    // MVP bridge: keep model-ready true so the JNI path can be validated now.
    // Integrate real ncnn net loading in the next milestone.
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
    hasTemplate_ = true;
    lastBox_ = bbox;
    return true;
}

bool NcnnTrackerImpl::track(const FrameBuffer& frame, TrackResult* outResult) {
    if (!hasTemplate_ || outResult == nullptr || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }

    outResult->ok = true;
    outResult->bbox = lastBox_;
    outResult->confidence = 0.50f;
    return true;
}

void NcnnTrackerImpl::reset() {
    hasTemplate_ = false;
    lastBox_ = TrackerBbox{};
}

const char* NcnnTrackerImpl::name() const {
    return "ncnn";
}

} // namespace dronetracker
