#include "tracker/RknnTrackerImpl.h"

#include <android/log.h>

namespace {
constexpr const char* kTag = "NativeTracker";
}

namespace dronetracker {

bool RknnTrackerImpl::loadModel(const std::string& modelParamPath, const std::string& modelBinPath) {
    (void)modelParamPath;
    (void)modelBinPath;
    initialized_ = true;
    __android_log_print(ANDROID_LOG_INFO, kTag, "backend=rknn loadModel stub-ready");
    return true;
}

bool RknnTrackerImpl::init(const FrameBuffer& frame, const TrackerBbox& bbox) {
    (void)bbox;
    if (!initialized_ || frame.width <= 0 || frame.height <= 0 || frame.yPlane == nullptr) {
        return false;
    }
    return true;
}

bool RknnTrackerImpl::track(const FrameBuffer& frame, TrackResult* outResult) {
    (void)frame;
    if (!initialized_ || outResult == nullptr) {
        return false;
    }
    outResult->ok = false;
    outResult->confidence = 0.0f;
    outResult->bbox = TrackerBbox{};
    return false;
}

bool RknnTrackerImpl::setPrior(const TrackerBbox& bbox) {
    (void)bbox;
    return false;
}

void RknnTrackerImpl::reset() {
    initialized_ = false;
}

const char* RknnTrackerImpl::name() const {
    return "rknn";
}

} // namespace dronetracker
