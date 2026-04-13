#include "tracker/NanoTrackerEngine.h"

#include <android/log.h>

#include "tracker/NcnnTrackerImpl.h"
#include "tracker/RknnTrackerImpl.h"

namespace {
constexpr const char* kTag = "NativeTracker";
}

namespace dronetracker {

NanoTrackerEngine& NanoTrackerEngine::instance() {
    static NanoTrackerEngine engine;
    return engine;
}

bool NanoTrackerEngine::init(TrackerBackend backend, const std::string& modelParamPath, const std::string& modelBinPath) {
    std::lock_guard<std::mutex> lock(mutex_);

    backend_ = backend;
    tracker_ = createTrackerLocked(backend);
    if (!tracker_) {
        modelReady_ = false;
        return false;
    }

    modelReady_ = tracker_->loadModel(modelParamPath, modelBinPath);
    __android_log_print(
        ANDROID_LOG_INFO,
        kTag,
        "engine init backend=%s ready=%d",
        tracker_->name(),
        modelReady_ ? 1 : 0);
    return modelReady_;
}

bool NanoTrackerEngine::initTarget(const FrameBuffer& frame, const TrackerBbox& bbox) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_ || !modelReady_) {
        return false;
    }
    return tracker_->init(frame, bbox);
}

bool NanoTrackerEngine::track(const FrameBuffer& frame, TrackResult* outResult) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_ || !modelReady_ || outResult == nullptr) {
        return false;
    }
    return tracker_->track(frame, outResult);
}

void NanoTrackerEngine::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracker_) {
        tracker_->reset();
    }
}

void NanoTrackerEngine::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    tracker_.reset();
    modelReady_ = false;
}

bool NanoTrackerEngine::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tracker_ != nullptr && modelReady_;
}

std::string NanoTrackerEngine::backendName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_) {
        return "none";
    }
    return tracker_->name();
}

std::unique_ptr<ITracker> NanoTrackerEngine::createTrackerLocked(TrackerBackend backend) const {
    switch (backend) {
    case TrackerBackend::kRknn:
        return std::make_unique<RknnTrackerImpl>();
    case TrackerBackend::kNcnn:
    default:
        return std::make_unique<NcnnTrackerImpl>();
    }
}

} // namespace dronetracker
