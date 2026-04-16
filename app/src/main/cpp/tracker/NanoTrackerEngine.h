#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "tracker/ITracker.h"

namespace dronetracker {

enum class TrackerBackend {
    kNcnn = 0,
    kRknn = 1,
};

class NanoTrackerEngine {
public:
    static NanoTrackerEngine& instance();

    bool init(TrackerBackend backend, const std::string& modelParamPath, const std::string& modelBinPath);
    bool initTarget(const FrameBuffer& frame, const TrackerBbox& bbox);
    bool track(const FrameBuffer& frame, TrackResult* outResult);
    void reset();
    void release();

    bool available() const;
    std::string backendName() const;

private:
    NanoTrackerEngine() = default;

    std::unique_ptr<ITracker> createTrackerLocked(TrackerBackend backend) const;

    mutable std::mutex mutex_;
    std::unique_ptr<ITracker> tracker_;
    TrackerBackend backend_ = TrackerBackend::kNcnn;
    std::string modelParamPath_;
    std::string modelBinPath_;
    bool modelReady_ = false;
};

} // namespace dronetracker
