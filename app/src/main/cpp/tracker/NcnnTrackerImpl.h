#pragma once

#include <string>

#include "tracker/ITracker.h"

namespace dronetracker {

class NcnnTrackerImpl final : public ITracker {
public:
    NcnnTrackerImpl() = default;
    ~NcnnTrackerImpl() override = default;

    bool loadModel(const std::string& modelParamPath, const std::string& modelBinPath) override;
    bool init(const FrameBuffer& frame, const TrackerBbox& bbox) override;
    bool track(const FrameBuffer& frame, TrackResult* outResult) override;
    void reset() override;
    const char* name() const override;

private:
    std::string modelParamPath_;
    std::string modelBinPath_;
    bool modelReady_ = false;
    bool hasTemplate_ = false;
    TrackerBbox lastBox_{};
};

} // namespace dronetracker
