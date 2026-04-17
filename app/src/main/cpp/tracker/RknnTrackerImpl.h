#pragma once

#include <string>

#include "tracker/ITracker.h"

namespace dronetracker {

class RknnTrackerImpl final : public ITracker {
public:
    RknnTrackerImpl() = default;
    ~RknnTrackerImpl() override = default;

    bool loadModel(const std::string& modelParamPath, const std::string& modelBinPath) override;
    bool init(const FrameBuffer& frame, const TrackerBbox& bbox) override;
    bool track(const FrameBuffer& frame, TrackResult* outResult) override;
    bool setPrior(const TrackerBbox& bbox) override;
    void reset() override;
    const char* name() const override;

private:
    bool initialized_ = false;
};

} // namespace dronetracker
