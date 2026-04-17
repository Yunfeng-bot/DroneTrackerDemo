#pragma once

#include <string>

#include "tracker/TrackerTypes.h"

namespace dronetracker {

class ITracker {
public:
    virtual ~ITracker() = default;

    virtual bool loadModel(const std::string& modelParamPath, const std::string& modelBinPath) = 0;

    // Template branch: runs once after target lock.
    virtual bool init(const FrameBuffer& frame, const TrackerBbox& bbox) = 0;

    // Search branch: runs for each frame in high frequency.
    virtual bool track(const FrameBuffer& frame, TrackResult* outResult) = 0;

    // Optional external prior update from upper fusion layer (for example Kalman prediction).
    virtual bool setPrior(const TrackerBbox& bbox) {
        (void)bbox;
        return false;
    }

    virtual void reset() = 0;

    virtual const char* name() const = 0;
};

} // namespace dronetracker
