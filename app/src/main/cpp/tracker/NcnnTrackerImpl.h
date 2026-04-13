#pragma once

#include <string>
#include <vector>

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
    bool extractNormalizedPatch(
        const FrameBuffer& frame,
        float centerX,
        float centerY,
        float boxW,
        float boxH,
        std::vector<float>* outPatch) const;

    float scorePatch(const std::vector<float>& patch) const;
    void updateTemplate(const std::vector<float>& patch);
    TrackerBbox clampBox(const TrackerBbox& box, int frameW, int frameH) const;

    std::string modelParamPath_;
    std::string modelBinPath_;
    bool modelReady_ = false;
    bool hasTemplate_ = false;
    TrackerBbox lastBox_{};
    std::vector<float> templatePatch_;
    std::vector<float> scratchPatch_;

    int patchSize_ = 48;
    float searchScale_ = 2.4f;
    float minScoreThreshold_ = 0.16f;
    float smoothAlpha_ = 0.60f;
    float templateUpdateRate_ = 0.04f;
};

} // namespace dronetracker
