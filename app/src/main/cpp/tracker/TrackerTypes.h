#pragma once

#include <cstdint>

namespace dronetracker {

struct FrameBuffer {
    const uint8_t* yPlane = nullptr;
    const uint8_t* uPlane = nullptr;
    const uint8_t* vPlane = nullptr;

    int width = 0;
    int height = 0;
    int rotation = 0;

    int yRowStride = 0;
    int uRowStride = 0;
    int vRowStride = 0;

    int yPixelStride = 1;
    int uPixelStride = 2;
    int vPixelStride = 2;
};

struct TrackerBbox {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

struct TrackResult {
    bool ok = false;
    float confidence = 0.0f;
    // Model-native similarity score in [0, 1], before upper-layer temporal/window fusion.
    float similarity = 0.0f;
    TrackerBbox bbox;
};

} // namespace dronetracker
