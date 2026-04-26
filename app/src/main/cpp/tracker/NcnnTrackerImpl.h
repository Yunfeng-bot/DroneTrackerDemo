#pragma once

#include <string>
#include <vector>

#include "tracker/ITracker.h"

#if defined(DRONETRACKER_HAVE_NCNN) && DRONETRACKER_HAVE_NCNN
#include <ncnn/net.h>
#endif

namespace dronetracker {

class NcnnTrackerImpl final : public ITracker {
public:
    NcnnTrackerImpl() = default;
    ~NcnnTrackerImpl() override = default;

    bool loadModel(const std::string& modelParamPath, const std::string& modelBinPath) override;
    bool init(const FrameBuffer& frame, const TrackerBbox& bbox) override;
    bool track(const FrameBuffer& frame, TrackResult* outResult) override;
    bool setPrior(const TrackerBbox& bbox) override;
    void reset() override;
    const char* name() const override;

private:
#if defined(DRONETRACKER_HAVE_NCNN) && DRONETRACKER_HAVE_NCNN
    enum class ModelMode {
        kEmbedding = 0,
        kSiamLike = 1,
    };

    bool resolveModelPaths(std::string* outParamPath, std::string* outBinPath) const;
    bool fileExists(const std::string& path) const;
    bool extractPatchToMat(
        const FrameBuffer& frame,
        float centerX,
        float centerY,
        float boxW,
        float boxH,
        int patchSize,
        ncnn::Mat* outMat) const;
    bool runEmbeddingFeature(const ncnn::Mat& patch, ncnn::Mat* outFeature) const;
    bool computeGapEmbedding(const ncnn::Mat& featureMap, std::vector<float>* outEmbedding) const;
    bool runSiamScore(const ncnn::Mat& templatePatch, const ncnn::Mat& searchPatch, float* outScore) const;
    bool resolveDualNetHeadPaths(
        const std::string& backboneParamPath,
        const std::string& backboneBinPath,
        std::string* outHeadParamPath,
        std::string* outHeadBinPath) const;
    void ensureDualHanningCache(int rows, int cols);
    float cosineSimilarity(const ncnn::Mat& a, const ncnn::Mat& b) const;
    float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) const;
    float reduceScore(const ncnn::Mat& out) const;
    void updateTemplateFeature(const ncnn::Mat& feature);
    float computeTemplateUpdateMahalanobis(const TrackerBbox& prev, const TrackerBbox& next) const;
    float computeTemplateEdgeRatio(const TrackerBbox& box, int frameW, int frameH) const;
    bool shouldUpdateTemplateFeature(
        const TrackerBbox& prev,
        const TrackerBbox& next,
        float similarity,
        int frameW,
        int frameH,
        float* outMahalanobis,
        float* outAnchorMahalanobis,
        float* outEdgeRatio) const;
    float cosineWindowValue(float centerOffset, float radius) const;
    float applyCosineWindow(
        float confidence,
        float candidateCx,
        float candidateCy,
        float prevCx,
        float prevCy,
        float searchRadiusX,
        float searchRadiusY) const;
#else
    bool extractNormalizedPatch(
        const FrameBuffer& frame,
        float centerX,
        float centerY,
        float boxW,
        float boxH,
        std::vector<float>* outPatch) const;

    float scorePatch(const std::vector<float>& patch) const;
    void updateTemplate(const std::vector<float>& patch);
#endif
    TrackerBbox clampBox(const TrackerBbox& box, int frameW, int frameH) const;

    std::string modelParamPath_;
    std::string modelBinPath_;
    bool modelReady_ = false;
    bool hasTemplate_ = false;
    TrackerBbox lastBox_{};
    TrackerBbox templateAnchorBox_{};
    bool hasTemplateAnchorBox_ = false;

#if defined(DRONETRACKER_HAVE_NCNN) && DRONETRACKER_HAVE_NCNN
    ModelMode modelMode_ = ModelMode::kEmbedding;
    bool useFp16Storage_ = true;
    bool useFp16Arithmetic_ = true;
    int templateInputSize_ = 127;
    int searchInputSize_ = 255;
    std::string templateInputBlob_;
    std::string searchInputBlob_;
    std::string scoreOutputBlob_;
    std::vector<std::string> inputBlobNames_;
    std::vector<std::string> outputBlobNames_;
    ncnn::Net net_;
    ncnn::Net netBackbone_;
    ncnn::Net netHead_;
    ncnn::Mat templateFeature_;
    std::vector<float> templateGapEmbedding_;
    ncnn::Mat templateInputMat_;
    bool useDualNetPipeline_ = false;
    std::string headParamPath_;
    std::string headBinPath_;
    std::string backboneFeatureBlob_;
    bool enableCosineWindow_ = true;
    float cosineWindowInfluence_ = 0.40f;
    std::vector<float> dualHanningY_;
    std::vector<float> dualHanningX_;
    int dualHanningRows_ = 0;
    int dualHanningCols_ = 0;
#endif

#if !(defined(DRONETRACKER_HAVE_NCNN) && DRONETRACKER_HAVE_NCNN)
    std::vector<float> templatePatch_;
    std::vector<float> scratchPatch_;
    int patchSize_ = 48;
#endif
    float searchScale_ = 2.4f;
    float minScoreThreshold_ = 0.16f;
    float smoothAlpha_ = 0.60f;
    float templateUpdateRate_ = 0.04f;
    float templateUpdateMahalThreshold_ = 9.0f;
    float templateUpdateMinSimilarity_ = 0.70f;
    float templateUpdateMaxEdgeRatio_ = 0.15f;
    float gapRecoverSimilarityThreshold_ = 0.85f;
};

} // namespace dronetracker

