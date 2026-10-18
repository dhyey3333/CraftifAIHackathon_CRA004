// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ParseConfigDetPgie] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // 84 = 4 (cx,cy,w,h) + 80 class scores
    // We need to identify dims at runtime.

    const unsigned int dimFeatures = layer0.inferDims.d[0]; // 84
    const unsigned int dimAnchors = layer0.inferDims.d[1];  // 8400
    const unsigned int numBoxCoords = 4u;

    // Sanity: features should be numBoxCoords + numClasses
    if (dimFeatures != numBoxCoords + numClasses) {
        fprintf(stderr, "[ParseConfigDetPgie] ERROR: dimFeatures=%u != 4+numClasses=%u\n",
                dimFeatures, numBoxCoords + numClasses);
        return false;
    }

    // Data layout: data[featureIdx * dimAnchors + anchorIdx]
    // For anchor i:
    //   cx = data[0 * dimAnchors + i]
    //   cy = data[1 * dimAnchors + i]
    //   w  = data[2 * dimAnchors + i]
    //   h  = data[3 * dimAnchors + i]
    //   class_score[c] = data[(4+c) * dimAnchors + i]
    // Scores are post-sigmoid probabilities from ultralytics export — threshold directly.

    for (unsigned int i = 0; i < dimAnchors; ++i) {
        // Find best class
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float s = data[(numBoxCoords + c) * dimAnchors + i];
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Per-class threshold
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.35f;
        if (maxScore < thr) continue;

        // Decode box (cx, cy, w, h) in network-input pixel coords
        float cx = data[0 * dimAnchors + i];
        float cy = data[1 * dimAnchors + i];
        float bw = data[2 * dimAnchors + i];
        float bh = data[3 * dimAnchors + i];

        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        // Clamp to network input
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;

        // Drop degenerate boxes
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}