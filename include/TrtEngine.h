#pragma once
// Optional TensorRT inference for the YOLO network (speed step 3, 26 Nov).
// Compiled always; without TensorRT (USE_TENSORRT not defined) Load() returns false and YOLOdetector keeps
// using OpenCV DNN. TensorRT headers stay inside TrtEngine.cc, so nothing else depends on them.
// Uses the name-based I/O API (TensorRT >= 8.5, incl. 10.x): setTensorAddress + enqueueV3.
#include <opencv2/core.hpp>
#include <string>

class TrtEngine
{
public:
    TrtEngine();
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    static bool Available();                        // built with TensorRT?
    bool Load(const std::string& enginePath);       // serialized engine from scripts/trt/build_engine.py
    bool Ok() const;

    // blob: 1x3x640x640 CV_32F (same as for OpenCV DNN). decoded: [1, N, 85] CV_32F, valid until the next call.
    bool Infer(const cv::Mat& blob, cv::Mat& decoded);

private:
    struct Impl;
    Impl* p;
};
