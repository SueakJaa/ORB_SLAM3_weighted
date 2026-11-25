#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

class YOLOdetector
{
public:
    YOLOdetector();

    bool LoadModel(const std::string &modelPath);

    // return mask (important for your pipeline)
    cv::Mat Infer(const cv::Mat &image);
    cv::Mat mConfidenceMap;
    int mnLastDetections = 0;   // dynamic-class boxes kept in the last Infer() call
    std::vector<cv::Rect> mvLastBoxes;   // those boxes (shrunk as in the confidence map), image pixels
    float mfPreMs = 0, mfFwdMs = 0, mfPostMs = 0;   // last Infer(): letterbox+blob, network, decode+mask [ms]
    bool  mbGpu = false;                            // CUDA backend in use
    bool  mbFp16 = false;                           // ... with half precision
private:
    cv::dnn::Net net;   // the model
};