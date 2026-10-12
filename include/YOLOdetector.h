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
private:
    cv::dnn::Net net;   // the model
};