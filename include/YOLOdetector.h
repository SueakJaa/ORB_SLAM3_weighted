#pragma once

#include <opencv2/opencv.hpp>
#include "TrtEngine.h"
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
    bool  mbTrt = false;                            // TensorRT engine used for the network (DSLAM_YOLO_TRT)
    TrtEngine mTrt;
private:
    cv::dnn::Net net;   // the model
    std::vector<std::string> mvOutNames;   // outputs requested per frame (only the decoded one if the model has it)
    // fast pre-processing (no resize needed, e.g. 640x480 frames): blob kept between frames, grey padding written
    // once, only the image rows converted per frame. Same values as letterbox + blobFromImage (checked at load).
    cv::Mat mBlob;
    cv::Size mBlobImgSize;
    int mBlobPx = -1, mBlobPy = -1;
    bool mbFastBlobOk = false;   // set by the check in LoadModel
    cv::Mat mLut;                // 1x256 float: blobFromImage's value for each 8-bit level
    bool FastBlob(const cv::Mat& image, int px, int py);
};