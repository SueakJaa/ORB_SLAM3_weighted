#include "YOLOdetector.h"
#include "DynamicConfig.h"
#include <iostream>
#include <opencv2/core/cuda.hpp>

YOLOdetector::YOLOdetector()
{
    if(!LoadModel(ORB_SLAM3::DynamicConfig::Get().yoloModel))
    {
        std::cerr << "FATAL: YOLO model not loaded!" << std::endl;
    }
}

bool YOLOdetector::LoadModel(const std::string &modelPath)
{
    net = cv::dnn::readNetFromONNX(modelPath);

    if (net.empty())
    {
        std::cerr << "❌ Failed to load YOLO model!" << std::endl;
        return false;
    }

    std::cout << "YOLO model loaded: " << modelPath << std::endl;

    // ===== Backend selection =====
    // CUDA is used only if requested AND OpenCV was built with CUDA AND a GPU is visible.
    // FP32 target (not FP16) so detections match the CPU version as closely as possible.
    bool useGpu = false;
    if(ORB_SLAM3::DynamicConfig::Get().yoloGpu)
    {
        int nDev = 0;
        try { nDev = cv::cuda::getCudaEnabledDeviceCount(); } catch(...) { nDev = 0; }
        if(nDev > 0)
        {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            useGpu = true;
        }
        else
        {
            std::cout << "[YOLO] CUDA requested but not available in this OpenCV build -> CPU" << std::endl;
        }
    }
    if(!useGpu)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    std::cout << "[YOLO] backend = " << (useGpu ? "CUDA (GPU)" : "OpenCV (CPU)") << std::endl;

    // Warm-up: the first forward pass initialises CUDA/cuDNN (can take ~1 s);
    // do it here so it is not counted in the per-frame timing.
    cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat blob;
    cv::dnn::blobFromImage(dummy, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);
    net.forward();

    return true;
}

cv::Mat YOLOdetector::Infer(const cv::Mat &image)
{
    if(net.empty())
    {
        std::cerr << "[YOLO] Model not loaded, returning safe mask" << std::endl;
        return cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    }

    const bool verbose = ORB_SLAM3::DynamicConfig::Get().verbose;
    if(verbose) std::cout << "[YOLO] Infer called" << std::endl;

    // ===== Preprocess =====
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::Mat confMap(image.rows, image.cols, CV_32F, cv::Scalar(0));
    net.setInput(blob);
    cv::Mat output = net.forward();

    if(verbose) std::cout << "[YOLO] Output size: " << output.size << std::endl;

    // ===== Create mask =====
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    float* data = (float*)output.data;

    const int num_detections = 25200;
    const int num_classes = 80;

    const float confThreshold = ORB_SLAM3::DynamicConfig::Get().yoloConf;
    const float shrink = 0.8f;  // boxes shrunk to 80% before building the confidence map

    for(int i = 0; i < num_detections; i++)
    {
        float obj_conf = data[4];

        if(obj_conf < confThreshold)
        {
            data += 85;
            continue;
        }

        float max_class_score = 0;
        int class_id = -1;

        for(int c = 0; c < num_classes; c++)
        {
            float score = data[5 + c];
            if(score > max_class_score)
            {
                max_class_score = score;
                class_id = c;
            }
        }

        float confidence = obj_conf * max_class_score;

        if(confidence < confThreshold)
        {
            data += 85;
            continue;
        }

        // dynamic classes
        if(class_id != 0 && class_id != 2 && class_id != 3 && class_id != 5 && class_id != 7)
        {
            data += 85;
            continue;
        }

        // ===== BOX =====
        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];

        int x = (cx - w/2) * image.cols / 640.0;
        int y = (cy - h/2) * image.rows / 640.0;
        int width  = w * image.cols / 640.0;
        int height = h * image.rows / 640.0;

        // clamp
        x = std::max(0, x);
        y = std::max(0, y);
        width = std::min(width, image.cols - x);
        height = std::min(height, image.rows - y);

        // ===== SHRINK =====
        int new_w = width * shrink;
        int new_h = height * shrink;

        int new_x = x + (width - new_w)/2;
        int new_y = y + (height - new_h)/2;

        // ===== DRAW MASK =====
        cv::rectangle(mask, cv::Rect(new_x, new_y, new_w, new_h), 0, -1);

        // ===== UPDATE CONF MAP (CORRECT PLACE) =====
        for(int yy = new_y; yy < new_y + new_h; yy++)
        {
            for(int xx = new_x; xx < new_x + new_w; xx++)
            {
                if(yy >= 0 && yy < image.rows && xx >= 0 && xx < image.cols)
                {
                    confMap.at<float>(yy, xx) = std::max(
                        confMap.at<float>(yy, xx),
                        confidence
                    );
                }
            }
        }

        // move pointer LAST
        data += 85;
    }

    // ✅ AFTER LOOP ONLY
    mConfidenceMap = confMap.clone();

    // optional smoothing
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 1);

    return mask;
}