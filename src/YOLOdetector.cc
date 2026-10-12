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
    // Letterbox (YOLOv5 standard): resize keeping the aspect ratio, pad to 640x640 with grey 114.
    // Stretching 640x480 to 640x640 instead distorts people so much that YOLOv5n's scores drop
    // below the 0.4 threshold (checked: best person score 0.33 stretched vs 0.71 letterboxed).
    const bool letterbox = ORB_SLAM3::DynamicConfig::Get().yoloLetterbox;
    float scale = 1.0f, padX = 0.0f, padY = 0.0f;
    cv::Mat input = image;
    if(letterbox)
    {
        scale = std::min(640.0f / image.cols, 640.0f / image.rows);
        const int nw = static_cast<int>(std::round(image.cols * scale));
        const int nh = static_cast<int>(std::round(image.rows * scale));
        const int px = (640 - nw) / 2;
        const int py = (640 - nh) / 2;
        padX = static_cast<float>(px);
        padY = static_cast<float>(py);
        cv::Mat resized;
        if(nw != image.cols || nh != image.rows)
            cv::resize(image, resized, cv::Size(nw, nh));
        else
            resized = image;
        input = cv::Mat(640, 640, image.type(), cv::Scalar(114, 114, 114, 0));
        resized.copyTo(input(cv::Rect(px, py, nw, nh)));
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::Mat confMap(image.rows, image.cols, CV_32F, cv::Scalar(0));
    net.setInput(blob);
    cv::Mat output = net.forward();

    if(verbose) std::cout << "[YOLO] Output size: " << output.size << std::endl;

    // The parser below expects the YOLOv5 layout [1, N, 85]. Warn once if the model differs
    // (e.g. a YOLOv8-style export [1, 84, 8400]), instead of silently finding nothing.
    static bool warned = false;
    if(output.dims != 3 || output.size[2] != 85)
    {
        if(!warned)
        {
            std::cerr << "[YOLO] Unexpected output shape " << output.size
                      << " (expected [1, N, 85], YOLOv5 export). No detections will be used." << std::endl;
            warned = true;
        }
        mConfidenceMap = confMap.clone();
        mnLastDetections = 0;
        return cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    }

    // ===== Create mask =====
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    float* data = (float*)output.data;

    const int num_detections = output.size[1];
    const int num_classes = 80;
    mnLastDetections = 0;

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

        int x, y, width, height;
        if(letterbox)
        {
            // undo padding and scale
            x = static_cast<int>((cx - w/2 - padX) / scale);
            y = static_cast<int>((cy - h/2 - padY) / scale);
            width  = static_cast<int>(w / scale);
            height = static_cast<int>(h / scale);
        }
        else
        {
            x = (cx - w/2) * image.cols / 640.0;
            y = (cy - h/2) * image.rows / 640.0;
            width  = w * image.cols / 640.0;
            height = h * image.rows / 640.0;
        }

        // clamp (keep the far edge, cut what is outside the image)
        int x2 = std::min(x + width,  image.cols);
        int y2 = std::min(y + height, image.rows);
        x = std::max(0, x);
        y = std::max(0, y);
        width  = x2 - x;
        height = y2 - y;
        if(width <= 0 || height <= 0)
        {
            data += 85;
            continue;
        }
        mnLastDetections++;

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