#include "YOLOdetector.h"
#include <iostream>

YOLOdetector::YOLOdetector()
{
    if(!LoadModel("/home/shirt/yolov5_models/yolov5n.onnx"))
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

    std::cout << "✅ YOLO model loaded!" << std::endl;
    return true;
}

cv::Mat YOLOdetector::Infer(const cv::Mat &image)
{
    if(net.empty())
    {
        std::cout << "[YOLO] Model not loaded, returning safe mask" << std::endl;
        return cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    }

    std::cout << "[YOLO] Infer called" << std::endl;

    // ===== Preprocess =====
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::Mat confMap(image.rows, image.cols, CV_32F, cv::Scalar(0));
    net.setInput(blob);
    cv::Mat output = net.forward();

    std::cout << "[YOLO] Output size: " << output.size << std::endl;

    // ===== Create mask =====
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    float* data = (float*)output.data;

    const int num_detections = 25200;
    const int num_classes = 80;

    float confThreshold = 0.4;
    float shrink = 0.8;

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