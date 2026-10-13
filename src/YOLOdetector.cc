#include "YOLOdetector.h"
#include "DynamicConfig.h"
#include <iostream>
#include <cmath>
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

    // Read ALL network outputs. Two YOLOv5 export styles exist:
    //  (a) decoded:  one output [1, N, 85] with boxes in pixels and sigmoid scores (current yolov5 export)
    //  (b) raw heads: three outputs [1, 3, ny, nx, 85] with logits (older export, e.g. yolov5 v5.0);
    //      net.forward() alone returns only one of them. They are decoded here exactly like
    //      YOLOv5's Detect layer (checked against an official decoded export: max diff 2e-4).
    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<float> dets;           // decoded rows of 85 floats: cx, cy, w, h, obj, 80 class scores
    bool haveDecoded = false;
    for(const cv::Mat& o : outs)
    {
        if(o.dims == 3 && o.size[2] == 85)
        {
            const float* p = o.ptr<float>();
            dets.assign(p, p + static_cast<size_t>(o.size[1]) * 85);
            haveDecoded = true;
            break;
        }
    }
    if(!haveDecoded)
    {
        static const float ANCHORS[3][6] = {{10,13, 16,30, 33,23},       // stride 8
                                            {30,61, 62,45, 59,119},      // stride 16
                                            {116,90, 156,198, 373,326}}; // stride 32
        auto sigm = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
        for(const cv::Mat& o : outs)
        {
            if(o.dims != 5 || o.size[1] != 3 || o.size[4] != 85) continue;
            const int ny = o.size[2], nx = o.size[3];
            const int stride = 640 / nx;
            const int si = stride == 8 ? 0 : stride == 16 ? 1 : stride == 32 ? 2 : -1;
            if(si < 0) continue;
            const float* p = o.ptr<float>();
            for(int a = 0; a < 3; a++)
                for(int gy = 0; gy < ny; gy++)
                    for(int gx = 0; gx < nx; gx++)
                    {
                        const float* q = p + ((static_cast<size_t>(a) * ny + gy) * nx + gx) * 85;
                        const size_t base = dets.size();
                        dets.resize(base + 85);
                        float* d = &dets[base];
                        for(int k = 0; k < 85; k++) d[k] = sigm(q[k]);
                        d[0] = (d[0] * 2.0f - 0.5f + gx) * stride;
                        d[1] = (d[1] * 2.0f - 0.5f + gy) * stride;
                        d[2] = (d[2] * 2.0f) * (d[2] * 2.0f) * ANCHORS[si][2 * a];
                        d[3] = (d[3] * 2.0f) * (d[3] * 2.0f) * ANCHORS[si][2 * a + 1];
                    }
        }
    }

    static bool reported = false;
    if(!reported)
    {
        std::cout << "[YOLO] outputs:";
        for(const cv::Mat& o : outs) std::cout << " [" << o.size << "]";
        std::cout << " -> " << (haveDecoded ? "decoded output used" : "raw heads decoded in code")
                  << ", " << dets.size() / 85 << " candidate boxes" << std::endl;
        reported = true;
    }

    if(dets.empty())
    {
        static bool warned = false;
        if(!warned)
        {
            std::cerr << "[YOLO] No usable output (expected [1, N, 85] or three [1, 3, ny, nx, 85] heads)."
                         " No detections will be used." << std::endl;
            warned = true;
        }
        mConfidenceMap = confMap.clone();
        mnLastDetections = 0;
        return cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    }

    // ===== Create mask =====
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    const float* data = dets.data();

    const int num_detections = static_cast<int>(dets.size() / 85);
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