// yolo_bench: where does YOLO's network time go? (speed step 2, 25 Nov)
//
//   ./tools/yolo_bench /home/shirt/yolov5_models/yolov5n.onnx [iterations] [engine.trt] [frame.png]
//   With an engine (scripts/trt/build_engine.py) it also times TensorRT and compares its decoded output with
//   OpenCV CUDA on the same input (a real frame if given, else random pixels).
//
// For CPU, CUDA FP32 and CUDA FP16 it times net.forward() two ways:
//   all4    all four ONNX outputs (what YOLOdetector does now: three raw heads + the decoded [1,25200,85])
//   heads   only the three raw heads (logits; does OpenCV then skip the decoding layers?)
//   decoded only the decoded output (the three raw heads are 2.1 M floats = 8.6 MB more to copy back per frame)
// and prints the slowest layers of the CUDA run (getPerfProfile), so a layer that falls back to the CPU shows up.
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include "TrtEngine.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Result { std::string name; double fwd_ms; double p95_ms; };

static Result timeIt(cv::dnn::Net& net, const cv::Mat& blob, const std::vector<std::string>& outNames,
                     const std::string& label, int iters)
{
    std::vector<cv::Mat> outs;
    for(int i = 0; i < 20; i++) { net.setInput(blob); net.forward(outs, outNames); }   // warm-up
    std::vector<double> t;
    for(int i = 0; i < iters; i++)
    {
        const auto t0 = clk::now();
        net.setInput(blob);
        net.forward(outs, outNames);
        t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    }
    std::vector<double> s = t;
    std::sort(s.begin(), s.end());
    return {label, median(t), s[std::min(s.size() - 1, (size_t)(0.95 * s.size()))]};
}

int main(int argc, char** argv)
{
    if(argc < 2) { std::cerr << "usage: yolo_bench model.onnx [iterations=200]\n"; return 1; }
    const std::string model = argv[1];
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const std::string enginePath = argc > 3 ? argv[3] : "";
    const std::string framePath = argc > 4 ? argv[4] : "";

    // same input as YOLOdetector: 640x480 frame letterboxed into 640x640
    cv::Mat frame(480, 640, CV_8UC3);
    cv::randu(frame, 0, 255);
    if(!framePath.empty())
    {
        cv::Mat f = cv::imread(framePath, cv::IMREAD_COLOR);
        if(f.size() == frame.size()) frame = f;
        else std::cerr << "frame " << framePath << " not read or not 640x480: random pixels used\n";
    }
    cv::Mat input(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    frame.copyTo(input(cv::Rect(0, 80, 640, 480)));
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);

    std::cout << "OpenCV " << CV_VERSION << ", threads " << cv::getNumThreads() << ", CUDA devices "
              << cv::cuda::getCudaEnabledDeviceCount() << "\n";

    struct Cfg { const char* name; cv::dnn::Backend b; cv::dnn::Target t; };
    const std::vector<Cfg> cfgs = {{"CPU", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
                                   {"CUDA", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
                                   {"CUDA_FP16", cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16}};
    std::vector<Result> res;
    cv::Mat cudaDecoded;   // OpenCV CUDA decoded output, to compare TensorRT with
    for(const Cfg& c : cfgs)
    {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model);
        net.setPreferableBackend(c.b);
        net.setPreferableTarget(c.t);
        const std::vector<std::string> all = net.getUnconnectedOutLayersNames();

        // find the decoded output ([1, N, 85])
        std::vector<cv::Mat> outs;
        net.setInput(blob);
        net.forward(outs, all);
        std::string decoded;
        std::vector<std::string> heads;
        for(size_t i = 0; i < outs.size(); i++)
        {
            if(outs[i].dims == 3) decoded = all[i];
            if(outs[i].dims == 5) heads.push_back(all[i]);
        }

        res.push_back(timeIt(net, blob, all, std::string(c.name) + " all4", iters));
        // raw heads only: if OpenCV then skips the decoding layers (model.24 Mul/Pow/...), decoding on the CPU
        // for the few cells above the objectness threshold would be cheaper
        if(heads.size() == 3)
            res.push_back(timeIt(net, blob, heads, std::string(c.name) + " heads", iters));
        if(!decoded.empty())
            res.push_back(timeIt(net, blob, {decoded}, std::string(c.name) + " decoded", iters));
        if(c.t == cv::dnn::DNN_TARGET_CUDA && !decoded.empty())
        {
            std::vector<cv::Mat> o;
            net.setInput(blob);
            net.forward(o, std::vector<std::string>{decoded});
            cudaDecoded = o[0].clone();
        }

        if(c.t == cv::dnn::DNN_TARGET_CUDA)
        {
            // slowest layers of the last (decoded-only) run; on the CUDA backend a layer it cannot run is
            // executed on the CPU, with copies around it, and usually shows up here
            std::vector<double> lt;
            const double total = net.getPerfProfile(lt) * 1000.0 / cv::getTickFrequency();
            const std::vector<std::string> names = net.getLayerNames();
            std::vector<size_t> idx(lt.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return lt[a] > lt[b]; });
            std::cout << "\nCUDA, slowest layers (last forward, total " << total << " ms):\n";
            for(size_t k = 0; k < std::min<size_t>(12, idx.size()); k++)
            {
                const size_t i = idx[k];
                if(i >= names.size()) continue;
                const std::string type = net.getLayer(names[i])->type;
                std::cout << "  " << lt[i] * 1000.0 / cv::getTickFrequency() << " ms  " << type << "  " << names[i] << "\n";
            }
        }
    }

    if(!enginePath.empty())
    {
        TrtEngine trt;
        if(!trt.Load(enginePath)) { std::cerr << "TensorRT engine not usable\n"; }
        else
        {
            cv::Mat out;
            for(int i = 0; i < 20; i++) trt.Infer(blob, out);   // warm-up
            std::vector<double> t;
            for(int i = 0; i < iters; i++)
            {
                const auto t0 = clk::now();
                trt.Infer(blob, out);
                t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
            }
            std::vector<double> s = t;
            std::sort(s.begin(), s.end());
            res.push_back({"TensorRT decoded", median(t), s[std::min(s.size() - 1, (size_t)(0.95 * s.size()))]});
            if(!cudaDecoded.empty() && cudaDecoded.total() == out.total())
            {
                // compare: whole output, and the person score obj*cls0 that YOLOdetector thresholds at 0.4
                const float* a = cudaDecoded.ptr<float>();
                const float* b = out.ptr<float>();
                const size_t n = out.total() / 85;
                double maxAll = 0, maxPerson = 0;
                int candA = 0, candB = 0, both = 0;
                for(size_t r = 0; r < n; r++)
                {
                    for(int k = 0; k < 85; k++) maxAll = std::max(maxAll, (double)std::fabs(a[r * 85 + k] - b[r * 85 + k]));
                    const double pa = a[r * 85 + 4] * a[r * 85 + 5], pb = b[r * 85 + 4] * b[r * 85 + 5];
                    maxPerson = std::max(maxPerson, std::fabs(pa - pb));
                    candA += pa >= 0.4; candB += pb >= 0.4; both += (pa >= 0.4 && pb >= 0.4);
                }
                std::cout << "\nTensorRT vs OpenCV CUDA, same input: max |diff| all values " << maxAll
                          << ", person score " << maxPerson << "; person candidates >= 0.4: OpenCV " << candA
                          << ", TensorRT " << candB << ", in both " << both << "\n";
            }
        }
    }

    std::cout << "\nforward() incl. copying outputs back, " << iters << " iterations\n";
    for(const Result& r : res)
        std::cout << "  " << r.name << ": median " << r.fwd_ms << " ms, p95 " << r.p95_ms << " ms\n";
    return 0;
}
