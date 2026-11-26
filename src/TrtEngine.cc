#include "TrtEngine.h"

#include <iostream>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>

namespace
{
class Logger : public nvinfer1::ILogger
{
    void log(Severity s, const char* msg) noexcept override
    {
        if(s <= Severity::kWARNING) std::cout << "[TensorRT] " << msg << std::endl;
    }
};
Logger gLogger;

bool cudaOk(cudaError_t e, const char* what)
{
    if(e == cudaSuccess) return true;
    std::cerr << "[TensorRT] CUDA error in " << what << ": " << cudaGetErrorString(e) << std::endl;
    return false;
}

size_t volume(const nvinfer1::Dims& d)
{
    size_t v = 1;
    for(int i = 0; i < d.nbDims; i++) v *= static_cast<size_t>(d.d[i] > 0 ? d.d[i] : 1);
    return v;
}
}   // namespace

struct TrtEngine::Impl
{
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<void*> deviceBuffers;   // one per I/O tensor
    void* dIn = nullptr;
    void* dOut = nullptr;
    size_t inBytes = 0, outBytes = 0;
    int outRows = 0;
    float* hOut = nullptr;              // pinned host memory for the decoded output
    bool ok = false;

    ~Impl()
    {
        for(void* b : deviceBuffers) if(b) cudaFree(b);
        if(hOut) cudaFreeHost(hOut);
        if(stream) cudaStreamDestroy(stream);
        delete context;
        delete engine;
        delete runtime;
    }
};

TrtEngine::TrtEngine() : p(new Impl) {}
TrtEngine::~TrtEngine() { delete p; }
bool TrtEngine::Available() { return true; }
bool TrtEngine::Ok() const { return p->ok; }

bool TrtEngine::Load(const std::string& enginePath)
{
    std::ifstream f(enginePath, std::ios::binary);
    if(!f) { std::cerr << "[TensorRT] cannot open engine " << enginePath << std::endl; return false; }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    p->runtime = nvinfer1::createInferRuntime(gLogger);
    if(!p->runtime) return false;
    p->engine = p->runtime->deserializeCudaEngine(data.data(), data.size());
    if(!p->engine) { std::cerr << "[TensorRT] engine does not deserialize (built with another TensorRT version or GPU?)" << std::endl; return false; }
    p->context = p->engine->createExecutionContext();
    if(!p->context) return false;
    if(!cudaOk(cudaStreamCreate(&p->stream), "cudaStreamCreate")) return false;

    const int n = p->engine->getNbIOTensors();
    p->deviceBuffers.assign(n, nullptr);
    for(int i = 0; i < n; i++)
    {
        const char* name = p->engine->getIOTensorName(i);
        const nvinfer1::Dims d = p->engine->getTensorShape(name);
        const bool isInput = p->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        if(p->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
        {
            std::cerr << "[TensorRT] tensor " << name << " is not float32" << std::endl;
            return false;
        }
        for(int k = 0; k < d.nbDims; k++)
            if(d.d[k] < 0) { std::cerr << "[TensorRT] dynamic shape in " << name << " not supported" << std::endl; return false; }
        const size_t bytes = volume(d) * sizeof(float);
        if(!cudaOk(cudaMalloc(&p->deviceBuffers[i], bytes), "cudaMalloc")) return false;
        if(!p->context->setTensorAddress(name, p->deviceBuffers[i])) return false;

        std::cout << "[TensorRT] " << (isInput ? "input " : "output ") << name << " [";
        for(int k = 0; k < d.nbDims; k++) std::cout << (k ? "x" : "") << d.d[k];
        std::cout << "]" << std::endl;

        if(isInput)
        {
            if(d.nbDims != 4 || d.d[1] != 3 || d.d[2] != 640 || d.d[3] != 640) { std::cerr << "[TensorRT] input must be 1x3x640x640" << std::endl; return false; }
            p->dIn = p->deviceBuffers[i];
            p->inBytes = bytes;
        }
        else if(d.nbDims == 3 && d.d[2] == 85)   // the decoded output; the three raw heads are left on the GPU
        {
            p->dOut = p->deviceBuffers[i];
            p->outBytes = bytes;
            p->outRows = d.d[1];
        }
    }
    if(!p->dIn || !p->dOut) { std::cerr << "[TensorRT] no 1x3x640x640 input or [1,N,85] output" << std::endl; return false; }
    if(!cudaOk(cudaMallocHost(reinterpret_cast<void**>(&p->hOut), p->outBytes), "cudaMallocHost")) return false;
    p->ok = true;
    return true;
}

bool TrtEngine::Infer(const cv::Mat& blob, cv::Mat& decoded)
{
    if(!p->ok || !blob.isContinuous() || blob.total() * blob.elemSize() != p->inBytes) return false;
    if(!cudaOk(cudaMemcpyAsync(p->dIn, blob.ptr(), p->inBytes, cudaMemcpyHostToDevice, p->stream), "copy in")) return false;
    if(!p->context->enqueueV3(p->stream)) { std::cerr << "[TensorRT] enqueueV3 failed" << std::endl; return false; }
    if(!cudaOk(cudaMemcpyAsync(p->hOut, p->dOut, p->outBytes, cudaMemcpyDeviceToHost, p->stream), "copy out")) return false;
    if(!cudaOk(cudaStreamSynchronize(p->stream), "sync")) return false;
    const int sz[3] = {1, p->outRows, 85};
    decoded = cv::Mat(3, sz, CV_32F, p->hOut);   // header on pinned memory, no copy
    return true;
}

#else   // built without TensorRT

struct TrtEngine::Impl { bool ok = false; };
TrtEngine::TrtEngine() : p(new Impl) {}
TrtEngine::~TrtEngine() { delete p; }
bool TrtEngine::Available() { return false; }
bool TrtEngine::Ok() const { return false; }
bool TrtEngine::Load(const std::string&)
{
    std::cerr << "[TensorRT] this build has no TensorRT (see scripts/trt/README.md); using OpenCV DNN" << std::endl;
    return false;
}
bool TrtEngine::Infer(const cv::Mat&, cv::Mat&) { return false; }

#endif
