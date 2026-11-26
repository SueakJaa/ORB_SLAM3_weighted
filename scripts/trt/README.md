# TensorRT for the YOLO network (optional)

Why: OpenCV DNN runs yolov5n on the GPU as a few hundred small kernel launches. On this PC (RTX 3080 Ti, WSL) the
network takes ~18 ms inside SLAM, almost all launch overhead (tools/yolo_bench, 25 Nov). TensorRT fuses the
layers into a few large kernels; expected network time a few ms. Pre- and post-processing stay the same code, so
only the network's numbers can differ (check below).

Without these steps nothing changes: the build finds no TensorRT and YOLO keeps using OpenCV DNN.

## Steps (from ~/ORB_SLAM3, one command at a time)

1. Install TensorRT for CUDA 11.8 (libraries + Python builder; no NVIDIA login needed):

       python3 --version
       pip3 install --user tensorrt-cu11

   If pip refuses with "externally-managed-environment", add `--break-system-packages`.
   Check: `python3 -c "import tensorrt; print(tensorrt.__version__)"`

2. Get the C++ headers for that version (pip has none):

       bash scripts/trt/fetch_headers.sh

3. Build the engine once (FP32, TF32 off):

       python3 scripts/trt/build_engine.py --onnx ~/yolov5_models/yolov5n.onnx --out ~/yolov5_models/yolov5n_fp32.engine

4. Rebuild; cmake should print `TensorRT: .../libnvinfer.so...`:

       cd build && cmake .. | grep -i tensorrt && make -j4 && cd ..

5. Compare with OpenCV on a real frame, and time it:

       ./tools/yolo_bench ~/yolov5_models/yolov5n.onnx 200 ~/yolov5_models/yolov5n_fp32.engine $(ls ~/datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/*.png | sed -n 300p)

   (frame 300 of walking_xyz, people in view.) The person candidates should agree (same count, "in both" equal to both counts) and the person-score
   difference should be tiny (~1e-4 or less).

6. Use it in SLAM: `DSLAM_YOLO_TRT=/home/shirt/yolov5_models/yolov5n_fp32.engine` (configs/speed_tum.yaml has
   variants). The log shows `[YOLO] backend = TensorRT` and `yolo_backend=tensorrt` in the [Timing] line.

An engine only loads with the TensorRT version and GPU it was built with; after updating TensorRT, rebuild it.
