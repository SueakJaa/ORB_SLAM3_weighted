#!/usr/bin/env python3
"""Build a TensorRT engine from the YOLOv5 ONNX model. Run once per GPU / TensorRT version (an engine only loads
with the TensorRT version and GPU it was built for).

    python3 scripts/trt/build_engine.py --onnx ~/yolov5_models/yolov5n.onnx --out ~/yolov5_models/yolov5n_fp32.engine
    python3 scripts/trt/build_engine.py --onnx ... --out ~/yolov5_models/yolov5n_fp16.engine --fp16

Default is strict FP32: TF32 (on by default on RTX 30xx/40xx) is switched off, so the network's numbers stay as
close as possible to OpenCV's. --fp16 allows half precision (faster, scores change slightly).
"""
import argparse
import os
import sys

try:
    import tensorrt as trt
except ImportError:
    sys.exit("python module 'tensorrt' not found: see scripts/trt/README.md step 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    a = ap.parse_args()
    onnx, out = os.path.expanduser(a.onnx), os.path.expanduser(a.out)

    print(f"TensorRT {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):   # needed in 8.x, ignored/deprecated in 10.x
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit("ONNX parse failed")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"input  {t.name} {tuple(t.shape)}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"output {t.name} {tuple(t.shape)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if a.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    print("building (takes a minute or two) ...")
    data = builder.build_serialized_network(network, config)
    if data is None:
        sys.exit("engine build failed")
    with open(out, "wb") as f:
        f.write(data)
    print(f"engine written: {out} ({os.path.getsize(out) / 1e6:.1f} MB, {'FP16' if a.fp16 else 'FP32 (no TF32)'})")


if __name__ == "__main__":
    main()
