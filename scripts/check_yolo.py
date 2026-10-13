#!/usr/bin/env python3
"""Check whether the YOLO model used by ORB-SLAM3 actually detects people on TUM frames.

Usage (from ~/ORB_SLAM3):
    python3 check_yolo.py /home/shirt/yolov5_models/yolov5n.onnx ~/datasets/rgbd_dataset_freiburg3_walking_xyz

It runs the model with exactly the same pre-processing as YOLOdetector::Infer
(blobFromImage, 1/255, 640x640, swapRB=true) and then parses the output with the same rule
as the C++ code (YOLOv5 layout [1, 25200, 85], obj_conf >= 0.4, obj_conf*class >= 0.4,
classes person/car/motorbike/bus/truck). It also prints the raw output shape, so a model
exported in a different layout (e.g. YOLOv8-style [1, 84, 8400]) is easy to spot.
"""
import os
import sys

import cv2
import numpy as np

DYNAMIC = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
CONF = 0.4


def cpp_style_parse(out):
    """Mirror of the C++ loop: walk N rows of 85 floats."""
    flat = out.reshape(-1)
    n_rows = flat.size // 85
    rows = flat[: n_rows * 85].reshape(n_rows, 85)
    obj = rows[:, 4]
    cls = rows[:, 5:].argmax(1)
    score = obj * rows[:, 5:].max(1)
    keep = (obj >= CONF) & (score >= CONF)
    dyn = keep & np.isin(cls, list(DYNAMIC))
    return n_rows, obj.max(), int(keep.sum()), int(dyn.sum()), rows[dyn]


ANCHORS = {8: [(10, 13), (16, 30), (33, 23)], 16: [(30, 61), (62, 45), (59, 119)],
           32: [(116, 90), (156, 198), (373, 326)]}


def run_net(net, blob):
    """All outputs -> decoded [N, 85] rows, same as the C++ code (decoded output, or raw heads)."""
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    for o in outs:
        if o.ndim == 3 and o.shape[2] == 85:
            return o.reshape(-1, 85), [o.shape for o in outs]
    rows = []
    for o in outs:
        if o.ndim != 5 or o.shape[1] != 3 or o.shape[4] != 85:
            continue
        ny, nx = o.shape[2], o.shape[3]
        stride = 640 // nx
        y = 1 / (1 + np.exp(-o[0]))
        gy, gx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        d = y.copy()
        for a in range(3):
            aw, ah = ANCHORS[stride][a]
            d[a, ..., 0] = (y[a, ..., 0] * 2 - 0.5 + gx) * stride
            d[a, ..., 1] = (y[a, ..., 1] * 2 - 0.5 + gy) * stride
            d[a, ..., 2] = (y[a, ..., 2] * 2) ** 2 * aw
            d[a, ..., 3] = (y[a, ..., 3] * 2) ** 2 * ah
        rows.append(d.reshape(-1, 85))
    return (np.concatenate(rows) if rows else np.zeros((0, 85), np.float32)), [o.shape for o in outs]


def letterbox(img):
    """Same as the new C++ code: keep aspect ratio, pad to 640x640 with grey 114."""
    h, w = img.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    px, py = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh)) if (nw, nh) != (w, h) else img[:, :, :3]
    return canvas, scale, px, py


def box_to_image(r, scale, px, py, w_img, h_img):
    """Same box mapping as the new C++ code (letterbox branch), with clamping."""
    cx, cy, w, h = r[:4]
    x, y = int((cx - w / 2 - px) / scale), int((cy - h / 2 - py) / scale)
    x2, y2 = min(x + int(w / scale), w_img), min(y + int(h / scale), h_img)
    x, y = max(0, x), max(0, y)
    return x, y, x2, y2


def main(model_path, seq_dir):
    print("OpenCV", cv2.__version__)
    net = cv2.dnn.readNetFromONNX(model_path)
    names = [l.split()[1] for l in open(os.path.join(seq_dir, "rgb.txt")) if l[0] != "#"]
    picks = [0, 100, 200, 300, 400, 500, 600, 700]
    for k in picks:
        if k >= len(names):
            continue
        img = cv2.imread(os.path.join(seq_dir, names[k]), cv2.IMREAD_UNCHANGED)
        h_img, w_img = img.shape[:2]
        # (a) old C++ pre-processing: stretch to 640x640
        out, shapes = run_net(net, cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False))
        n_rows, obj_max, _, n_dyn_s, _ = cpp_style_parse(out)
        # (b) new C++ pre-processing: letterbox
        lb, scale, px, py = letterbox(img)
        out_l, _ = run_net(net, cv2.dnn.blobFromImage(lb, 1 / 255.0, (640, 640), swapRB=True, crop=False))
        n_rows, obj_max_l, _, n_dyn_l, rows_l = cpp_style_parse(out_l)
        if k == picks[0]:
            print("model outputs:", shapes)
        line = (f"frame {k:4d}"
                f"  STRETCH max obj {obj_max:.2f} dynamic boxes {n_dyn_s:3d}"
                f"  | LETTERBOX max obj {obj_max_l:.2f} dynamic boxes {n_dyn_l:3d}")
        if n_dyn_l:
            r = rows_l[rows_l[:, 4].argmax()]
            line += "  best box (x1,y1,x2,y2) = %d,%d,%d,%d" % box_to_image(r, scale, px, py, w_img, h_img)
            vis = img[:, :, :3].copy()
            for r in rows_l:
                x1, y1, x2, y2 = box_to_image(r, scale, px, py, w_img, h_img)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f"check_yolo_frame{k}.png", vis)
        print(line)

    print("\nWhat to look for:")
    print("  model outputs (1, 25200, 85) or three (1, 3, ny, nx, 85) -> both handled by the C++ code (v14)")
    print("  max obj must be between 0 and 1 (it is a probability after decoding)")
    print("  STRETCH 0 boxes but LETTERBOX > 0  -> confirms the pre-processing bug fixed in v13")
    print("  check_yolo_frame*.png             -> red boxes (letterbox) should sit on the people")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
