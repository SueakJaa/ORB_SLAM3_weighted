#!/usr/bin/env python3
"""Scan a whole TUM sequence with the SLAM's YOLO model and show WHAT it detects.

Usage (from ~/ORB_SLAM3):
    python3 yolo_scan.py <model.onnx> <sequence dir> [step] [conf]
    python3 yolo_scan.py ~/yolov5_models/yolov5n.onnx ~/datasets/rgbd_dataset_freiburg1_room

Same pre-processing and decoding as YOLOdetector.cc (letterbox, decoded or raw-head output,
obj >= conf and obj*class >= conf, classes person/car/motorbike/bus/truck), plus NMS so each
object is counted once. Prints how many frames have a detection, which classes, the confidence
spread and box sizes, and saves annotated frames to yolo_scan_<sequence>/ so you can look at them.
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

DYNAMIC = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
ANCHORS = {8: [(10, 13), (16, 30), (33, 23)], 16: [(30, 61), (62, 45), (59, 119)],
           32: [(116, 90), (156, 198), (373, 326)]}
MAX_SAVED = 40


def run_net(net, blob):
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    for o in outs:
        if o.ndim == 3 and o.shape[2] == 85:
            return o.reshape(-1, 85)
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
    return np.concatenate(rows) if rows else np.zeros((0, 85), np.float32)


def letterbox(img):
    h, w = img.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    px, py = (640 - nw) // 2, (640 - nh) // 2
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(img[:, :, :3], (nw, nh))
    return canvas, scale, px, py


def detect(net, img, conf):
    lb, scale, px, py = letterbox(img)
    rows = run_net(net, cv2.dnn.blobFromImage(lb, 1 / 255.0, (640, 640), swapRB=True, crop=False))
    obj = rows[:, 4]
    cls = rows[:, 5:].argmax(1)
    score = obj * rows[:, 5:].max(1)
    keep = (obj >= conf) & (score >= conf) & np.isin(cls, list(DYNAMIC))
    rows, cls, score = rows[keep], cls[keep], score[keep]
    h_img, w_img = img.shape[:2]
    boxes = []
    for r in rows:
        cx, cy, w, h = r[:4]
        x1, y1 = (cx - w / 2 - px) / scale, (cy - h / 2 - py) / scale
        boxes.append([int(max(0, x1)), int(max(0, y1)), int(min(w / scale, w_img)), int(min(h / scale, h_img))])
    idx = cv2.dnn.NMSBoxes(boxes, score.tolist(), conf, 0.45) if boxes else []
    idx = np.array(idx).reshape(-1)
    return [(boxes[i], int(cls[i]), float(score[i])) for i in idx]


def main(model, seq, step=1, conf=0.4):
    net = cv2.dnn.readNetFromONNX(model)
    names = [l.split()[1] for l in open(os.path.join(seq, "rgb.txt")) if l.strip() and l[0] != "#"]
    tag = os.path.basename(os.path.normpath(seq)).replace("rgbd_dataset_", "")
    outdir = f"yolo_scan_{tag}"
    os.makedirs(outdir, exist_ok=True)
    n_frames, n_hit, saved = 0, 0, 0
    classes, confs, areas, hit_frames = Counter(), [], [], []
    for k in range(0, len(names), step):
        img = cv2.imread(os.path.join(seq, names[k]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        n_frames += 1
        dets = detect(net, img, conf)
        if not dets:
            continue
        n_hit += 1
        hit_frames.append(k)
        for (x, y, w, h), c, s in dets:
            classes[DYNAMIC[c]] += 1
            confs.append(s)
            areas.append(100.0 * w * h / (img.shape[0] * img.shape[1]))
        if saved < MAX_SAVED and (saved == 0 or k - hit_frames[-2 if len(hit_frames) > 1 else -1] > 5 or saved < 10):
            vis = img.copy()
            for (x, y, w, h), c, s in dets:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, f"{DYNAMIC[c]} {s:.2f}", (x + 3, max(15, y + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, f"frame {k}", (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imwrite(os.path.join(outdir, f"frame{k:05d}.png"), vis)
            saved += 1
        if n_frames % 200 == 0:
            print(f"  ... {n_frames} frames scanned", flush=True)

    print(f"\n=== {tag}  (conf >= {conf}, every {step}. frame)")
    print(f"frames with a detection: {n_hit}/{n_frames} = {100.0 * n_hit / max(n_frames, 1):.1f} %")
    if not n_hit:
        return
    print("classes:", ", ".join(f"{c} {n}" for c, n in classes.most_common()))
    c = np.array(confs)
    print(f"confidence: min {c.min():.2f}  median {np.median(c):.2f}  max {c.max():.2f}  "
          f"| share >= 0.5: {100 * (c >= 0.5).mean():.0f} %  >= 0.6: {100 * (c >= 0.6).mean():.0f} %  >= 0.7: {100 * (c >= 0.7).mean():.0f} %")
    a = np.array(areas)
    print(f"box size (% of image): median {np.median(a):.1f}  max {a.max():.1f}")
    # how long do detections last? real people persist, false positives flicker
    runs, cur = [], 1
    for p, q in zip(hit_frames, hit_frames[1:]):
        if q - p <= step:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    r = np.array(runs) * step
    print(f"detection streaks: {len(runs)}  | length in frames: median {np.median(r):.0f}  max {r.max()}"
          f"  | streaks of 1-2 frames: {100 * (r <= 2).mean():.0f} %")
    print(f"annotated frames saved in {outdir}/ ({saved} images)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.4)
