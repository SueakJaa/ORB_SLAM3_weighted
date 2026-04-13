from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys

# ====== INPUT ARGUMENT ======
# usage: python3 mask_dataset.py /path/to/dataset

if len(sys.argv) != 2:
    print("Usage: python3 mask_dataset.py /path/to/dataset")
    sys.exit(1)

dataset_path = sys.argv[1]
input_dir = os.path.join(dataset_path, "rgb")

dataset_name = os.path.basename(dataset_path)

output_dir = f"/home/shirt/masked_{dataset_name}"
rgb_out = os.path.join(output_dir, "rgb")

print("Input:", input_dir)
print("Output:", output_dir)

os.makedirs(rgb_out, exist_ok=True)

print("Saving directory confirmed:", rgb_out)
# ====== LOAD MODEL ======
model = YOLO("yolov8n-seg.pt")

files = sorted(os.listdir(input_dir))

for file in files:
    if not file.endswith(".png"):
        continue

    path = os.path.join(input_dir, file)
    img = cv2.imread(path)

    if img is None:
        print("❌ Failed:", file)
        continue

    results = model(img, conf=0.5)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # ====== SEGMENTATION ======
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for i in range(len(masks)):
            if int(classes[i]) != 0:  # person only
                continue

            m = masks[i]
            m = cv2.resize(m, (img.shape[1], img.shape[0]))
            m = m > 0.5

            if np.sum(m) < 2000:
                continue

            mask[m] = 255

    # expand mask
    mask = cv2.dilate(mask, None, iterations=3)

    # convert to binary mask
    binary_mask = np.ones_like(mask) * 255
    binary_mask[mask > 0] = 0

    # ✅ SAVE MASK (CORRECT)
    save_path = os.path.join(rgb_out, file)
    print("Saving to:", save_path)
    cv2.imwrite(save_path, binary_mask)

    print("Processed:", file)

print("DONE")