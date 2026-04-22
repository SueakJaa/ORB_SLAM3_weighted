from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt")

img = cv2.imread("/home/shirt/datasets/rgbd_dataset_freiburg3_walking_xyz/rgb/1341846342.347894.png")

results = model(img)

# Show result
annotated = results[0].plot()
cv2.imshow("result", annotated)
cv2.waitKey(0)
