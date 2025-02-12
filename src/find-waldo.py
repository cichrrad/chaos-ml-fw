import cv2
import sys
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

if len(sys.argv) < 2:
    print("Usage: python find_waldo.py <image_path> [output_file]")
    sys.exit(1)

image_path = sys.argv[1]
output_file = sys.argv[2]
MODEL_PATH = "path/to/weights/best.pt"

# Load trained YOLO model
model = YOLO(MODEL_PATH)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from {image_path}")
    sys.exit(1)

height, width, _ = image.shape
imgsz = max(width, height)
print(f"Processing image: {image_path} ({width}x{height} pixels), using imgsz={imgsz}")

results = model(image_path, imgsz=imgsz, conf=0.8, iou=0.4)

result = results[0]

result.save(filename=output_file)

print(f"Detection completed! Result saved as: {output_file}")
