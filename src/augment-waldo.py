import cv2
import os
import glob
import numpy as np
import albumentations as A
from tqdm import tqdm

# Paths
INPUT_FOLDER = "path/to/input/imgs"
OUTPUT_FOLDER = "destination/for/output/imgs"
LABELS_FOLDER = "path/to/input/labels"
OUTPUT_LABELS_FOLDER = "destination/for/transformed/labels"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)

augmentations = A.Compose([
    A.HorizontalFlip(p=0.3),  # Flip 30% of images
    A.Rotate(limit=30, p=0.5),  # Rotate up to ±30 degrees
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.Perspective(scale=(0.05, 0.15), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=10, max_width=10, p=0.3),
], bbox_params=A.BboxParams(format="yolo", label_fields=["category_ids"]))

image_files = glob.glob(INPUT_FOLDER + "*.jpg")

for img_path in tqdm(image_files, desc="Augmenting images"):
    image = cv2.imread(img_path)
    image_name = os.path.basename(img_path)

    label_path = LABELS_FOLDER + image_name.replace(".jpg", ".txt")
    if not os.path.exists(label_path):
        continue  # Skip images without labels

    with open(label_path, "r") as f:
        lines = f.readlines()

    boxes = []
    category_ids = []
    for line in lines:
        parts = line.strip().split()
        category_ids.append(int(parts[0]))
        boxes.append([float(x) for x in parts[1:]])  # x_center, y_center, width, height

    for i in range(5):
        augmented = augmentations(image=image, bboxes=boxes, category_ids=category_ids)
        aug_img = augmented["image"]
        aug_boxes = augmented["bboxes"]

        aug_img_name = f"{image_name.split('.')[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, aug_img_name), aug_img)

        aug_label_path = os.path.join(OUTPUT_LABELS_FOLDER, aug_img_name.replace(".jpg", ".txt"))
        with open(aug_label_path, "w") as f:
            for box, cls in zip(aug_boxes, category_ids):
                f.write(f"{cls} " + " ".join(map(str, box)) + "\n")

print("Data augmentation completed!")
