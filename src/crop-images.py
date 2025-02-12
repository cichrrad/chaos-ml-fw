import cv2
import os
import glob
import numpy as np

# Paths
INPUT_FOLDER = "path/to/input/imgs"
OUTPUT_FOLDER = "destination/for/output/imgs"
LABELS_FOLDER = "path/to/input/labels"
OUTPUT_LABELS_FOLDER = "destination/for/transformed/labels"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)

CROP_SIZE = 256

image_files = glob.glob(INPUT_FOLDER + "*.jpg")

for img_path in image_files:
    image = cv2.imread(img_path)
    image_name = os.path.basename(img_path)

    label_path = LABELS_FOLDER + image_name.replace(".jpg", ".txt")
    if not os.path.exists(label_path):
        continue  # Skip images without labels

    with open(label_path, "r") as f:
        lines = f.readlines()

    img_h, img_w, _ = image.shape

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        x_pixel = int(x_center * img_w)
        y_pixel = int(y_center * img_h)
        w_pixel = int(box_width * img_w)
        h_pixel = int(box_height * img_h)

        x1 = max(0, x_pixel - CROP_SIZE // 2)
        y1 = max(0, y_pixel - CROP_SIZE // 2)
        x2 = min(img_w, x1 + CROP_SIZE)
        y2 = min(img_h, y1 + CROP_SIZE)

        cropped_img = image[y1:y2, x1:x2]

        cropped_img_name = f"{image_name.split('.')[0]}_cropped.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, cropped_img_name), cropped_img)

        new_x = (x_pixel - x1) / (x2 - x1)
        new_y = (y_pixel - y1) / (y2 - y1)
        new_w = w_pixel / (x2 - x1)
        new_h = h_pixel / (y2 - y1)

        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        new_w = max(0.0, min(1.0, new_w))
        new_h = max(0.0, min(1.0, new_h))

        aug_label_path = os.path.join(OUTPUT_LABELS_FOLDER, cropped_img_name.replace(".jpg", ".txt"))
        with open(aug_label_path, "w") as f:
            f.write(f"{class_id} {new_x} {new_y} {new_w} {new_h}\n")

print("Done cropping!")
