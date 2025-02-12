# Paths Configuration for Local Training & Inference

## Training Scripts

### Augmentation Script (`src/augment-waldo.py`)
- **INPUT_FOLDER** – Path to the folder containing images to apply augmentations.
- **OUTPUT_FOLDER** – Path to store the generated augmented images.
- **LABELS_FOLDER** – Path to the folder containing labels of the original images.
- **OUTPUT_LABELS_FOLDER** – Path to store labels of the generated images.

### Cropping Script (`src/crop-images.py`)
- **INPUT_FOLDER** – Path to the folder containing images to crop.
- **OUTPUT_FOLDER** – Path to store the cropped images.
- **LABELS_FOLDER** – Path to the folder containing labels of the original images.
- **OUTPUT_LABELS_FOLDER** – Path to store labels of the cropped images.

### Training Script (`src/train.py`)
- **MODEL_PATH** – Path to the YOLO model.
- **DATA_PATH** – Path to the `data.yaml` file.

## Local Run Script

### Inference Script (`src/find-waldo.py`)
- **MODEL_PATH** – Path to the YOLO model weights.

---

Ensure all paths are correctly set before running the scripts.
