import torch
from ultralytics import YOLO

MODEL_PATH = "path/to/model.pt"
DATA_PATH = "path/to/data.yaml"
model = YOLO(MODEL_PATH)


device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


#[MAKE ALL LAYERS TRAINABLE]========================================

#for param in model.model.parameters():
#    param.requires_grad = True  # Ensure all layers are trainable

#===================================================================

#[TWEAK TRAINING PARAMS]============================================

#model.train(
#   data=DATA_PATH,
#   epochs=300,
#   batch=16,
#   imgsz=256,
#   optimizer="AdamW",
#   lr0=1e-4,
#   lrf=0.01,
#   mosaic=0.5,
#   mixup=0.2,
#   device=device
#)

#===================================================================

#[USE tune() FUNC]==================================================

#results = model.tune(
#    data="waldo-dataset/data.yaml",
#    epochs=100,
#    iterations=50,  # Number of tuning cycles
#    optimizer="AdamW",
#    imgsz=256,
#    project="tuning-waldo",
#)

#===================================================================

#[DEFAULT]

model.train(
    data=DATA_PATH,
    epochs=300,
    batch=16,
    imgsz=256,
    device=device
)
