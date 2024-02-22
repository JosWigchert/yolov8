from ultralytics import YOLO
from multiprocessing import freeze_support
from torch import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO("runs/segment/train18/weights/best.pt")

# Train the model
if __name__ == "__main__":
    freeze_support()
    model.export(
        format="onnx",
        imgsz=[512, 512],
        opset=12,
        simplify=True,
        device=device,
    )  # export the model to ONNX format
