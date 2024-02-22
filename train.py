from ultralytics import YOLO
from multiprocessing import freeze_support
from torch import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
if __name__ == "__main__":
    freeze_support()
    results = model.train(data="custom.yaml")
