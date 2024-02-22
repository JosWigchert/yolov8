import os
import json
import random
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from uuid import uuid4

import label_studio_sdk
from label_studio_ml.model import LabelStudioMLBase

from YOLOv8Segmentation import YOLOV8Segmentation

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv(
    "LABEL_STUDIO_API_KEY", "08aa8df0e07618412c91ab9e03fcab2f2a6a216d"
)
MODEL_BACKUP = os.getenv("MODEL_BACKUP", "best.pt")


class YOLOV8(YOLOV8Segmentation):
    pass
