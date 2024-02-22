from ultralytics import YOLO
from multiprocessing import freeze_support
from torch import torch
from PIL import Image
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO("runs/segment/train21/weights/best.pt")
image = "datasets/custom/val/images/a8d69716-Screenshot_20240123_09522407.png"

# Train the model
if __name__ == "__main__":
    freeze_support()
    results = model.predict(
        Image.open(image),
        device=device,
    )
    result = results[0]

    masks = results[0].masks.xy
    print(results[0].masks.xyn[0] * 100)
    img = cv2.imread(image)
    for mask in masks:
        for i in range(len(mask)):
            (u, v) = (int(mask[i][0]), int(mask[i][1]))
            # print((u,v))
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
    cv2.imshow("Marked Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
