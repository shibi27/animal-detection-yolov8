Animal Detection Using YOLOv8

This project implements object detection for animals using YOLOv8 and a custom dataset. It trains a YOLOv8 model, validates it, and performs inference on test images.

Table of Contents

Overview

Installation

Dataset

Training

Validation

Inference

Results

License

Overview

This project uses the Roboflow API to download a custom animal dataset and train a YOLOv8 model for object detection. It also supports inference on new images and displays bounding boxes with class labels and confidence scores.

Installation

Install the required Python packages:

!pip install roboflow
!pip install ultralytics opencv-python matplotlib


Mount Google Drive to access test images:

from google.colab import drive
drive.mount('/content/drive')

Dataset

The dataset is hosted on Roboflow and contains images of different animals. It is downloaded in YOLOv8 format using:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("shibi-p-xmggq").project("animal-detection-using-yolov8-cvuyo")
version = project.version(2)
dataset = version.download("yolov8")


The dataset directory includes:

train/ – Training images and labels

valid/ – Validation images and labels

test/ – Test images (optional)

Training

The YOLOv8 model (yolov8n.pt) is trained on the dataset with the following settings:

from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLO("yolov8n.pt")
model.train(
    data="/content/Animal-Detection-Using-YOLOv8-2/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=device
)

Validation

After training, model performance is evaluated using the val() method:

metrics = model.val()
print(f"mAP50: {metrics.box.map:.3f}")
print(f"mAP50-95: {metrics.box.map50:.3f}")


mAP50 – Mean Average Precision at IoU threshold 0.5

mAP50-95 – Mean Average Precision averaged over IoU thresholds 0.5 to 0.95

Inference

Run inference on images stored in Google Drive:

import cv2
import glob
import matplotlib.pyplot as plt

test_images_folder = "/content/drive/My Drive/test"

for image_path in glob.glob(f"{test_images_folder}/*.jpg"):
    results = model(image_path)
    img = cv2.imread(image_path)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = model.names[label]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(img, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

Results

Bounding boxes with class labels and confidence scores are displayed on test images.

Example metrics from validation:

mAP50: 0.85
mAP50-95: 0.62

License

This project is for educational purposes. Dataset usage is subject to Roboflow terms.
