🐾 Animal Detection Using YOLOv8

Detect and classify animals in images using YOLOv8 and a custom dataset. This project trains a YOLOv8 model, evaluates its performance, and performs inference on test images with bounding boxes and confidence scores.

🌟 Features

✅ Detect multiple animal species in images

✅ Custom dataset from Roboflow

✅ Training with YOLOv8 (PyTorch backend)

✅ High mAP detection with bounding boxes and confidence

✅ Easy inference on Google Drive images

📸 Demo

Here’s an example of predictions:

Bounding boxes with animal class names and confidence scores.

🛠️ Installation

Clone this repository and install dependencies:

!pip install roboflow
!pip install ultralytics opencv-python matplotlib


Mount Google Drive to access test images:

from google.colab import drive
drive.mount('/content/drive')

📂 Dataset

The dataset is hosted on Roboflow:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("shibi-p-xmggq").project("animal-detection-using-yolov8-cvuyo")
version = project.version(2)
dataset = version.download("yolov8")


Dataset contains:

train/ – Training images and labels

valid/ – Validation images and labels

test/ – Test images

🚀 Training

Train YOLOv8 on the dataset:

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

📊 Validation

Evaluate the trained model:

metrics = model.val()
print(f"mAP50: {metrics.box.map:.3f}")
print(f"mAP50-95: {metrics.box.map50:.3f}")


mAP50 – Mean Average Precision at IoU 0.5

mAP50-95 – Mean Average Precision averaged over IoU thresholds 0.5–0.95

🖼️ Inference

Run inference on test images stored in Google Drive:

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

📈 Results

Bounding boxes with class names and confidence scores are displayed on images.

Example metrics:

mAP50: 0.85
mAP50-95: 0.62

⚙️ Requirements

Python ≥ 3.8

PyTorch

Ultralytics YOLOv8

OpenCV, Matplotlib

📄 License

This project is for educational purposes only. Dataset usage is subject to Roboflow terms.
