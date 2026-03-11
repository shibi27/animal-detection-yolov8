# 🐾 Animal Detection using YOLOv8

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Deep Learning](https://img.shields.io/badge/Computer%20Vision-Object%20Detection-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A **Deep Learning based Animal Detection System** built using **YOLOv8, PyTorch, and Roboflow**.
The model detects animals in images and displays **bounding boxes with confidence scores**.

---

# 🚀 Run in Google Colab (Recommended)

To avoid dependency issues, **run the notebook in Google Colab**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GMFZwjVozt6AIffmmEz7l0lkmS1u36TK?usp=sharing)

⚠️ Running in Colab ensures all required libraries install correctly and prevents environment errors.

---

# 📌 Project Overview

This project trains a **YOLOv8 object detection model** to detect animals in images.

The system performs:

✔ Animal detection
✔ Bounding box visualization
✔ Confidence score prediction
✔ Model evaluation using mAP metrics
✔ Inference on custom test images

The dataset is prepared using **Roboflow** and trained using **Ultralytics YOLOv8**.

---

# 🛠 Tech Stack

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming Language    |
| YOLOv8       | Object Detection Model  |
| PyTorch      | Deep Learning Framework |
| Roboflow     | Dataset Management      |
| OpenCV       | Image Processing        |
| Matplotlib   | Visualization           |
| Google Colab | Model Training          |

---

# 📊 Model Training

Model used: **YOLOv8 Nano (yolov8n)**

| Parameter  | Value       |
| ---------- | ----------- |
| Epochs     | 25          |
| Image Size | 640         |
| Batch Size | 16          |
| Framework  | Ultralytics |

---

# 📈 Model Performance

Example evaluation metrics:

```id="qb1ff8"
mAP50: 0.82
mAP50-95: 0.61
```

Metrics used:

* **mAP50**
* **mAP50-95**

These measure the accuracy of object detection predictions.

---

# 🖼 Detection Results

Example detections produced by the trained model.

![Detection Result](results/Test_1.jpg)

![Detection Result](results/gb.jpg)

Each detection includes:

* Animal class label
* Bounding box
* Confidence score

---

# ▶️ Running the Project

## 1️⃣ Open in Google Colab

Click the **Open in Colab** button above.

---

## 2️⃣ Install Dependencies

The notebook automatically installs required libraries:

```id="hsiqlq"
pip install roboflow
pip install ultralytics opencv-python matplotlib
```

---

## 3️⃣ Upload Test Images

Place your test images inside the following folder:

```id="yrhn49"
/content/drive/MyDrive/sample_test_images
```

Example images:

```id="zy7t0b"
dog.jpg
cow.jpg
horse.jpg
```

---

## 4️⃣ Update Image Path in Notebook

Make sure the path in the notebook matches:

```id="agbdg5"
test_images_folder = "/content/drive/MyDrive/sample_test_images"
```

---

# 📦 Dataset

Dataset created and exported using **Roboflow**.

Dataset preparation steps:

* Image annotation
* Data preprocessing
* Export in **YOLOv8 format**

---

# 🔮 Future Improvements

* Real-time **webcam animal detection**
* **Wildlife monitoring system**
* Integration with **object tracking**
* Edge deployment using **Raspberry Pi**

---
