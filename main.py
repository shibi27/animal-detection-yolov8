!pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import torch
import cv2
import os
import glob
import matplotlib.pyplot as plt

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

metrics = model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

test_images_folder = "/content/drive/MyDrive/test"

for image_path in glob.glob(f"{test_images_folder}/*.jpg"):

    results = model(image_path)
    img = cv2.imread(image_path)

    print(f"\nResults for: {os.path.basename(image_path)}")

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = model.names[label]

            print(f"Detected: {class_name} | Confidence: {confidence:.2f}")

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)

            label_text = f"{class_name} {confidence:.2f}"

            text_size = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                2
            )[0]

            cv2.rectangle(
                img,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                (0,255,0),
                -1
            )

            cv2.putText(
                img,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,0),
                2
            )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

print("\n✅ Inference completed! Check the displayed results.")
