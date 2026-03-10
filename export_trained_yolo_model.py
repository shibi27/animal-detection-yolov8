import shutil
import os

best_model = "/content/drive/MyDrive/YoloTraining/animal_model/weights/best.pt"
last_model = "/content/drive/MyDrive/YoloTraining/animal_model/weights/last.pt"

save_folder = "/content/drive/MyDrive/YoloModels"
os.makedirs(save_folder, exist_ok=True)

shutil.copy(best_model, save_folder)
shutil.copy(last_model, save_folder)

print("✅ Model saved successfully!")
print("Saved location:", save_folder)
