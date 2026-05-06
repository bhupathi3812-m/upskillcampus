from ultralytics import YOLO

import os

model=YOLO("models/best.pt")

output_folder="outputs"

os.makedirs(output_folder,exist_ok=True)

results=model.predict(
        source="dataset/images/val",
        imgsz=520,
        project="outputs",
        conf=0.25,
        save=True,
        name="result"
)

print("Detection Completed! Check output folder")


