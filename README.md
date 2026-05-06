# 🌱 Crop & Weed Detection System

## 📌 Overview
This project uses YOLOv8 to detect crops and weeds from agricultural field images.

## 🚀 Features
- Data validation and cleaning
- Exploratory Data Analysis (EDA)
- YOLOv8 model training
- Prediction and evaluation

## 📂 Project Structure
```
crop-weed-detection/
│
├── data/raw/
├── src/
├── dataset/
├── runs/
├── requirements.txt
└── README.md
```

## ⚙️ Installation
```bash
pip install -r requirements.txt
```

## ▶️ Run Training
```bash
python src/training.py
```

## 📊 Output
- Trained model: runs/detect/train/weights/best.pt
- Predictions saved in runs/

## 🧠 Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Pandas, Matplotlib, Seaborn

## 📌 Note
Place your dataset inside `data/raw/` with images and YOLO format labels.

## ✨ Author
Your Name
