# 📦 ML Scrap Classification – End-to-End Pipeline

## 🧠 Overview
This project is a mini end-to-end machine learning pipeline for classifying scrap materials in real-time, simulating an industrial scrap-sorting scenario.

We use a CNN model (ResNet18) with transfer learning to classify 5 types of materials:
- Cardboard
- Glass
- Metal
- Paper
- Plastic

## 🗃 Dataset Used
- **TrashNet Dataset**: https://github.com/garythung/trashnet
- Includes labeled images for each scrap material class.
- Preprocessed and augmented during training (resizing, normalization).

## 🏗 Architecture
- **Model**: ResNet18 (transfer learning)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Deployment Format**: ONNX

## 🧪 Real-Time Simulation
A simulation script (`simulate.py`) mimics real-time image capture from a conveyor belt. It:
- Classifies each frame from a folder
- Logs predictions and confidence to a CSV
- Flags low-confidence predictions

## 📁 Project Structure
ML-Scrap-Classification/
├── data/ # Sample dataset (or download from TrashNet)
├── models/ # Trained .pth and ONNX models
├── results/ # Output predictions CSV
├── src/
│ ├── train.py # Model training script
│ ├── inference.py # Single image prediction using ONNX
│ └── simulate.py # Real-time simulation loop
├── README.md # This documentation
├── performance_report.md # Key metrics and visualization summary
└── ML_Scrap_Classifier_Colab.ipynb # Google Colab version


## 🛠 How to Run
### 1. Train the Model
```bash
cd src
python train.py
2. Inference on a Single Image
python inference.py ../data/plastic/sample_plastic.jpg
3. Run Simulation
python simulate.py
📊 Metrics
See performance_report.md for accuracy, precision, recall, confusion matrix, and visualizations.

Made for Machine Learning Internship Task – AlfaStack