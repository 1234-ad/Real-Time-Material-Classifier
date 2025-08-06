import onnxruntime as ort
import numpy as np
import cv2
import torch
from torchvision import transforms
import sys

# Class names (edit if needed)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Preprocessing function
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0).numpy()
    return tensor

# Inference function
def predict(image_path, model_path='../models/scrap_classifier.onnx'):
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    image_tensor = preprocess(image_path)
    outputs = ort_session.run(None, {input_name: image_tensor})[0]
    confidences = torch.softmax(torch.tensor(outputs[0]), dim=0)
    pred_idx = torch.argmax(confidences).item()
    pred_label = classes[pred_idx]
    pred_conf = confidences[pred_idx].item()
    return pred_label, pred_conf

# Example usage
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "../data/plastic/sample_plastic.jpg"
    label, conf = predict(image_path)
    print(f"Prediction: {label} ({conf:.2f})")
