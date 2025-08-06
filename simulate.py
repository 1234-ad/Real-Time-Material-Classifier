import os
import csv
import time
from inference import predict

# Path to folder containing test images
frame_folder = '../data/plastic'  # Replace with any class folder or test folder
output_csv = '../results/simulation_output.csv'
confidence_threshold = 0.6  # Flag if below this

# Ensure results folder exists
os.makedirs('../results', exist_ok=True)

# Open CSV to write
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Predicted Class', 'Confidence', 'Low Confidence Flag'])

    for frame in sorted(os.listdir(frame_folder)):
        if frame.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(frame_folder, frame)
            label, conf = predict(image_path)
            low_flag = 'YES' if conf < confidence_threshold else 'NO'
            print(f"{frame} → {label} ({conf:.2f}) | Low Confidence: {low_flag}")
            writer.writerow([frame, label, f"{conf:.2f}", low_flag])
            time.sleep(1)  # Simulate time delay like real-time camera feed
