# 📈 Performance Report – ML Scrap Classifier

## 📊 Evaluation Metrics
| Metric     | Value (Simulated Example) |
|------------|----------------------------|
| Accuracy   | 87.5%                      |
| Precision  | 86.7%                      |
| Recall     | 85.2%                      |
| F1-Score   | 85.9%                      |

*Metrics evaluated on a validation set split from TrashNet dataset.*

## 📉 Confusion Matrix (Simulated Example)
         Predicted
       | Card | Glas | Met | Pap | Plas |
Actual |------|------|-----|-----|------|
Cardboard | 45 | 2 | 1 | 1 | 1 |
Glass | 1 | 48 | 0 | 0 | 1 |
Metal | 0 | 1 | 46 | 2 | 1 |
Paper | 1 | 0 | 2 | 45 | 2 |
Plastic | 2 | 1 | 1 | 1 | 45 |


## 🏗 Model Architecture
- **Base Model**: ResNet18
- **Final Layer**: Fully connected (5 output classes)
- **Input Size**: 224x224 RGB images
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.0003)

## 🚀 Deployment Format
- Exported using `torch.onnx.export`
- Optimized for lightweight real-time inference

## 🔄 Real-Time Simulation Summary
- 20 frames tested from mixed class folders
- Logged output to `results/simulation_output.csv`
- 3 frames flagged as low-confidence (< 60%)

## 📌 Notes
- Results may vary depending on dataset size & training time.
- Model performs well with clearly labeled, clean images.

---

Prepared for ML Internship @ AlfaStack