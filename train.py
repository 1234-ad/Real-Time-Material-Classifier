import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import nn, optim
import os

# Paths
data_dir = '../data'
model_out = '../models/scrap_classifier.pth'
onxx_out = '../models/scrap_classifier.onnx'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset & Dataloader
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model Setup
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training Loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Save Model
os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), model_out)
print("Model saved to:", model_out)

# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    onxx_out,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("ONNX model exported to:", onxx_out)
