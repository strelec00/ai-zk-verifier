import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights


# Postavke
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# ✅ Transformacije za slike (normalizacija i augmentacija)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Učitavanje podataka (pretpostavljamo da su slike u 'data/train' i 'data/test')
train_dataset = ImageFolder(root='data/train', transform=transform)
test_dataset = ImageFolder(root='data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Korištenje pretreniranog ResNet-18 modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Zamjena posljednjeg sloja za binarnu klasifikaciju
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Definicija loss funkcije i optimizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Treniranje modela
def train():
    model.train()
    # Unutar funkcije train
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # Spremanje modela
    torch.save(model.state_dict(), "cat_dog_classifier.pth")
    print("Model saved!")

    # Export to ONNX format
    model.eval()  # Set model to evaluation mode for exporting
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Dummy input with the same shape as input images
    onnx_output_path = "cat_dog_classifier.onnx"
    
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_output_path, input_names=['input'], output_names=['output'], opset_version=12)
    print(f"Model exported to {onnx_output_path}")

if __name__ == "__main__":
    train()
