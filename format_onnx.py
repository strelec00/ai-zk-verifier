import torch
from torchvision import models
from torchvision.models import ResNet18_Weights

# Postavke uređaja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Učitavanje treniranog modela
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cat_dog_classifier.pth"))
model = model.to(device)
model.eval()

# Funkcija za eksport modela u ONNX format
def export_to_onnx(model, filename="cat_dog_classifier.onnx"):
    # Kreiraj dummy input koji odgovara obliku koji očekuje model
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # 1 slike, 3 kanala, 224x224 piksela

    # Export modela u ONNX format
    torch.onnx.export(model, dummy_input, filename, export_params=True, opset_version=12,
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model has been successfully exported to {filename}")

if __name__ == "__main__":
    export_to_onnx(model, "cat_dog_classifier.onnx")
