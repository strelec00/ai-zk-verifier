import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.models import ResNet18_Weights
import sys

# Postavke uređaja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Učitavanje treniranog modela
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cat_dog_classifier.pth"))
model = model.to(device)
model.eval()

# Transformacije za test sliku
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcija za predviđanje slike
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Dodaj batch dimenziju

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        confidence = probabilities[0][predicted].item()  # Povjerenje za predviđenu klasu

    class_names = ["Cat", "Dog"]
    print(f"Predicted: {class_names[predicted.item()]}, Confidence: {confidence:.4f}")

    # Provjera povjerenja (ako povjerenje nije dovoljno visoko, prijavi grešku)
    if confidence < 0.999:  # Prag povjerenja 70%
        raise ValueError("Slika nije prepoznata kao pas ili mačka (nisko povjerenje).")

if __name__ == "__main__":
    image_path = sys.argv[1]  # Putanja do slike iz komandne linije
    try:
        predict(image_path)
    except ValueError as e:
        print(f"Greška: {str(e)}")
