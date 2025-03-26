import onnxruntime as ort
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

# Postavke uređaja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Učitavanje ONNX modela
onnx_model_path = "cat_dog_classifier.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Transformacije za test sliku
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcija za predviđanje slike koristeći ONNX
def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Dodaj batch dimenziju
    image = image.numpy()  # Pretvori u numpy array

    # Priprema za ONNX model
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Pokretanje inferencije
    result = ort_session.run([output_name], {input_name: image})

    # Softmax za vjerojatnost
    probabilities = torch.nn.functional.softmax(torch.tensor(result[0]), dim=1)
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
