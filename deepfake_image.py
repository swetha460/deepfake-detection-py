import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

# -------------------------------
# Load model
# -------------------------------
def load_model(model_path):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Single output neuron
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# -------------------------------
# Predict image with debug
# -------------------------------
def predict_image(model, image_path, threshold=0.5):
    img = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img)
        raw_output = outputs.item()
        predicted = (torch.sigmoid(outputs) > threshold).int()
    print(f"Image: {image_path}")
    print(f"Raw output (logit): {raw_output}")
    print(f"Sigmoid(output): {torch.sigmoid(outputs).item()}")
    return "Real" if predicted.item() == 1 else "Fake"

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection for Image (Debug)")
    parser.add_argument("--model", type=str, default="deepfake_detector.pth", help="Path to trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    model = load_model(args.model)
    result = predict_image(model, args.image)
    print(f"Predicted: {result}")
