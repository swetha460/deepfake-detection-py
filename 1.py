import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import argparse

# -------------------------------
# Load your model (1-output)
# -------------------------------
def load_model(model_path):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 1)  # single output: probability of Real
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

# -------------------------------
# Grad-CAM++ visualization
# -------------------------------
def gradcam_pp(model, image_tensor):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.layer4[-1]]  # last conv layer
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]  # single-output model

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]

    img = image_tensor.squeeze().permute(1,2,0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    cam_image = show_cam_on_image(img, grayscale_cam)

    cv2.imshow('Grad-CAM++', cam_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------
# Main prediction function
# -------------------------------
def predict(image_path, model_path, threshold=0.5):
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)
        prob_real = torch.sigmoid(output).item()  # single-output model
        print(f"[DEBUG] Probability of Real: {prob_real:.4f}")

    # Inverted logic: high probability = Real, low probability = Fake
    if prob_real < threshold:  # < threshold = Fake
        print(f"{image_path} → FAKE, generating Grad-CAM++...")
        gradcam_pp(model, image_tensor)
    else:
        print(f"{image_path} → REAL")

# -------------------------------
# Command line interface
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for Fake probability")
    args = parser.parse_args()

    predict(args.image, args.model, threshold=args.threshold)
