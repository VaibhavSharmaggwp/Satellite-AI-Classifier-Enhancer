import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use standard ResNet18 (this matches what you trained in train.py)
    model = models.resnet18(weights=None) # We don't need ImageNet weights because we're loading YOURS
    
    # Adjust the final layer to match our 10 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    return model, device

if __name__ == "__main__":
    model, device = get_model()
    print(f"✅ Model architecture matches training. Device: {device}")