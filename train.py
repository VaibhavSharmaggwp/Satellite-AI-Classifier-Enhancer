import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_loaders
from model_setup import get_model


import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use a standard ResNet18 for Classification
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Change the last layer to match our 10 EuroSAT classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model.to(device), device

def main():
    # 1. Setup Data and Model
    train_loader, val_loader, _, classes = get_loaders(batch_size=32)
    model, device = get_model(num_classes=len(classes))
    
    # 2. Define "Teacher" (Loss) and "Student" (Optimizer)
    # CrossEntropyLoss is best for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Setup AMP for your RTX 3060 (Saves VRAM and reduces heat)
    scaler = torch.amp.GradScaler('cuda')

    print("🚀 Starting Training on RTX 3060...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass (Learning)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5 | Average Loss: {running_loss/len(train_loader):.4f}")
    
    # Save the learned weights
    torch.save(model.state_dict(), "land_use_model.pth")
    print("✅ Training complete! Model saved.")

if __name__ == "__main__":
    main()