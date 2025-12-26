import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_loaders
from model_setup import get_model

def evaluate():
    # 1 Step 
    _,_, test_loader, classes = get_loaders(batch_size=16)
    model, device = get_model(num_classes=len(classes))

    #2 Load the trained weights "Brain" of the model
    model.load_state_dict(torch.load("land_use_model.pth"))
    model.eval()

    # 3. Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # 4. Predict
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 5. Visualize
    images = images.cpu().numpy()
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        # Un-normalize image for viewing) 
        img = np.transpose(images[i], (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)


        color = 'green' if predicted[i] == labels[i] else 'red'
        plt.imshow(img)
        plt.title(f"Actual: {classes[labels[i]]}\nPred: {classes[predicted[i]]}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()        
