import torchvision
import matplotlib.pyplot as plt
import numpy as np

def download_and_show_data():
    print("Step 1: Downloading EuroSAT dataset...")
    
    # Corrected the typo 'touchvision' to 'torchvision'
    dataset = torchvision.datasets.EuroSAT(
        root='./data',
        download=True,
        transform=None  # We keep it as raw images for visualization
    )

    # dataset.classes gives us the human-readable names
    class_names = dataset.classes
    print(f" Success! Found {len(dataset)} images in {len(class_names)} categories.")

    # Let's see one of each or 5 random ones
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.suptitle('Sample Satellite Images from EuroSAT', fontsize=16)

    for i in range(5):
        # Grab a random image
        idx = np.random.randint(0, len(dataset))
        img, label = dataset[idx]
        
        axes[i].imshow(img)
        axes[i].set_title(class_names[label])
        axes[i].axis('off') 

    plt.show()

if __name__ == "__main__":
    download_and_show_data()