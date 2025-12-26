import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_loaders(batch_size = 32):
    # Define transformations for the dataset
    # We normalize using ImageNet standards because our ResNet backbone was trained on them
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load the full dataset
    full_dataset = datasets.EuroSAT(root='./data', download=True, transform=data_transforms)
    # 3. Split: 80% Train, 10% Validation, 10% Test

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    # Create Loaders
    # num_workers=4 helps load data faster using your CPU's extra cores
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_dataset.classes

if __name__ == "__main__":
    train, val, test, classes = get_loaders()
    print(f"✅ Data partitioned! Train: {len(train.dataset)} images, Val: {len(val.dataset)}, Test: {len(test.dataset)}")
