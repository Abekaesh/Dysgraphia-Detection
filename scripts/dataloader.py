import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DysgraphiaDataset(Dataset):
    def __init__(self, root_dir, transform=None, load_to_memory=False):
        self.root_dir = root_dir
        self.transform = transform
        self.load_to_memory = load_to_memory
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        self.file_list = []
        self.labels = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.file_list.append(os.path.join(class_path, img_name))
                    self.labels.append(self.label_encoder.transform([class_name])[0])
        
        self.labels = np.array(self.labels)
        self.images = None
        if self.load_to_memory:
            self._load_images_to_memory()

    def _load_images_to_memory(self):
        print("Loading images to memory...")
        self.images = []
        for img_path in self.file_list:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            self.images.append(image.numpy())
        self.images = np.array(self.images)
        print("Images loaded to memory.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_to_memory:
            image = torch.from_numpy(self.images[idx])
        else:
            image = Image.open(self.file_list[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
        label = self.labels[idx]
        return image, label

def load_dysgraphia_dataset(root_dir, batch_size=32, train_split=0.7, val_split=0.15, test_split=0.15, load_to_memory=False):
    # Ensure splits sum to 1
    assert abs(train_split + val_split + test_split - 1.0) < 1e-5, "Splits must sum to 1"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = DysgraphiaDataset(root_dir=root_dir, transform=transform, load_to_memory=load_to_memory)

    # Print class information
    print("Classes:", full_dataset.classes)
    print("Class encodings:", full_dataset.label_encoder.transform(full_dataset.classes))

    # Perform stratified split
    indices = np.arange(len(full_dataset))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_split, 
        stratify=full_dataset.labels, 
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=val_split/(train_split+val_split), 
        stratify=full_dataset.labels[train_idx], 
        random_state=42
    )

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset.label_encoder

"""
# Example usage
root_dir = 'D:\Research projects\Final year project\Dysgraphia\DATASET DYSGRAPHIA HANDWRITING\DATASET DYSGRAPHIA HANDWRITING'  # Replace with your actual dataset path
train_loader, val_loader, test_loader, label_encoder = load_dysgraphia_dataset(root_dir, load_to_memory=True)

# Print sizes of each split
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Function to count labels in a dataset
def count_labels(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(label_encoder.inverse_transform(unique), counts))

# Print class distribution in each split
print("Train set class distribution:", count_labels(train_loader.dataset))
print("Validation set class distribution:", count_labels(val_loader.dataset))
print("Test set class distribution:", count_labels(test_loader.dataset))

# Iterate through the train loader
for images, labels in train_loader:
    print("Batch shape:", images.shape)
    print("Labels:", labels)
    print("Decoded labels:", label_encoder.inverse_transform(labels))
    break  # Just print the first batch and break

"""