import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import load_dysgraphia_dataset
from torchvision import models
from model import CNNRNNModel
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load the dataset
root_dir = 'D:\Research projects\Final year project\Dysgraphia\DATASET DYSGRAPHIA HANDWRITING\DATASET DYSGRAPHIA HANDWRITING'  # Replace with your actual dataset path
train_loader, val_loader, test_loader, label_encoder = load_dysgraphia_dataset(
    root_dir, 
    batch_size=32, 
    train_split=0.7, 
    val_split=0.15, 
    test_split=0.15, 
    load_to_memory=True
)

# Print dataset information
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"Number of classes: {len(label_encoder.classes_)}")

'''

# Define the model (example using a pre-trained ResNet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
model = model.to(device)
'''
print("Resnet18 + RNN hidden size 64")
input_shape = (224, 224, 3)  # Image size (height, width, channels)
num_classes = 2  # Binary classification
cnn_out_features = 512  # Output features of the CNN (ResNet18)
rnn_hidden_size = 64 # Hidden size for LSTM
num_rnn_layers = 2  # Number of LSTM layers

# Instantiate the model
model = CNNRNNModel(num_classes=num_classes, cnn_out_features=cnn_out_features, rnn_hidden_size=rnn_hidden_size)
model = model.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # Convert labels to LongTensor
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(train_loader), 100. * correct / total

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)  # Convert labels to LongTensor
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(val_loader), 100. * correct / total

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print()

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100. * correct / total:.2f}%")