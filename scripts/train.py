import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataloader import load_dysgraphia_dataset
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt
import os
from model import CNNRNNModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

'''
# Define the model (ResNet18)
model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
model = model.to(device)
'''
input_shape = (224, 224, 3)  # Image size (height, width, channels)
num_classes = 2  # Binary classification
cnn_out_features = 512  # Output features of the CNN (ResNet18)
rnn_hidden_size = 128 # Hidden size for LSTM
num_rnn_layers = 2  # Number of LSTM layers

# Instantiate the model
model = CNNRNNModel(num_classes=num_classes, cnn_out_features=cnn_out_features, rnn_hidden_size=rnn_hidden_size)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
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
            inputs, labels = inputs.to(device), labels.to(device,dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(val_loader), 100. * correct / total

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop
num_epochs = 100  # Set a high number, early stopping will prevent overfitting
early_stopping = EarlyStopping(patience=10)
train_losses, train_accs, val_losses, val_accs = [], [], [], []
best_val_loss = float('inf')

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    epoch_time = time.time() - epoch_start_time
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Epoch Time: {epoch_time:.2f} seconds")
    print()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/resnet18_rnn_128.pth')
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

# Plot and save accuracy and loss graphs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('graphs/training_graphs_resnet18_rnn128.eps', format='eps')
plt.savefig('graphs/training_graphs_resnet18_rnn128.png')
plt.close()

# Load the best model for testing
model.load_state_dict(torch.load('models/resnet18_rnn_128.pth'))

# Test the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Save the final model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}, 'models/final_model_resnet18_rnn_128.pth')

print("Training completed. Model and graphs saved.")