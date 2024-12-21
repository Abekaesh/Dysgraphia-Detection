import torch
import torch.nn as nn
from torchvision import models

# Define the combined CNN-RNN model
class CNNRNNModel(nn.Module):
    def __init__(self, num_classes=2, cnn_out_features=512, rnn_hidden_size=128, num_rnn_layers=2):
        super(CNNRNNModel, self).__init__()

        # Pre-trained CNN model for feature extraction
        #self.cnn = models.resnet18(weights='IMAGENET1K_V1')
        self.cnn = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()  # Remove the final classification layer
        self.cnn_out_features = cnn_out_features

        # LSTM to process the extracted features from the CNN
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,  # Feature size output from CNN
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer for binary classification
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # *2 for bidirectional LSTM

    def forward(self, x):
        # Step 1: Extract features using the CNN
        batch_size, channels, height, width = x.size()
        # Pass input through the CNN (ResNet18)
        cnn_features = self.cnn(x)

        # Step 2: Reshape for RNN (sequence_length=1, feature_size=cnn_out_features)
        cnn_features = cnn_features.view(batch_size, 1, self.cnn_out_features)

        # Step 3: Pass through the RNN (LSTM)
        rnn_out, _ = self.lstm(cnn_features)

        # Step 4: Take the last output from the LSTM for classification
        rnn_out = rnn_out[:, -1, :]  # Take the last time step's output

        # Step 5: Pass through fully connected layer
        out = self.fc(rnn_out)
        return out
"""
# Define model parameters
input_shape = (224, 224, 3)  # Image size (height, width, channels)
num_classes = 2  # Binary classification
cnn_out_features = 512  # Output features of the CNN (ResNet18)
rnn_hidden_size = 128  # Hidden size for LSTM
num_rnn_layers = 2  # Number of LSTM layers

# Instantiate the model
model = CNNRNNModel(num_classes=num_classes, cnn_out_features=cnn_out_features, rnn_hidden_size=rnn_hidden_size)
print(model)

# Check if CUDA is available and move the model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
"""