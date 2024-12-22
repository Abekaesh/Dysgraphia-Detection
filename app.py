from flask import Flask, render_template, request, jsonify
import torch
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from model import CNNLSTMModel
import torchvision.transforms as transforms

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_shape = (224, 224, 3)  # Image size (height, width, channels)
num_classes = 2  # Binary classification
cnn_out_features = 512  # Output features of the CNN (ResNet18)
rnn_hidden_size = 128 # Hidden size for LSTM
num_rnn_layers = 2  # Number of LSTM layers

# Instantiate the model
model = CNNLSTMModel(num_classes=num_classes, rnn_hidden_size=rnn_hidden_size)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'final_model_mobilenet_rnn_128_0.0001.pth')
model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Placeholder function for preprocessing
def preprocess_image(image_path):
    # Replace this with your actual preprocessing logic
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)

# Placeholder function for prediction
def predict_class(preprocessed_image):
    # Replace this with your actual prediction logic
    # This is just a dummy prediction
    with torch.no_grad():
        outputs = model(preprocessed_image)
        proba = torch.nn.functional.sigmoid(outputs)
        print(proba)
        confidence, predicted = proba.max(1)
        classes = ["Low Potential Dysgraphia", "Potential Dysgraphia"]
    return classes[predicted], confidence

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(file_path)
            
            # Make prediction
            predicted_class, confidence = predict_class(preprocessed_image)
            
            # Prepare image for display
            with open(file_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': confidence.item(),
                'image_data': encoded_string
            })
    return render_template('upload (1).html')

if __name__ == '__main__':
    app.run(debug=True)
