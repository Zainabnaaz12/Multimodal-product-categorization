import os
import joblib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- 1. CONFIGURATION ---
app = Flask(__name__)
# Configure a secret key for session management (e.g., flash messages)
app.secret_key = 'your_super_secret_key' 
# Set the folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    """Check if a filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 2. LOAD MODELS & PYTORCH PIPELINE ---
# Use joblib to load the pre-trained models
try:
    multimodal_model = joblib.load('multimodal_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error: Model file not found. Please ensure all .pkl files are in the directory with app.py. Missing: {e}")
    # This will cause the app to fail gracefully on startup
    exit()

# Load pre-trained PyTorch ResNet50 for image feature extraction
# Check for CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a ResNet50 model without the final classification layer
base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
image_model = nn.Sequential(*list(base_model.children())[:-1])
image_model.to(device)
image_model.eval()  # Set the model to evaluation mode

# Define the image transformations required by the PyTorch model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_image_features(image):
    """
    Takes a PIL Image, preprocesses it, and extracts features using PyTorch ResNet50.
    """
    if image is None:
        return None
    
    # Preprocess the image and add a batch dimension
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Move the input to the appropriate device
    input_batch = input_batch.to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        features = image_model(input_batch)

    # The output is a tensor of shape [1, 2048, 1, 1]. Flatten it and convert to numpy array.
    features = features.cpu().numpy().flatten().reshape(1, -1)
    
    return features

# --- 3. FLASK ROUTES ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload form page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, runs the prediction, and redirects to the results page."""
    
    # Check if a file was uploaded
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file part')
        return redirect(url_for('index'))
    
    # Get the uploaded file and product title
    file = request.files['file']
    product_title = request.form['product_title']
    
    # Check if the file is valid and the product title is not empty
    if file and allowed_file(file.filename) and product_title:
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Open the image file
            pil_image = Image.open(filepath).convert('RGB')
            
            # Extract image features using the PyTorch function
            image_features = extract_image_features(pil_image)
            
            # Transform text features using TF-IDF vectorizer
            text_features = tfidf_vectorizer.transform([product_title]).toarray()
            
            # Combine the features
            combined_features = np.concatenate([image_features, text_features], axis=1)
            
            # Predict the category using the multimodal model
            prediction = multimodal_model.predict(combined_features)
            predicted_category = label_encoder.inverse_transform(prediction)[0]

            # Redirect to the results page with the predicted category and image filename
            return redirect(url_for('result', category=predicted_category, image_path=filename))
        
        except Exception as e:
            flash(f'An error occurred during prediction: {e}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type or missing product title.')
        return redirect(url_for('index'))

@app.route('/result')
def result():
    """Renders the result page with the prediction and uploaded image."""
    category = request.args.get('category')
    image_path = request.args.get('image_path')
    return render_template('result.html', category=category, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
