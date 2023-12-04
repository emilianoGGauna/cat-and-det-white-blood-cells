import collections
import json
import os
import pandas as pd
import random
import seaborn as sns
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from tqdm import tqdm
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

# effdet and efficientnet_pytorch are from third-party libraries for efficientdet and efficientnet models respectively
from efficientnet_pytorch import EfficientNet

# albumentations for augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Functional transforms from torchvision
from torchvision.transforms import functional as F

import cv2

from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename

# Assuming DEVICE is defined globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
config = {
    'img_width': 180,
    'img_height': 180,
    'data_dir': r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\CSV Y IMAG\cropped_images",
}

test_transforms = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
])

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_transforms():
    train_transform = A.Compose([
        A.RandomResizedCrop(height=config['img_height'], width=config['img_width'], p=1.0),
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ], p=1)

    test_transforms = transforms.Compose([
        transforms.Resize((config['img_width'], config['img_height'])),
        transforms.ToTensor(),
    ])

    return train_transform, test_transforms

def albumentations_transform(image, transform):
    image_np = np.array(image)
    
    # Check if the transform is from albumentations or PyTorch and apply it correctly
    if isinstance(transform, A.core.composition.Compose):
        augmented = transform(image=image_np)
        image_torch = augmented['image']
    else:  # Assuming it's a PyTorch transform
        image_torch = transform(Image.fromarray(image_np))

    return image_torch.float()

def train_transform(x):
    return albumentations_transform(x, train_transforms)

def test_transform(x):
    return albumentations_transform(x, test_transforms)

def build_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model = model.to(DEVICE)
    return model

def load_all_models(model_paths, num_classes):
    models = []
    for path in model_paths:
        model = build_model(num_classes=num_classes)
        model.load_state_dict(torch.load(path))
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode
        models.append(model)
        print(f"Model loaded from {path}")
    return models

def load_detection_model(model_path):
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_bounding_boxes(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(image_tensor)

    if "boxes" in prediction[0] and "labels" in prediction[0]:
        boxes = prediction[0]['boxes'].numpy()
        labels = prediction[0]['labels'].numpy()
        filtered_boxes = []

        for i, box in enumerate(boxes):
            # Process only if the label is 1
            if labels[i] == 1:
                keep = True
                for j, other_box in enumerate(boxes):
                    # Check only other boxes with label 1
                    if labels[j] == 1 and i != j:
                        if is_inside_and_full(box, other_box):
                            keep = False
                            break
                if keep:
                    filtered_boxes.append(box)

        return np.array(filtered_boxes)
    else:
        return []

def is_inside_and_full(inside_box, outside_box):
    """
    Check if inside_box is completely within outside_box.
    Returns True if inside_box is inside outside_box.
    """
    inside_condition = all(inside_box[i] <= outside_box[i] for i in [0, 1]) and all(inside_box[i] >= outside_box[i] for i in [2, 3])
    return inside_condition
    
def predict_category(models: List[nn.Module], image, class_labels, test_transform):
    # Apply the test transformation
    image_tensor = test_transform(image)
    image_tensor = image_tensor.to(DEVICE)

    predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor.unsqueeze(0).to(DEVICE))
            _, preds = outputs.max(1)
            predictions.append(class_labels[preds.item()])
    print(predictions)
    # Choose the most frequent prediction
    most_common = Counter(predictions).most_common(1)
    return most_common[0][0], predictions

# Assuming that the build_model function is defined as in the initial code snippet you provided

def load_classification_models(model_paths, num_classes):
    models = []
    for path in model_paths:
        model = build_model(num_classes)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    return models

def majority_vote(predictions):
    # Aggregate all predictions and choose the most common one
    counter = Counter(predictions)
    majority_label, _ = counter.most_common(1)[0]
    return majority_label

def predict_with_models(models, image, class_labels):
    # Transform and normalize the image as required by the models
    image = TF.resize(image, size=(180, 180))
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = image_tensor.to(DEVICE)

    predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor.unsqueeze(0))
            _, preds = outputs.max(1)
            predictions.append(class_labels[preds.item()])
    print(predictions)
    return majority_vote(predictions), predictions

def draw_and_crop_boxes_on_image(image_path, boxes, models, class_labels, test_transform):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = 15
    font = ImageFont.truetype("arial.ttf", font_size)
    
    for box in boxes:
        cropped_image = image.crop(box)
        category_label, pred = predict_category(models, cropped_image, class_labels, test_transform)
        
        draw.rectangle(list(box), outline="red", width=2)
        
        # Estimate text size
        estimated_text_length = len(category_label) * font_size
        text_height = font_size + 4  # Adding a small buffer
        background_rectangle = [box[0], box[1] - text_height - 10, box[0] + estimated_text_length + 10, box[1]]
        draw.rectangle(background_rectangle, fill="red")
        draw.text((box[0] + 5, box[1] - text_height - 5), category_label, fill="white", font=font)
    
    processed_image_path = os.path.join(UPLOAD_FOLDER, "processed_" + os.path.basename(image_path))
    image.save(processed_image_path)

    return image, pred


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            process_image(filepath)  # Define this function
            return render_template('result.html', filename='processed_'+filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_image(image_path):
    # Load models and transforms (if not already loaded)
    classification_model_paths = [
        r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\cat_model_full_data.pth"
    ]# Your model paths ]
    num_classes = 6
    class_labels = ['BAS', 'BNE', 'EO', 'ERB', 'LY', 'MO']

    # Assuming models are loaded outside this function or check if they need to be loaded
    global classification_models
    global detection_model
    if 'classification_models' not in globals():
        classification_models = load_all_models(classification_model_paths, num_classes)
    if 'detection_model' not in globals():
        detection_model_path = r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\det_20model.pth"
        detection_model = load_detection_model(detection_model_path)

    # Load test transforms
    _, test_transforms = get_transforms()
    prediction =[]
    # Perform detection and classification
    boxes = detect_bounding_boxes(detection_model, image_path)
    if len(boxes) > 0:
        processed_image, pred = draw_and_crop_boxes_on_image(image_path, boxes, classification_models, class_labels, test_transforms)

    else:
        print(f"No boxes detected in image: {image_path}")
        processed_image = Image.open(image_path)

    # Save the processed image back to the uploads folder
    processed_image_path = os.path.join(UPLOAD_FOLDER, "processed_" + os.path.basename(image_path))
    processed_image.save(processed_image_path)  # Save the Image object

    return processed_image_path, pred

if __name__ == '__main__':
    app.run(debug=True)