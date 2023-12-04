import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold

# effdet and efficientnet_pytorch are from third-party libraries for efficientdet and efficientnet models respectively
from efficientnet_pytorch import EfficientNet

# albumentations for augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Load the dataframe
df = pd.read_csv(r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\cleaned_bloodsmear_df.csv")

# Drop NA values
df.dropna(inplace=True)

# Remove rows where region_shape_attributes is {} or values are 
df = df[~(df['region_shape_attributes'].str.contains("0"))]

# Remove rows where region_attributes is 'RBC'
df = df[df['region_attributes'] != 'RBC']

# Path to the directory containing all images
image_directory = "C:\\Users\\emili\\OneDrive\\Escritorio\\BLOODSMEARS\\CSV Y IMAG\\ALL IMAGES"

# List of filenames to be removed from the dataframe
files_to_remove = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Construct the full path of the image
    img_path = os.path.join(image_directory, row['filename'])  # assuming 'filename' is the column name containing the image filenames
    
    # Check if the image exists in the specified path
    if not os.path.exists(img_path):
        # If it doesn't exist, append to files_to_remove list and print it
        files_to_remove.append(row['filename'])

# Remove rows with filenames that don't exist in the specified path
df = df[~df['filename'].isin(files_to_remove)]

print(df.info)
def has_x_key(row):
    try:
        # Assuming 'region_attributes_shape' is a stringified JSON
        region_attributes = json.loads(row['region_shape_attributes'])
        return 'x' in region_attributes
    except json.JSONDecodeError:
        # Handle cases where JSON decoding fails
        return False

# Apply the function to filter the DataFrame
filtered_df = df[df.apply(has_x_key, axis=1)]

df= filtered_df

print(df.info())

df.to_csv('df.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class BloodCellDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        bbox_str = self.dataframe.iloc[idx, 2]
        if isinstance(bbox_str, str):
            bbox = json.loads(bbox_str)
            boxes = [bbox['x'], bbox['y'], bbox['x'] + bbox['width'], bbox['y'] + bbox['height']]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 'region_shape_attributes' to be a string, but got {type(bbox_str)} at index {idx}")

        labels = torch.ones((1,), dtype=torch.int64)
        target = {"boxes": boxes.unsqueeze(0), "labels": labels}
        if self.transform:
            image = self.transform(image)
        return image, target


def get_datasets(df, image_dir, transform, train_frac=0.8):
    df_sample = df.sample(1980, replace=True)
    train_df = df_sample.sample(frac=train_frac)
    val_df = df_sample.drop(train_df.index)
    train_dataset = BloodCellDataset(train_df, image_dir, transform)
    val_dataset = BloodCellDataset(val_df, image_dir, transform)
    return train_dataset, val_dataset


def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(DEVICE)
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    # Add tqdm progress bar
    for images, targets in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return total_loss

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area by using the formula: union(A,B) = A + B - intersection(A,B)
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou

def calculate_metrics(detections: List[Dict], ground_truths: List[Dict], iou_threshold=0.5):
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for prediction, ground_truth in zip(detections, ground_truths):
        gt_boxes = ground_truth['boxes']
        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        
        matched_gt = []
        
        for pred_box, pred_score in zip(pred_boxes, pred_scores):
            # Assume that we start with a false positive
            false_positive = True
            
            for idx, gt_box in enumerate(gt_boxes):
                if iou(pred_box, gt_box) >= iou_threshold:
                    if idx not in matched_gt:
                        matched_gt.append(idx)
                        total_true_positives += 1
                        false_positive = False
                        break
            
            if false_positive:
                total_false_positives += 1
        
        total_false_negatives += len(gt_boxes) - len(matched_gt)

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def evaluate(model, data_loader, device):
    model.eval()
    detections = []
    ground_truths = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                detections.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })
                ground_truths.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })

    metrics = calculate_metrics(detections, ground_truths)
    return metrics




def visualize_prediction(img, target, epoch, img_idx, base_path="C:\\Users\\emili\\OneDrive\\Escritorio\\BLOODSMEARS\\CSV Y IMAG\\DET_IMAGES_IMPROVEMENT", display_labels=True, display_scores=False, color='red'):
    # Construct the path for the current epoch
    epoch_path = os.path.join(base_path, f"epoch_{epoch}")
    os.makedirs(epoch_path, exist_ok=True)
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())
    
    boxes = target['boxes'].cpu().numpy()
    labels = target.get('labels', None)
    scores = target.get('scores', None)

    for idx, box in enumerate(boxes):
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor=color,
            linewidth=2
        )
        ax.add_patch(rect)

        # If labels are provided and display_labels is True
        if labels is not None and display_labels:
            ax.text(box[0], box[1], str(labels[idx].item()), color=color, fontweight='bold')
        
        # If scores are provided and display_scores is True
        if scores is not None and display_scores:
            ax.text(box[2], box[3], f"{scores[idx].item():.2f}", color=color, fontweight='bold')
    
    # Remove axis
    ax.axis('off')

    # Save the figure
    image_path = os.path.join(epoch_path, f"image_{img_idx}.png")
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to avoid display
    
def main(df, image_dir, num_epochs=10, batch_size=10, lr=0.005, k_folds=3):
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    # Splitting the dataset into k folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kf.split(df)):
        print(f"Fold {fold + 1}")

        train_dataset, val_dataset = get_datasets(df.iloc[train_ids], image_dir, transform, 0.8)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = get_model(2)  # Assuming 1 class + background
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            total_loss = train_one_epoch(model, optimizer, train_loader, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

            # Evaluate the model after training
            metrics = evaluate(model, val_loader, device)
            print(f"Metrics after Epoch {epoch + 1}/{num_epochs}: {metrics}")

            # Visualization and other steps if necessary

        fold_results.append(metrics)  # Store the metrics for each fold

    # Calculate and print average performance across folds
    avg_performance = sum(fold_results) / k_folds
    print(f"Average performance across {k_folds} folds: {avg_performance}")

    # Save the final model (or models for each fold, if necessary)
    model_save_path = "C:\\Users\\emili\\OneDrive\\Escritorio\\BLOODSMEARS\\det_1800model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Assuming `df` and `image_dir` are defined earlier in your code
    main(df, "C:\\Users\\emili\\OneDrive\\Escritorio\\BLOODSMEARS\\CSV Y IMAG\\ALL IMAGES")