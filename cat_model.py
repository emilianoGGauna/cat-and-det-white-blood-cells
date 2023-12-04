import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader, Subset, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, auc
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# effdet and efficientnet_pytorch are from third-party libraries for efficientdet and efficientnet models respectively
from efficientnet_pytorch import EfficientNet

# albumentations for augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Functional transforms from torchvision
from torchvision.transforms import functional as F

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_transforms = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
])

# Define constants
config = {
    'img_width': 180,
    'img_height': 180,
    'batch_size': 32,
    'data_dir': r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\CSV Y IMAG\CROPPED",
    'max_images_per_class': 500,
    'epochs': 12,
}

# Set the names of the folders in 'cropped_images' as the category labels
config['labels'] = [folder.name for folder in Path(config['data_dir']).iterdir() if folder.is_dir()]

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

def load_data():
    global train_transforms, test_transforms

    # Get the train and test transforms
    train_transforms, test_transforms = get_transforms()

    full_dataset = datasets.ImageFolder(config['data_dir'], loader=pil_loader)
    full_dataset_classes = full_dataset.classes

    # Create labels array for stratified sampling
    labels = np.array([full_dataset.targets[i] for i in range(len(full_dataset))])

    # Stratified Shuffle Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, val_index in sss.split(np.zeros(len(labels)), labels):
        train_dataset = Subset(full_dataset, train_index)
        val_dataset = Subset(full_dataset, val_index)

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    return train_dataset, val_dataset, full_dataset, full_dataset_classes

def get_dataloaders(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def build_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model = model.to(DEVICE)
    return model

def train_and_validate(model, train_loader, val_loader, full_dataset_classes):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    print(full_dataset_classes)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        # Training Phase 
        model.train()
        running_loss, running_corrects = 0.0, 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f"Train Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Validation Phase 
        model.eval()
        running_loss, running_corrects = 0.0, 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        
        print(f"Validation Epoch: {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # After training for all epochs, compute the confusion matrix and ROC curve
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print('Confusion Matrix:')
    print(cm)

    # Compute precision, recall, F1-score, and support
    classification_report_str = classification_report(all_labels, all_predictions, target_names=full_dataset_classes)
    print('Classification Report:')
    print(classification_report_str)

    # Optional: Compute ROC-AUC for binary classification
    # Note: This is applicable if your problem is a binary classification task
    # For multi-class, you would need a different approach
    if len(full_dataset_classes) == 2:
        # Assuming the model outputs probabilities for each class
        # You might need to adjust this part depending on your model's output
        probs = torch.nn.functional.softmax(outputs, dim=1)
        roc_auc = roc_auc_score(all_labels, probs[:, 1])  # Assuming index 1 is the positive class
        print(f'ROC-AUC Score: {roc_auc:.4f}')

    return history

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img_t = test_transform(img)
    img_t = img_t.unsqueeze(0)
    return img_t

def predict_image(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def save_model(model, filepath=r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS"):
    """
    Save the model to the specified filepath.
    
    Parameters:
    model (torch.nn.Module): The trained model
    filepath (str): The filepath where the model should be saved
    
    Returns:
    None
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    
def run_single_fold(train_dataset, val_dataset, full_dataset_classes):
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)
    
    model = build_model(num_classes=len(full_dataset_classes))
    history = train_and_validate(model, train_loader, val_loader, full_dataset_classes)
    
    return history, model


def main():
    # Load the full dataset
    full_dataset = datasets.ImageFolder(config['data_dir'], transform=train_transform, loader=pil_loader)
    full_dataset_classes = full_dataset.classes

    # Split the dataset into training and validation sets
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply test transformations for the validation dataset
    val_dataset.dataset.transform = test_transform
  
    # Get dataloaders for the training and validation datasets
    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset)

    # Build and train the model
    model = build_model(num_classes=len(full_dataset_classes))
    history = train_and_validate(model, train_loader, val_loader, full_dataset_classes)

    # Save the trained model
    save_path = r"C:\Users\emili\OneDrive\Escritorio\BLOODSMEARS\cat_model_full_data.pth"  # Updated to include file extension
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    save_model(model, filepath=save_path)

    # Optionally, you can plot the training history
    plot_history(history)

if __name__ == "__main__":
    main()
