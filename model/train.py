import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import GlaucomaModel
from utils.preprocessing import preprocess_image

class GlaucomaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preprocess=True):
        """
        Dataset for Glaucoma classification
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (0: Normal, 1: Suspicious, 2: Early, 3: Advanced)
            transform: Albumentations transformations
            preprocess: Whether to apply preprocessing pipeline
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Apply preprocessing if required
        if self.preprocess:
            # Apply preprocessing pipeline without normalization
            # Extract ROI, apply CLAHE, denoise
            from utils.preprocessing import extract_roi, apply_clahe, denoise_image
            image = extract_roi(image)
            image = apply_clahe(image)
            image = denoise_image(image)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def get_transforms(mode: str) -> A.Compose:
    """
    Get image transformations for training or validation
    
    Args:
        mode: 'train' or 'val'
        
    Returns:
        Albumentations transformations
    """
    if mode == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    save_dir: str
) -> Dict:
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_dir: Directory to save model checkpoints
        
    Returns:
        Dictionary containing training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize variables
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"Saved best model with F1 score: {val_f1:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_f1': val_f1,
            'history': history
        }, os.path.join(save_dir, 'checkpoint.pt'))
    
    return history

def plot_training_history(history: Dict, save_path: str) -> None:
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.plot(history['val_f1'], label='Validation F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Training and Validation Metrics')
    ax2.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define paths
    data_dir = "../data/processed"
    save_dir = "../model/weights"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    # This is a placeholder - you'll need to adapt this to your actual data structure
    # For example, you might have a CSV file with image paths and labels
    # Or you might need to scan directories to find images
    
    # Example: Loading from a CSV file
    # df = pd.read_csv("../data/processed/dataset.csv")
    # image_paths = df['image_path'].values
    # labels = df['label'].values
    
    # For demonstration, let's assume we have lists of image paths and labels
    # You'll need to replace this with your actual data loading code
    image_paths = []  # List of image file paths
    labels = []       # List of corresponding labels
    
    # Split data into train, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Create datasets
    train_transforms = get_transforms('train')
    val_transforms = get_transforms('val')
    
    train_dataset = GlaucomaDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = GlaucomaDataset(val_paths, val_labels, transform=val_transforms)
    test_dataset = GlaucomaDataset(test_paths, test_labels, transform=val_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = GlaucomaModel(num_classes=4, model_name="efficientnet_b0")
    model.to(device)
    
    # Define loss function with class weights (if needed)
    # You can calculate class weights based on your dataset distribution
    # For example:
    # class_weights = torch.tensor([1.0, 2.0, 2.0, 3.0], device=device)  # Example weights
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=device,
        save_dir=save_dir
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels_list.extend(labels.numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels_list, test_preds)
    test_f1 = f1_score(test_labels_list, test_preds, average='weighted')
    conf_matrix = confusion_matrix(test_labels_list, test_preds)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Export model to ONNX format
    model.export_onnx(os.path.join(save_dir, 'model.onnx'))
    print("Model exported to ONNX format")

if __name__ == "__main__":
    main()