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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import GlaucomaModel
from utils.dataset_processor import combine_datasets, split_dataset

class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Dataset for Retinal Disease classification
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (0: Normal, 1: Glaucoma, 2: DR, 3: Pathological)
            transform: Albumentations transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
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
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    num_epochs: int = 10,
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    save_path: str = "model_weights.pth"
) -> Dict:
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on
        save_path: Path to save the best model weights
        
    Returns:
        Dictionary containing training history
    """
    # Initialize variables
    best_val_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    # Move model to device
    model = model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc="Training")
        
        for inputs, targets in train_pbar:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val F1: {val_f1:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        print("-" * 50)
    
    return history

def plot_training_history(history: Dict, save_path: str = "training_history.png"):
    """
    Plot training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot(history['train_loss'], label='Train')
    axs[0, 0].plot(history['val_loss'], label='Validation')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot accuracy
    axs[0, 1].plot(history['train_acc'], label='Train')
    axs[0, 1].plot(history['val_acc'], label='Validation')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    
    # Plot F1 score
    axs[1, 0].plot(history['train_f1'], label='Train')
    axs[1, 0].plot(history['val_f1'], label='Validation')
    axs[1, 0].set_title('F1 Score')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].legend()
    
    # Save plot
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
    
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process datasets
    print("Loading and processing datasets...")
    image_paths, labels = combine_datasets()
    print(f"Total images: {len(image_paths)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths, labels
    )
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")
    
    # Create datasets
    train_dataset = RetinalDataset(
        train_paths, train_labels, transform=get_transforms('train')
    )
    val_dataset = RetinalDataset(
        val_paths, val_labels, transform=get_transforms('val')
    )
    test_dataset = RetinalDataset(
        test_paths, test_labels, transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )
    
    # Create model
    num_classes = len(np.unique(labels))
    model = GlaucomaModel(num_classes=num_classes, model_name="efficientnet_b0")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        device=device,
        save_path=os.path.join(output_dir, "best_model.pth")
    )
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Calculate test metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    conf_matrix = confusion_matrix(test_targets, test_preds)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save test metrics
    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
    
    # Save model in ONNX format for deployment
    model.export_onnx(os.path.join(output_dir, "model.onnx"))
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()