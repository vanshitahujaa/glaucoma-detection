import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
import glob
from typing import List, Tuple

# Import directly from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model import GlaucomaModel

class SimpleDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if loading fails
            print(f"Failed to load image: {img_path}")
            return torch.zeros((3, 224, 224)), self.labels[idx]
            
        # Resize and convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        # Convert to tensor and normalize
        image = image / 255.0
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = torch.FloatTensor(image)
        
        return image, self.labels[idx]

def get_sample_dataset():
    """Get a sample dataset from the 'all' folder"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "all", "images")
    
    image_paths = []
    labels = []
    
    # Process healthy images
    healthy_images = glob.glob(os.path.join(data_dir, '*_h.jpg'))
    image_paths.extend(healthy_images[:10])  # Take only 10 samples
    labels.extend([0] * len(healthy_images[:10]))  # 0 for Normal
    
    # Process glaucoma images
    glaucoma_images = glob.glob(os.path.join(data_dir, '*_g.jpg'))
    image_paths.extend(glaucoma_images[:10])  # Take only 10 samples
    labels.extend([1] * len(glaucoma_images[:10]))  # 1 for Glaucoma
    
    # Process DR images
    dr_images = glob.glob(os.path.join(data_dir, '*_dr.JPG'))
    image_paths.extend(dr_images[:10])  # Take only 10 samples
    labels.extend([2] * len(dr_images[:10]))  # 2 for DR
    
    print(f"Total sample images: {len(image_paths)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return image_paths, labels

def train_quick_model():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get sample dataset
    image_paths, labels = get_sample_dataset()
    
    # Split into train and validation
    train_size = int(0.8 * len(image_paths))
    train_paths = image_paths[:train_size]
    train_labels = labels[:train_size]
    val_paths = image_paths[train_size:]
    val_labels = labels[train_size:]
    
    # Create datasets
    train_dataset = SimpleDataset(train_paths, train_labels)
    val_dataset = SimpleDataset(val_paths, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model
    num_classes = len(np.unique(labels))
    model = GlaucomaModel(num_classes=num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Train for a few epochs
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            targets = torch.tensor(targets).to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = torch.tensor(targets).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "quick_model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'quick_model.pth')}")
    
    return model

if __name__ == "__main__":
    train_quick_model()