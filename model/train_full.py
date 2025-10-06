import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import glob
import random
import time
import sys

# Import model
from model import GlaucomaModel

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_dataset():
    """Load and prepare the complete dataset"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    
    # Collect all images
    image_paths = []
    labels = []
    
    # Class mapping
    class_map = {
        'h': 0,  # Normal
        'g': 1,  # Glaucoma
        'dr': 2, # DR
        'p': 3   # Pathological (if exists)
    }
    
    # Process 'all' dataset
    all_dir = os.path.join(data_dir, 'all', 'images')
    if os.path.exists(all_dir):
        for img_path in glob.glob(os.path.join(all_dir, '*.jpg')) + glob.glob(os.path.join(all_dir, '*.JPG')):
            filename = os.path.basename(img_path).lower()
            for class_code, class_idx in class_map.items():
                if f'_{class_code}.' in filename:
                    image_paths.append(img_path)
                    labels.append(class_idx)
                    break
    
    # Process REFUGE2 dataset
    refuge_dir = os.path.join(data_dir, 'REFUGE2')
    if os.path.exists(refuge_dir):
        # Normal images
        normal_dir = os.path.join(refuge_dir, 'Training400', 'Non-Glaucoma')
        if os.path.exists(normal_dir):
            for img_path in glob.glob(os.path.join(normal_dir, '*.jpg')):
                image_paths.append(img_path)
                labels.append(0)  # Normal
        
        # Glaucoma images
        glaucoma_dir = os.path.join(refuge_dir, 'Training400', 'Glaucoma')
        if os.path.exists(glaucoma_dir):
            for img_path in glob.glob(os.path.join(glaucoma_dir, '*.jpg')):
                image_paths.append(img_path)
                labels.append(1)  # Glaucoma
    
    # Process PALM dataset (if it contains relevant classes)
    palm_dir = os.path.join(data_dir, 'PALM')
    if os.path.exists(palm_dir):
        # Add PALM dataset processing if needed
        pass
    
    # Split into train, validation, test
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Print dataset statistics
    print(f"Total images: {len(image_paths)}")
    print(f"Training: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Count class distribution
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Class distribution:")
    for class_idx, count in class_counts.items():
        class_name = ["Normal", "Glaucoma", "DR", "Pathological"][class_idx]
        print(f"  {class_name}: {count} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='model/output/full_model.pth'):
    """Train the model"""
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in train_bar:
            inputs = inputs.to(device)
            targets = torch.tensor(targets).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = torch.tensor(targets).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds, 
                                        target_names=["Normal", "Glaucoma", "DR", "Pathological"])
    
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return accuracy, conf_matrix, class_report

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='model/output/training_history.png'):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs('model/output', exist_ok=True)
    
    # Load dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_dataset()
    
    # Create datasets
    train_dataset = RetinalDataset(train_paths, train_labels, transform=get_transforms('train'))
    val_dataset = RetinalDataset(val_paths, val_labels, transform=get_transforms('val'))
    test_dataset = RetinalDataset(test_paths, test_labels, transform=get_transforms('val'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    model = GlaucomaModel(num_classes=4).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=10, save_path='model/output/full_model.pth'
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('model/output/full_model.pth'))
    
    # Evaluate model
    print("Evaluating model on test set...")
    accuracy, conf_matrix, class_report = evaluate_model(model, test_loader)
    
    # Save evaluation results
    with open('model/output/evaluation_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()