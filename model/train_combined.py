import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import random

# Import directly
from model import GlaucomaModel

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define constants
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 4
IMAGE_SIZE = 224
OUTPUT_DIR = "model/output"
MODEL_NAME = "efficientnet_b0"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the label
            placeholder = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return placeholder, self.labels[idx]

def load_all_datasets():
    """Load and combine all available datasets"""
    print("Loading all datasets...")
    
    # Define class mapping
    class_mapping = {
        "normal": 0,
        "h": 0,      # Normal/Healthy
        "g": 1,      # Glaucoma
        "dr": 2,     # Diabetic Retinopathy
        "pathological": 3  # Other pathological conditions
    }
    
    all_image_paths = []
    all_labels = []
    
    # Load from 'all' dataset
    print("Loading 'all' dataset...")
    all_dir = os.path.join("data", "raw", "all", "images")
    if os.path.exists(all_dir):
        for img_path in glob.glob(os.path.join(all_dir, "*.jpg")) + glob.glob(os.path.join(all_dir, "*.JPG")):
            filename = os.path.basename(img_path).lower()
            for class_key in class_mapping.keys():
                if f"_{class_key}" in filename or f"_{class_key}." in filename:
                    all_image_paths.append(img_path)
                    all_labels.append(class_mapping[class_key])
                    break
    
    # Load from 'REFUGE2' dataset
    print("Loading 'REFUGE2' dataset...")
    refuge_dir = os.path.join("data", "raw", "REFUGE2")
    if os.path.exists(refuge_dir):
        # Assuming REFUGE2 has a structure with glaucoma and non-glaucoma folders
        glaucoma_dir = os.path.join(refuge_dir, "glaucoma")
        normal_dir = os.path.join(refuge_dir, "non-glaucoma")
        
        if os.path.exists(glaucoma_dir):
            for img_path in glob.glob(os.path.join(glaucoma_dir, "*.jpg")) + glob.glob(os.path.join(glaucoma_dir, "*.png")):
                all_image_paths.append(img_path)
                all_labels.append(class_mapping["g"])  # Glaucoma
                
        if os.path.exists(normal_dir):
            for img_path in glob.glob(os.path.join(normal_dir, "*.jpg")) + glob.glob(os.path.join(normal_dir, "*.png")):
                all_image_paths.append(img_path)
                all_labels.append(class_mapping["normal"])  # Normal
    
    # Load from 'PALM' dataset
    print("Loading 'PALM' dataset...")
    palm_dir = os.path.join("data", "raw", "PALM")
    if os.path.exists(palm_dir):
        # Assuming PALM has pathological images
        for img_path in glob.glob(os.path.join(palm_dir, "**", "*.jpg"), recursive=True) + \
                       glob.glob(os.path.join(palm_dir, "**", "*.png"), recursive=True):
            all_image_paths.append(img_path)
            all_labels.append(class_mapping["pathological"])  # Pathological
    
    # Load from 'archive' dataset
    print("Loading 'archive' dataset...")
    archive_dir = os.path.join("data", "raw", "archive")
    if os.path.exists(archive_dir):
        # Check for common structures in retinal datasets
        for condition in ["normal", "glaucoma", "diabetic_retinopathy", "dr"]:
            condition_dir = os.path.join(archive_dir, condition)
            if os.path.exists(condition_dir):
                for img_path in glob.glob(os.path.join(condition_dir, "*.jpg")) + \
                               glob.glob(os.path.join(condition_dir, "*.png")):
                    all_image_paths.append(img_path)
                    
                    if condition == "normal":
                        all_labels.append(class_mapping["normal"])
                    elif condition == "glaucoma":
                        all_labels.append(class_mapping["g"])
                    elif condition in ["diabetic_retinopathy", "dr"]:
                        all_labels.append(class_mapping["dr"])
    
    # Print dataset statistics
    print(f"Total images loaded: {len(all_image_paths)}")
    class_counts = {}
    for label in all_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    class_names = ["Normal", "Glaucoma", "DR", "Pathological"]
    print("Class distribution:")
    for class_idx, count in class_counts.items():
        print(f"  {class_names[class_idx]}: {count} images")
    
    return all_image_paths, all_labels

def train_model():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load all datasets
    image_paths, labels = load_all_datasets()
    
    # Create dataset splits
    dataset_size = len(image_paths)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Create indices for the splits
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create datasets
    train_dataset = RetinalDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=train_transform
    )
    
    val_dataset = RetinalDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=val_transform
    )
    
    test_dataset = RetinalDataset(
        [image_paths[i] for i in test_indices],
        [labels[i] for i in test_indices],
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize model
    model = GlaucomaModel(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        train_loss = train_loss / len(train_dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc="Validation")
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        val_loss = val_loss / len(val_dataset)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "combined_model.pth"))
            print(f"Model saved to {os.path.join(OUTPUT_DIR, 'combined_model.pth')}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_combined_model.pth"))
    
    # Save model in safetensors format for compatibility
    try:
        import safetensors.torch
        safetensors.torch.save_file(model.state_dict(), os.path.join(OUTPUT_DIR, "combined_model.safetensors"))
        print(f"Model saved in safetensors format to {os.path.join(OUTPUT_DIR, 'combined_model.safetensors')}")
    except ImportError:
        print("safetensors not available, skipping safetensors export")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_training_history.png"))
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    test_loss = test_loss / len(test_dataset)
    test_acc = 100. * correct / total
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    class_names = ["Normal", "Glaucoma", "DR", "Pathological"]
    print("Per-class accuracy:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            print(f"  {class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"  {class_names[i]}: N/A (no test samples)")

if __name__ == "__main__":
    train_model()