import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import glob
from sklearn.model_selection import train_test_split

def process_all_dataset(data_dir: str = '../data/raw/all') -> Tuple[List[str], List[int]]:
    """
    Process the 'all' dataset which contains images with suffixes:
    - _h.jpg: Healthy/Normal
    - _g.jpg: Glaucoma
    - _dr.JPG: Diabetic Retinopathy
    
    Args:
        data_dir: Directory containing the 'all' dataset
        
    Returns:
        Tuple containing:
            - List of image paths
            - List of labels (0: Normal, 1: Glaucoma, 2: DR)
    """
    image_paths = []
    labels = []
    
    # Get all images from the images directory
    images_dir = os.path.join(data_dir, 'images')
    
    # Process healthy images
    healthy_images = glob.glob(os.path.join(images_dir, '*_h.jpg'))
    image_paths.extend(healthy_images)
    labels.extend([0] * len(healthy_images))  # 0 for Normal
    
    # Process glaucoma images
    glaucoma_images = glob.glob(os.path.join(images_dir, '*_g.jpg'))
    image_paths.extend(glaucoma_images)
    labels.extend([1] * len(glaucoma_images))  # 1 for Glaucoma
    
    # Process DR images
    dr_images = glob.glob(os.path.join(images_dir, '*_dr.JPG'))
    image_paths.extend(dr_images)
    labels.extend([2] * len(dr_images))  # 2 for DR
    
    return image_paths, labels

def process_refuge_dataset(data_dir: str = '../data/raw/REFUGE2') -> Tuple[List[str], List[int]]:
    """
    Process the REFUGE2 dataset
    
    Args:
        data_dir: Directory containing the REFUGE2 dataset
        
    Returns:
        Tuple containing:
            - List of image paths
            - List of labels (0: Normal, 1: Glaucoma)
    """
    image_paths = []
    labels = []
    
    # Process training data
    train_dir = os.path.join(data_dir, 'Train')
    
    # REFUGE1-train
    refuge1_train_dir = os.path.join(train_dir, 'REFUGE1-train')
    if os.path.exists(refuge1_train_dir):
        # Read the labels from the CSV file if available
        csv_path = os.path.join(refuge1_train_dir, 'glaucoma_labels.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(refuge1_train_dir, row['filename'])
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(int(row['label']))
        else:
            # If no CSV, assume directory structure with 'glaucoma' and 'non-glaucoma' subdirectories
            glaucoma_dir = os.path.join(refuge1_train_dir, 'glaucoma')
            non_glaucoma_dir = os.path.join(refuge1_train_dir, 'non-glaucoma')
            
            if os.path.exists(glaucoma_dir):
                glaucoma_images = glob.glob(os.path.join(glaucoma_dir, '*.jpg'))
                image_paths.extend(glaucoma_images)
                labels.extend([1] * len(glaucoma_images))
            
            if os.path.exists(non_glaucoma_dir):
                non_glaucoma_images = glob.glob(os.path.join(non_glaucoma_dir, '*.jpg'))
                image_paths.extend(non_glaucoma_images)
                labels.extend([0] * len(non_glaucoma_images))
    
    # Process validation data
    val_dir = os.path.join(data_dir, 'Validation')
    val_images_dir = os.path.join(val_dir, 'Images')
    
    if os.path.exists(val_images_dir):
        # Read the glaucoma.csv file
        csv_path = os.path.join(val_dir, 'glaucoma.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(val_images_dir, row['ImgName'])
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(int(row['Label']))
    
    return image_paths, labels

def process_palm_dataset(data_dir: str = '../data/raw/PALM') -> Tuple[List[str], List[int]]:
    """
    Process the PALM dataset
    
    Args:
        data_dir: Directory containing the PALM dataset
        
    Returns:
        Tuple containing:
            - List of image paths
            - List of labels (0: Normal, 1: Pathological)
    """
    image_paths = []
    labels = []
    
    # Process training data
    train_dir = os.path.join(data_dir, 'Train')
    train_images_dir = os.path.join(train_dir, 'PALM-Training400')
    
    if os.path.exists(train_images_dir):
        # Read the labels from the Excel file if available
        excel_path = os.path.join(train_dir, 'PM_Label_and_Fovea_Location.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            for _, row in df.iterrows():
                img_name = row['imgName']
                img_path = os.path.join(train_images_dir, img_name)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    # Assuming 0 is normal and 1 is pathological
                    labels.append(1 if row['label'] == 1 else 0)
    
    return image_paths, labels

def combine_datasets() -> Tuple[List[str], List[int]]:
    """
    Combine all datasets into a single dataset
    
    Returns:
        Tuple containing:
            - List of image paths
            - List of labels (0: Normal, 1: Glaucoma, 2: DR, 3: Pathological)
    """
    all_image_paths = []
    all_labels = []
    
    # Process 'all' dataset
    all_paths, all_labels_list = process_all_dataset()
    all_image_paths.extend(all_paths)
    all_labels.extend(all_labels_list)
    
    # Process REFUGE2 dataset
    refuge_paths, refuge_labels = process_refuge_dataset()
    all_image_paths.extend(refuge_paths)
    # Map REFUGE2 labels (0: Normal, 1: Glaucoma) to our combined labels
    all_labels.extend(refuge_labels)
    
    # Process PALM dataset
    palm_paths, palm_labels = process_palm_dataset()
    all_image_paths.extend(palm_paths)
    # Map PALM labels (0: Normal, 1: Pathological) to our combined labels
    # We'll map pathological to a new category (3)
    palm_mapped_labels = [3 if label == 1 else 0 for label in palm_labels]
    all_labels.extend(palm_mapped_labels)
    
    return all_image_paths, all_labels

def split_dataset(
    image_paths: List[str], 
    labels: List[int], 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        test_size: Proportion of the dataset to include in the test split
        val_size: Proportion of the training dataset to include in the validation split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing:
            - Training image paths
            - Training labels
            - Validation image paths
            - Validation labels
            - Test image paths
            - Test labels
    """
    # First split into train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Then split train+val into train and val
    # Calculate validation size relative to train+val size
    relative_val_size = val_size / (1 - test_size)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=relative_val_size, 
        random_state=random_state,
        stratify=train_val_labels
    )
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

if __name__ == "__main__":
    # Test the dataset processing
    image_paths, labels = combine_datasets()
    print(f"Total images: {len(image_paths)}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Split the dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths, labels
    )
    
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")