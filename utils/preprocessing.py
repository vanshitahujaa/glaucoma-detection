import cv2
import numpy as np
from typing import Tuple, Union, Optional
import torch

def extract_roi(image: np.ndarray) -> np.ndarray:
    """
    Extract Region of Interest (ROI) - Optic Disc from fundus image
    
    Args:
        image: Input fundus image
        
    Returns:
        ROI extracted image
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply adaptive thresholding to find bright regions (potential optic disc)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return the original image
    if not contours:
        return image
    
    # Find the largest contour (likely to be the optic disc)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add some padding around the ROI
    padding = int(max(w, h) * 0.2)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    
    # Extract ROI
    roi = image[y_start:y_end, x_start:x_end]
    
    return roi

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Input image
        
    Returns:
        CLAHE enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return result

def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply non-local means denoising
    
    Args:
        image: Input image
        
    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image using ImageNet mean and std
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Normalized image as numpy array (CHW format)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # Convert to float and scale to [0, 1]
    image_float = image_resized.astype(np.float32) / 255.0
    
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_float - mean) / std
    
    # Transpose from HWC to CHW format for PyTorch
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    return image_chw

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply full preprocessing pipeline to fundus image
    
    Args:
        image: Input fundus image
        
    Returns:
        Preprocessed image ready for model input
    """
    # Extract ROI (optic disc)
    roi = extract_roi(image)
    
    # Apply CLAHE for contrast enhancement
    enhanced = apply_clahe(roi)
    
    # Denoise the image
    denoised = denoise_image(enhanced)
    
    # Normalize the image
    normalized = normalize_image(denoised)
    
    return normalized