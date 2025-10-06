import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from typing import Union, List, Tuple

def generate_gradcam(model, preprocessed_img: np.ndarray, target_layer_name: str = None) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for model explainability
    
    Args:
        model: The trained model
        preprocessed_img: Preprocessed image as numpy array (CHW format)
        target_layer_name: Name of the target layer for Grad-CAM (if None, uses the last convolutional layer)
        
    Returns:
        Heatmap visualization as numpy array
    """
    # Convert preprocessed image to tensor and add batch dimension if needed
    if isinstance(preprocessed_img, np.ndarray):
        input_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0).float()
    else:
        input_tensor = preprocessed_img.unsqueeze(0) if preprocessed_img.dim() == 3 else preprocessed_img
    
    # Move to the same device as the model
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    # Determine target layer
    if target_layer_name is None:
        # For EfficientNet
        if hasattr(model.backbone, 'conv_head'):
            target_layer = [model.backbone.conv_head]
        # For MobileNetV2
        elif hasattr(model.backbone, 'conv_head'):
            target_layer = [model.backbone.conv_head]
        else:
            # Fallback to the last convolutional layer
            for name, module in reversed(list(model.backbone.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = [module]
                    break
    else:
        # Get the specified target layer
        target_layer = [dict([*model.backbone.named_modules()])[target_layer_name]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=torch.cuda.is_available())
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
    grayscale_cam = grayscale_cam[0, :]  # First image in batch
    
    # Get original image for visualization
    # Convert from CHW to HWC and denormalize
    orig_img = preprocessed_img.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    orig_img = (orig_img * std + mean)
    orig_img = np.clip(orig_img, 0, 1)
    
    # Overlay heatmap on original image
    visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
    
    return visualization