import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Tuple, List, Dict, Union

class GlaucomaModel(nn.Module):
    def __init__(self, num_classes: int = 4, model_name: str = "efficientnet_b0"):
        """
        Initialize the Retinal Disease classification model
        
        Args:
            num_classes: Number of classes for classification (Normal, Glaucoma, DR, Pathological)
            model_name: Name of the backbone model to use (efficientnet_b0 or mobilenetv2_100)
        """
        super(GlaucomaModel, self).__init__()
        
        # Load the backbone model
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Get the number of features in the last layer
        if 'efficientnet' in model_name:
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'mobilenetv2' in model_name:
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Create a new classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Class names
        self.class_names = ["Normal", "Glaucoma", "DR", "Pathological"]
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        features = self.backbone(x)
        return self.classifier(features)
    
    def load_model(self, model_path: str) -> None:
        """Load model weights from a checkpoint"""
        self.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def predict(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Make a prediction on a preprocessed image
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Tuple containing:
                - Predicted class name
                - Dictionary of class probabilities
        """
        # Convert numpy array to tensor and add batch dimension
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0).float()
        
        # Move to device
        image = image.to(self.device)
        
        # Set model to evaluation mode
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self(image)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Convert probabilities to dictionary
        probs_dict = {
            class_name: float(prob) 
            for class_name, prob in zip(self.class_names, probabilities.cpu().numpy())
        }
        
        return self.class_names[predicted_class], probs_dict
    
    def export_onnx(self, save_path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> None:
        """
        Export the model to ONNX format for faster inference
        
        Args:
            save_path: Path to save the ONNX model
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
        """
        dummy_input = torch.randn(input_shape, device=self.device)
        torch.onnx.export(
            self,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )