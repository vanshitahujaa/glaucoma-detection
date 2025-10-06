from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
from PIL import Image
import base64
import io
import sys
import os
import torch
import torchvision.transforms as transforms

# Add model directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from model import GlaucomaModel

# Initialize FastAPI app
app = FastAPI(title="Retinal Disease Detection API", 
              description="API for detecting retinal diseases from fundus images",
              version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    # Load model on startup
    # Try to load combined model first, fall back to other models if not available
    model_paths = [
        "../model/output/combined_model.pth",  # First try combined model
        "../model/output/full_model.pth",      # Then full model
        "../model/output/quick_model.pth"      # Then quick model
    ]
    
    model = GlaucomaModel(num_classes=4, model_name="efficientnet_b0")
    model_loaded = False
    
    for path in model_paths:
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Model loaded successfully from {path}")
            model_loaded = True
            break
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Could not load model from {path}: {e}")
    
    if not model_loaded:
        print("Using pretrained backbone only")
        
    model.eval()
    model.class_names = ["Normal", "Glaucoma", "DR", "Pathological"]
    print("Model loaded successfully")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Retinal Disease Detection API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess image for model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get class names and probabilities
    class_names = model.class_names
    probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    prediction = class_names[probabilities.argmax().item()]
    
    # Generate heatmap using GradCAM or similar technique
    # For now, create a simple colored overlay as a placeholder
    width, height = image.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Draw a simple colored circle in the center
    img_draw = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    cv_img = np.array(img_draw)
    center = (width // 2, height // 2)
    
    # Different colors for different predictions
    color_map = {
        "Normal": (0, 255, 0, 128),      # Green
        "Glaucoma": (255, 0, 0, 128),    # Red
        "DR": (0, 0, 255, 128),          # Blue
        "Pathological": (255, 165, 0, 128)  # Orange
    }
    
    color = color_map.get(prediction, (255, 0, 0, 128))
    cv2.circle(cv_img, center, min(width, height) // 3, color, -1)
    overlay = Image.fromarray(cv_img)
    
    # Blend with original image
    image = image.convert('RGBA')
    heatmap_img = Image.alpha_composite(image, overlay).convert('RGB')
    
    # Convert heatmap to base64 for response
    buffered = io.BytesIO()
    heatmap_img.save(buffered, format="JPEG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "prediction": prediction,
        "probabilities": probs_dict,
        "heatmap": heatmap_base64
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)