from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
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

# Add model directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from model import GlaucomaModel


# ------------------------------------------------------------
# LIFESPAN EVENT HANDLER (replaces deprecated @app.on_event)
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown logic.
    """
    global model

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "output"))

    print("\n🔍 Looking for models in:", MODEL_DIR)
    if os.path.exists(MODEL_DIR):
        print("📁 Available files:", os.listdir(MODEL_DIR))
    else:
        print("❌ Model directory not found!")

    # Define possible model paths
    model_paths = [
        os.path.join(MODEL_DIR, "combined_model.pth"),
        os.path.join(MODEL_DIR, "full_model.pth"),
        os.path.join(MODEL_DIR, "quick_model.pth")
    ]

    # Initialize model
    model = GlaucomaModel(num_classes=4, model_name="efficientnet_b0")
    model_loaded = False

    # Try loading available models
    for path in model_paths:
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"✅ Model loaded successfully from: {path}")
            model_loaded = True
            break
        except (FileNotFoundError, RuntimeError) as e:
            print(f"⚠️ Could not load model from {path}: {e}")

    if not model_loaded:
        print("⚠️ No trained model found — using pretrained backbone only.")

    model.eval()
    model.class_names = ["Normal", "Glaucoma", "DR", "Pathological"]
    print("✅ Model setup complete.\n")

    # Startup complete
    yield

    # Shutdown logic (optional)
    print("🛑 Shutting down Retinal Disease Detection API...")


# ------------------------------------------------------------
# FASTAPI APP INITIALIZATION
# ------------------------------------------------------------
app = FastAPI(
    title="Retinal Disease Detection API",
    description="API for detecting retinal diseases from fundus images",
    version="1.0.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------
# CORS CONFIGURATION
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Retinal Disease Detection API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model

    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess for model
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Model prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        class_names = model.class_names
        probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        prediction = class_names[probabilities.argmax().item()]

        # Create simple visual heatmap (placeholder)
        width, height = image.size
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        center = (width // 2, height // 2)

        color_map = {
            "Normal": (0, 255, 0, 128),      # Green
            "Glaucoma": (255, 0, 0, 128),    # Red
            "DR": (0, 0, 255, 128),          # Blue
            "Pathological": (255, 165, 0, 128)  # Orange
        }
        color = color_map.get(prediction, (255, 0, 0, 128))
        cv2.circle(overlay, center, min(width, height) // 3, color, -1)

        # Combine overlay with original image
        image_rgba = image.convert('RGBA')
        heatmap_img = Image.alpha_composite(image_rgba, Image.fromarray(overlay)).convert('RGB')

        # Convert to base64 for response
        buffered = io.BytesIO()
        heatmap_img.save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "prediction": prediction,
            "probabilities": probs_dict,
            "heatmap": heatmap_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
