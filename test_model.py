import requests
import os
import sys
import json
from PIL import Image
import base64
import io

def test_model(image_path):
    """Test the model with a sample image"""
    url = "http://localhost:8000/predict/"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    # Open the image file
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        
        # Make the request
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Print the results
            print(f"Prediction: {result['prediction']}")
            print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")
            print("Heatmap received successfully")
            
            # Save the heatmap
            heatmap_data = base64.b64decode(result['heatmap'])
            heatmap_img = Image.open(io.BytesIO(heatmap_data))
            output_path = "heatmap_result.jpg"
            heatmap_img.save(output_path)
            print(f"Heatmap saved to {output_path}")
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test images for each class
        test_images = {
            "normal": "/Users/vanshitahuja/Documents/trae_projects/Project_DIP/data/raw/all/images/01_h.jpg",
            "glaucoma": "/Users/vanshitahuja/Documents/trae_projects/Project_DIP/data/raw/all/images/01_g.jpg",
            "dr": "/Users/vanshitahuja/Documents/trae_projects/Project_DIP/data/raw/all/images/01_dr.JPG"
        }
        
        # Test with all sample images
        for class_name, img_path in test_images.items():
            print(f"\nTesting {class_name} image: {img_path}")
            test_model(img_path)