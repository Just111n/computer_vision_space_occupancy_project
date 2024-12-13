# cv.py
import cv2
import torch
# Additional imports as needed (e.g., for YOLO, CSRNet)
import time
import numpy as np
import os
from ultralytics import YOLO

def load_model():
    """
    Load the model used for occupancy detection.
    Placeholder: Adjust this to load YOLO/CSRNet or any other model.
    """
    # Example placeholder for loading model, replace with actual model loading code
    model = None  # Replace with YOLO or CSRNet model load logic
    # TODO: return model here
    model_dir = "./ml_models/yolo"
    model_fn = "custom_yolo11m_aug.pt"

    # model = (net, classes)
    model = YOLO(os.path.join(model_dir, model_fn))
    # model = YOLO('yolo11m.pt')

    print("Model loaded successfully.")
    return model

def process_image_with_model(image_path, model):
    """
    Process the image to analyze occupancy using the loaded model.
    
    Parameters:
    - image_path (str): Path to the uploaded image.
    - model: The loaded model for occupancy detection.
    
    Returns:
    - dict: A dictionary containing occupancy data, e.g., count, heatmap path.
    """
    # Load the image
    image = cv2.imread(image_path)
 

    result = model.predict(image, conf=0.5)[0]

    people = 0
    chairs = 0
    for cls_id in result.boxes.cls:
        if result.names[cls_id.item()] == 'person':
            people += 1
        elif result.names[cls_id.item()] == 'chair':
            chairs += 1

    # print(image.shape)  # Example: Output image shape (height, width, channels)
    print(result.orig_img.shape)  # Example: Output image shape (height, width, channels)
    

    # save image
    result_path_ls = image_path.split('/')[:]
    result_path_ls[1] = 'outputs'
    result_path_ls[2] = 'output_' + result_path_ls[2]
    result_path = "/".join(result_path_ls)
    # print(result_path)

    # Ensure the directory exists
    output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn’t exist

    # cv2.imwrite(result_path, image)
    result.save(result_path)

    # Placeholder for actual processing using the model
    # Here you could use the model to detect objects and count them
    occupancy_count = people  # Replace with actual count derived from model
    print(f"Processed image at {result_path} and found {occupancy_count} individuals and {chairs} chairs.")
    
    # Example of returning data (expand this with heatmaps or density maps if needed)
    return {'count': occupancy_count, 'percentage': occupancy_count/chairs*100, 'seats': chairs}
