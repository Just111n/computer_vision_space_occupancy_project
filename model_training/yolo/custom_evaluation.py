from ultralytics import YOLO
import torch
import argparse
from train import batch_read_images_from_dir, batch_get_filenames_from_dir
import yaml
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('model_file', help='Path to YOLO model file')

args = parser.parse_args()

model_fn = args.model_file
model_name = model_fn.split('.')[0]

assert os.path.exists(model_fn), 'Model does not exist'

model = YOLO(model_fn)
device = 0 if torch.cuda.is_available() else 'cpu'     # use GPU if available, otherwise use CPU

eval_dir = 'yolo_models/eval'
eval_results = model.val(data='data.yaml', device=device, name=model_name, project=eval_dir, save_json=True)

model_p = eval_results.results_dict['metrics/precision(B)']
model_r = eval_results.results_dict['metrics/recall(B)']
model_f1 = 2 / (1/model_p + 1/model_r)
print("R2 score:", model_f1)
results_dir = [dn for dn in sorted(os.listdir(eval_dir)) if model_name in dn and os.path.isdir(os.path.join(eval_dir, dn))][-1]
print(results_dir)

def calculate_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union area
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    # IoU
    return intersection / union if union != 0 else 0

def parse_yolo_labels(label_file, image_width, image_height):
    """
    Parse YOLO label file and convert to absolute pixel coordinates.
    
    Args:
        label_file (str): Path to the YOLO label file (.txt).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        
    Returns:
        list[dict]: List of ground truth bounding boxes with class IDs and coordinates.
    """
    ground_truth_boxes = []
    
    with open(label_file, "r") as file:
        for line in file:
            # Parse the line
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert normalized to absolute coordinates
            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height
            
            # Add the bounding box to the list
            ground_truth_boxes.append({
                "class_id": int(class_id),
                "bbox": [x1, y1, x2, y2]
            })
    
    return ground_truth_boxes

def parse_dataset_labels(label_dir, image_sizes):
    """
    Parse all YOLO label files in a directory.
    
    Args:
        label_dir (str): Directory containing YOLO label files.
        image_sizes (dict): Dictionary mapping image file names to (width, height).
        
    Returns:
        dict: Mapping of image file names to ground truth boxes.
    """
    dataset_labels = {}
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            # Get corresponding image dimensions
            image_file = label_file.replace(".txt", ".jpg")  # or ".png"
            image_width, image_height = image_sizes[image_file]
            
            # Parse the label file
            gt_boxes = parse_yolo_labels(os.path.join(label_dir, label_file), image_width, image_height)
            dataset_labels[image_file] = gt_boxes
    
    return dataset_labels

# names = [name + '.jpg' for name in batch_get_filenames_from_dir('./data/images')['val']]
# image_sizes = [img.shape for img in batch_read_images_from_dir('./data/images')['val']]

