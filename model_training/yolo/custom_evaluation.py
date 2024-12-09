from ultralytics import YOLO
import torch
import argparse
from train import create_yaml_config
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('model_file', help='Path to YOLO model file')

args = parser.parse_args()

model = YOLO(args.model_file)

device = 0 if torch.cuda.is_available() else 'cpu'     # use GPU if available, otherwise use CPU

val_results = model.val(data='data.yaml', device=device)

model_p = val_results.results_dict['metrics/precision(B)']
model_r = val_results.results_dict['metrics/recall(B)']
model_f1 = 2 / (1/model_p + 1/model_r)
print(model_f1)

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



