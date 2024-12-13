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