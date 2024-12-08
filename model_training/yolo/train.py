import torch
import cv2
import numpy as np
import os
import json
import yaml
from ultralytics import YOLO

def prepare_dir(dir: str):
    """
    Prepare directory for rewriting.

    Create directories if they do not exist already. Otherwise, clear the contents 
    of the directory.
    
    :param dir: Directory path
    :type dir: str
    :return: None
    :rtype: None
    """
    if os.path.isdir(dir):
        try:
            for file in os.listdir(dir):
                os.remove(os.path.join(dir, file))
        except Exception as e:
            print(f'Error removing {dir}: {e}')
    os.makedirs(dir, exist_ok=True)

def batch_get_filenames_from_dir(src_dir: str):
    subdir_names_paths = [(d, os.path.join(src_dir, d)) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    out = dict()
    for dn, dp in subdir_names_paths:
        name_ls = [fn.split('.')[0] for fn in os.listdir(dp)]
        out[dn] = name_ls
    
    return out

def batch_read_images_from_dir(src_dir: str):
    subdir_names_paths = [(d, os.path.join(src_dir, d)) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    out = dict()
    for dn, dp in subdir_names_paths:
        img_ls = [cv2.imread(os.path.join(dp, fn)) for fn in os.listdir(dp)]
        img_ls = [img for img in img_ls if img is not None]
        out[dn] = img_ls
    
    return out

def batch_read_masks_from_dir(src_dir: str):
    subdir_names_paths = [(d, os.path.join(src_dir, d)) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    out = dict()
    for dn, dp in subdir_names_paths:
        mask_ls = [cv2.imread(os.path.join(dp, fn), cv2.IMREAD_GRAYSCALE) for fn in os.listdir(dp)]
        mask_ls = [mask for mask in mask_ls if mask is not None]
        out[dn] = mask_ls
    
    return out

def batch_read_mappings_from_dir(src_dir: str):
    subdir_names_paths = [(d, os.path.join(src_dir, d)) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    out = dict()
    for dn, dp in subdir_names_paths:
        json_ls = []
        for fn in os.listdir(dp):
            with open(os.path.join(dp, fn), 'r') as json_file:
                mapping = json.load(json_file)
                json_file.close()
            mapping = {int(poly_id): class_name for poly_id, class_name in mapping.items()}
            json_ls.append(mapping)
        out[dn] = json_ls
        
    return out

def get_unique_class_names_from_map(mapping_dict: dict[str, list[dict[int, str]]]):
    class_name_ls = [class_name for mappings in mapping_dict.values() 
                     for mapping in mappings 
                     for class_name in mapping.values()]
    
    return np.unique(class_name_ls).tolist()

def write_yolo_annotations(name_ls: list[str], mask_ls: list[np.ndarray], mapping_ls: list[dict[int, str]], dst_dir: str, yolo_mapping: dict[int, str]):
    prepare_dir(dst_dir)
    class2id = {v: k for k,v in yolo_mapping.items()}

    for name, mask, mapping in zip(name_ls, mask_ls, mapping_ls):
        # print(img.shape, mask.shape, mapping)
        height, width = mask.shape

        annotation_lines = []
        for pixel_value, class_name in mapping.items():
            # print(pixel_value, class_name)
            contours, _ = cv2.findContours((mask == pixel_value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                bbox_width = w / width
                bbox_height = h / height

                class_id = class2id[class_name]
                annotation_lines.append(f'{class_id} {x_center} {y_center} {bbox_width} {bbox_height}')

        with open(os.path.join(dst_dir, f'{name}.txt'), 'w') as label_file:
            label_file.write('\n'.join(annotation_lines))
            label_file.close()

def create_yaml_config(img_train_path: str, img_val_path: str, class_ls: list[str]):
    config = {
        'train': img_train_path,
        'val': img_val_path,
        'nc': len(class_ls),
        'names': class_ls
    }

    with open('data.yaml', 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
        yaml_file.close()

if __name__ == '__main__':
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = YOLO('yolo11n.pt')
    print(type(model))

    metrics = model.val(data='data.yaml', epochs=10, batch=16)
    print('Original:', metrics.box.maps)

    names = batch_get_filenames_from_dir(r'..\data\images')
    images = batch_read_images_from_dir(r'..\data\images')
    masks = batch_read_masks_from_dir(r'..\data\masks')
    mappings = batch_read_mappings_from_dir(r'..\data\mappings')

    data_unique_classes = get_unique_class_names_from_map(mappings)
    yolo_unique_classes = list(model.names.values())
    new_classes = sorted(list(set(data_unique_classes) - set(yolo_unique_classes)))
    classes_ls = yolo_unique_classes + new_classes
    classes_dict = {i: cls for i, cls in enumerate(classes_ls)}
    
    for dset in masks.keys():
        write_yolo_annotations(names[dset], masks[dset], mappings[dset], rf'..\data\labels\{dset}', classes_dict)

    create_yaml_config(r'..\data\images\train', r'..\data\images\val', classes_ls)

    model.train(data='data.yaml', epochs=10, batch=16, save=False, name='custom_yolo11n')
    metrics = model.val(data='data.yaml', epochs=10, batch=16)
    print('Finetuned:', metrics.box.maps)
    # model.export(format='onnx')
