# cv.py
import cv2
import torch
# Additional imports as needed (e.g., for YOLO, CSRNet)
import time
import numpy as np
import os

def load_model():
    """
    Load the model used for occupancy detection.
    Placeholder: Adjust this to load YOLO/CSRNet or any other model.
    """
    # Example placeholder for loading model, replace with actual model loading code
    model = None  # Replace with YOLO or CSRNet model load logic
    # TODO: return model here
    model_dir = "./ml_models/yolo/"
    net = cv2.dnn.readNetFromONNX(model_dir + "yolov5n.onnx")
    file = open(model_dir + "coco.txt","r")
    classes = file.read().split('\n')

    model = (net, classes)

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
    # TODO: apply CV here
    # TODO: rewrite code if needed
    # unpack model
    net, classes = model
    if image is None:
        return # something?
    
    # image = cv2.resize(image, (640,640))   # can try diff dimensions
    # TODO: find params that work
    blob = cv2.dnn.blobFromImage(image,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    image_width, image_height = image.shape[1], image.shape[0]
    x_scale = image_width/640
    y_scale = image_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)


    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)

    
    people = 0
    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        if (label=="person"):
            people += 1
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(image,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(image, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0,255),2)

    #   Adding a 5-second delay
    print("Processing image, please wait...")
    # time.sleep(5)  # 5-second delay

    print(image.shape)  # Example: Output image shape (height, width, channels)
    

    # save image
    result_path_ls = image_path.split('/')[:]
    result_path_ls[1] = 'outputs'
    result_path_ls[2] = 'output_' + result_path_ls[2]
    result_path = "/".join(result_path_ls)
    print(result_path)

    # Ensure the directory exists
    output_dir = os.path.dirname(result_path)
    os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn’t exist

    cv2.imwrite(result_path, image)

    # Placeholder for actual processing using the model
    # Here you could use the model to detect objects and count them
    occupancy_count = people  # Replace with actual count derived from model
    print(f"Processed image at {result_path} and found {occupancy_count} individuals.")
    
    # Example of returning data (expand this with heatmaps or density maps if needed)
    return {'count': occupancy_count}
