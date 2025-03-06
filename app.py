from flask import Flask, render_template, request, redirect, url_for
import os
from cv import load_model, process_image_with_model  # Import the functions from cv.py

# import torch
# from transformers import SegformerForSemanticSegmentation
# from torchvision import transforms
# from PIL import Image
# from test import apply_segmentation_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder to save uploaded images

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the model once when the app starts
model = load_model()


# # Load the model once when the app starts
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically select GPU or CPU
# semantic_segnmentation_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# num_classes = 3
# semantic_segnmentation_model.decode_head.classifier = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
# semantic_segnmentation_model.config.num_labels = num_classes

# # Load custom checkpoint
# checkpoint = torch.load("final_model.pth", map_location=device)
# semantic_segnmentation_model.load_state_dict(checkpoint['model_state_dict'])
# semantic_segnmentation_model.eval()
# semantic_segnmentation_model.to(device)

# # Define the transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    image = request.files['image']
    if image.filename == '':
        return "No selected file", 400
    
    # Save the uploaded image to the specified folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    print("Image saved to:", image_path)
    
    
    
    
    

   

    

   
    occupancy_data = process_image_with_model(image_path, model )

        
    data = {
         "occupancy_data": occupancy_data,
    }


    
        
    

    return render_template('result.html', data=data, image_path=image_path, image_filename=image.filename)

if __name__ == '__main__':
    app.run(debug=True)
