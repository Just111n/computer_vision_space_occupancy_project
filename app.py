from flask import Flask, render_template, request, redirect, url_for
import os
from cv import load_model, process_image_with_model  # Import the functions from cv.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Folder to save uploaded images

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the model once when the app starts
model = load_model()

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
    max_people = request.form.get("max_people")
    
    # Placeholder for image processing (occupancy analysis)
    # You can add your YOLO/CSRNet model processing here
    # Use the model to process the image
    occupancy_data = process_image_with_model(image_path, model)

    # TODO Put Spatial Occupancy calculations here

    data = {
        "image_path": image_path,
        "occupancy_data": occupancy_data,
        "max_people": max_people
    }

    return render_template('result.html', data=data, image_path=image_path, image_filename=image.filename)

if __name__ == '__main__':
    app.run(debug=True)
