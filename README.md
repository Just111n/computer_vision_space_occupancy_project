
# Occupancy WebApp

This project is a web application built with Flask to monitor room occupancy. It enables users to upload images of a public space, processes them using computer vision models (e.g., YOLO, CSRNet), and provides insights such as occupancy count, heatmaps, and density maps.

## Features

- Upload an image of a room or public space.
- Process the image with computer vision models (YOLO/CSRNet) to detect and count individuals.
- Display occupancy data, including heatmaps and density maps, for easy analysis.
  
## Project Structure

```plaintext
Occupancy-WebApp/
├── app.py                # Main Flask app file
├── static/               # Folder for static files (CSS, JavaScript)
│   └── uploads/          # Folder for uploaded images
├── templates/            # Folder for HTML templates
├── requirements.txt      # Python dependencies
└── environment.yml       # Conda environment file
```

## Getting Started

### Prerequisites

- **Python** 3.8 or higher
- **Conda** for environment management
- **Git** for version control

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/Occupancy-WebApp.git
   cd Occupancy-WebApp
   ```

2. **Create and Activate Conda Environment**:
   - Run the following command to create and activate the virtual environment:

     ```bash
     conda env create -f environment.yml --name computer_vision_space_occupancy_project_env
     conda activate computer_vision_space_occupancy_project_env
     ```

3. **Install Dependencies**:
   - If not using `environment.yml`, install dependencies directly from `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Flask Application**:

   ```bash
   python app.py
   ```

   The app will be available at `http://127.0.0.1:5000/`.

### Folder Structure

- `app.py`: Contains the Flask routes and logic for uploading, processing, and displaying images.
- `static/uploads/`: Stores images uploaded by users.
- `templates/`: Contains HTML files for rendering the UI (e.g., `upload.html` and `result.html`).

### Workflow

1. **Upload an Image**:
   - Go to the homepage (`/`), select an image file, and submit it.

2. **Image Processing**:
   - The app processes the image to detect and count individuals using YOLO/CSRNet models (integration ongoing).

3. **Display Results**:
   - Occupancy count and visualizations (e.g., heatmaps, density maps) are displayed on the results page.

### Development Notes

- **Model Integration**:
  - Implement YOLO/CSRNet processing in `app.py` under the `process_image` function.
  - Add any helper functions for loading models, image processing, and generating outputs.
  
- **UI Enhancements**:
  - Customize templates in `templates/` for improved user experience.
  - Add CSS and JavaScript in `static/` as needed.

### Contribution Guide

1. **Branching**:
   - Create feature branches for each task (e.g., `feature/model-integration`).
   - Submit pull requests for review before merging to `main`.

2. **Code Style**:
   - Follow Python PEP 8 guidelines.
   - Document functions and modules thoroughly for team readability.

3. **Dependencies**:
   - Add new packages to `requirements.txt` or `environment.yml` and commit changes.

### Future Enhancements

- Implement real-time video processing for continuous occupancy tracking.
- Add authentication for secure access to the app.
- Integrate cloud storage for handling large numbers of uploaded images.

### License

This project is for internal development and testing purposes only. Further distribution is restricted.

---

Let me know if you need further adjustments or any additional information!
