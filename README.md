# Occupancy WebApp

This project is a web application built with Flask to monitor room occupancy. Users can upload images of public spaces, and the app processes them using computer vision models (e.g., YOLO, CSRNet) to provide insights such as occupancy count, heatmaps, and density maps.

## Features

- Upload an image of a room or public space.
- Process the image with computer vision models (YOLO/CSRNet) to detect and count individuals.
- Display occupancy data, including heatmaps and density maps, for easy analysis.

---

## Project Structure

```plaintext
Occupancy-WebApp/
├── app.py                # Main Flask app file
├── static/               # Folder for static files (CSS, JavaScript)
│   └── uploads/          # Folder for uploaded images
├── templates/            # Folder for HTML templates
├── requirements.txt      # Python dependencies
├── Dockerfile            # Configuration for building the Docker image
├── docker-compose.yml    # Docker Compose configuration for production
└── README.md             # Project documentation
```

---

## Running in Development Mode (Python)

### Prerequisites

- **Python** 3.11
- **pip** for dependency management
- **Git** for version control

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Just111n/computer_vision_space_occupancy_project.git
   cd computer_vision_space_occupancy_project
   ```

2. **Create and Activate a Virtual Environment**:

   On Windows:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS/Linux:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask Application**:

   ```bash
   python app.py
   ```

   The app will be available at **`http://127.0.0.1:5000/`**.

---

## Running in Production Mode (Docker Compose with Gunicorn)

### Prerequisites For Production

- **Docker** installed
- **Docker Compose** installed

### Setup Instructions For Production

1. **Pull the Docker Image** from GitHub Container Registry:

   ```bash
   docker pull ghcr.io/just111n/computer_vision_space_occupancy_project-web:latest
   ```

2. **Run the Application Using `docker-compose.ghcr.yml`**:

   Instead of using the default `docker-compose.yml`, use `docker-compose.ghcr.yml`, which is configured for the **pulled image** from GHCR:

   ```bash
   docker-compose -f docker-compose.ghcr.yml up -d
   ```

   - `-f docker-compose.ghcr.yml`: Specifies the **correct compose file**.
   - `-d`: Runs the container **in detached mode** (in the background).

3. **Access the App**:

   Open your browser and go to **`http://localhost:5000/`**.

4. **Stopping the Application**:

   ```bash
   docker-compose -f docker-compose.ghcr.yml down
   ```

---

### Additional Commands:

- **Pull the Latest Image and Restart the Application**:

  ```bash
  docker pull ghcr.io/just111n/computer_vision_space_occupancy_project-web:latest
  docker-compose -f docker-compose.ghcr.yml up -d
  ```

- **Force a Clean Restart** (Stops, Pulls, and Starts Fresh):

  ```bash
  docker-compose -f docker-compose.ghcr.yml down
  docker pull ghcr.io/just111n/computer_vision_space_occupancy_project-web:latest
  docker-compose -f docker-compose.ghcr.yml up -d
  ```

This setup ensures your **Docker container is deployed from GHCR** using a dedicated `docker-compose.ghcr.yml` file. 

---

## Folder Structure

- **`app.py`**: Contains the Flask routes and logic for uploading, processing, and displaying images.
- **`static/uploads/`**: Stores images uploaded by users.
- **`templates/`**: Contains HTML files for rendering the UI (e.g., `upload.html` and `result.html`).
- **`Dockerfile`**: Defines how the Flask app is built and run inside a container.
- **`docker-compose.yml`**: Specifies how to run the Flask app in production using Gunicorn.

---

## Workflow

1. **Upload an Image**:
   - Go to the homepage (`/`), select an image file, and submit it.

2. **Image Processing**:
   - The app processes the image to detect and count individuals using YOLO/CSRNet models.

3. **Display Results**:
   - Occupancy count and visualizations (e.g., heatmaps, density maps) are displayed on the results page.

---

## Contribution Guide

1. **Branching**:
   - Create feature branches for each task (e.g., `feature/model-integration`).
   - Submit pull requests for review before merging to `main`.

2. **Code Style**:
   - Follow Python PEP 8 guidelines.
   - Document functions and modules for readability.

3. **Dependencies**:
   - Add new packages to `requirements.txt` and commit changes.

---

## Future Enhancements

- Implement real-time video processing for continuous occupancy tracking.
- Add authentication for secure access to the app.
- Integrate cloud storage for handling large numbers of uploaded images.

---

### License

This project is for internal development and testing purposes only. Further distribution is restricted.
