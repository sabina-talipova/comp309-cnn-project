# COMP309 CNN Project
This project is a Flask-based web application for object recognition, designed to serve predictions based on a dataset of images. The application uses a machine learning model trained on image data to classify and recognize objects.

## Key Features:
REST API endpoint for uploading images and receiving predictions.
Integration with Docker for easy deployment and containerization.
Lightweight, modular structure for ease of development and scaling.

## How to Run:
- Clone the repository.
- Build the Docker image: `docker build -t flask-app .`
- Run the container: `docker run -d -p 5000:5000 flask-app`
- Access the app via `http://localhost:5000/`
