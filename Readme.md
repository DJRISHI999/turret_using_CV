# Turret Using Computer Vision

This project implements a turret system using computer vision techniques. The system uses OpenCV for face detection and recognition, and it can track and identify faces in real-time.

## Project Structure

- `main.py`: The main script that runs the turret system, detects faces, and recognizes them using a trained model.
- `Image_collecter.py`: A script to collect grayscale images of faces for training. It captures images from the webcam and saves them with incremental IDs.
- `Training.py`: A script to train the face recognition model using the collected images and save the trained model to a YAML file.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/DJRISHI999/turret_using_CV.git
   cd turret_using_CV
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```sh
   pip install opencv-python opencv-contrib-python numpy
   ```

## Usage

### Pre-run

1. Open Lib folder under your virtual environment by default ".venv"
2. Open the cv2 folder inside it
3. Open the data folder
4. Copy haarcascade_frontalface_default.xml file to the main directory

### Collect Images

1. Run the `Image_collecter.py` script to collect images for training:
   ```sh
   python Image_collecter.py
   ```

2. Enter the ID for the person when prompted. The script will capture 30 grayscale images of the person's face and save them in the `images` directory.

### Train Model

1. Run the `Training.py` script to train the face recognition model:
   ```sh
   python Training.py
   ```

2. The script will load the images from the `images` directory, train the model, and save the trained model to `trained_model.yaml`.

### Run Turret System

1. Run the `main.py` script to start the turret system:
   ```sh
   python main.py
   ```

2. The script will open the webcam, detect faces, and recognize them using the trained model. It will display the camera feed with detected faces and their names.


## Acknowledgements

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)