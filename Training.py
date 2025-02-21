import cv2
import os
import numpy as np

def get_images_and_labels(image_dir):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        # Read the image and convert it to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Extract the ID from the image filename
        person_id = int(os.path.split(image_path)[-1].split('_')[1])
        # Append the image and ID to the lists
        face_samples.append(img)
        ids.append(person_id)

    return face_samples, ids

def train_model():
    image_dir = 'images'
    face_samples, ids = get_images_and_labels(image_dir)

    # Create the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the face samples and IDs
    recognizer.train(face_samples, np.array(ids))

    # Save the trained model to a YAML file
    model_path = 'trained_model.yaml'
    recognizer.save(model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()