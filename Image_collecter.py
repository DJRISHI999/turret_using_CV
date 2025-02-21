import cv2
import os

def collect_images():
    # Ask for the ID
    person_id = input("Enter the ID for the person: ")

    # Create a directory to save images if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load the Haar Cascade for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    img_count = 0  # Counter for the number of images taken

    while img_count < 30:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Select the largest face (closest to the camera)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            (x, y, w, h) = largest_face

            # Draw a rectangle around the largest face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Save the grayscale image of the largest face
            face_img = gray[y:y+h, x:x+w]
            img_path = f'images/person_{person_id}_{img_count + 1}.jpg'
            cv2.imwrite(img_path, face_img)
            print(f'Saved {img_path}')
            img_count += 1

        # Display the frame with the detected face
        cv2.imshow('Grayscale Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {img_count} images for person ID {person_id}")

if __name__ == "__main__":
    collect_images()