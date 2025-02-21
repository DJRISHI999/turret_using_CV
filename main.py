import cv2
import numpy as np
import math

def main():
    names = ["None", "Dhruv"]  # List of names with 0th index as "None"
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Load the Haar Cascade for frontal face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yaml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Get the dimensions of the frame
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Draw a "+" sign spanning the entire screen along the x and y axes
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 2)
        cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 255), 2)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces and recognize them
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(face_img)
            if confidence < 70:
                name = names[id] if id < len(names) else "Unknown"
                color = (0, 255, 0)  # Green for recognized faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unrecognized faces

                # Draw x and y axes within the bounding box of the unknown face
                face_center_x, face_center_y = x + w // 2, y + h // 2
                cv2.line(frame, (face_center_x, y), (face_center_x, y + h), (0, 0, 255), 2)
                cv2.line(frame, (x, face_center_y), (x + w, face_center_y), (0, 0, 255), 2)

                # Draw a line from the center of the bounding box to the origin (center of the frame)
                cv2.line(frame, (face_center_x, face_center_y), (center_x, center_y), (0, 0, 255), 2)

                # Calculate the angle between the line and the horizontal axis
                delta_x = face_center_x - center_x
                delta_y = face_center_y - center_y
                angle = math.degrees(math.atan2(delta_y, delta_x))

                # Calculate the height of the right-angled triangle
                distance = math.sqrt(delta_x**2 + delta_y**2)
                height_of_triangle = abs(distance * math.sin(math.radians(angle)))

                cv2.putText(frame, f'Angle: {angle:.2f}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f'Height: {height_of_triangle:.2f}', (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()