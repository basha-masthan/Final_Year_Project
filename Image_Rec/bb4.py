import cv2
import face_recognition
import numpy as np
import os

KNOWN_FACES_DIR = r"C:\Users\basha\OneDrive\Documents\code\Final_Year_Project\Pre-Trained_CNN\Image_Rec\img"

known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])
    else:
        print(f"No face found in {filename}, skipping...")

# Use OpenCV CUDA if available
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

if USE_CUDA:
    print("Using GPU acceleration with OpenCV CUDA.")

cap = cv2.VideoCapture(0)
last_recognized_name = None  # Track last recognized face

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to CUDA (if GPU is available)
    if USE_CUDA:
        frame_gpu = cv2.cuda_GpuMat()
        frame_gpu.upload(frame)
        frame = frame_gpu.download()

    # Resize frame to improve performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")  # CNN uses GPU
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_recognized_name = None

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        if name != last_recognized_name:
            print(f"Match Found: {name}")
            last_recognized_name = name  # Update last recognized face

        # Scale back face location for original frame size
        top, right, bottom, left = [v * 2 for v in face_location]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        current_recognized_name = name

    if not face_encodings:
        last_recognized_name = None

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
# print("Starting...")

# if USE_CUDA:
#     print("Using GPU acceleration with OpenCV CUDA.")


# import cv2
# print(cv2.getBuildInformation())


# import dlib
# print("DLIB CUDA:", dlib.DLIB_USE_CUDA)

