from ultralytics import YOLO
import face_recognition
import cv2
import numpy as np
import os
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model_path = r"C:\Users\basha\OneDrive\Documents\code\Final_Year_Project\Pre-Trained_CNN\Image_Rec\yolov8n-face.pt"
model = YOLO(model_path)
model.to(device)  


KNOWN_FACES_DIR = r"C:\Users\basha\OneDrive\Documents\code\Final_Year_Project\Pre-Trained_CNN\Image_Rec\img"

known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    img_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])
    else:
        print(f"No face found in {filename}, skipping...")

print(f"✅ Loaded {len(known_face_encodings)} known faces.")


cap = cv2.VideoCapture(0)

last_recognized_names = set()  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    face_locations = []
    face_encodings = []
    recognized_names = set()  

    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            
            encoding = face_recognition.face_encodings(rgb_face)
            if encoding:
                face_encodings.append(encoding[0])
                face_locations.append((y1, x2, y2, x1))  

    
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        recognized_names.add(name)

        
        if name not in last_recognized_names:
            print(f"Match Found: {name}")

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 500, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 500, 0), 2)

    
    last_recognized_names = recognized_names

    cv2.imshow("YOLOv8 Face Recognition (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
