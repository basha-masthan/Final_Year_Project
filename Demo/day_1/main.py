import cv2
import face_recognition
from flask import Flask, render_template, request, Response
import numpy as np
import os

app = Flask(__name__)

# Path to pre-loaded student image (replace with your image path)
REFERENCE_IMAGE_PATH = "student_reference.jpg"

# Load the reference image and encode it
def load_reference_image():
    ref_image = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
    ref_encoding = face_recognition.face_encodings(ref_image)[0]  # Get face encoding
    return ref_encoding

# Capture live image from webcam
def capture_live_image():
    cap = cv2.VideoCapture(0)  # Open webcam
    ret, frame = cap.read()  # Capture one frame
    if ret:
        # Save the frame temporarily
        cv2.imwrite("live_image.jpg", frame)
    cap.release()
    return "live_image.jpg"

# Compare the reference and live images
def verify_identity(ref_encoding, live_image_path):
    live_image = face_recognition.load_image_file(live_image_path)
    live_encodings = face_recognition.face_encodings(live_image)
    
    if len(live_encodings) == 0:
        return False, "No face detected in live image!"
    
    # Compare the first detected face
    result = face_recognition.compare_faces([ref_encoding], live_encodings[0])
    return result[0], "Verification successful" if result[0] else "Identity mismatch!"

# Flask route for the exam start page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle verification
@app.route('/verify', methods=['POST'])
def verify():
    ref_encoding = load_reference_image()
    live_image_path = capture_live_image()
    match, message = verify_identity(ref_encoding, live_image_path)
    
    # Clean up temporary live image
    os.remove(live_image_path)
    
    if match:
        return "Verified! Exam can start."
    else:
        return f"Failed: {message}"

# Stream webcam feed to the browser
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)