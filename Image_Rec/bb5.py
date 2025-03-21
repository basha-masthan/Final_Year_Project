from insightface.app import FaceAnalysis
import cv2


app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

    cv2.imshow("InsightFace Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
