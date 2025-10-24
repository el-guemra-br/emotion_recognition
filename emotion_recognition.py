import cv2
from fer import FER

# Initialize the emotion detector
emotion_detector = FER(mtcnn=True)  # Use MTCNN for better face detection

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame for emotions
    results = emotion_detector.detect_emotions(frame)

    # Annotate frame with detected emotions
    for result in results:
        (x, y, w, h) = result["box"]
        emotion, score = emotion_detector.top_emotion(frame[y:y+h, x:x+w])
        label = f"{emotion} ({score:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
