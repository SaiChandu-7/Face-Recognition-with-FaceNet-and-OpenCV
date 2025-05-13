import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not accessible")
else:
    print("✅ Camera opened successfully")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
            break

cap.release()
cv2.destroyAllWindows()
