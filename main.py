import os
from preprocess import Preprocess
from keras_facenet import FaceNet
import numpy as np
import cv2
import config
import pyttsx3
from datetime import datetime

ATTENDANCE_DIR = os.path.join(os.getcwd(), "attendance_history")

class FaceRecognition:
    def __init__(self):
        print("[Log] Initializing system...")
        self.embedder = FaceNet()
        self.preprocess = Preprocess(database_path=config.database_path)
        self.database = self.init_database()
        self.recognized_today = set()
        self.engine = pyttsx3.init()
        self.set_voice()

    def set_voice(self):
        self.engine.setProperty('volume', 1.0)
        self.engine.setProperty('rate', 150)

    def speak(self, text, loud=False):
        self.engine.setProperty('volume', 1.0 if loud else 0.9)
        self.engine.setProperty('rate', 130 if loud else 150)
        self.engine.say(text)
        self.engine.runAndWait()

    def init_database(self):
        data = {}
        for file in os.listdir(config.database_path):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                data[name] = np.load(os.path.join(config.database_path, file))
        print("[Log] Database initialized.")
        return data

    def log_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        date_folder = os.path.join(ATTENDANCE_DIR, date_str)
        os.makedirs(date_folder, exist_ok=True)

        file_path = os.path.join(date_folder, f"{name}_{time_str}.txt")
        with open(file_path, "w") as f:
            f.write(f"Name: {name}\nTime: {now.strftime('%H:%M:%S')}\nDate: {date_str}")
        print(f"[Log] Attendance saved to {file_path}")

    def recognize_faces_in_video(self, video_path=0):
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces, coords = self.preprocess.getFace(frame)
            if faces is not None:
                for i, face in enumerate(faces):
                    emb = self.embedder.embeddings([face])[0]
                    name, similarity = self.find_match(emb)

                    x1, y1, x2, y2 = coords[i]

                    if name is not None:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if name not in self.recognized_today:
                            self.speak(f"{name}, your attendance is recorded.", loud=True)
                            self.recognized_today.add(name)
                            self.log_attendance(name)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        self.speak("Face not registered.", loud=True)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == 27:  # ESC key to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    def find_match(self, embedding):
        min_dist = float('inf')
        identity = None

        for name, db_embeds in self.database.items():
            for db_emb in db_embeds:
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        print(f"[Debug] Closest match: {identity}, Distance: {min_dist:.4f}")

        if min_dist < config.threshold:
            return identity, min_dist
        else:
            return None, None

if __name__ == "__main__":
    FR = FaceRecognition()
    FR.recognize_faces_in_video()
