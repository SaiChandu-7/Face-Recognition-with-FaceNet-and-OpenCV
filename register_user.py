import cv2
import numpy as np
import os
import argparse
from preprocess import Preprocess
import config
import sys

try:
    import winsound
    beep = True
except ImportError:
    beep = False  # winsound only available on Windows


def register_user(name):
    print(f"[Log] Starting registration for: {name}")

    preprocess = Preprocess(database_path=config.database_path)
    cap = cv2.VideoCapture(0)
    collected_embeddings = []
    count = 0
    max_count = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces, coords = preprocess.getFace(frame)
        display_frame = frame.copy()

        if faces is not None:
            for face in faces:
                embedding = preprocess.embedding(face)
                collected_embeddings.append(embedding)
                count += 1
                print(f"[Log] Captured face {count}/{max_count}")

                if beep:
                    winsound.Beep(1000, 150)  # frequency, duration in ms

                cv2.putText(display_frame, f"Captured {count}/{max_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Registering Face", face)

        cv2.imshow("Camera Feed", display_frame)

        if cv2.waitKey(100) == 27 or count >= max_count:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save embeddings
    save_path = os.path.join(config.database_path, f"{name}.npy")
    if not os.path.exists(config.database_path):
        os.makedirs(config.database_path)z

    np.save(save_path, np.array(collected_embeddings))
    print(f"[Log] Saved embeddings to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Name of the person to register")
    args = parser.parse_args()

    register_user(args.name)
