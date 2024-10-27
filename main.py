import cv2
import numpy as np
from deepface import DeepFace
import os
import time
from datetime import datetime
import logging
import json
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    filename='face_recognition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DATA_FILE = "face_encodings.json"
BACKUP_DIR = "backups"
MIN_FACE_SIZE = (80, 80)
SIMILARITY_THRESHOLD = 0.50
REGISTRATION_TIME = 15
DETECTION_TIME = 20
MIN_REQUIRED_FRAMES = 5


class FaceRecognitionSystem:
    def __init__(self):
        self.face_encodings = self.load_face_encodings()
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.executor = ThreadPoolExecutor(max_workers=4)

        if not os.path.exists(BACKUP_DIR):
            os.makedirs(BACKUP_DIR)

    def load_face_encodings(self):
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                    return {name: np.array(encoding) for name, encoding in data.items()}
            return {}
        except Exception as e:
            logging.error(f"Error loading face encodings: {e}")
            return {}

    def save_face_encodings(self):
        try:
            data_to_save = {name: encoding.tolist() for name, encoding in self.face_encodings.items()}

            with open(DATA_FILE, 'w') as f:
                json.dump(data_to_save, f)

            backup_file = os.path.join(BACKUP_DIR,
                                       f"face_encodings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(backup_file, 'w') as f:
                json.dump(data_to_save, f)

            print("✅ Data saved successfully")
            logging.info("Data saved and backup created")
        except Exception as e:
            print(f"❌ Error: {e}")
            logging.error(f"Error saving data: {e}")

    def preprocess_face(self, frame):
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def get_face_encoding(self, frame):
        try:
            preprocessed_frame = self.preprocess_face(frame)
            face_analysis = DeepFace.represent(
                preprocessed_frame,
                model_name='VGG-Face',
                enforce_detection=True,
                detector_backend='opencv'
            )
            if isinstance(face_analysis, list) and len(face_analysis) > 0:
                return face_analysis[0].get('embedding')
            return None
        except Exception as e:
            logging.warning(f"Error processing frame: {e}")
            return None

    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Camera could not be opened")
            return True
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            print(f"❌ Error initializing camera: {e}")
            return False

    def register_face(self, name):
        frames = []
        encodings = []
        start_time = time.time()
        face_found = False

        print(f"\n👤 Collecting face data for {name}...")
        print("📸 Look at the camera and turn your face in different angles")

        while time.time() - start_time < REGISTRATION_TIME:
            ret, frame = self.cap.read()
            if not ret:
                continue

            remaining_time = int(REGISTRATION_TIME - (time.time() - start_time))
            frame_copy = frame.copy()

            progress = int((time.time() - start_time) / REGISTRATION_TIME * 30)
            progress_bar = "▓" * progress + "░" * (30 - progress)
            cv2.putText(frame_copy, f"Time Remaining: {remaining_time}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 1:
                face_found = True
                x, y, w, h = faces[0]
                margin = int(0.2 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)

                face_frame = frame[y:y + h, x:x + w]
                frames.append(face_frame)

                encoding = self.get_face_encoding(face_frame)
                if encoding is not None:
                    encodings.append(encoding)

                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_copy, f"Face detected! Samples: {len(encodings)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif len(faces) > 1:
                cv2.putText(frame_copy, "Multiple faces detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame_copy, "No face detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Registration', frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Registration')

        if not face_found or len(encodings) < MIN_REQUIRED_FRAMES:
            print("❌ Insufficient face data collected")
            return False

        mean_encoding = np.mean(encodings, axis=0)
        self.face_encodings[name] = mean_encoding
        self.save_face_encodings()
        print(f"✅ {name} successfully registered! ({len(encodings)} samples)")
        return True

    def recognize_face(self, face_encoding):
        if face_encoding is None:
            return None, 0.0

        matches = []
        for name, stored_encoding in self.face_encodings.items():
            try:
                if len(face_encoding) != len(stored_encoding):
                    continue

                similarity = np.dot(stored_encoding, face_encoding) / (
                        np.linalg.norm(stored_encoding) * np.linalg.norm(face_encoding))
                matches.append((name, similarity))

            except Exception as e:
                logging.error(f"Error comparing faces: {e}")
                continue

        if not matches:
            return None, 0.0

        best_match = max(matches, key=lambda x: x[1])
        if best_match[1] >= SIMILARITY_THRESHOLD:
            return best_match
        return None, best_match[1]

    def detect_faces(self):
        self.face_encodings = self.load_face_encodings()

        start_time = time.time()
        detections = {}

        print("\n🔍 Face detection mode started...")
        print(f"⏱️  Duration: {DETECTION_TIME} seconds")

        while time.time() - start_time < DETECTION_TIME:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame_copy = frame.copy()
            remaining_time = int(DETECTION_TIME - (time.time() - start_time))

            progress = int((time.time() - start_time) / DETECTION_TIME * 30)
            progress_bar = "▓" * progress + "░" * (30 - progress)
            cv2.putText(frame_copy, f"Time Remaining: {remaining_time}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            future_results = []

            for (x, y, w, h) in faces:
                margin = int(0.2 * w)
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(frame.shape[1] - x, w + 2 * margin)
                h = min(frame.shape[0] - y, h + 2 * margin)

                face_frame = frame[y:y + h, x:x + w]

                future = self.executor.submit(self.get_face_encoding, face_frame)
                future_results.append((future, (x, y, w, h)))

            for future, (x, y, w, h) in future_results:
                try:
                    face_encoding = future.result(timeout=1.0)
                    if face_encoding is not None:
                        name, score = self.recognize_face(face_encoding)

                        if name:
                            if name not in detections:
                                detections[name] = {"count": 0, "total_score": 0}
                            detections[name]["count"] += 1
                            detections[name]["total_score"] += score

                            label = f"✅ {name}: {score:.2%}"
                            color = (0, 255, 0)
                        else:
                            label = f"❌ Unknown: {score:.2%}"
                            color = (0, 0, 255)

                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame_copy, (x, y + h - 35), (x + w, y + h), color, cv2.FILLED)
                        cv2.putText(frame_copy, label, (x + 6, y + h - 6),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                except Exception as e:
                    logging.error(f"Error processing face: {e}")
                    continue

            cv2.imshow('Face Detection', frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Face Detection')

        print("\n📊 Results:")
        for name, stats in detections.items():
            avg_score = stats["total_score"] / stats["count"]
            print(f"👤 {name}: detected {stats['count']} times (average accuracy: {avg_score:.2%})")

    def cleanup(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def show_menu(self):
        menu = """
╔════════════════════════════════════════╗
║           FACE RECOGNITION SYSTEM      ║
╠════════════════════════════════════════╣
║ 1. Register a new face                 ║
║ 2. Face detection mode                 ║
║ 3. List registered people              ║
║ 4. Clear data                          ║
║ q. Quit                                ║
╚════════════════════════════════════════╝
"""
        print(menu)
        return input("Select: ").lower()

    def run(self):
        print("\n🔥 Face Recognition System is starting...")

        if not self.initialize_camera():
            return

        try:
            while True:
                choice = self.show_menu()

                if choice == 'q':
                    print("\n👋 Goodbye! The program is ending.")
                    break

                elif choice == '1':
                    name = input("\n👤 Enter the name: ")
                    if name.strip():
                        self.register_face(name)
                    else:
                        print("❌ Name cannot be empty!")

                elif choice == '2':
                    if not self.face_encodings:
                        print("❌ No registered people!")
                        continue
                    self.detect_faces()

                elif choice == '3':
                    if self.face_encodings:
                        print("\n📋 Registered people:")
                        for i, name in enumerate(self.face_encodings.keys(), 1):
                            print(f"{i}. 👤 {name}")
                    else:
                        print("❌ No registered people!")

                elif choice == '4':
                    confirm = input("❗️ Are you sure you want to clear data? (yes/no): ").lower()
                    if confirm == 'yes':
                        self.face_encodings = {}
                        self.save_face_encodings()
                        print("✅ All data cleared.")
                    else:
                        print("❌ Data clearing canceled.")

                else:
                    print("❌ Invalid command! Please try again.")

        except KeyboardInterrupt:
            print("\n❗️ Interrupt signal received. Cleaning up...")
        except Exception as e:
            logging.error(f"Error in main program: {e}")
            print(f"❌ Unknown error: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.run()
