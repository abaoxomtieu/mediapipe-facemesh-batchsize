import numpy as np
import cv2
import mediapipe as mp
import threading


class FacialLandmark:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        # self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True) # 478 keypoint
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) # 468 keypoint
        self.image = None
        self.results = None
        self.landmarks = None
        self.lock = threading.Lock()

    def process(self, image):
        self.image = image
        self.results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.results.multi_face_landmarks:
            self.landmarks = self.results.multi_face_landmarks[0]
            return True
        return False

    def get_landmarks_coordinate(self):
        if self.landmarks:
            return np.array(
                [[landmark.x, landmark.y] for landmark in self.landmarks.landmark]
            )
        return np.zeros((468, 2))


class MultiThreadExtractor:
    def __init__(self, facial_landmark):
        self.facial_landmark = facial_landmark

    def extract_coordinate_multi_thread(self, images):
        results = [None] * len(images)
        threads = [
            threading.Thread(target=self._extract_coordinate, args=(image, results, i))
            for i, image in enumerate(images)
        ]
        [thread.start() for thread in threads]
        [thread.join() for thread in threads]
        return np.array(results)

    def _extract_coordinate(self, image, results, index):
        with self.facial_landmark.lock:
            self.facial_landmark.process(image)
            results[index] = self.facial_landmark.get_landmarks_coordinate()


F = FacialLandmark()

# Initialize MultiThreadExtractor instance with the FacialLandmark instance
Extractor = MultiThreadExtractor(F).extract_coordinate_multi_thread

"""
Import Extractor function from the MultiThreadExtractor class and call it with the images as input to get the landmarks of the images.
Usage:
    Extractor(images)
Condition:
    - The images must be in the same shape.
    - INPUT: batch size x height x width x 3(num channels)

OUTPUT : batch size x 468(number of landmark point) x 2(x,y)

Example:
    images = [cv2.imread("./imagenormal.jpg"), cv2.imread("./imagenormal.jpg")]
    results = Extractor(images)
    frame = cv2.imread("./imagenormal.jpg")
    h, w, _ = frame.shape
    for x, y in results[0]:
        cv2.circle(frame, (int(x * w), int(y * h)), 2, (0, 255, 0), -1)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

"""
