import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

class Preprocessor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.lip_landmarks = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]

    def extract_lips(self, frame):
        """Extract and align lip region from a frame"""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get lip landmark coordinates
        lip_points = []
        for landmark_idx in self.lip_landmarks:
            landmark = landmarks[landmark_idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points)
        
        # Get bounding box and expand it slightly
        x, y, w, h = cv2.boundingRect(lip_points)
        padding = 15
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(w + 2*padding, frame.shape[1] - x), min(h + 2*padding, frame.shape[0] - y)
        
        # Crop and resize
        lip_roi = frame[y:y+h, x:x+w]
        lip_roi = cv2.resize(lip_roi, Config.FRAME_SIZE)
        lip_roi = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)
        
        return lip_roi

    def process_video(self, video_path):
        """Process a video file to extract lip frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            lip_frame = self.extract_lips(frame)
            if lip_frame is not None:
                frames.append(lip_frame)
        
        cap.release()
        
        # Pad or truncate to fixed number of frames
        if len(frames) > Config.FRAMES_PER_SAMPLE:
            frames = frames[:Config.FRAMES_PER_SAMPLE]
        else:
            padding = np.zeros((Config.FRAMES_PER_SAMPLE - len(frames), *Config.FRAME_SIZE))
            frames.extend(padding)
            
        return np.array(frames)


