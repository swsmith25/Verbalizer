import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocess import Preprocessor
from configure import Config
from tqdm import tqdm

class LipReaderDataset:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_data(self):
        """Load and preprocess the dataset"""
        video_paths = []
        labels = []
        
        # Walk through data directory
        for root, _, files in os.walk(Config.DATA_DIR):
            for file in files:
                if file.endswith('.mpg') or file.endswith('.mp4'):
                    label = os.path.basename(root)
                    video_paths.append(os.path.join(root, file))
                    labels.append(label)
        
        # Create label mappings
        unique_labels = sorted(list(set(labels)))
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        
        # Process videos
        X = []
        y = []
        
        print("Processing videos...")
        for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
            frames = self.preprocessor.process_video(video_path)
            X.append(frames)
            y.append(self.class_to_idx[label])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = to_categorical(np.array(y), num_classes=len(self.class_to_idx))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-Config.TRAIN_SPLIT, random_state=42
        )
        
        return (X_train, y_train), (X_test, y_test)