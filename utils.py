import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_lip_movement(video_path, output_path=None):
    """Visualize lip movement from a video"""
    preprocessor = LipPreprocessor()
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        lip_frame = preprocessor.extract_lips(frame)
        if lip_frame is not None:
            frames.append(lip_frame)
    
    cap.release()
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.axis('off')
    
    def update(i):
        ax.clear()
        ax.imshow(frames[i], cmap='gray')
        ax.set_title(f"Frame {i+1}/{len(frames)}")
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)
    
    if output_path:
        anim.save(output_path, writer='ffmpeg', fps=10)
    
    return anim

def predict_from_video(model, video_path, class_mapping):
    """Make prediction on a single video"""
    preprocessor = LipPreprocessor()
    frames = preprocessor.process_video(video_path)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    
    # Add channel dimension if needed
    if frames.ndim == 4:
        frames = np.expand_dims(frames, axis=-1)
    
    # Predict
    probabilities = model.predict(frames)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_mapping[predicted_idx]
    
    return predicted_class, probabilities