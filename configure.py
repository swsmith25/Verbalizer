# Lip Reading Configuration Module
# This module contains configuration settings for a lip reading model using the GRID corpus.
#  Holds constants for project

import os


class Config:
    # Dataset paths
    DATA_DIR = "data/grid_corpus"
    TRAIN_SPLIT = 0.8
    
    # Preprocessing
    FRAME_SIZE = (128, 128)  # Height, Width
    FRAMES_PER_SAMPLE = 29  # GRID corpus uses 29 frames per sample
    CHANNELS = 1  # Grayscale
    
    # Model parameters
    NUM_CLASSES = 51  # GRID corpus has 51 classes (including silence)
    LSTM_UNITS = 128
    DENSE_UNITS = 64
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5
    
    # Output
    OUTPUT_DIR = "output"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "lip_reader_model.h5")
    
    @staticmethod
    def create_dirs():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        