from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPool3D, BatchNormalization,
    TimeDistributed, Conv2D, MaxPool2D, Flatten,
    LSTM, Dense, Dropout
)

def create_lip_reader_model():
    """Create a 3D CNN + LSTM model for lip reading"""
    input_shape = (Config.FRAMES_PER_SAMPLE, *Config.FRAME_SIZE, Config.CHANNELS)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 3D Convolutional layers
    x = Conv3D(32, (3, 5, 5), activation='relu', padding='same')(inputs)
    x = MaxPool3D(pool_size=(1, 2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPool3D(pool_size=(1, 2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(96, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPool3D(pool_size=(1, 2, 2))(x)
    x = BatchNormalization()(x)
    
    # TimeDistributed 2D convolutions
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPool2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Flatten())(x)
    
    # Recurrent layers
    x = LSTM(Config.LSTM_UNITS, return_sequences=True)(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    x = LSTM(Config.LSTM_UNITS)(x)
    x = Dropout(Config.DROPOUT_RATE)(x)
    
    # Dense layers
    x = Dense(Config.DENSE_UNITS, activation='relu')(x)
    outputs = Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model