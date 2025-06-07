import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from load_data import LipReaderDataset
from model import create_lip_reader_model
from configure import Config

def train():
    # Initialize
    Config.create_dirs()
    
    # Load data
    dataset = LipReaderDataset()
    (X_train, y_train), (X_test, y_test) = dataset.load_data()
    
    # Create model
    model = create_lip_reader_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            Config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == "__main__":
    model, history = train()