import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.data_preprocessing import preprocess
import os

MODEL_PATH = "models/lstm_model.h5"


def build_model(input_shape):
    model = Sequential()

    # LSTM layer (core)
    model.add(LSTM(64, input_shape=input_shape))

    # Output layer (binary risk for now)
    model.add(Dense(3, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train():
    # Load data
    X, y = preprocess()

    if len(X) == 0:
        print(" Not enough data to train model.")
        return

    print("\n Training model...\n")

    # Build model
    model = build_model((X.shape[1], X.shape[2]))

    # Train
    model.fit(X, y, epochs=10, batch_size=4)

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    # Save model
    model.save(MODEL_PATH)

    print("\nModel trained and saved successfully!\n")


if __name__ == "__main__":
    train()