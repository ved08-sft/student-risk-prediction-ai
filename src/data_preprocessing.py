import pandas as pd
import numpy as np

DATA_FILE = "data/dataset.csv"


def load_data():
    df = pd.read_csv(DATA_FILE)
    print("\n Data Loaded Successfully!\n")
    return df


def normalize_data(df):
    # Normalize values between 0 and 1
    return (df - df.min()) / (df.max() - df.min())


def create_sequences(data, time_steps=3):
    X = []
    y = []

    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])

        # Extract next week data
        next_week = data[i + time_steps]

        # --- Create 3 risks (simple logic for now) ---

        # Academic Risk
        academic_risk = 1 if (next_week[0] > 0.7 and next_week[8] < 0.3) else 0

        # Burnout Risk
        burnout_risk = 1 if (next_week[0] > 0.7 and next_week[4] < 0.4) else 0

        # Career Risk
        career_risk = 1 if (next_week[7] > 0.7 and next_week[8] < 0.3) else 0

        y.append([academic_risk, burnout_risk, career_risk])

    return np.array(X), np.array(y)

def preprocess():
    df = load_data()

    # Convert dataframe to numpy
    data = df.values

    # Normalize
    data = normalize_data(df).values

    # Create sequences
    X, y = create_sequences(data)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y