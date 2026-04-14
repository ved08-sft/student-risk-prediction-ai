import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import preprocess

MODEL_PATH = "models/lstm_model.h5"


def load_trained_model():
    model = load_model(MODEL_PATH)
    return model


def interpret(score):
    if score > 0.7:
        return "High Risk ⚠️"
    elif score > 0.4:
        return "Moderate Risk ⚠️"
    else:
        return "Low Risk ✅"


def predict_risk():
    # Load processed data
    X, y = preprocess()

    if len(X) == 0:
        print("❌ Not enough data to predict.")
        return

    # Take the latest sequence
    latest_data = X[-1]

    # Reshape for model (1 sample)
    latest_data = np.expand_dims(latest_data, axis=0)

    # Load model
    model = load_trained_model()

    # ✅ Predict (NOW INSIDE FUNCTION)
    prediction = model.predict(latest_data)[0]

    academic, burnout, career = prediction

    print("\n🔍 Risk Prediction Result:\n")
    print(f"📘 Academic Risk: {academic:.2f}")
    print(f"🔥 Burnout Risk: {burnout:.2f}")
    print(f"💼 Career Risk: {career:.2f}")

    print("\n📊 Interpretation:\n")
    print("Academic:", interpret(academic))
    print("Burnout:", interpret(burnout))
    print("Career:", interpret(career))


if __name__ == "__main__":
    predict_risk()