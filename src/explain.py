import shap
import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import preprocess

MODEL_PATH = "models/lstm_model.h5"


def explain_prediction():
    # Load data
    X, y = preprocess()

    if len(X) == 0:
        print("❌ Not enough data")
        return

    # Flatten time-series for SHAP
    X_flat = X.reshape(X.shape[0], -1)

    # Load model
    model = load_model(MODEL_PATH)

    # Wrapper function (VERY IMPORTANT)
    def model_predict(data):
        data = data.reshape(data.shape[0], X.shape[1], X.shape[2])
        return model.predict(data)

    # Use small background sample
    background = X_flat[:10]

    # Create explainer
    explainer = shap.KernelExplainer(model_predict, background)

    # Explain latest sample
    sample = X_flat[-1:]
    shap_values = explainer.shap_values(sample)

    print("\n🔍 SHAP Feature Contributions:\n")

    feature_names = [
        "stress", "anxiety", "mood", "emotional_clarity",
        "sleep_hours", "energy", "routine",
        "procrastination", "study_hours", "task_completion"
    ]

    # Since we flattened, repeat feature names for time steps
    feature_names = feature_names * X.shape[1]

    # Take first output (academic risk for simplicity)
    contributions = shap_values[0][0]

    # Get top 5 important features
    indices = np.argsort(np.abs(contributions))[-5:]

    print("\n🔥 Top Contributing Features:\n")

    for i in reversed(indices):
        print(f"{feature_names[i]}: {contributions[i]:.6f}")

if __name__ == "__main__":
    explain_prediction()