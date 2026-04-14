import pandas as pd
import os

# File where data will be stored
DATA_FILE = "data/dataset.csv"


def collect_weekly_data():
    print("\n--- Weekly Behavioral Data Collection ---\n")

    data = {}

    # Psychological & Emotional
    data["stress"] = int(input("Stress level (1-5): "))
    data["anxiety"] = int(input("Anxiety level (1-5): "))
    data["mood"] = int(input("Mood (-2 to +2): "))
    data["emotional_clarity"] = int(input("Emotional clarity (1-5): "))

    # Physical & Health
    data["sleep_hours"] = float(input("Sleep hours: "))
    data["energy"] = int(input("Energy level (1-5): "))

    # Behavioral & Routine
    data["routine"] = int(input("Routine regularity (1-5): "))
    data["procrastination"] = int(input("Procrastination (1-5): "))

    # Academic / Career
    data["study_hours"] = float(input("Study hours: "))
    data["task_completion"] = float(input("Task completion ratio (0-1): "))

    return data


def save_data(data):
    df = pd.DataFrame([data])

    # Check if file exists
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE, index=False)
    else:
        df.to_csv(DATA_FILE, mode='a', header=False, index=False)

    print("\n Data saved successfully!!\n")


def run_input_system():
    data = collect_weekly_data()
    save_data(data)