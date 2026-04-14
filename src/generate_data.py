import pandas as pd
import numpy as np
import os

DATA_FILE = "data/dataset.csv"


def generate_fake_data(num_weeks=60):
    data = []

    for i in range(num_weeks):
        week = {
            "stress": np.random.randint(1, 6),
            "anxiety": np.random.randint(1, 6),
            "mood": np.random.randint(-2, 3),
            "emotional_clarity": np.random.randint(1, 6),
            "sleep_hours": np.random.uniform(4, 9),
            "energy": np.random.randint(1, 6),
            "routine": np.random.randint(1, 6),
            "procrastination": np.random.randint(1, 6),
            "study_hours": np.random.uniform(0, 6),
            "task_completion": np.random.uniform(0, 1)
        }

        data.append(week)

    df = pd.DataFrame(data)

    # Create folder if not exists
    os.makedirs("data", exist_ok=True)

    df.to_csv(DATA_FILE, index=False)

    print(f"\n✅ Generated {num_weeks} weeks of data!\n")


if __name__ == "__main__":
    generate_fake_data()