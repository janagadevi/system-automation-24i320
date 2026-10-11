import csv
import random
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# All files are saved inside this project folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sensor_data.csv"
OUTPUT_FILE = BASE_DIR / "output.csv"
LOG_FILE = BASE_DIR / "day1_log.txt"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_FILE = OUTPUT_DIR / "decision_tree_model.joblib"
REPORT_FILE = OUTPUT_DIR / "training_report.txt"

NUM_READINGS = 250
FEATURES = ["temperature", "pressure", "vibration", "rpm", "current"]


def generate_reading():
    """Generate one simulated reading for the five sensors."""
    return {
        "temperature": round(random.uniform(25, 100), 2),
        "pressure": round(random.uniform(1, 10), 2),
        "vibration": round(random.uniform(0.1, 15), 2),
        "rpm": random.randint(500, 3000),
        "current": round(random.uniform(1, 25), 2),
    }


def label_reading(reading):
    """Day 2 labels based on clear, predefined sensor thresholds."""
    if (
        reading["temperature"] >= 85
        or reading["pressure"] >= 8.5
        or reading["vibration"] >= 10
        or reading["current"] >= 22
        or reading["rpm"] < 700
    ):
        return "FAILURE"
    if (
        reading["temperature"] >= 70
        or reading["pressure"] >= 7
        or reading["vibration"] >= 7
        or reading["current"] >= 18
        or reading["rpm"] < 1000
        or reading["rpm"] > 2700
    ):
        return "WARNING"
    return "NORMAL"


# Day 1 rule engine: lower priority number is checked first.
RULES = [
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] >= 85,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] >= 10,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] >= 8.5,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] >= 22,
        "action": "SEND_WARNING",
    },
    {
        "name": "LOW_RPM",
        "priority": 5,
        "condition": lambda r: r["rpm"] < 700,
        "action": "CHECK_MOTOR",
    },
    {
        "name": "SENSOR_WARNING",
        "priority": 6,
        "condition": lambda r: label_reading(r) == "WARNING",
        "action": "SCHEDULE_MAINTENANCE",
    },
]


def evaluate_rules(reading):
    for rule in sorted(RULES, key=lambda item: item["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"]
    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


def main():
    random.seed()  # Use new random readings each run.
    readings = [generate_reading() for _ in range(NUM_READINGS)]

    # Save the generated sensor readings and their threshold-based labels.
    with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURES + ["state"])
        writer.writeheader()
        for reading in readings:
            writer.writerow({**reading, "state": label_reading(reading)})

    # Train a Day 2 Decision Tree from the generated CSV.
    data = pd.read_csv(DATA_FILE)
    X = data[FEATURES]
    y = data["state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    REPORT_FILE.write_text(
        f"Dataset: {DATA_FILE.name}\n"
        f"Rows: {len(data)}\nTraining rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Features: {', '.join(FEATURES)}\n"
        f"Target: state\nAccuracy: {accuracy:.3f}\n\n"
        f"Classification report:\n"
        f"{classification_report(y_test, predictions, zero_division=0)}",
        encoding="utf-8",
    )

    # Run each reading through the Day 1 rule engine and Day 2 model.
    output_rows = []
    with LOG_FILE.open("w", encoding="utf-8") as log:
        for number, reading in enumerate(readings, start=1):
            rule_name, action = evaluate_rules(reading)
            predicted_state = model.predict(pd.DataFrame([reading], columns=FEATURES))[0]
            threshold_state = label_reading(reading)
            row = {
                "reading_no": number,
                **reading,
                "threshold_state": threshold_state,
                "predicted_state": predicted_state,
                "rule": rule_name,
                "action": action,
            }
            output_rows.append(row)
            log_entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **row,
            }
            log.write(json.dumps(log_entry) + "\n")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSYSTEM AUTOMATION: DAY 1 + DAY 2")
    print("--------------------------------")
    print(f"Sensor readings generated : {NUM_READINGS}")
    print(f"Training/test accuracy    : {accuracy:.2%}")
    print(f"Sensor data CSV           : {DATA_FILE}")
    print(f"Combined results CSV      : {OUTPUT_FILE}")
    print(f"Day 1 automation log      : {LOG_FILE}")
    print(f"Trained model             : {MODEL_FILE}")
    print(f"Training report           : {REPORT_FILE}")
    print("\nFirst 10 results:")
    print(pd.read_csv(OUTPUT_FILE).head(10).to_string(index=False))
    print("\nDone! Open output.csv to view all sensor readings and decisions.")


if __name__ == "__main__":
    main()
