"""
Robotic Arm Joint Health Monitoring System
23IF02 System Automation - Bonus Challenge

Synthetic coursework demo only. Never use this code to control real machinery.
Run: python main.py
"""
import csv
import random
import re
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"
DATASET_FILE = DATA_DIR / "historical_joint_readings.csv"
MAINTENANCE_LOG = LOG_DIR / "maintenance_notes.txt"
OUTPUT_FILE = REPORT_DIR / "arm_decisions.csv"
MODEL_REPORT = REPORT_DIR / "model_report.txt"

FEATURES = ["torque_nm", "temperature_c", "vibration_mm_s"]
JOINTS = ["shoulder", "elbow", "wrist"]
ARMS = [f"ARM-{i:02d}" for i in range(1, 7)]

# Illustrative limits for this synthetic assignment only.
SAFETY_LIMITS = {"torque_nm": 95.0, "temperature_c": 90.0, "vibration_mm_s": 12.0}
WARNING_LIMITS = {"torque_nm": 75.0, "temperature_c": 75.0, "vibration_mm_s": 7.0}
RELEVANT_KEYWORDS = [
    "wear", "loose", "overheat", "overheating", "noise", "vibration",
    "lubrication", "crack", "friction", "drift", "damaged", "abnormal"
]
SEVERE_KEYWORDS = ["crack", "overheat", "overheating", "damaged"]


def make_synthetic_dataset(rows=300, seed=42):
    """Create historical labelled readings for model training."""
    rng = random.Random(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arm_id", "joint", *FEATURES, "wear_label"])
        writer.writeheader()
        for i in range(rows):
            worn = rng.random() < 0.35
            if worn:
                torque, temp, vibration = rng.uniform(65, 100), rng.uniform(65, 95), rng.uniform(5, 14)
            else:
                torque, temp, vibration = rng.uniform(25, 75), rng.uniform(30, 75), rng.uniform(0.5, 7.5)
            label = "WEAR" if torque >= 72 and (temp >= 70 or vibration >= 6.5) else "NORMAL"
            writer.writerow({
                "arm_id": ARMS[i % len(ARMS)], "joint": JOINTS[i % len(JOINTS)],
                "torque_nm": round(torque, 2), "temperature_c": round(temp, 2),
                "vibration_mm_s": round(vibration, 2), "wear_label": label
            })


def create_sample_maintenance_log():
    """Write 10 sample free-text notes, including irrelevant notes."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    notes = [
        "ARM-01 shoulder: unusual vibration and a loose joint observed.",
        "ARM-02 elbow: temperature is higher than normal during operation.",
        "ARM-03 wrist: routine inspection completed, no abnormal noise.",
        "ARM-04 shoulder: visible crack near the joint cover; stop and inspect.",
        "ARM-05 elbow: lubrication may be low; friction noise heard.",
        "ARM-06 wrist: technician noticed sensor readings drifting suddenly.",
        "ARM-01 elbow: assembly line lunch schedule changed.",
        "ARM-02 shoulder: replacement parts delivery expected tomorrow.",
        "ARM-03 elbow: repeated abnormal vibration during movement.",
        "ARM-04 wrist: workstation floor cleaned; no joint issue reported."
    ]
    MAINTENANCE_LOG.write_text("\n".join(notes) + "\n", encoding="utf-8")


def load_notes():
    """Group free-text maintenance notes by arm ID."""
    notes_by_arm = {arm: [] for arm in ARMS}
    if MAINTENANCE_LOG.exists():
        for line in MAINTENANCE_LOG.read_text(encoding="utf-8").splitlines():
            match = re.search(r"\b(ARM-\d{2})\b", line.upper())
            if match and match.group(1) in notes_by_arm:
                notes_by_arm[match.group(1)].append(line)
    return notes_by_arm


def note_signals(notes):
    text = " ".join(notes).lower()
    relevant = [word for word in RELEVANT_KEYWORDS if word in text]
    severe = [word for word in SEVERE_KEYWORDS if word in text]
    return relevant, severe


def safety_stop(readings):
    """Immediate deterministic check; never uses the ML model."""
    for r in readings:
        if (r["torque_nm"] >= SAFETY_LIMITS["torque_nm"]
                or r["temperature_c"] >= SAFETY_LIMITS["temperature_c"]
                or r["vibration_mm_s"] >= SAFETY_LIMITS["vibration_mm_s"]):
            return True, f"Safety threshold exceeded at {r['joint']}"
    return False, "No immediate safety threshold crossed"


def detect_spoofing(readings):
    """
    Simple inconsistency checks: implausibly low torque paired with high heat/
    vibration, or a torque outlier compared with the arm's other joint readings.
    This detects only selected crude attacks, not every possible spoof.
    """
    torques = sorted(r["torque_nm"] for r in readings)
    median_torque = torques[len(torques) // 2]
    flags = []
    for r in readings:
        reasons = []
        if r["torque_nm"] < 20 and (
            r["temperature_c"] >= WARNING_LIMITS["temperature_c"]
            or r["vibration_mm_s"] >= WARNING_LIMITS["vibration_mm_s"]
        ):
            reasons.append("low torque conflicts with high heat/vibration")
        if abs(r["torque_nm"] - median_torque) > 45:
            reasons.append("torque differs sharply from the arm's joint median")
        if reasons:
            flags.append(f"{r['joint']}: " + "; ".join(reasons))
    return flags


def decide_arm(readings, model, notes):
    # Safety takes highest priority and is checked before model prediction.
    stop, stop_reason = safety_stop(readings)
    spoof_flags = detect_spoofing(readings)
    relevant_notes, severe_notes = note_signals(notes)

    predictions = model.predict(pd.DataFrame(
        [{f: r[f] for f in FEATURES} for r in readings], columns=FEATURES
    ))
    predicted_wear = any(p == "WEAR" for p in predictions)
    warning = any(
        r["torque_nm"] >= WARNING_LIMITS["torque_nm"]
        or r["temperature_c"] >= WARNING_LIMITS["temperature_c"]
        or r["vibration_mm_s"] >= WARNING_LIMITS["vibration_mm_s"]
        for r in readings
    )

    if stop:
        decision, reasons = "EMERGENCY_STOP", [stop_reason]
    elif spoof_flags:
        decision, reasons = "SCHEDULE_MAINTENANCE", ["Possible spoofed/inconsistent sensor data", *spoof_flags]
    elif severe_notes:
        decision, reasons = "SCHEDULE_MAINTENANCE", ["Severe maintenance note: " + ", ".join(severe_notes)]
    elif predicted_wear or relevant_notes:
        decision, reasons = "SCHEDULE_MAINTENANCE", []
        if predicted_wear:
            reasons.append("ML model predicts wear in at least one joint")
        if relevant_notes:
            reasons.append("Maintenance notes contain: " + ", ".join(relevant_notes))
    elif warning:
        decision, reasons = "MONITOR", ["At least one joint crossed a warning limit"]
    else:
        decision, reasons = "SAFE", ["No safety, warning, spoof, wear, or relevant-note signal"]

    return {
        "decision": decision,
        "predicted_wear_joints": sum(p == "WEAR" for p in predictions),
        "relevant_note_keywords": ", ".join(relevant_notes) or "none",
        "spoof_flags": " | ".join(spoof_flags) or "none",
        "reason": "; ".join(reasons)
    }


def make_current_readings():
    readings = {}
    for arm in ARMS:
        readings[arm] = []
        for joint in JOINTS:
            readings[arm].append({
                "joint": joint,
                "torque_nm": round(random.uniform(35, 78), 2),
                "temperature_c": round(random.uniform(35, 78), 2),
                "vibration_mm_s": round(random.uniform(1, 8), 2)
            })
    return readings


def train_model():
    data = pd.read_csv(DATASET_FILE)
    X, y = data[FEATURES], data["wear_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REPORT.write_text(
        f"Dataset rows: {len(data)}\nTraining rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\nAccuracy: {accuracy_score(y_test, pred):.3f}\n\n"
        f"{classification_report(y_test, pred, zero_division=0)}",
        encoding="utf-8"
    )
    return model


def demonstrate_four_outcomes(model):
    """Run one controlled example for each required decision."""
    examples = [
        ("SAFE", [
            {"joint": j, "torque_nm": 45.0, "temperature_c": 45.0, "vibration_mm_s": 2.0}
            for j in JOINTS
        ], []),
        ("MONITOR", [
            {"joint": j, "torque_nm": 65.0, "temperature_c": 78.0, "vibration_mm_s": 5.0}
            for j in JOINTS
        ], []),
        ("SCHEDULE_MAINTENANCE", [
            {"joint": j, "torque_nm": 55.0, "temperature_c": 50.0, "vibration_mm_s": 3.0}
            for j in JOINTS
        ], ["ARM-99 shoulder: loose joint observed."]),
        ("EMERGENCY_STOP", [
            {"joint": "shoulder", "torque_nm": 99.0, "temperature_c": 55.0, "vibration_mm_s": 4.0},
            {"joint": "elbow", "torque_nm": 45.0, "temperature_c": 45.0, "vibration_mm_s": 2.0},
            {"joint": "wrist", "torque_nm": 45.0, "temperature_c": 45.0, "vibration_mm_s": 2.0}
        ], [])
    ]
    results = []
    for expected, readings, notes in examples:
        result = decide_arm(readings, model, notes)
        results.append({"example": expected, "actual_decision": result["decision"], "reason": result["reason"]})
    return results


def main():
    random.seed()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(rows=300)
    create_sample_maintenance_log()
    model = train_model()
    notes_by_arm = load_notes()
    current = make_current_readings()

    output_rows = []
    for arm, readings in current.items():
        result = decide_arm(readings, model, notes_by_arm[arm])
        output_rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "arm_id": arm,
            "decision": result["decision"],
            "predicted_wear_joints": result["predicted_wear_joints"],
            "relevant_note_keywords": result["relevant_note_keywords"],
            "spoof_flags": result["spoof_flags"],
            "reason": result["reason"],
            "joint_readings_json": json.dumps(readings)
        })
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    demos = demonstrate_four_outcomes(model)
    demo_file = REPORT_DIR / "four_outcome_examples.csv"
    with demo_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=demos[0].keys())
        writer.writeheader()
        writer.writerows(demos)

    explanation = """Robotic Arm Joint Health Monitoring System — Design Notes

1. Safety thresholds
This synthetic demo uses immediate-stop thresholds of torque >= 95 Nm,
temperature >= 90 C, or vibration >= 12 mm/s at any shoulder, elbow, or wrist
joint. These are illustrative coursework thresholds only. A real robot requires
manufacturer-approved limits and a formal safety review. safety_stop() is
deterministic and independent of the ML model.

2. Prediction and maintenance notes
The program creates 300 synthetic labelled historical readings. A Decision Tree
learns NORMAL versus WEAR from torque, temperature, and vibration. Free-text
technician notes are scanned for maintenance keywords. An emergency threshold
takes priority; otherwise predicted wear or relevant notes schedule maintenance.
Warning-range readings lead to MONITOR; otherwise the arm is SAFE.

3. Spoofed-sensor defence
The demo flags very low torque combined with high heat/vibration, or a torque
value far from the median torque across the arm's three joints. This may catch
some crude low-value masking and outliers. It will not catch a coordinated spoof
that keeps readings plausible, and it cannot prove which sensor is correct.
Real deployments need authenticated sensors, time-series checks, redundant
measurements, and calibrated physical limits.

Synthetic coursework demonstration only; do not use to control a real robot.
"""
    (REPORT_DIR / "design_explanation.txt").write_text(explanation, encoding="utf-8")

    print("\\nROBOTIC ARM JOINT HEALTH MONITOR")
    print("--------------------------------")
    print(f"Historical labelled dataset: {DATASET_FILE}")
    print(f"Maintenance notes:           {MAINTENANCE_LOG}")
    print(f"Model report:                {MODEL_REPORT}")
    print(f"Arm decisions:               {OUTPUT_FILE}")
    print(f"Four outcome examples:       {demo_file}")
    print("\\nCurrent decisions:")
    print(pd.read_csv(OUTPUT_FILE)[["arm_id", "decision", "reason"]].to_string(index=False))
    print("\\nRequired outcome demonstration:")
    print(pd.read_csv(demo_file).to_string(index=False))
    print("\\nFinished. Read reports/design_explanation.txt before submission.")


if __name__ == "__main__":
    main()
