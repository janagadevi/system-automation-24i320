# Robotic Arm Joint Health Monitoring System

**Course:** 23IF02 System Automation — Bonus Challenge

A Python coursework simulation for joint health monitoring. It includes a deterministic emergency safety check, synthetic historical data, Decision Tree wear prediction, free-text maintenance-note scanning, basic sensor-spoof inconsistency checks, and one final decision per arm.

> **Safety note:** This uses synthetic data and illustrative thresholds. Never use it to control real industrial equipment.

## Run it
Install Python 3.10+ and then run:

```bash
python -m pip install -r requirements.txt
python main.py
```

The program generates 300 labelled historical readings, 10 sample maintenance notes, a model report, one decision per arm, four outcome examples, and a design explanation.

## Project files
- `main.py` — complete pipeline
- `data/historical_joint_readings.csv` — generated historical training data
- `logs/maintenance_notes.txt` — 10 sample technician notes
- `reports/arm_decisions.csv` — current final decision per arm
- `reports/four_outcome_examples.csv` — examples of all four required outcomes
- `reports/model_report.txt` — test accuracy and classification report
- `reports/design_explanation.txt` — required written explanation

## Upload to GitHub
Create an empty repository on GitHub. Open a terminal in this folder and run:

```bash
git init
git add .
git commit -m "Add robotic arm joint health monitoring system"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example remote URL with your own repository URL.
