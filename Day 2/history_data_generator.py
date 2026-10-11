import csv
import random


NUM_READINGS = 250


filename = "sensor_data.csv"


with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

   
    writer.writerow([
        "temp",
        "pressure",
        "vibration",
        "rpm",
        "current",
        "state"
    ])

    for _ in range(NUM_READINGS):
        temp = round(random.uniform(25, 100), 2)
        pressure = round(random.uniform(1, 10), 2)
        vibration = round(random.uniform(0.1, 15), 2)
        rpm = random.randint(500, 3000)
        current = round(random.uniform(1, 20), 2)

       
        state = random.choice(["NORMAL", "WARNING", "FAULT"])

        writer.writerow([
            temp,
            pressure,
            vibration,
            rpm,
            current,
            state
        ])

print(f"{NUM_READINGS} readings generated successfully!")
print(f"Output file: {filename}")