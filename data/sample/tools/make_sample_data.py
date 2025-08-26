# tools/make_sample_data.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
SAMPLE.mkdir(parents=True, exist_ok=True)

# Çok minik oyuncak veri (2 yarış / 4 sonuç)
drivers = pd.DataFrame([
    {"driverId": 1, "driverRef": "max_verstappen", "code": "VER", "forename": "Max", "surname": "Verstappen", "nationality": "Dutch", "url": "-"},
    {"driverId": 2, "driverRef": "lewis_hamilton", "code": "HAM", "forename": "Lewis", "surname": "Hamilton", "nationality": "British", "url": "-"},
])

constructors = pd.DataFrame([
    {"constructorId": 1, "constructorRef": "red_bull", "name": "Red Bull", "nationality": "Austrian", "url": "-"},
    {"constructorId": 2, "constructorRef": "mercedes", "name": "Mercedes", "nationality": "German", "url": "-"},
])

races = pd.DataFrame([
    {"raceId": 101, "year": 2023, "round": 1, "circuitId": 1, "name": "Sample GP", "date": "2023-03-01", "time": "14:00:00", "url": "-"},
    {"raceId": 102, "year": 2023, "round": 2, "circuitId": 1, "name": "Sample GP 2", "date": "2023-03-10", "time": "14:00:00", "url": "-"},
])

results = pd.DataFrame([
    {"resultId": 1001, "raceId": 101, "driverId": 1, "constructorId": 1, "grid": 1, "position": 1, "positionOrder": 1, "points": 25, "laps": 57, "statusId": 1},
    {"resultId": 1002, "raceId": 101, "driverId": 2, "constructorId": 2, "grid": 2, "position": 2, "positionOrder": 2, "points": 18, "laps": 57, "statusId": 1},
    {"resultId": 1003, "raceId": 102, "driverId": 1, "constructorId": 1, "grid": 2, "position": 2, "positionOrder": 2, "points": 18, "laps": 56, "statusId": 1},
    {"resultId": 1004, "raceId": 102, "driverId": 2, "constructorId": 2, "grid": 1, "position": 1, "positionOrder": 1, "points": 25, "laps": 56, "statusId": 1},
])

circuits = pd.DataFrame([
    {"circuitId": 1, "circuitRef": "sample", "name": "Sample Circuit", "location": "Nowhere", "country": "XX", "lat": 0.0, "lng": 0.0, "alt": 0, "url": "-"},
])

# Kaydet
drivers.to_csv(SAMPLE / "drivers.csv", index=False)
constructors.to_csv(SAMPLE / "constructors.csv", index=False)
races.to_csv(SAMPLE / "races.csv", index=False)
results.to_csv(SAMPLE / "results.csv", index=False)
circuits.to_csv(SAMPLE / "circuits.csv", index=False)

print(f"Sample CSV'ler yazıldı → {SAMPLE}")