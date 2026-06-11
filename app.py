
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Load model artifacts
BASE = os.path.dirname(__file__)
model    = joblib.load(os.path.join(BASE, "model/best_model.pkl"))
scaler   = joblib.load(os.path.join(BASE, "model/scaler.pkl"))
features = joblib.load(os.path.join(BASE, "model/features.pkl"))

with open(os.path.join(BASE, "model/meta.json")) as f:
    meta = json.load(f)

SEASON_MAP  = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEATHER_MAP = {1: "Clear/Partly Cloudy", 2: "Mist/Cloudy", 3: "Light Rain/Snow", 4: "Heavy Rain/Fog"}

def build_feature_vector(form):
    hr       = int(form["hr"])
    weekday  = int(form["weekday"])
    holiday  = int(form["holiday"])
    season   = int(form["season"])
    yr       = int(form["yr"])
    mnth     = int(form["mnth"])
    weather  = int(form["weathersit"])
    temp     = float(form["temp"]) / 41.0          # normalise (max 41°C)
    atemp    = float(form["atemp"]) / 50.0
    hum      = float(form["hum"]) / 100.0
    windspeed= float(form["windspeed"]) / 67.0
    working  = 1 if (weekday in range(1, 6) and holiday == 0) else 0

    # Estimated lag/rolling values from typical demand at this hour
    hour_avg = {
        0:20,1:12,2:8,3:6,4:8,5:30,6:80,7:200,8:220,
        9:100,10:80,11:100,12:130,13:120,14:110,15:130,
        16:180,17:230,18:200,19:140,20:100,21:80,22:60,23:40
    }
    lag1  = hour_avg.get(hr, 100) * (1 + 0.3 * yr)
    lag24 = lag1 * np.random.uniform(0.9, 1.1)
    roll3 = lag1 * 1.05
    roll24= lag1 * 0.95

    is_peak    = 1 if hr in [7, 8, 17, 18] else 0
    is_weekend = 1 if weekday in [0, 6] else 0
    temp_hum   = temp * hum

    vec = [
        season, yr, mnth, hr, holiday, weekday, working, weather,
        temp, atemp, hum, windspeed,
        lag1, lag24, roll3, roll24,
        is_peak, is_weekend, temp_hum
    ]
    return np.array(vec).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html", meta=meta)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = build_feature_vector(data)
        pred = int(np.clip(model.predict(X)[0], 1, 977))

        # Demand level label
        if pred < 50:    level, color = "Very Low",  "#64748b"
        elif pred < 150: level, color = "Low",        "#3b82f6"
        elif pred < 300: level, color = "Moderate",   "#10b981"
        elif pred < 500: level, color = "High",        "#f59e0b"
        else:            level, color = "Very High",   "#ef4444"

        tip_map = {
            "Very Low":  "Quiet period — minimal fleet needed.",
            "Low":       "Light demand — standard fleet deployment.",
            "Moderate":  "Steady demand — ensure regular availability.",
            "High":      "Strong demand — increase fleet and reduce rebalancing time.",
            "Very High": "Peak demand — deploy full fleet and monitor station capacity.",
        }

        return jsonify({
            "prediction": pred,
            "level": level,
            "color": color,
            "tip": tip_map[level],
            "season": SEASON_MAP.get(int(data["season"]), ""),
            "weather": WEATHER_MAP.get(int(data["weathersit"]), ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/metrics")
def metrics():
    return jsonify(meta)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
