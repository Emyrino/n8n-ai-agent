from flask import Flask, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import requests, random, os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable cross-origin access so n8n or other services can call this API

# URL for Mega Millions historical data (NY Lottery API)
DATA_URL = "https://data.ny.gov/resource/5xaw-6ayf.json?$limit=1000"

def fetch_draws():
    try:
        res = requests.get(DATA_URL)
        res.raise_for_status()
        data = res.json()
        draws = []
        for d in data:
            nums = [int(n) for n in d["winning_numbers"].split()]
            mega = int(d["mega_ball"])
            draws.append(nums + [mega])
        return draws
    except Exception as e:
        print(f"Error fetching draws: {e}")
        return []


def train_and_predict(draws):
    df = pd.DataFrame(draws, columns=["N1", "N2", "N3", "N4", "N5", "Mega"])
    X, y = df[["N1", "N2", "N3", "N4", "N5"]], df["Mega"]

    # Model to learn Mega Ball pattern
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    # Predict new Mega number based on pattern
    if len(X) >= 50:
        mean_input = np.mean(X.tail(50).values, axis=0).reshape(1, -1)
    else:
        mean_input = np.mean(X.values, axis=0).reshape(1, -1)

    predicted_mega = int(round(model.predict(mean_input)[0]))
    predicted_mega = max(1, min(predicted_mega, 25))

    # Generate weighted main numbers
    freq = pd.Series(np.concatenate(X.values)).value_counts(normalize=True)
    nums = np.random.choice(freq.index, size=5, replace=False, p=freq.values)
    nums = sorted(nums)

    confidence = round(freq.loc[nums].mean() * 100, 2)
    return nums, predicted_mega, confidence


@app.route("/predict", methods=["GET"])
def predict():
    draws = fetch_draws()
    if not draws:
        return jsonify({"error": "No data"}), 500

    nums, mega, confidence = train_and_predict(draws)
    return jsonify({
        "prediction": list(map(int, nums)),
        "megaBall": int(mega),
        "confidence": confidence,
        "drawsUsed": len(draws),
        "model": "GradientBoosting (AI Pattern Learner)"
    })


if __name__ == "__main__":
    # Render assigns a dynamic port, so we use the environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
