from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import requests

crop_model = joblib.load("../models/crop_recommendation.pkl")
soil_encoder = joblib.load("../models/soil_encoder.pkl")
crop_encoder = joblib.load("../models/crop_encoder.pkl")
yield_model = joblib.load("../models/yield_prediction.pkl")
price_model = joblib.load("../models/price_prediction.pkl")


try:
    location_encoder = joblib.load("../models/location_encoder.pkl")
    crop_encoder_reg = joblib.load("../models/crop_encoder_reg.pkl")
except:
    location_encoder, crop_encoder_reg = None, None

OPENWEATHER_API_KEY = "114f0357f8477afe2772b2e408b48cb9"   
OPENWEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather(location):
    """Fetch weather data (temperature, humidity) for a given location"""
    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(OPENWEATHER_URL, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    weather = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0.0)  
    }
    return weather


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌱 Smart Agricultural Advisor API is running with weather integration!"})

@app.route("/recommend", methods=["POST"])
def recommend_crop():
    try:
        data = request.get_json()

        if "soil_type" not in data or "location" not in data:
            return jsonify({"error": "Provide both soil_type and location"}), 400

        soil_type = data["soil_type"]
        location = data["location"]

        weather = get_weather(location)
        if not weather:
            return jsonify({"error": f"Could not fetch weather for {location}"}), 400

        rainfall = float(weather["rainfall"])
        temperature = float(weather["temperature"])
        humidity = float(weather["humidity"])

        soil_encoded = soil_encoder.transform([soil_type])[0]

        X = pd.DataFrame([[soil_encoded, rainfall, temperature, humidity]],
                         columns=["soil_type", "rainfall", "temperature", "humidity"])

        probs = crop_model.predict_proba(X)[0]
        top_indices = np.argsort(probs)[-3:][::-1]
        top_crops = crop_encoder.inverse_transform(top_indices)

        recommendations = [
            {"crop": crop, "probability": round(float(probs[i]), 2)}
            for i, crop in zip(top_indices, top_crops)
        ]

        return jsonify({
            "location": location,
            "weather": weather,
            "recommended_crops": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict_yield_price():
    try:
        data = request.get_json()

        required = ["location", "year", "crop", "soil_type"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        location = str(data["location"])
        year = int(data["year"])
        crop = str(data["crop"])
        soil_type = str(data["soil_type"])

        weather = get_weather(location)
        if not weather:
            return jsonify({"error": f"Could not fetch weather for {location}"}), 400

        rainfall = float(weather["rainfall"])
        temperature = float(weather["temperature"])
        humidity = float(weather["humidity"])

        if location_encoder and crop_encoder_reg:
            location_encoded = location_encoder.transform([location])[0]
            crop_encoded = crop_encoder_reg.transform([crop])[0]
        else:
            location_encoded = pd.Series([location]).astype("category").cat.codes[0]
            crop_encoded = pd.Series([crop]).astype("category").cat.codes[0]

        X = pd.DataFrame([[location_encoded, year, crop_encoded, rainfall, temperature, humidity]],
                         columns=["location", "year", "crop", "rainfall", "temperature", "humidity"])

        yield_pred = yield_model.predict(X)[0]
        price_pred = price_model.predict(X)[0]

        return jsonify({
            "location": location,
            "crop": crop,
            "weather": weather,
            "predicted_yield": round(float(yield_pred), 2),
            "predicted_price": round(float(price_pred), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
