import os
import joblib
import numpy as np
import pandas as pd

from config import MODELS_DIR


def load_models():

    soil_model = joblib.load(os.path.join(MODELS_DIR, "soil_model.pkl"))
    weather_model = joblib.load(os.path.join(MODELS_DIR, "weather_model.pkl"))

    soil_encoder = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))

    soil_features = joblib.load(os.path.join(MODELS_DIR, "soil_features.pkl"))
    weather_features = joblib.load(os.path.join(MODELS_DIR, "weather_features.pkl"))

    return soil_model, weather_model, soil_encoder, soil_features, weather_features


def recommend_crop(input_data):

    soil_model, weather_model, encoder, soil_features, weather_features = load_models()

    df = pd.DataFrame([input_data])

    soil_X = df[soil_features]
    weather_X = df[weather_features]

    soil_probs = soil_model.predict_proba(soil_X)[0]
    weather_probs = weather_model.predict_proba(weather_X)[0]

    final_scores = (0.6 * soil_probs) + (0.4 * weather_probs)

    top_indices = np.argsort(final_scores)[::-1][:3]

    crops = encoder.inverse_transform(top_indices)

    results = []

    for i in range(3):
        results.append({
            "crop": crops[i],
            "score": round(final_scores[top_indices[i]], 3)
        })

    return results


if __name__ == "__main__":

    sample_input = {
        "ph": 6.5,
        "n": 40,
        "p": 30,
        "k": 35,
        "zn": 5,
        "s": 10,
        "gwettop": 0.4,

        "avg_temp_max": 30,
        "avg_temp_min": 18,
        "temp_range": 12,
        "avg_rainfall": 100,
        "total_rainfall": 400,
        "avg_humidity": 60,
        "humidity_variation": 10,
        "cloud_amt": 30,
        "ws2m_range": 4,
        "ps": 1010
    }

    recommendations = recommend_crop(sample_input)

    print("\nTop Crop Recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['crop']} (score: {rec['score']})")


## Predicting the best crop to grow based on soil and weather characteristics using a decision layer that combines outputs from both models.