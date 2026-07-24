import os
import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SOIL_WEATHER_PATH = os.path.join(DATA_RAW_DIR, "fao_yield.csv")
AGRI_PATH = os.path.join(DATA_RAW_DIR, "crop.csv")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


crop_model = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
crop_encoder = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
features = joblib.load(os.path.join(MODELS_DIR, "crop_features.pkl"))

yield_map = joblib.load(os.path.join(MODELS_DIR, "yield_map.pkl"))


def run_pipeline(user_input):
  

    area = user_input["area"]

    model_input = user_input.copy()
    del model_input["area"]

    df = pd.DataFrame([model_input])
    X = df[features]

    probs = crop_model.predict_proba(X)[0]
    classes = crop_encoder.classes_

    top_indices = probs.argsort()[::-1][:3]

    results = []

    for idx in top_indices:
        crop = classes[idx]
        confidence = probs[idx]

        crop_key = crop.lower()

        if crop_key in yield_map:
            yield_per_ha = yield_map[crop_key]
            total_yield = yield_per_ha * area
        else:
            yield_per_ha = None
            total_yield = None

        results.append({
            "crop": crop,
            "confidence": round(float(confidence), 3),
            "yield_per_hectare": round(yield_per_ha, 2) if yield_per_ha else None,
            "total_yield": round(total_yield, 2) if total_yield else None
        })

    return results


if __name__ == "__main__":

    sample_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 26,
        "humidity": 65,
        "ph": 6.5,
        "rainfall": 210,
        "area": 2
    }



    output = run_pipeline(sample_input)

    print("\n=====  FARMWISE OUTPUT =====\n")

    for i, res in enumerate(output, 1):
        print(f"{i}. Crop: {res['crop']}")
        print(f"   Confidence: {res['confidence']}")
        print(f"   Yield/ha: {res['yield_per_hectare']}")
        print(f"   Total Yield: {res['total_yield']}\n")