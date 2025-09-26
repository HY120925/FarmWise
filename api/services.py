# ============================
# 🌱 FarmWise Services
# ============================

import joblib
import pandas as pd

# ============================
# 📂 Load models & encoders
# ============================
try:
    crop_model = joblib.load("../models/crop_model.pkl")
    crop_encoder = joblib.load("../models/crop_encoder.pkl")
    yield_model = joblib.load("../models/yield_model.pkl")

    fertilizer_model = joblib.load("../models/fertilizer_used_model.pkl")
    water_model = joblib.load("../models/water_usage_model.pkl")

    encoders = joblib.load("../models/encoders.pkl")
    required_features = joblib.load("../models/features.pkl")
except Exception as e:
    print("⚠️ Warning: Could not load models:", e)
    crop_model = yield_model = fertilizer_model = water_model = None
    crop_encoder = encoders = required_features = None


# ============================
# 🔄 Preprocess Input
# ============================
def preprocess_input(input_dict: dict):
    if required_features is None:
        raise ValueError("❌ No feature list found (features.pkl missing)")

    df = pd.DataFrame([input_dict])

    # encode categorical features
    for col, le in encoders.items():
        if col in df:
            df[col] = le.transform(df[col].astype(str))

    df = df.reindex(columns=required_features, fill_value=0)
    return df


# ============================
# 🌾 Prediction Functions
# ============================
def recommend_crop(input_data: dict):
    df = preprocess_input(input_data)
    pred = crop_model.predict(df)
    crop_name = crop_encoder.inverse_transform(pred)[0]
    return crop_name


def predict_yield(input_data: dict):
    df = preprocess_input(input_data)
    pred = yield_model.predict(df)[0]
    return f"{pred:.2f} tons/ha"


def predict_resources(input_data: dict):
    df = preprocess_input(input_data)
    resources = {}

    if fertilizer_model:
        pred = fertilizer_model.predict(df)[0]
        resources["fertilizer_used"] = f"{pred:.2f} tons/ha"

    if water_model:
        pred = water_model.predict(df)[0]
        resources["water_usage"] = f"{pred:.2f} cubic meters/ha"

    return resources


# ============================
# 📝 Main Advisory
# ============================
def generate_advisory(input_data: dict):
    crop = recommend_crop(input_data)
    yield_val = predict_yield(input_data)
    resources = predict_resources(input_data)

    return {
        "advisor_report": {
            "Recommended Crop": crop,
            "Expected Yield": yield_val,
            "Resource Requirements": {
                "Crop": crop,
                "Fertilizer Used": resources.get("fertilizer_used", "N/A"),
                "Yield Tons": yield_val.replace(" tons/ha", ""),
                "Water Usage": resources.get("water_usage", "N/A"),
            },
        }
    }
