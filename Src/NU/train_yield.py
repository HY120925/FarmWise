import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR


def train_yield_models():

    print("Loading datasets...")

    crop_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "crop_training.csv"))
    agri_df = pd.read_csv(os.path.join(DATA_RAW_DIR, "agriculture.csv"))

    crop_df.columns = [col.strip().lower() for col in crop_df.columns]
    agri_df.columns = [col.strip().lower().replace(" ", "_") for col in agri_df.columns]

    agri_df = agri_df.rename(columns={
        "crop_type": "crop",
        "farm_area(acres)": "farm_area",
        "fertilizer_used(tons)": "fertilizer",
        "yield(tons)": "yield",
        "water_usage(cubic_meters)": "water_usage"
    })

    df = pd.merge(crop_df, agri_df, on="crop", how="inner")

    print("Merged dataset shape:", df.shape)

    df = df[df["farm_area"] > 0]

    df["yield_per_acre"] = df["yield"] / df["farm_area"]
    df["fertilizer_per_acre"] = df["fertilizer"] / df["farm_area"]
    df["water_per_acre"] = df["water_usage"] / df["farm_area"]

    df["yield_per_acre"] = np.log1p(df["yield_per_acre"])
    df["fertilizer_per_acre"] = np.log1p(df["fertilizer_per_acre"])
    df["water_per_acre"] = np.log1p(df["water_per_acre"])

    soil_features = [
        "ph", "n", "p", "k", "zn", "s", "gwettop"
    ]

    weather_features = [
        "avg_temp_max", "avg_temp_min", "temp_range",
        "avg_rainfall", "total_rainfall",
        "avg_humidity", "humidity_variation",
        "cloud_amt", "ws2m_range", "ps"
    ]

    features = soil_features + weather_features

    crop_encoder = LabelEncoder()
    df["crop_encoded"] = crop_encoder.fit_transform(df["crop"])

    X = df[features + ["crop_encoded"]]

    y_yield = df["yield_per_acre"]
    y_fert = df["fertilizer_per_acre"]
    y_water = df["water_per_acre"]

    X_train, X_test, y1_train, y1_test = train_test_split(
        X, y_yield, test_size=0.2, random_state=42
    )

    _, _, y2_train, y2_test = train_test_split(
        X, y_fert, test_size=0.2, random_state=42
    )

    _, _, y3_train, y3_test = train_test_split(
        X, y_water, test_size=0.2, random_state=42
    )

    def build_model():
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_lambda=1,
            random_state=42
        )

    print("\nTraining Yield Model...")
    yield_model = build_model()
    yield_model.fit(X_train, y1_train)

    print("\nTraining Fertilizer Model...")
    fert_model = build_model()
    fert_model.fit(X_train, y2_train)

    print("\nTraining Water Model...")
    water_model = build_model()
    water_model.fit(X_train, y3_train)

    def evaluate(model, y_test, name):
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\n{name} Results:")
        print("RMSE:", round(rmse, 3))
        print("R2 Score:", round(r2, 3))

    evaluate(yield_model, y1_test, "Yield")
    evaluate(fert_model, y2_test, "Fertilizer")
    evaluate(water_model, y3_test, "Water")
    
    joblib.dump(yield_model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(fert_model, os.path.join(MODELS_DIR, "fertilizer_model.pkl"))
    joblib.dump(water_model, os.path.join(MODELS_DIR, "water_model.pkl"))

    joblib.dump(crop_encoder, os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "yield_features.pkl"))

    print("\nAll models saved successfully!")


if __name__ == "__main__":
    train_yield_models()