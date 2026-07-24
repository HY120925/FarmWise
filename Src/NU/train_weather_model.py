#Not used in final version 

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

from xgboost import XGBClassifier
from config import DATA_PROCESSED_DIR, MODELS_DIR


def train_weather_model():

    print("Loading processed dataset...")
    df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "crop_training.csv"))

    weather_features = [
        "avg_temp_max",
        "avg_temp_min",
        "temp_range",
        "avg_rainfall",
        "total_rainfall",
        "avg_humidity",
        "humidity_variation",
        "cloud_amt",
        "ws2m_range",
        "ps"
    ]

    X = df[weather_features]
    y = df["crop"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Classes:", encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.2,
        reg_lambda=1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    print("\nTraining weather model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nWeather Model Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(model, os.path.join(MODELS_DIR, "weather_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "weather_encoder.pkl"))
    joblib.dump(weather_features, os.path.join(MODELS_DIR, "weather_features.pkl"))

    print("\nWeather model saved successfully.")


if __name__ == "__main__":
    train_weather_model()
