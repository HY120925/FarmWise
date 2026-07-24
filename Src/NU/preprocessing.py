## Feature engineering and preprocessing for crop recommendation dataset

import pandas as pd
from config import SOIL_WEATHER_PATH, DATA_PROCESSED_DIR

def preprocess_crop_dataset():

    df = pd.read_csv(SOIL_WEATHER_PATH)

    if "Soilcolor" in df.columns:
        df = df.drop(columns=["Soilcolor"])

    df = df.rename(columns={"label": "crop"})

    df.columns = [col.strip().lower() for col in df.columns]


    df["avg_temp_max"] = df[
        ["t2m_max-w", "t2m_max-sp", "t2m_max-su", "t2m_max-au"]
    ].mean(axis=1)

    df["avg_temp_min"] = df[
        ["t2m_min-w", "t2m_min-sp", "t2m_min-su", "t2m_min-au"]
    ].mean(axis=1)

    df["temp_range"] = df["avg_temp_max"] - df["avg_temp_min"]

    df["avg_rainfall"] = df[
        ["prectotcorr-w", "prectotcorr-sp", "prectotcorr-su", "prectotcorr-au"]
    ].mean(axis=1)

    df["total_rainfall"] = df[
        ["prectotcorr-w", "prectotcorr-sp", "prectotcorr-su", "prectotcorr-au"]
    ].sum(axis=1)

    df["avg_humidity"] = df[
        ["qv2m-w", "qv2m-sp", "qv2m-su", "qv2m-au"]
    ].mean(axis=1)

    df["humidity_variation"] = df[
        ["qv2m-w", "qv2m-sp", "qv2m-su", "qv2m-au"]
    ].std(axis=1)

    seasonal_cols = [
        "t2m_max-w","t2m_max-sp","t2m_max-su","t2m_max-au",
        "t2m_min-w","t2m_min-sp","t2m_min-su","t2m_min-au",
        "prectotcorr-w","prectotcorr-sp","prectotcorr-su","prectotcorr-au",
        "qv2m-w","qv2m-sp","qv2m-su","qv2m-au"
    ]

    df = df.drop(columns=seasonal_cols)

    output_path = f"{DATA_PROCESSED_DIR}/crop_training.csv"

    df.to_csv(output_path, index=False)

    print(f"Feature engineered Dataset saved in {output_path}.")

if __name__ == "__main__":
    preprocess_crop_dataset()