import os
import pandas as pd
import joblib
from config import DATA_RAW_DIR, MODELS_DIR


def build_yield_map():

    print("Loading FAO dataset...")

    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "fao_yield.csv"))

    df.columns = [col.strip().lower() for col in df.columns]

    df = df.rename(columns={
        "item": "crop",
        "value": "yield"
    })

    df = df[["crop", "yield"]]


    df["yield"] = df["yield"].astype(str).str.replace(" kg/ha", "", regex=False)

    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")

    df = df.dropna(subset=["yield"])

    df["crop"] = df["crop"].str.lower()

    yield_map = df.groupby("crop")["yield"].mean().to_dict()

    print("Number of crops in yield map:", len(yield_map))

    joblib.dump(yield_map, os.path.join(MODELS_DIR, "yield_map.pkl"))

    print("Yield map saved successfully!")


if __name__ == "__main__":
    build_yield_map()