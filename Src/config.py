## Path Management for data and models

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SOIL_WEATHER_PATH = os.path.join(DATA_RAW_DIR, "fao_yield.csv")
AGRI_PATH = os.path.join(DATA_RAW_DIR, "crop.csv")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)