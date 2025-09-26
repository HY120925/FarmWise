import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = pd.read_csv("../data/processed/final_training_dataset.csv")

print("✅ Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())
print("Unique soil types:", df["soil_type"].unique())
print("Unique regions:", df["region"].unique())

target_crop = "crop"
target_yield = "yield_tons"
target_resources = ["fertilizer_used", "water_usage"]

# ============================
# 🎯 Label Encoding
# ============================
encoders = {}
for col in ["soil_type", "region", "crop"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "../models/encoders.pkl")
print("✅ Encoders trained & saved.")

# ============================
# 🌾 Crop Classification
# ============================

# Drop irrigation_type since we don't use it as input
X = df.drop([target_crop, target_yield] + target_resources + ["irrigation_type"], axis=1)
y_crop = df[target_crop]

crop_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, random_state=42, class_weight="balanced"
)
crop_model.fit(X, y_crop)

joblib.dump(crop_model, "../models/crop_model.pkl")
joblib.dump(encoders["crop"], "../models/crop_encoder.pkl")
print("✅ Crop model trained & saved.")

# ============================
# 🌱 Yield Regression
# ============================
y_yield = df[target_yield]
yield_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
yield_model.fit(X, y_yield)

joblib.dump(yield_model, "../models/yield_model.pkl")
print("✅ Yield model trained & saved.")

# ============================
# 🔧 Resource Models
# ============================
for resource in target_resources:
    y_res = df[resource]
    model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X, y_res)
    joblib.dump(model, f"../models/{resource}_model.pkl")
    print(f"✅ {resource} model trained & saved.")

# ============================
# 💾 Save feature list
# ============================
joblib.dump(list(X.columns), "../models/features.pkl")
print("🎉 All models retrained and saved successfully!")
