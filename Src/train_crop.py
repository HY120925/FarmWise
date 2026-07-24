
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost import XGBClassifier
from config import DATA_RAW_DIR, MODELS_DIR


def train_crop_model():

    print("Loading dataset...")

    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "crop.csv"))

    df.columns = [col.strip().lower() for col in df.columns]

    df = df.rename(columns={
        "nitrogen": "n",
        "phosphorus": "p",
        "potassium": "k",
        "temperature": "temperature",
        "humidity": "humidity",
        "ph_value": "ph",
        "rainfall": "rainfall",
        "crop": "crop"
    })

    features = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    X = df[features]
    y = df["crop"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("Number of crops:", len(encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist"
    )

    print("\nRunning Cross Validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring="f1_macro"
    )

    print("CV F1 Scores:", cv_scores)
    print("Mean CV F1:", round(np.mean(cv_scores), 3))

    print("\nTraining final model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Results:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 3))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "crop_encoder.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "crop_features.pkl"))

    print("\n Model saved successfully!")


if __name__ == "__main__":
    train_crop_model()

## Crop Recommendation Model Training Script 
## Uses: xgbclassifier, labelencoder, stratifiedkfold, cross_val_score, classification_report