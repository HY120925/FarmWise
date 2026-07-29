# FarmWise — AI-Powered Crop Recommendation System

Predicts the top crops a farmer should plant based on soil and weather conditions, using a trained XGBoost model served through a live Flask web app.

## Demo

> Add a screenshot or short GIF of the web app here once deployed.
> If you deploy this on Render/Railway, put the live link here too.

## Features

- Predicts top crop recommendations from soil parameters (Nitrogen, Phosphorus, Potassium, pH) and weather data (humidity, rainfall)
- Trained on 1,400 real agricultural samples across 14 crop classes
- Integrates FAO yield data to estimate expected per-hectare yield for each recommended crop
- Simple web interface — enter soil/weather values, get instant predictions

## Tech Stack

- **Model:** XGBoost (hyperparameter-tuned, cross-validated)
- **Backend:** Flask
- **Data processing:** Pandas, NumPy, Scikit-learn
- **Frontend:** HTML/CSS/JS (served via Flask templates)

## Results

- **100% accuracy** on the 14-class crop classification task (test set)
- Model trained and validated with hyperparameter tuning and cross-validation to avoid overfitting

## Project Structure

```
FarmWise/
├── Src/              # Data processing and model training scripts
├── models/           # Trained model artifacts
├── static/           # CSS/JS assets
├── templates/        # HTML templates for the Flask app
├── app.py            # Flask application entry point
└── requirements.txt  # Python dependencies
```

## Setup

```bash
git clone https://github.com/Yogesh-Vats-11/FarmWise.git
cd FarmWise
pip install -r requirements.txt
python app.py
```

Then open the local URL printed in the terminal (usually `http://127.0.0.1:5000`).

## What I'd Improve Next

- Add automated tests for the prediction pipeline
- Deploy a live public demo
- Expand the dataset beyond 1,400 samples to improve generalization
- Add a confidence score display alongside each crop recommendation

## Author

**Yogesh** — B.Tech, AI & Machine Learning, Amity University Gurugram
[GitHub](https://github.com/Yogesh-Vats-11) · [LinkedIn](https://www.linkedin.com/in/yogesh-22a62b27a/)