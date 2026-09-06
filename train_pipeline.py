import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = "saved_models"
DATA_DIR = "data"

def train_crop_recommender():
    os.makedirs(MODELS_DIR, exist_ok=True)
    crop_path = os.path.join(DATA_DIR, "Crop_recommendation.csv")
    
    if os.path.exists(crop_path):
        df = pd.read_csv(crop_path)
    else:
        # Fallback synthetic dataset if CSV is missing
        df = pd.DataFrame({
            'N': [90, 85, 60, 74, 78, 69, 69, 94, 89, 68],
            'P': [42, 58, 55, 35, 42, 55, 55, 53, 54, 58],
            'K': [43, 41, 44, 40, 42, 38, 38, 40, 38, 38],
            'temperature': [20.8, 21.7, 23.0, 26.4, 20.1, 23.0, 22.7, 20.2, 24.5, 23.6],
            'humidity': [82.0, 80.3, 82.1, 80.1, 81.6, 83.3, 82.6, 82.8, 83.5, 83.0],
            'ph': [6.5, 7.0, 7.8, 6.9, 7.6, 7.0, 5.7, 5.7, 6.6, 6.3],
            'rainfall': [202.9, 226.6, 263.9, 242.8, 262.7, 251.0, 271.3, 226.7, 230.4, 272.8],
            'label': ['rice', 'rice', 'rice', 'rice', 'rice', 'rice', 'rice', 'rice', 'rice', 'rice']
        })

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    crop_encoder = LabelEncoder()
    y_encoded = crop_encoder.fit_transform(y)

    # Random Forest with 100 decision trees
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    joblib.dump(model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(crop_encoder, os.path.join(MODELS_DIR, "crop_encoder.pkl"))


def train_fertilizer_classifier():
    os.makedirs(MODELS_DIR, exist_ok=True)
    fert_path = os.path.join(DATA_DIR, "Fertilizer Prediction.csv")

    if os.path.exists(fert_path):
        df = pd.read_csv(fert_path)
        df.columns = [c.strip() for c in df.columns]
    else:
        # Fallback synthetic dataset
        df = pd.DataFrame({
            'Temparature': [26, 25, 29, 34, 32, 26, 25, 28, 26, 29],
            'Humidity': [52, 54, 52, 65, 62, 54, 50, 54, 52, 58],
            'Moisture': [38, 35, 45, 62, 34, 35, 32, 39, 38, 40],
            'Soil Type': ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey', 'Sandy', 'Loamy', 'Black', 'Red', 'Clayey'],
            'Crop Type': ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses'],
            'Nitrogen': [37, 12, 7, 22, 22, 36, 9, 22, 13, 14],
            'Potassium': [0, 0, 9, 0, 0, 0, 10, 0, 0, 7],
            'Phosphorous': [0, 36, 30, 20, 20, 0, 13, 18, 40, 19],
            'Fertilizer Name': ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', 'Urea', 'DAP', '28-28', 'DAP', '14-35-14']
        })

    soil_encoder = LabelEncoder()
    crop_type_encoder = LabelEncoder()
    fert_encoder = LabelEncoder()

    df['Soil Type'] = soil_encoder.fit_transform(df['Soil Type'])
    df['Crop Type'] = crop_type_encoder.fit_transform(df['Crop Type'])
    y = fert_encoder.fit_transform(df['Fertilizer Name'])

    feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    X = df[feature_cols]

    # Random Forest multi-class fertilizer classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(MODELS_DIR, "fert_model.pkl"))
    joblib.dump(soil_encoder, os.path.join(MODELS_DIR, "soil_encoder.pkl"))
    joblib.dump(crop_type_encoder, os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    joblib.dump(fert_encoder, os.path.join(MODELS_DIR, "fert_encoder.pkl"))


def train_yield_regressor():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Calibration dataset for baseline crop yield potential
    df = pd.DataFrame({
        'N': [40, 60, 80, 100, 120, 45, 65, 85, 105, 125],
        'P': [20, 30, 40, 50, 60, 25, 35, 45, 55, 65],
        'K': [20, 30, 40, 50, 60, 25, 35, 45, 55, 65],
        'ph': [6.5, 6.8, 7.0, 6.2, 6.5, 6.4, 6.9, 7.1, 6.0, 6.6],
        'rainfall': [100, 150, 200, 120, 180, 110, 160, 210, 130, 190],
        'crop_type': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'yield': [2.8, 3.6, 4.2, 4.8, 5.1, 3.0, 3.9, 4.5, 4.9, 5.4]
    })

    X = df[['N', 'P', 'K', 'ph', 'rainfall', 'crop_type']]
    y = df['yield']

    # Random Forest regressor for yield estimation
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    yield_crop_encoder = LabelEncoder()
    yield_crop_encoder.fit(['Default Crop', 'Alternative Crop'])

    joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "yield_features.pkl"))
    joblib.dump(yield_crop_encoder, os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
