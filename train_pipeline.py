import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = "saved_models"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_all_models():
    np.random.seed(42)
    n_samples = 1500

    # 1. Dataset Generation & Loading
    crop_path = os.path.join(DATA_DIR, "Crop_recommendation.csv")
    if os.path.exists(crop_path):
        df_crop_raw = pd.read_csv(crop_path)
    else:
        df_crop_raw = pd.DataFrame({
            'N': [90, 85, 60, 74, 78, 69, 69, 94, 89, 68],
            'P': [42, 58, 55, 35, 42, 55, 55, 53, 54, 58],
            'K': [43, 41, 44, 40, 42, 38, 38, 40, 38, 38],
            'temperature': [20.8, 21.7, 23.0, 26.4, 20.1, 23.0, 22.7, 20.2, 24.5, 23.6],
            'humidity': [82.0, 80.3, 82.1, 80.1, 81.6, 83.3, 82.6, 82.8, 83.5, 83.0],
            'ph': [6.5, 7.0, 7.8, 6.9, 7.6, 7.0, 5.7, 5.7, 6.6, 6.3],
            'rainfall': [202.9, 226.6, 263.9, 242.8, 262.7, 251.0, 271.3, 226.7, 230.4, 272.8],
            'label': ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate']
        })

    # Train Crop Recommender (Model 1)
    X_crop = df_crop_raw[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_crop = df_crop_raw['label']
    crop_encoder = LabelEncoder().fit(y_crop)
    y_crop_enc = crop_encoder.transform(y_crop)

    crop_model = RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42)
    crop_model.fit(X_crop, y_crop_enc)
    joblib.dump(crop_model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(crop_encoder, os.path.join(MODELS_DIR, "crop_encoder.pkl"))

    # Train Fertilizer Classifier (Model 2)
    fert_path = os.path.join(DATA_DIR, "Fertilizer Prediction.csv")
    if os.path.exists(fert_path):
        df_fert = pd.read_csv(fert_path)
        df_fert.columns = [c.strip() for c in df_fert.columns]
    else:
        df_fert = pd.DataFrame({
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

    soil_encoder = LabelEncoder().fit(df_fert['Soil Type'])
    crop_type_encoder = LabelEncoder().fit(df_fert['Crop Type'])
    fert_encoder = LabelEncoder().fit(df_fert['Fertilizer Name'])

    df_fert['Soil Type'] = soil_encoder.transform(df_fert['Soil Type'])
    df_fert['Crop Type'] = crop_type_encoder.transform(df_fert['Crop Type'])
    y_fert = fert_encoder.transform(df_fert['Fertilizer Name'])
    X_fert = df_fert[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]

    fert_model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
    fert_model.fit(X_fert, y_fert)
    joblib.dump(fert_model, os.path.join(MODELS_DIR, "fert_model.pkl"))
    joblib.dump(soil_encoder, os.path.join(MODELS_DIR, "soil_encoder.pkl"))
    joblib.dump(crop_type_encoder, os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    joblib.dump(fert_encoder, os.path.join(MODELS_DIR, "fert_encoder.pkl"))

    # Train Expected Yield Regressor (Model 3)
    df_yield = pd.DataFrame({
        'N': np.random.uniform(20, 140, n_samples),
        'P': np.random.uniform(10, 90, n_samples),
        'K': np.random.uniform(10, 100, n_samples),
        'ph': np.random.uniform(5.0, 8.5, n_samples),
        'rainfall': np.random.uniform(60, 300, n_samples),
        'crop_type': np.random.choice([0, 1], n_samples)
    })
    df_yield['yield'] = (
        (df_yield['N'] * 0.014) + (df_yield['P'] * 0.012) + (df_yield['K'] * 0.01) +
        (df_yield['rainfall'] * 0.005) - abs(df_yield['ph'] - 6.5) * 0.25 + np.random.normal(0, 0.2, n_samples)
    ).clip(1.2, 8.5)

    X_yield = df_yield[['N', 'P', 'K', 'ph', 'rainfall', 'crop_type']]
    y_yield = df_yield['yield']
    yield_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
    yield_model.fit(X_yield, y_yield)
    yield_crop_encoder = LabelEncoder().fit(['Default Crop', 'Alternative Crop'])

    joblib.dump(yield_model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(list(X_yield.columns), os.path.join(MODELS_DIR, "yield_features.pkl"))
    joblib.dump(yield_crop_encoder, os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))

    # Train Smart Irrigation Volume Predictor (Model 4)
    df_irrig = pd.DataFrame({
        'temperature': np.random.uniform(18, 42, n_samples),
        'humidity': np.random.uniform(25, 95, n_samples),
        'rainfall': np.random.uniform(30, 280, n_samples),
        'soil_encoded': np.random.choice(range(len(soil_encoder.classes_)), n_samples)
    })
    df_irrig['irrigation_mm'] = (
        (df_irrig['temperature'] * 1.8) + ((100 - df_irrig['humidity']) * 0.9) -
        (df_irrig['rainfall'] * 0.35) + np.random.normal(0, 3, n_samples)
    ).clip(15, 180)

    X_irrig = df_irrig[['temperature', 'humidity', 'rainfall', 'soil_encoded']]
    irrig_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
    irrig_model.fit(X_irrig, df_irrig['irrigation_mm'])
    joblib.dump(irrig_model, os.path.join(MODELS_DIR, "irrigation_model.pkl"))

    # Train Future Mandi Crop Price Predictor (Model 5)
    df_price = pd.DataFrame({
        'yield_t_acre': np.random.uniform(1.2, 7.5, n_samples),
        'temperature': np.random.uniform(18, 40, n_samples),
        'rainfall': np.random.uniform(40, 280, n_samples),
        'crop_encoded': np.random.choice(range(len(crop_encoder.classes_)), n_samples)
    })
    df_price['market_price_per_quintal'] = (
        2400 + (df_price['yield_t_acre'] * -90) + (df_price['temperature'] * 20) + np.random.normal(0, 100, n_samples)
    ).clip(1500, 6200)

    X_price = df_price[['yield_t_acre', 'temperature', 'rainfall', 'crop_encoded']]
    price_model = LinearRegression().fit(X_price, df_price['market_price_per_quintal'])
    joblib.dump(price_model, os.path.join(MODELS_DIR, "price_model.pkl"))

    print("All ML models trained and saved to:", MODELS_DIR)

if __name__ == "__main__":
    train_all_models()
