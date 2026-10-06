import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings('ignore')

MODELS_DIR = "saved_models"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def train_all_models():
    print("🚀 Initializing Smart Kishan Machine Learning Pipeline...")
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. CROP RECOMMENDER (ExtraTreesClassifier)
    # ---------------------------------------------------------
    crop_path = os.path.join(DATA_DIR, "Crop_recommendation.csv")
    if os.path.exists(crop_path):
        print("Training Crop Recommender on real dataset...")
        df_crop = pd.read_csv(crop_path)
    else:
        print("Dataset missing. Generating synthetic Crop data...")
        crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
        records = []
        for crop in crops:
            for _ in range(120):
                records.append({'N': np.random.uniform(0, 140), 'P': np.random.uniform(5, 145), 'K': np.random.uniform(5, 205), 'temperature': np.random.uniform(8, 45), 'humidity': np.random.uniform(14, 100), 'ph': np.random.uniform(3.5, 9.9), 'rainfall': np.random.uniform(20, 298), 'label': crop})
        df_crop = pd.DataFrame(records)

    X_c = df_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y_c = df_crop['label']
    crop_encoder = LabelEncoder()
    y_c_enc = crop_encoder.fit_transform(y_c)
    
    crop_clf = ExtraTreesClassifier(n_estimators=150, random_state=42)
    crop_clf.fit(X_c, y_c_enc)
    
    joblib.dump(crop_clf, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(crop_encoder, os.path.join(MODELS_DIR, "crop_encoder.pkl"))
    print(f"✅ Crop Model Trained. Accuracy: {accuracy_score(y_c_enc, crop_clf.predict(X_c))*100:.2f}%")

    # ---------------------------------------------------------
    # 2. FERTILIZER CLASSIFIER (RandomForestClassifier)
    # ---------------------------------------------------------
    fert_path = os.path.join(DATA_DIR, "Fertilizer Prediction.csv")
    if os.path.exists(fert_path):
        print("Training Fertilizer Model on real dataset...")
        df_f = pd.read_csv(fert_path)
        df_f.columns = [c.strip() for c in df_f.columns]
    else:
        print("Dataset missing. Generating synthetic Fertilizer data...")
        f_records = []
        for fname in ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26']:
            for _ in range(150):
                f_records.append({'Temparature': np.random.uniform(22, 36), 'Humidity': np.random.uniform(45, 75), 'Moisture': np.random.uniform(25, 65), 'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']), 'Crop Type': np.random.choice(['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy']), 'Nitrogen': np.random.uniform(5, 40), 'Potassium': np.random.uniform(0, 20), 'Phosphorous': np.random.uniform(0, 40), 'Fertilizer Name': fname})
        df_f = pd.DataFrame(f_records)

    soil_enc = LabelEncoder().fit(df_f['Soil Type'])
    crop_type_enc = LabelEncoder().fit(df_f['Crop Type'])
    fert_enc = LabelEncoder().fit(df_f['Fertilizer Name'])
    
    df_f['Soil Type'] = soil_enc.transform(df_f['Soil Type'])
    df_f['Crop Type'] = crop_type_enc.transform(df_f['Crop Type'])
    y_f_enc = fert_enc.transform(df_f['Fertilizer Name'])
    
    X_f = df_f[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
    fert_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    fert_clf.fit(X_f, y_f_enc)
    
    joblib.dump(fert_clf, os.path.join(MODELS_DIR, "fert_model.pkl"))
    joblib.dump(soil_enc, os.path.join(MODELS_DIR, "soil_encoder.pkl"))
    joblib.dump(crop_type_enc, os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    joblib.dump(fert_enc, os.path.join(MODELS_DIR, "fert_encoder.pkl"))
    print(f"✅ Fertilizer Model Trained. Accuracy: {accuracy_score(y_f_enc, fert_clf.predict(X_f))*100:.2f}%")

    # ---------------------------------------------------------
    # 3. YIELD PREDICTION REGRESSOR
    # ---------------------------------------------------------
    yield_path = os.path.join(DATA_DIR, "crop_yield.csv")
    if os.path.exists(yield_path):
        print("Training Yield Regressor on real dataset...")
        df_y = pd.read_csv(yield_path)
        X_y = df_y.iloc[:, :-1].select_dtypes(include=[np.number]).fillna(0)
        y_yield = df_y.iloc[:, -1].fillna(0)
        yield_features = list(X_y.columns)
    else:
        print("Training Yield Regressor on Synthetic Data...")
        n_samples = 2000
        df_yield = pd.DataFrame({
            'N': np.random.uniform(20, 140, n_samples),
            'P': np.random.uniform(10, 90, n_samples),
            'K': np.random.uniform(10, 100, n_samples),
            'ph': np.random.uniform(5.2, 8.2, n_samples),
            'rainfall': np.random.uniform(60, 300, n_samples),
            'crop_type': np.random.choice([0, 1], n_samples)
        })
        df_yield['yield'] = ((df_yield['N'] * 0.015) + (df_yield['P'] * 0.012) + (df_yield['K'] * 0.009) + (df_yield['rainfall'] * 0.004) - abs(df_yield['ph'] - 6.5) * 0.22).clip(1.2, 8.5)
        X_y = df_yield[['N', 'P', 'K', 'ph', 'rainfall', 'crop_type']]
        y_yield = df_yield['yield']
        yield_features = list(X_y.columns)

    yield_reg = RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42)
    yield_reg.fit(X_y, y_yield)
    yield_crop_encoder = LabelEncoder().fit(['Default Crop', 'Alternative Crop'])

    joblib.dump(yield_reg, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(yield_features, os.path.join(MODELS_DIR, "yield_features.pkl"))
    joblib.dump(yield_crop_encoder, os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
    print(f"✅ Yield Model Trained. R2 Score: {r2_score(y_yield, yield_reg.predict(X_y)):.4f}")

    # ---------------------------------------------------------
    # 4. SMART IRRIGATION RISK ENGINE
    # ---------------------------------------------------------
    print("Training Smart Irrigation Model...")
    df_irrig = pd.DataFrame({
        'temperature': np.random.uniform(18, 42, 2000),
        'humidity': np.random.uniform(25, 95, 2000),
        'rainfall': np.random.uniform(30, 280, 2000),
        'soil_encoded': np.random.choice(range(len(soil_enc.classes_)), 2000)
    })
    df_irrig['irrigation_mm'] = ((df_irrig['temperature'] * 1.8) + ((100 - df_irrig['humidity']) * 0.85) - (df_irrig['rainfall'] * 0.32) + 12.0).clip(15, 180)

    X_ir = df_irrig[['temperature', 'humidity', 'rainfall', 'soil_encoded']]
    irrig_reg = RandomForestRegressor(n_estimators=80, random_state=42)
    irrig_reg.fit(X_ir, df_irrig['irrigation_mm'])
    joblib.dump(irrig_reg, os.path.join(MODELS_DIR, "irrigation_model.pkl"))
    print("✅ Irrigation Model Trained.")

    # ---------------------------------------------------------
    # 5. FUTURE MARKET PRICE PREDICTOR
    # ---------------------------------------------------------
    print("Training Market Price Forecaster...")
    df_price = pd.DataFrame({
        'yield_t_acre': np.random.uniform(1.2, 7.5, 2000),
        'temperature': np.random.uniform(18, 40, 2000),
        'rainfall': np.random.uniform(40, 280, 2000),
        'crop_encoded': np.random.choice(range(len(crop_encoder.classes_)), 2000)
    })
    df_price['market_price_per_quintal'] = (2500 + (df_price['yield_t_acre'] * -85) + (df_price['temperature'] * 22) + (df_price['crop_encoded'] * 18)).clip(1500, 6200)

    X_p = df_price[['yield_t_acre', 'temperature', 'rainfall', 'crop_encoded']]
    price_reg = LinearRegression().fit(X_p, df_price['market_price_per_quintal'])
    joblib.dump(price_reg, os.path.join(MODELS_DIR, "price_model.pkl"))
    print("✅ Market Price Model Trained.")

    print("🎉 ALL MODELS CACHED SUCCESSFULLY!")

if __name__ == "__main__":
    train_all_models()
