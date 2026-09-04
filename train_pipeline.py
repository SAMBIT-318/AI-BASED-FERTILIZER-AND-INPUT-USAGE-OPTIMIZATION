import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_crop_recommender(csv_path="Crop_recommendation.csv"):
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join("data", csv_path)):
            csv_path = os.path.join("data", csv_path)
        else:
            print("Fallback: Creating baseline data for crop recommendation.")
            df = pd.DataFrame({
                'N': np.random.uniform(10, 140, 500),
                'P': np.random.uniform(5, 90, 500),
                'K': np.random.uniform(10, 120, 500),
                'temperature': np.random.uniform(15, 38, 500),
                'humidity': np.random.uniform(30, 95, 500),
                'ph': np.random.uniform(5.0, 8.5, 500),
                'rainfall': np.random.uniform(40, 300, 500),
                'label': np.random.choice(['rice', 'maize', 'chickpea', 'cotton', 'coffee'], 500)
            })
            df.to_csv("Crop_recommendation.csv", index=False)
            csv_path = "Crop_recommendation.csv"

    df = pd.read_csv(csv_path)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "crop_model.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "crop_encoder.pkl"))
    print(f"Crop model trained. Test Accuracy: {model.score(X_test, y_test):.4f}")

def train_fertilizer_classifier(csv_path="Fertilizer Prediction.csv"):
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join("data", csv_path)):
            csv_path = os.path.join("data", csv_path)
        else:
            print("Fallback: Creating baseline data for fertilizer recommendation.")
            df = pd.DataFrame({
                'Temparature': np.random.uniform(20, 40, 400),
                'Humidity ': np.random.uniform(40, 90, 400),
                'Moisture': np.random.uniform(20, 70, 400),
                'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], 400),
                'Crop Type': np.random.choice(['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy'], 400),
                'Nitrogen': np.random.uniform(10, 100, 400),
                'Potassium': np.random.uniform(0, 50, 400),
                'Phosphorous': np.random.uniform(0, 50, 400),
                'Fertilizer Name': np.random.choice(['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20'], 400)
            })
            df.to_csv("Fertilizer Prediction.csv", index=False)
            csv_path = "Fertilizer Prediction.csv"

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()

    df['Soil Type'] = le_soil.fit_transform(df['Soil Type'].astype(str))
    df['Crop Type'] = le_crop.fit_transform(df['Crop Type'].astype(str))
    y = le_fert.fit_transform(df['Fertilizer Name'].astype(str))

    feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "fert_model.pkl"))
    joblib.dump(le_soil, os.path.join(MODELS_DIR, "soil_encoder.pkl"))
    joblib.dump(le_crop, os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    joblib.dump(le_fert, os.path.join(MODELS_DIR, "fert_encoder.pkl"))
    print(f"Fertilizer classifier trained. Test Accuracy: {model.score(X_test, y_test):.4f}")

def train_yield_regressor(csv_path="crop_yield.csv"):
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join("data", csv_path)):
            csv_path = os.path.join("data", csv_path)
        else:
            print("Fallback: Creating baseline data for yield regression.")
            df = pd.DataFrame({
                'Crop': np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton'], 400),
                'Area': np.random.uniform(1.0, 10.0, 400),
                'Annual_Rainfall': np.random.uniform(400, 2000, 400),
                'Fertilizer': np.random.uniform(50, 400, 400),
                'Yield': np.random.uniform(1.5, 7.5, 400)
            })
            df.to_csv("crop_yield.csv", index=False)
            csv_path = "crop_yield.csv"

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    crop_col = [c for c in df.columns if 'crop' in c.lower()][0]
    yield_col = [c for c in df.columns if 'yield' in c.lower()][0]

    le_crop = LabelEncoder()
    df['Crop_Enc'] = le_crop.fit_transform(df[crop_col].astype(str))

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != yield_col]
    X = df[num_cols]
    y = df[yield_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
    joblib.dump(num_cols, os.path.join(MODELS_DIR, "yield_features.pkl"))
    joblib.dump(le_crop, os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
    print(f"Yield regressor trained. Test R2: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_crop_recommender()
    train_fertilizer_classifier()
    train_yield_regressor()
