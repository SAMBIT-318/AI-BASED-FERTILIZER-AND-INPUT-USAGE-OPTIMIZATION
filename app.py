import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from optimizer import optimize_fertilizer_blend
from train_pipeline import train_crop_recommender, train_fertilizer_classifier, train_yield_regressor

st.set_page_config(page_title="AI Precision Fertilizer & Input Optimizer", page_icon="🌾", layout="wide")

MODELS_DIR = "saved_models"

def ensure_models():
    if not os.path.exists(os.path.join(MODELS_DIR, "crop_model.pkl")):
        train_crop_recommender()
    if not os.path.exists(os.path.join(MODELS_DIR, "fert_model.pkl")):
        train_fertilizer_classifier()
    if not os.path.exists(os.path.join(MODELS_DIR, "yield_model.pkl")):
        train_yield_regressor()

ensure_models()

# Load models and encoders
crop_model = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
crop_encoder = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))

fert_model = joblib.load(os.path.join(MODELS_DIR, "fert_model.pkl"))
soil_encoder = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))
crop_type_encoder = joblib.load(os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
fert_encoder = joblib.load(os.path.join(MODELS_DIR, "fert_encoder.pkl"))

yield_model = joblib.load(os.path.join(MODELS_DIR, "yield_model.pkl"))
yield_features = joblib.load(os.path.join(MODELS_DIR, "yield_features.pkl"))
yield_crop_encoder = joblib.load(os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))

st.title("🌾 AI Based Fertilizer and Input Usage Optimization")
st.markdown("Precision Soil Chemistry Diagnostics • Multi-Model Inference • Budget-Constrained Linear Programming")

# Sidebar Controls
st.sidebar.header("📍 Farm & Soil Test Inputs")
soil_n = st.sidebar.slider("Soil Nitrogen (N) [mg/kg]", 0.0, 140.0, 50.0)
soil_p = st.sidebar.slider("Soil Phosphorus (P) [mg/kg]", 5.0, 100.0, 30.0)
soil_k = st.sidebar.slider("Soil Potassium (K) [mg/kg]", 5.0, 140.0, 35.0)
soil_ph = st.sidebar.slider("Soil pH", 4.5, 9.0, 6.5, 0.1)
soil_moist = st.sidebar.slider("Soil Moisture (%)", 10.0, 90.0, 45.0)

st.sidebar.header("🌦️ Weather & Parcel Metadata")
temp = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 26.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 68.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 400.0, 150.0)

soil_type_list = list(soil_encoder.classes_)
crop_type_list = list(crop_type_encoder.classes_)

sel_soil = st.sidebar.selectbox("Soil Texture Type", soil_type_list)
sel_crop = st.sidebar.selectbox("Current Cultivated Crop", crop_type_list)
land_area = st.sidebar.number_input("Field Acreage / Area (Hectares)", 0.5, 50.0, 2.0, 0.5)
budget_cap = st.sidebar.number_input("Farmer Budget / Wallet Cap (INR / $)", 1000.0, 500000.0, 15000.0, 500.0)

tab1, tab2, tab3 = st.tabs(["🚀 Complete Advisory & Optimization", "📊 Dataset Analytics", "📑 Architecture & References"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. AI Diagnostic & Suitability Prediction")
        
        # Crop Recommendation
        crop_input = pd.DataFrame([{
            'N': soil_n, 'P': soil_p, 'K': soil_k,
            'temperature': temp, 'humidity': humidity,
            'ph': soil_ph, 'rainfall': rainfall
        }])
        pred_crop_idx = crop_model.predict(crop_input)[0]
        rec_crop_name = crop_encoder.inverse_transform([pred_crop_idx])[0]

        # Fertilizer Type Classification
        fert_input = pd.DataFrame([{
            'Temparature': temp, 'Humidity': humidity, 'Moisture': soil_moist,
            'Soil Type': soil_encoder.transform([sel_soil])[0],
            'Crop Type': crop_type_encoder.transform([sel_crop])[0],
            'Nitrogen': soil_n, 'Potassium': soil_k, 'Phosphorous': soil_p
        }])
        pred_fert_idx = fert_model.predict(fert_input)[0]
        rec_fert_class = fert_encoder.inverse_transform([pred_fert_idx])[0]

        # Deficit calculation
        benchmark_npk = {'N': 110.0, 'P': 55.0, 'K': 60.0}
        deficit_n = max(0.0, benchmark_npk['N'] - (soil_n * 0.6))
        deficit_p = max(0.0, benchmark_npk['P'] - (soil_p * 0.5))
        deficit_k = max(0.0, benchmark_npk['K'] - (soil_k * 0.5))

        c1, c2 = st.columns(2)
        c1.metric("Agronomic Best Crop", rec_crop_name.capitalize())
        c2.metric("Dominant Fertilizer Needed", str(rec_fert_class))

        st.markdown("**Calculated Macronutrient Deficits (per Hectare):**")
        d1, d2, d3 = st.columns(3)
        d1.metric("N Deficit", f"{deficit_n:.1f} kg/ha")
        d2.metric("P Deficit", f"{deficit_p:.1f} kg/ha")
        d3.metric("K Deficit", f"{deficit_k:.1f} kg/ha")

    with col2:
        st.subheader("2. Budget-Constrained Linear Programming (PuLP)")
        opt = optimize_fertilizer_blend(
            req_n=deficit_n,
            req_p=deficit_p,
            req_k=deficit_k,
            budget_cap=budget_cap,
            land_area=land_area
        )

        st.success(f"Solver Outcome: **{opt['status']}**")
        sc1, sc2 = st.columns(2)
        sc1.metric("Total Optimized Cost", f"${opt['total_cost']:,.2f}")
        sc2.metric("Wallet Capacity Used", f"{opt['budget_utilized_pct']}%")

        st.markdown("#### Recommended Input Dosage Breakdown")
        breakdown = {
            "Input Resource": ["Urea (Synthetic N)", "DAP (Phosphatic)", "MOP (Potash)", "Complex 14-35-14", "Organic Compost"],
            "Total Quantity (kg)": [opt['urea_kg'], opt['dap_kg'], opt['mop_kg'], opt['complex_kg'], opt['compost_kg']],
            "Per Hectare Dose (kg/ha)": [
                round(opt['urea_kg'] / land_area, 2),
                round(opt['dap_kg'] / land_area, 2),
                round(opt['mop_kg'] / land_area, 2),
                round(opt['complex_kg'] / land_area, 2),
                round(opt['compost_kg'] / land_area, 2)
            ]
        }
        st.dataframe(pd.DataFrame(breakdown), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Nutrient Delivery Fulfillment vs. Demand")
    nc1, nc2, nc3 = st.columns(3)
    nc1.metric("Total N Supplied", f"{opt['supplied_n']} kg", delta=f"Target: {opt['target_n']} kg")
    nc2.metric("Total P Supplied", f"{opt['supplied_p']} kg", delta=f"Target: {opt['target_p']} kg")
    nc3.metric("Total K Supplied", f"{opt['supplied_k']} kg", delta=f"Target: {opt['target_k']} kg")

    if opt['unmet_n'] > 0 or opt['unmet_p'] > 0 or opt['unmet_k'] > 0:
        st.warning(f"⚠️ Budget Cap Reached: Unmet deficits — N: {opt['unmet_n']} kg, P: {opt['unmet_p']} kg, K: {opt['unmet_k']} kg.")
    else:
        st.info("✅ Full nutrient requirement satisfied within budget limits without excess chemical leaching.")

with tab2:
    st.subheader("Dataset Exploratory Insights")
    data_sel = st.radio("Select dataset to view", ["Crop Recommendation Data", "Fertilizer Data"])
    if data_sel == "Crop Recommendation Data" and os.path.exists("Crop_recommendation.csv"):
        st.dataframe(pd.read_csv("Crop_recommendation.csv").head(15), use_container_width=True)
    elif data_sel == "Fertilizer Data" and os.path.exists("Fertilizer Prediction.csv"):
        st.dataframe(pd.read_csv("Fertilizer Prediction.csv").head(15), use_container_width=True)
    else:
        st.info("Upload dataset files to the directory to explore full records.")

with tab3:
    st.markdown("""
    ### Project Architecture
    - **Data Pipeline**: Preprocessing, categorical encoding, and feature scaling from Indian agronomic datasets.
    - **Machine Learning**: Random Forest and XGBoost classifiers for crop and fertilizer prediction.
    - **Optimization**: PuLP linear programming modeling bounded multi-nutrient allocation within economic constraints.
    - **Institution**: Centurion University of Technology and Management
    - **Guide**: Dr. Sujata Chakravarty
    """)
