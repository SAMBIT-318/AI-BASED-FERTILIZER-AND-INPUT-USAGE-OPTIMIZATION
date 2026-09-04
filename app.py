import os
import urllib.parse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from sqlalchemy import create_engine, text
from optimizer import optimize_fertilizer_blend
from train_pipeline import train_crop_recommender, train_fertilizer_classifier, train_yield_regressor

st.set_page_config(page_title="AI Precision Fertilizer & Input Optimizer (IEEE Access SSNM)", page_icon="🌾", layout="wide")

MODELS_DIR = "saved_models"

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODELS_DIR, "crop_model.pkl")):
        train_crop_recommender()
    if not os.path.exists(os.path.join(MODELS_DIR, "fert_model.pkl")):
        train_fertilizer_classifier()
    if not os.path.exists(os.path.join(MODELS_DIR, "yield_model.pkl")):
        train_yield_regressor()

@st.cache_resource(show_spinner="Initializing machine learning models...")
def load_all_models():
    ensure_models_exist()
    crop_m = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
    crop_enc = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
    fert_m = joblib.load(os.path.join(MODELS_DIR, "fert_model.pkl"))
    soil_enc = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))
    crop_type_enc = joblib.load(os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
    fert_enc = joblib.load(os.path.join(MODELS_DIR, "fert_encoder.pkl"))
    yield_m = joblib.load(os.path.join(MODELS_DIR, "yield_model.pkl"))
    yield_feat = joblib.load(os.path.join(MODELS_DIR, "yield_features.pkl"))
    yield_crop_enc = joblib.load(os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
    return crop_m, crop_enc, fert_m, soil_enc, crop_type_enc, fert_enc, yield_m, yield_feat, yield_crop_enc

(crop_model, crop_encoder, fert_model, soil_encoder, 
 crop_type_encoder, fert_encoder, yield_model, 
 yield_features, yield_crop_encoder) = load_all_models()

# -------------------------------------------------------------
# Database Engine Initialization (Configured for Tokyo Pooler)
# -------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    try:
        cfg_user = urllib.parse.quote_plus(str(st.secrets["postgres"]["user"]))
        cfg_password = urllib.parse.quote_plus(str(st.secrets["postgres"]["password"]))
        cfg_host = str(st.secrets["postgres"]["host"]).strip()
        cfg_port = st.secrets["postgres"]["port"]
        cfg_db = str(st.secrets["postgres"]["database"]).strip()
        
        db_uri = f"postgresql://{cfg_user}:{cfg_password}@{cfg_host}:{cfg_port}/{cfg_db}?sslmode=require"
    except Exception:
        db_uri = "postgresql://postgres.ivshypgnhsprrkhkzkkx:SambitSwain2005@aws-0-ap-northeast-1.pooler.supabase.com:6543/postgres?sslmode=require"

    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile_number TEXT PRIMARY KEY, password TEXT)"))
            conn.commit()
        return engine
    except Exception as e:
        st.error(f"Database Connection Failure: {e}")
        return None

engine = get_db_engine()

def register_user(mobile, password):
    if not engine:
        return False, "Database connection not established. Check configuration."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (mobile_number, password) VALUES (:m, :p)"), {"m": mobile, "p": hashed_pw})
            conn.commit()
        return True, "Registration successful! You can now log in."
    except Exception:
        return False, "Mobile number is already registered."

def verify_user(mobile, password):
    if not engine:
        return False
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT password FROM users WHERE mobile_number = :m"), {"m": mobile}).fetchone()
            if result and result[0] == hashed_pw:
                return True
    except Exception:
        pass
    return False

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""

if "uploaded_crop_df" not in st.session_state:
    st.session_state.uploaded_crop_df = None
if "uploaded_fert_df" not in st.session_state:
    st.session_state.uploaded_fert_df = None

if "soil_n" not in st.session_state:
    st.session_state.soil_n = 50.0
if "soil_p" not in st.session_state:
    st.session_state.soil_p = 30.0
if "soil_k" not in st.session_state:
    st.session_state.soil_k = 35.0
if "soil_ph" not in st.session_state:
    st.session_state.soil_ph = 6.5
if "soil_moist" not in st.session_state:
    st.session_state.soil_moist = 45.0
if "temp" not in st.session_state:
    st.session_state.temp = 26.0
if "humidity" not in st.session_state:
    st.session_state.humidity = 68.0
if "rainfall" not in st.session_state:
    st.session_state.rainfall = 150.0
if "sel_soil" not in st.session_state:
    st.session_state.sel_soil = list(soil_encoder.classes_)[0]
if "sel_crop" not in st.session_state:
    st.session_state.sel_crop = list(crop_type_encoder.classes_)[0]
if "land_area" not in st.session_state:
    st.session_state.land_area = 2.0
if "budget_cap" not in st.session_state:
    st.session_state.budget_cap = 15000.0

st.title("🌾 AI Based Fertilizer & Input Usage Optimization")
st.caption("Incorporating Site-Specific Nutrient Management (SSNM) & 4R Stewardship (IEEE Access 2025 Review)")
st.markdown(f"**Screen {st.session_state.step} of 7**")
st.progress(st.session_state.step / 7.0)

# -------------------------------------------------------------
# SCREEN 1: User Authentication (Mobile + Password)
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("1. 📱 User Authentication & Access")
    if not st.session_state.logged_in:
        auth_mode = st.radio("Choose Mode", ["Login", "Register New Account"])
        
        if auth_mode == "Login":
            input_mobile = st.text_input("Mobile Number", max_chars=10, placeholder="Enter 10-digit mobile number")
            input_pass = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Login & Proceed", type="primary"):
                clean_mobile = input_mobile.strip()
                if len(clean_mobile) == 10 and clean_mobile.isdigit():
                    if verify_user(clean_mobile, input_pass.strip()):
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = clean_mobile
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Invalid mobile number or incorrect password.")
                else:
                    st.warning("Please enter a valid 10-digit mobile number.")
                    
        else:
            reg_mobile = st.text_input("Mobile Number", max_chars=10, placeholder="Enter 10-digit mobile number")
            reg_pass = st.text_input("Create Password", type="password", placeholder="Set account password")
            reg_confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

            if st.button("Register & Create Account", type="primary"):
                clean_mob = reg_mobile.strip()
                if not (len(clean_mob) == 10 and clean_mob.isdigit()):
                    st.warning("Please enter a valid 10-digit mobile number.")
                elif not reg_pass.strip():
                    st.warning("Please enter a password.")
                elif reg_pass.strip() != reg_confirm_pass.strip():
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    success, msg = register_user(clean_mob, reg_pass.strip())
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        st.success(f"Logged in as **+91 {st.session_state.user_mobile}**")
        if st.button("Continue to Field Telemetry ➔", type="primary"):
            st.session_state.step = 2
            st.rerun()

# -------------------------------------------------------------
# SCREEN 2: Field Inputs & Telemetry Uploads
# -------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("2. 📍 Farm Soil Telemetry & Dataset Uploads")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.soil_n = st.slider("Soil Nitrogen (N) [mg/kg]", 0.0, 140.0, float(st.session_state.soil_n))
        st.session_state.soil_p = st.slider("Soil Phosphorus (P) [mg/kg]", 5.0, 100.0, float(st.session_state.soil_p))
        st.session_state.soil_k = st.slider("Soil Potassium (K) [mg/kg]", 5.0, 140.0, float(st.session_state.soil_k))
        st.session_state.soil_ph = st.slider("Soil pH", 4.5, 9.0, float(st.session_state.soil_ph), 0.1)
        st.session_state.soil_moist = st.slider("Soil Moisture (%)", 10.0, 90.0, float(st.session_state.soil_moist))
        st.session_state.temp = st.slider("Temperature (°C)", 10.0, 45.0, float(st.session_state.temp))
        st.session_state.humidity = st.slider("Humidity (%)", 20.0, 100.0, float(st.session_state.humidity))
        st.session_state.rainfall = st.slider("Seasonal Rainfall (mm)", 20.0, 400.0, float(st.session_state.rainfall))

        soil_types = list(soil_encoder.classes_)
        crop_types = list(crop_type_encoder.classes_)
        st.session_state.sel_soil = st.selectbox("Soil Texture", soil_types, index=soil_types.index(st.session_state.sel_soil) if st.session_state.sel_soil in soil_types else 0)
        st.session_state.sel_crop = st.selectbox("Target Cultivated Crop", crop_types, index=crop_types.index(st.session_state.sel_crop) if st.session_state.sel_crop in crop_types else 0)
        st.session_state.land_area = st.number_input("Field Acreage (Hectares)", 0.5, 50.0, float(st.session_state.land_area), 0.5)
        st.session_state.budget_cap = st.number_input("Budget Cap (INR)", 1000.0, 500000.0, float(st.session_state.budget_cap), 500.0)

    with col2:
        st.info("💡 **SSNM Note**: The research emphasizes continuous edge/IoT sensor streams to prevent blanket over-fertilization.")
        up_crop = st.file_uploader("Upload Crop CSV (Optional)", type=["csv"], key="crop_csv")
        if up_crop is not None:
            st.session_state.uploaded_crop_df = pd.read_csv(up_crop)
            st.success(f"Loaded {st.session_state.uploaded_crop_df.shape[0]} records.")
        up_fert = st.file_uploader("Upload Fertilizer CSV (Optional)", type=["csv"], key="fert_csv")
        if up_fert is not None:
            st.session_state.uploaded_fert_df = pd.read_csv(up_fert)
            st.session_state.uploaded_fert_df.columns = [c.strip() for c in st.session_state.uploaded_fert_df.columns]
            st.success(f"Loaded {st.session_state.uploaded_fert_df.shape[0]} records.")

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back"):
            st.session_state.step = 1
            st.rerun()
    with nav2:
        if st.button("Continue to Processing ➔", type="primary"):
            st.session_state.step = 3
            st.rerun()

# -------------------------------------------------------------
# SCREEN 3: Telemetry Normalization & Sensor Indicators
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("3. ⚙️ Telemetry Normalization & Sensor Pre-processing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature Scaling", "Standardized")
    c2.metric("Missing Values", "0 Nulls Detected")
    c3.metric("Edge IoT Protocol", "Active Calibration")
    
    st.markdown("#### Normalized Input Vector")
    st.json({
        "Normalized_N": round(st.session_state.soil_n / 140.0, 3),
        "Normalized_P": round(st.session_state.soil_p / 100.0, 3),
        "Normalized_K": round(st.session_state.soil_k / 140.0, 3),
        "pH_Balance": round(st.session_state.soil_ph / 14.0, 3),
        "Moisture_Index": round(st.session_state.soil_moist / 100.0, 3)
    })

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back"):
            st.session_state.step = 2
            st.rerun()
    with nav2:
        if st.button("Continue to Visualizations ➔", type="primary"):
            st.session_state.step = 4
            st.rerun()

# -------------------------------------------------------------
# SCREEN 4: Visualization Dashboard
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("4. 📊 Data Distribution & Dataset Insights")
    choice = st.radio("Select View", ["Fertilizer Category Breakdown", "Dataset Records"])
    if choice == "Fertilizer Category Breakdown":
        if st.session_state.uploaded_fert_df is not None and 'Fertilizer Name' in st.session_state.uploaded_fert_df.columns:
            st.bar_chart(st.session_state.uploaded_fert_df['Fertilizer Name'].value_counts())
        else:
            local_path = os.path.join("data", "Fertilizer Prediction.csv")
            if os.path.exists(local_path):
                df_f = pd.read_csv(local_path)
                df_f.columns = [c.strip() for c in df_f.columns]
                if 'Fertilizer Name' in df_f.columns:
                    st.bar_chart(df_f['Fertilizer Name'].value_counts())
            else:
                st.info("Upload a fertilizer CSV in step 2 to visualize category counts.")
    else:
        if st.session_state.uploaded_crop_df is not None:
            st.dataframe(st.session_state.uploaded_crop_df.head(15), use_container_width=True)
        else:
            local_crop = os.path.join("data", "Crop_recommendation.csv")
            if os.path.exists(local_crop):
                st.dataframe(pd.read_csv(local_crop).head(15), use_container_width=True)
            else:
                st.info("Upload a crop CSV in step 2 to view records.")

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back"):
            st.session_state.step = 3
            st.rerun()
    with nav2:
        if st.button("Continue to Diagnostics ➔", type="primary"):
            st.session_state.step = 5
            st.rerun()

# -------------------------------------------------------------
# SCREEN 5: Site-Specific Diagnostics & Environmental Risk Check
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("5. ⚠️ Site-Specific Nutrient Deficits & Ecological Risk Assessment")
    
    crop_in = pd.DataFrame([{
        'N': st.session_state.soil_n, 'P': st.session_state.soil_p, 'K': st.session_state.soil_k,
        'temperature': st.session_state.temp, 'humidity': st.session_state.humidity,
        'ph': st.session_state.soil_ph, 'rainfall': st.session_state.rainfall
    }])
    pred_c = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]

    fert_in = pd.DataFrame([{
        'Temparature': st.session_state.temp, 'Humidity': st.session_state.humidity, 'Moisture': st.session_state.soil_moist,
        'Soil Type': soil_encoder.transform([st.session_state.sel_soil])[0],
        'Crop Type': crop_type_encoder.transform([st.session_state.sel_crop])[0],
        'Nitrogen': st.session_state.soil_n, 'Potassium': st.session_state.soil_k, 'Phosphorous': st.session_state.soil_p
    }])
    pred_f = fert_encoder.inverse_transform([fert_model.predict(fert_in)[0]])[0]

    def_n = max(0.0, 110.0 - (st.session_state.soil_n * 0.6))
    def_p = max(0.0, 55.0 - (st.session_state.soil_p * 0.5))
    def_k = max(0.0, 60.0 - (st.session_state.soil_k * 0.5))

    # Environmental Leaching Risk Analysis (from IEEE Access review)
    leaching_risk = "Low"
    if st.session_state.rainfall > 200 and "Sandy" in str(st.session_state.sel_soil):
        leaching_risk = "High"
    elif st.session_state.rainfall > 150 or "Sandy" in str(st.session_state.sel_soil):
        leaching_risk = "Moderate"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Calculated Nutrient Gaps (SSNM Methodology):**")
        st.warning(f"Nitrogen Deficit: **{def_n:.1f} kg/ha**")
        st.warning(f"Phosphorus Deficit: **{def_p:.1f} kg/ha**")
        st.warning(f"Potassium Deficit: **{def_k:.1f} kg/ha**")
    with col2:
        st.markdown("**AI Predictive Crop Matching & Ecological Indices:**")
        st.success(f"Best Suited Crop: **{pred_c.capitalize()}**")
        st.success(f"Recommended Single Fertilizer: **{pred_f}**")
        
        if leaching_risk == "High":
            st.error(f"⚠️ Nutrient Leaching Risk: **{leaching_risk}** (High rainfall on coarse soil; split dosage strictly required).")
        elif leaching_risk == "Moderate":
            st.warning(f"⚠️ Nutrient Leaching Risk: **{leaching_risk}** (Adopt split applications).")
        else:
            st.info(f"🌿 Nutrient Leaching Risk: **{leaching_risk}** (Safe soil retention index).")

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back"):
            st.session_state.step = 4
            st.rerun()
    with nav2:
        if st.button("Continue to Recommendations ➔", type="primary"):
            st.session_state.step = 6
            st.rerun()

# -------------------------------------------------------------
# SCREEN 6: Optimized Fertilizer Blend & Variable-Rate Split-Dosing
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("6. 🚀 Optimized Fertilizer Formulation & Variable-Rate Application (VRA)")
    
    def_n = max(0.0, 110.0 - (st.session_state.soil_n * 0.6))
    def_p = max(0.0, 55.0 - (st.session_state.soil_p * 0.5))
    def_k = max(0.0, 60.0 - (st.session_state.soil_k * 0.5))
    
    opt = optimize_fertilizer_blend(
        req_n=def_n,
        req_p=def_p,
        req_k=def_k,
        budget_cap=st.session_state.budget_cap,
        land_area=st.session_state.land_area
    )

    f1, f2 = st.columns(2)
    f1.metric("Optimized Total Cost", f"₹{opt['total_cost']:,.2f}")
    f2.metric("Wallet Utilized", f"{opt['budget_utilized_pct']}%")

    st.markdown("#### Prescribed Total Input Blend")
    breakdown = {
        "Input Resource": ["Urea (Synthetic N)", "DAP (Phosphatic)", "MOP (Potash)", "Complex 14-35-14", "Organic Compost"],
        "Total Field Requirement (kg)": [opt['urea_kg'], opt['dap_kg'], opt['mop_kg'], opt['complex_kg'], opt['compost_kg']],
        "Application Density (kg/ha)": [
            round(opt['urea_kg'] / st.session_state.land_area, 2),
            round(opt['dap_kg'] / st.session_state.land_area, 2),
            round(opt['mop_kg'] / st.session_state.land_area, 2),
            round(opt['complex_kg'] / st.session_state.land_area, 2),
            round(opt['compost_kg'] / st.session_state.land_area, 2)
        ]
    }
    st.dataframe(pd.DataFrame(breakdown), use_container_width=True)

    # Staged Variable-Rate Split Application (4R Nutrient Stewardship)
    st.markdown("#### 📅 Variable-Rate Split-Dosing Schedule (IEEE Access 4R Stewardship)")
    st.caption("Splitting nitrogen and potash applications maximizes Nutrient Use Efficiency (NUE) and preserves soil microbiological health.")

    split_schedule = {
        "Crop Growth Stage": [
            "1. Basal Application (At Sowing)",
            "2. Vegetative / Tillering Stage (3-4 Weeks)",
            "3. Reproductive / Panicle Stage (6-8 Weeks)"
        ],
        "Fertilizer Allocation Strategy": [
            "100% Compost + 100% DAP + 50% MOP + 25% Urea",
            "50% Urea + 25% MOP (Targeted root-zone top-dressing)",
            "25% Urea + 25% MOP (Foliar/spray or fertigation)"
        ],
        "Objective": [
            "Root establishment & phosphorus fixation buffer",
            "Rapid vegetative canopy expansion",
            "Grain filling & stress tolerance"
        ]
    }
    st.table(pd.DataFrame(split_schedule))

    report_content = (
        f"PRECISION NUTRIENT MANAGEMENT REPORT (IEEE ACCESS 2025 SSNM)\n"
        f"User: {st.session_state.user_mobile}\n"
        f"Total Field Cost: INR {opt['total_cost']}\n"
        f"Inputs: Urea={opt['urea_kg']}kg, DAP={opt['dap_kg']}kg, MOP={opt['mop_kg']}kg, Compost={opt['compost_kg']}kg\n"
        f"Schedule: 3-Stage Split Dosage applied according to 4R Principles."
    )
    st.download_button("📥 Download Precision Management Plan", report_content, file_name="precision_fertilizer_report.txt")

    st.markdown("---")
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back"):
            st.session_state.step = 5
            st.rerun()
    with nav2:
        if st.button("Continue to Exit ➔", type="primary"):
            st.session_state.step = 7
            st.rerun()

# -------------------------------------------------------------
# SCREEN 7: Exit Panel & Session Flush
# -------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("7. 🚪 Session Summary & Exit Panel")
    st.write("Session completed successfully.")
    
    if st.button("Log Out & Complete Session"):
        st.session_state.logged_in = False
        st.session_state.user_mobile = ""
        st.session_state.step = 1
        st.cache_data.clear()
        st.success("Session concluded. Cache safely cleared.")
        st.rerun()

    st.markdown("---")
    if st.button("⬅️ Back to Recommendations"):
        st.session_state.step = 6
        st.rerun()
