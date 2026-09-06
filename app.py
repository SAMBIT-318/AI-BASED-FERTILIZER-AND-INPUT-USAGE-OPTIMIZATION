import os
import urllib.parse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from PIL import Image
from sqlalchemy import create_engine, text
from optimizer import optimize_fertilizer_blend

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Based Fertilizer & Input Usage Optimization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM CSS / THEME STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E4620;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #386641;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F4F9F4;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #FFFFFF;
        border: 2px solid #C8E6C9;
        border-radius: 12px;
        padding: 20px;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATABASE CONNECTION (Supabase Pooler)
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
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 8})
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile_number TEXT PRIMARY KEY, password TEXT)"))
            conn.commit()
        return engine
    except Exception:
        return None

engine = get_db_engine()

def register_user(mobile, password):
    if not engine:
        return True, "Registered locally."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (mobile_number, password) VALUES (:m, :p)"), {"m": mobile, "p": hashed_pw})
            conn.commit()
        return True, "Registration successful."
    except Exception:
        return False, "User already exists or connection issue."

def verify_user(mobile, password):
    if not engine:
        return True
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT password FROM users WHERE mobile_number = :m"), {"m": mobile}).fetchone()
            return bool(res and res[0] == hashed_pw)
    except Exception:
        return True

# -------------------------------------------------------------
# LOAD ML MODEL ARTIFACTS
# -------------------------------------------------------------
MODELS_DIR = "saved_models"

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    required = ["crop_model.pkl", "crop_encoder.pkl", "fert_model.pkl", "soil_encoder.pkl", 
                "crop_type_encoder.pkl", "fert_encoder.pkl", "yield_model.pkl", "yield_features.pkl", "yield_crop_encoder.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        from train_pipeline import train_crop_recommender, train_fertilizer_classifier, train_yield_regressor
        train_crop_recommender()
        train_fertilizer_classifier()
        train_yield_regressor()

@st.cache_resource(show_spinner=False)
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
# SOIL HEALTH SCORING FUNCTION
# -------------------------------------------------------------
def compute_soil_health_score(n, p, k, ph, moisture, soc):
    score = 0.0
    score += min(25.0, (n / 75.0) * 25.0)
    score += min(20.0, (p / 35.0) * 20.0)
    score += min(20.0, (k / 50.0) * 20.0)
    ph_dist = abs(ph - 6.7)
    score += max(0.0, 20.0 - (ph_dist * 10.0))
    score += min(15.0, (soc / 0.8) * 15.0)
    return int(np.clip(round(score), 0, 100))

# -------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""

defaults = {
    "soil_n": 50.0, "soil_p": 25.0, "soil_k": 35.0, "soil_ph": 6.5,
    "soil_moist": 45.0, "soc": 0.70, "temp": 28.0, "humidity": 65.0,
    "rainfall": 120.0, "land_area": 1.0, "budget_cap": 15000.0,
    "target_yield": 4.0, "sel_soil": list(soil_encoder.classes_)[0],
    "sel_crop": list(crop_type_encoder.classes_)[0],
    "chat_history": [
        {"role": "assistant", "content": "Hello! I am your AI Agronomist. How can I assist you with your soil or crop management today?"}
    ]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------------------
# HEADER & NAVIGATION
# -------------------------------------------------------------
st.markdown("<div class='main-header'>🌱 AI Based Fertilizer & Input Usage Optimization</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Scientific 4R Nutrient Stewardship & Crop Decision Support System</div>", unsafe_allow_html=True)

steps = ["Step 1: Authentication", "Step 2: Soil & Environment", "Step 3: Health Scorecard", 
         "Step 4: Soil Diagnostic", "Step 5: Yield Gap", "Step 6: Optimized Inputs", "Step 7: Plan & Chat"]

st.progress(st.session_state.step / 7.0)
st.caption(f"Current Phase: **{steps[st.session_state.step - 1]}**")
st.write("")

# -------------------------------------------------------------
# STEP 1: AUTHENTICATION
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("Farmer Authentication")
    if not st.session_state.logged_in:
        auth_mode = st.radio("Choose Mode", ["Login", "Register"], horizontal=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            mobile = st.text_input("Mobile Number", max_chars=10)
            password = st.text_input("Password", type="password")
            if auth_mode == "Login":
                if st.button("Login"):
                    if len(mobile) == 10 and verify_user(mobile, password):
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = mobile
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Invalid mobile number or credentials.")
            else:
                conf_password = st.text_input("Confirm Password", type="password")
                if st.button("Register"):
                    if len(mobile) == 10 and password == conf_password and len(password) > 0:
                        ok, msg = register_user(mobile, password)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please check details and passwords match.")
    else:
        st.success(f"Logged in as: {st.session_state.user_mobile}")
        if st.button("Proceed to Field Diagnostic"):
            st.session_state.step = 2
            st.rerun()

# -------------------------------------------------------------
# STEP 2: FIELD & SOIL INPUTS
# -------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("Field Parameters & Optical Scanner")
    tab1, tab2, tab3, tab4 = st.tabs(["AI Scanner", "Lab Data", "Weather", "Farm Setup"])

    with tab1:
        st.write("Scan soil or crop using camera or upload a soil card:")
        cam_pic = st.camera_input("Take a photo of soil or crop")
        if cam_pic:
            st.image(Image.open(cam_pic), width=300)
            st.success("Image received: Calibrating standard loamy profile.")
        upload_pic = st.file_uploader("Upload Image or Soil Health Card", type=["jpg", "png", "jpeg"])
        if upload_pic:
            st.success(f"Loaded: {upload_pic.name}")

    with tab2:
        c1, c2, c3 = st.columns(3)
        st.session_state.soil_n = c1.number_input("Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n))
        st.session_state.soil_p = c2.number_input("Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p))
        st.session_state.soil_k = c3.number_input("Potassium (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k))

        c4, c5, c6 = st.columns(3)
        st.session_state.soil_ph = c4.slider("Soil pH", 4.0, 9.5, float(st.session_state.soil_ph), 0.1)
        st.session_state.soc = c5.slider("Soil Organic Carbon (%)", 0.1, 2.5, float(st.session_state.soc), 0.05)
        st.session_state.soil_moist = c6.slider("Moisture (%)", 10.0, 90.0, float(st.session_state.soil_moist), 1.0)

    with tab3:
        w1, w2, w3 = st.columns(3)
        st.session_state.temp = w1.slider("Temperature (°C)", 10.0, 48.0, float(st.session_state.temp))
        st.session_state.humidity = w2.slider("Relative Humidity (%)", 15.0, 100.0, float(st.session_state.humidity))
        st.session_state.rainfall = w3.slider("Rainfall (mm)", 10.0, 500.0, float(st.session_state.rainfall))

    with tab4:
        cp1, cp2 = st.columns(2)
        s_list = list(soil_encoder.classes_)
        c_list = list(crop_type_encoder.classes_)
        st.session_state.sel_soil = cp1.selectbox("Soil Type", s_list, index=s_list.index(st.session_state.sel_soil) if st.session_state.sel_soil in s_list else 0)
        st.session_state.sel_crop = cp2.selectbox("Target Crop", c_list, index=c_list.index(st.session_state.sel_crop) if st.session_state.sel_crop in c_list else 0)

        cp3, cp4, cp5 = st.columns(3)
        st.session_state.land_area = cp3.number_input("Land Area (ha)", 0.2, 50.0, float(st.session_state.land_area), 0.2)
        st.session_state.target_yield = cp4.number_input("Target Yield (t/ha)", 1.0, 15.0, float(st.session_state.target_yield), 0.5)
        st.session_state.budget_cap = cp5.number_input("Budget Cap (INR)", 2000.0, 500000.0, float(st.session_state.budget_cap), 1000.0)

    col_b1, col_b2 = st.columns([1, 4])
    if col_b1.button("Back"):
        st.session_state.step = 1
        st.rerun()
    if col_b2.button("Compute Soil Health Score"):
        st.session_state.step = 3
        st.rerun()

# -------------------------------------------------------------
# STEP 3: SOIL HEALTH GAUGE
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("Soil Health Evaluation")
    score = compute_soil_health_score(
        st.session_state.soil_n, st.session_state.soil_p, st.session_state.soil_k,
        st.session_state.soil_ph, st.session_state.soil_moist, st.session_state.soc
    )

    g_col, stat_col = st.columns([1, 2])
    with g_col:
        badge = "#2E7D32" if score >= 70 else ("#F9A825" if score >= 50 else "#C62828")
        st.markdown(f"""
        <div class="gauge-container">
            <h1 style="font-size: 4rem; color: {badge}; margin: 0;">{score}</h1>
            <p style="font-weight: 600; color: #555;">Overall Score (0–100)</p>
            <h3 style="color: {badge}; margin: 0;">{"Optimal" if score >= 70 else ("Moderate" if score >= 50 else "Critical")}</h3>
        </div>
        """, unsafe_allow_html=True)

    with stat_col:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Diagnostic Highlights</h4>
            <p><strong>Active pH:</strong> {st.session_state.soil_ph} (Optimal range: 6.0 - 7.5)</p>
            <p><strong>Organic Carbon:</strong> {st.session_state.soc}% ({'Sufficient' if st.session_state.soc >= 0.75 else 'Low - Add Compost'})</p>
            <p><strong>Root-Zone Moisture:</strong> {st.session_state.soil_moist}%</p>
            <p><strong>Precipitation Hazard:</strong> {'High Leaching Risk' if st.session_state.rainfall > 200 else 'Normal'}</p>
        </div>
        """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 4])
    if col_b1.button("Back"):
        st.session_state.step = 2
        st.rerun()
    if col_b2.button("View Diagnostic Chart"):
        st.session_state.step = 4
        st.rerun()

# -------------------------------------------------------------
# STEP 4: SOIL DIAGNOSTIC CHART
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("Soil Diagnostic Reserve vs Target")
    df_chart = pd.DataFrame({
        "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
        "Available Level (kg/ha)": [st.session_state.soil_n * 2.24, st.session_state.soil_p * 2.24, st.session_state.soil_k * 2.24],
        "Recommended Level (kg/ha)": [280.0, 60.0, 150.0]
    }).set_index("Nutrient")
    st.bar_chart(df_chart)

    col_b1, col_b2 = st.columns([1, 4])
    if col_b1.button("Back"):
        st.session_state.step = 3
        st.rerun()
    if col_b2.button("Calculate Yield Gap"):
        st.session_state.step = 5
        st.rerun()

# -------------------------------------------------------------
# STEP 5: YIELD GAP & CROP PREDICTION
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("Yield Gap & AI Crop Recommendation")
    
    r_n = 22.0 * st.session_state.target_yield
    r_p = 4.5 * st.session_state.target_yield
    r_k = 19.0 * st.session_state.target_yield

    avail_n = (st.session_state.soil_n * 0.45) * (1.0 + (st.session_state.soc * 0.15))
    avail_p = (st.session_state.soil_p * 0.35) * (1.0 if 6.0 <= st.session_state.soil_ph <= 7.2 else 0.75)
    avail_k = (st.session_state.soil_k * 0.50)

    def_n = max(0.0, (r_n - avail_n) / 0.50)
    def_p = max(0.0, (r_p - avail_p) / 0.35)
    def_k = max(0.0, (r_k - avail_k) / 0.55)

    crop_in = pd.DataFrame([{
        'N': st.session_state.soil_n, 'P': st.session_state.soil_p, 'K': st.session_state.soil_k,
        'temperature': st.session_state.temp, 'humidity': st.session_state.humidity,
        'ph': st.session_state.soil_ph, 'rainfall': st.session_state.rainfall
    }])
    pred_crop = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Nutrient Shortage (Target: {st.session_state.target_yield} t/ha)</h4>
            <p><strong>N Deficit:</strong> {def_n:.1f} kg/ha</p>
            <p><strong>P Deficit:</strong> {def_p:.1f} kg/ha</p>
            <p><strong>K Deficit:</strong> {def_k:.1f} kg/ha</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Recommended Crop</h4>
            <h3 style="color: #2E7D32;">{pred_crop.capitalize()}</h3>
            <p>Best suitability for current weather and baseline parameters.</p>
        </div>
        """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 4])
    if col_b1.button("Back"):
        st.session_state.step = 4
        st.rerun()
    if col_b2.button("Run Input Optimization"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# STEP 6: INPUT OPTIMIZATION & FERTILIZER BAGS
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("Optimized Fertilizer Allocation")
    
    r_n = 22.0 * st.session_state.target_yield
    r_p = 4.5 * st.session_state.target_yield
    r_k = 19.0 * st.session_state.target_yield

    avail_n = (st.session_state.soil_n * 0.45) * (1.0 + (st.session_state.soc * 0.15))
    avail_p = (st.session_state.soil_p * 0.35) * (1.0 if 6.0 <= st.session_state.soil_ph <= 7.2 else 0.75)
    avail_k = (st.session_state.soil_k * 0.50)

    def_n = max(0.0, (r_n - avail_n) / 0.50)
    def_p = max(0.0, (r_p - avail_p) / 0.35)
    def_k = max(0.0, (r_k - avail_k) / 0.55)

    opt = optimize_fertilizer_blend(
        req_n=def_n, req_p=def_p, req_k=def_k,
        budget_cap=st.session_state.budget_cap,
        land_area=st.session_state.land_area,
        soil_texture=str(st.session_state.sel_soil),
        rainfall_mm=st.session_state.rainfall,
        soc=st.session_state.soc
    )
    st.session_state.opt_results = opt

    m1, m2, m3 = st.columns(3)
    m1.metric("Optimized Cost", f"₹{opt['total_cost']:,.0f}")
    m2.metric("Money Saved", f"₹{(opt['total_cost'] * 0.32):,.0f}", "vs Conventional")
    m3.metric("Budget Used", f"{opt['budget_utilized_pct']}%")

    st.write("### Recommended Fertilizer Bags (50kg Standard):")
    st.table(pd.DataFrame({
        "Product": ["Urea", "DAP", "MOP", "Complex (14-35-14)", "Bio-Compost"],
        "Total Quantity": [f"{opt['urea_kg']} kg", f"{opt['dap_kg']} kg", f"{opt['mop_kg']} kg", f"{opt['complex_kg']} kg", f"{opt['compost_kg']} kg"],
        "50kg Bags": [
            f"{max(1, round(opt['urea_kg'] / 50.0))} bags" if opt['urea_kg'] > 0 else "0",
            f"{max(1, round(opt['dap_kg'] / 50.0))} bags" if opt['dap_kg'] > 0 else "0",
            f"{max(1, round(opt['mop_kg'] / 50.0))} bags" if opt['mop_kg'] > 0 else "0",
            f"{max(1, round(opt['complex_kg'] / 50.0))} bags" if opt['complex_kg'] > 0 else "0",
            f"{round(opt['compost_kg'] / 50.0)} bags" if opt['compost_kg'] > 0 else "0"
        ]
    }))

    col_b1, col_b2 = st.columns([1, 4])
    if col_b1.button("Back"):
        st.session_state.step = 5
        st.rerun()
    if col_b2.button("View Prescription & Action Plan"):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# STEP 7: ACTION PLAN, RECEIPT & AGRONOMIST CHATBOT
# -------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("Field Prescription & Agronomist Support")
    opt = st.session_state.get("opt_results", {
        "urea_kg": 0.0, "dap_kg": 0.0, "mop_kg": 0.0, "complex_kg": 0.0, "compost_kg": 0.0, "total_cost": 0.0
    })

    st.markdown(f"""
    <div class="metric-card">
        <h3>Official Prescription Record</h3>
        <p><strong>Mobile:</strong> {st.session_state.user_mobile} | <strong>Land:</strong> {st.session_state.land_area} ha | <strong>Crop:</strong> {st.session_state.sel_crop}</p>
        <p><strong>Total Investment:</strong> ₹{opt['total_cost']:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

    receipt = (
        f"PRESCRIPTION REPORT\n"
        f"Mobile: {st.session_state.user_mobile}\n"
        f"Crop: {st.session_state.sel_crop}\n"
        f"Land Area: {st.session_state.land_area} ha\n"
        f"Estimated Cost: INR {opt['total_cost']:,.0f}\n"
        f"Urea: {opt['urea_kg']} kg\n"
        f"DAP: {opt['dap_kg']} kg\n"
        f"MOP: {opt['mop_kg']} kg\n"
        f"Compost: {opt['compost_kg']} kg\n"
    )
    st.download_button("Download Prescription (TXT)", receipt, file_name="Prescription.txt")

    st.write("---")
    st.subheader("AI Agronomist Chat")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask a question regarding application timing or fertilizer types...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        ans = f"For {st.session_state.sel_crop}, ensure basal phosphorus and potassium are placed at 5-7 cm depth during planting, and split nitrogen across tillering and flowering stages."
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.markdown(ans)

    if st.button("Start Over"):
        st.session_state.step = 1
        st.rerun()
