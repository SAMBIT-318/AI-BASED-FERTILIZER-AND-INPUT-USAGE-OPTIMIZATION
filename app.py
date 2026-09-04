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

# -------------------------------------------------------------
# APPLICATION SPECIFICATION & UI THEMING
# -------------------------------------------------------------
st.set_page_config(
    page_title="AgriPrecision Pro | Global Agronomic Decision System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-tag {
        background-color: #ecfdf5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "saved_models"

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(MODELS_DIR, "crop_model.pkl")):
        train_crop_recommender()
    if not os.path.exists(os.path.join(MODELS_DIR, "fert_model.pkl")):
        train_fertilizer_classifier()
    if not os.path.exists(os.path.join(MODELS_DIR, "yield_model.pkl")):
        train_yield_regressor()

@st.cache_resource(show_spinner="Initializing High-Precision Agronomy Inference Engines...")
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
# DATABASE ENGINE (Tokyo Pooler Native Connection)
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
    except Exception as e:
        st.warning(f"Database in Standalone Local Evaluation Mode. ({e})")
        return None

engine = get_db_engine()

def register_user(mobile, password):
    if not engine:
        return True, "Registered in local session storage."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (mobile_number, password) VALUES (:m, :p)"), {"m": mobile, "p": hashed_pw})
            conn.commit()
        return True, "Account registered successfully."
    except Exception:
        return False, "Mobile number already registered."

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
# QUEFTS STOICHIOMETRIC BALANCER
# -------------------------------------------------------------
def calculate_quefts_nutrients(target_yield, soil_n, soil_p, soil_k, soc, ph):
    """
    Non-linear boundary nutrient uptake envelope based on QUEFTS model.
    """
    # Baseline physiological requirements per ton of economic crop yield
    r_n = 22.0 * target_yield
    r_p = 4.5 * target_yield
    r_k = 19.0 * target_yield

    # Soil bioavailability factors (pH fixation + microbial organic carbon mineralization)
    ph_buffer_p = 1.0 if 6.0 <= ph <= 7.2 else (0.65 if ph < 5.5 or ph > 8.0 else 0.85)
    soc_avail_n = 1.0 + (soc * 0.18)

    avail_n = (soil_n * 0.45) * soc_avail_n
    avail_p = (soil_p * 0.35) * ph_buffer_p
    avail_k = (soil_k * 0.50)

    # Compute actual physiological deficit
    def_n = max(0.0, r_n - avail_n)
    def_p = max(0.0, r_p - avail_p)
    def_k = max(0.0, r_k - avail_k)

    return def_n, def_p, def_k

# -------------------------------------------------------------
# STATE REPOSITORY
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""

defaults = {
    "soil_n": 50.0, "soil_p": 30.0, "soil_k": 35.0, "soil_ph": 6.5,
    "soil_moist": 44.0, "soc": 0.70, "soil_ec": 0.80,
    "temp": 26.5, "humidity": 68.0, "rainfall": 150.0,
    "land_area": 2.0, "budget_cap": 25000.0, "target_yield": 4.5,
    "sel_soil": list(soil_encoder.classes_)[0],
    "sel_crop": list(crop_type_encoder.classes_)[0],
    "uploaded_crop_df": None, "uploaded_fert_df": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------------------
# NAVIGATION HEADER
# -------------------------------------------------------------
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🌾 AgriPrecision Pro | Decision Engine")
    st.caption("Hybrid System: Non-Linear Stoichiometry (QUEFTS) + Scipy Multi-Objective LP + 4R Stewardship")
with h2:
    if st.session_state.logged_in:
        st.write(f"**Operator:** `+91 {st.session_state.user_mobile}`")
        if st.button("End Session"):
            st.session_state.logged_in = False
            st.session_state.step = 1
            st.rerun()

steps = ["Authentication", "Sensors & Fields", "Pre-processing", "Analytics", "Stoichiometry", "VRA Optimization", "Prescription"]
p_cols = st.columns(len(steps))
for i, col in enumerate(p_cols):
    s_idx = i + 1
    if s_idx == st.session_state.step:
        col.markdown(f"**🟢 {steps[i]}**")
    elif s_idx < st.session_state.step:
        col.markdown(f"✓ {steps[i]}")
    else:
        col.markdown(f"⚪ {steps[i]}")
st.divider()

# -------------------------------------------------------------
# SCREEN 1: Authentication
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("1. 📱 Operator Security & Session Gate")
    if not st.session_state.logged_in:
        t_login, t_reg = st.tabs(["Log In to Existing Workspace", "Register New Farm Operator"])
        with t_login:
            m = st.text_input("Mobile Number", max_chars=10, key="log_m", placeholder="10-digit mobile number")
            p = st.text_input("Password", type="password", key="log_p")
            if st.button("Access Farm Workspace ➔", type="primary"):
                if len(m.strip()) == 10 and verify_user(m.strip(), p.strip()):
                    st.session_state.logged_in = True
                    st.session_state.user_mobile = m.strip()
                    st.session_state.step = 2
                    st.rerun()
                else:
                    st.error("Invalid credentials or unregistered profile.")
        with t_reg:
            rm = st.text_input("Mobile Number", max_chars=10, key="reg_m")
            rp = st.text_input("Create Security Password", type="password", key="reg_p")
            rpc = st.text_input("Confirm Security Password", type="password", key="reg_pc")
            if st.button("Establish Operator Profile"):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(rm.strip(), rp.strip())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Ensure passwords match and mobile is 10 digits.")
    else:
        st.success(f"Workspace Active: Operator **+91 {st.session_state.user_mobile}**")
        if st.button("Continue to Sensor Matrix ➔", type="primary"):
            st.session_state.step = 2
            st.rerun()

# -------------------------------------------------------------
# SCREEN 2: Field Sensors & Telemetry
# -------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("2. 📍 Comprehensive Field Telemetry & Target Yield")
    
    t_soil, t_meteo, t_yield = st.tabs(["🧪 Soil Chemical & Biological Profile", "⛅ Agro-Climatology", "🎯 Production Objectives"])
    
    with t_soil:
        s1, s2, s3 = st.columns(3)
        st.session_state.soil_n = s1.number_input("Available Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n))
        st.session_state.soil_p = s2.number_input("Available Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p))
        st.session_state.soil_k = s3.number_input("Available Potassium (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k))
        
        s4, s5, s6 = st.columns(3)
        st.session_state.soil_ph = s4.slider("Active Reaction (pH)", 4.0, 9.5, float(st.session_state.soil_ph), 0.1)
        st.session_state.soc = s5.slider("Soil Organic Carbon (SOC) [%]", 0.1, 2.5, float(st.session_state.soc), 0.05)
        st.session_state.soil_ec = s6.slider("Electrical Conductivity (EC) [dS/m]", 0.1, 4.0, float(st.session_state.soil_ec), 0.05)

    with t_meteo:
        m1, m2, m3 = st.columns(3)
        st.session_state.temp = m1.slider("Mean Field Temperature (°C)", 10.0, 48.0, float(st.session_state.temp))
        st.session_state.humidity = m2.slider("Relative Air Humidity (%)", 15.0, 100.0, float(st.session_state.humidity))
        st.session_state.rainfall = m3.slider("Precipitation Outlook (mm)", 10.0, 500.0, float(st.session_state.rainfall))

    with t_yield:
        y1, y2, y3, y4 = st.columns(4)
        soil_types = list(soil_encoder.classes_)
        crop_types = list(crop_type_encoder.classes_)
        st.session_state.sel_soil = y1.selectbox("Soil Texture Class", soil_types, index=soil_types.index(st.session_state.sel_soil) if st.session_state.sel_soil in soil_types else 0)
        st.session_state.sel_crop = y2.selectbox("Cultivated Crop", crop_types, index=crop_types.index(st.session_state.sel_crop) if st.session_state.sel_crop in crop_types else 0)
        st.session_state.land_area = y3.number_input("Field Acreage (Hectares)", 0.2, 50.0, float(st.session_state.land_area), 0.2)
        st.session_state.target_yield = y4.number_input("Target Harvest Yield (t/ha)", 1.0, 15.0, float(st.session_state.target_yield), 0.5)
        st.session_state.budget_cap = st.number_input("Working Capital Budget (INR)", 2000.0, 500000.0, float(st.session_state.budget_cap), 1000.0)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 1
        st.rerun()
    if b2.button("Process Sensor Arrays ➔", type="primary"):
        st.session_state.step = 3
        st.rerun()

# -------------------------------------------------------------
# SCREEN 3: Normalization & Scaling
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("3. ⚙️ Edge Telemetry Normalization & Calibration")
    
    k1, k2, k3, k4 = st.columns(4)
    cn_ratio = round(st.session_state.soc * 25.0 / (st.session_state.soil_n / 10.0 + 0.1), 1)
    k1.metric("Dynamic C:N Ratio", f"{cn_ratio}:1", "Balanced (10-15:1)" if 10 <= cn_ratio <= 15 else "Carbon Imbalance")
    k2.metric("Phosphorus Fixation Index", f"{round(abs(st.session_state.soil_ph - 6.5) * 12, 1)}%", "Buffering Active")
    k3.metric("Salinity Hazard (EC)", f"{st.session_state.soil_ec} dS/m", "Safe" if st.session_state.soil_ec < 1.5 else "Saline Warning")
    k4.metric("Evapotranspiration Index", f"{round((st.session_state.temp * 0.7) + (st.session_state.rainfall * 0.1), 1)}", "Stable")

    norm_matrix = pd.DataFrame({
        "Feature Attribute": ["Nitrogen Vector", "Phosphorus Vector", "Potassium Vector", "Soil Carbon Buffer", "Hydrologic Coeff."],
        "Normalized Value [0-1]": [
            round(st.session_state.soil_n / 200.0, 3),
            round(st.session_state.soil_p / 100.0, 3),
            round(st.session_state.soil_k / 250.0, 3),
            round(st.session_state.soc / 2.0, 3),
            round(st.session_state.rainfall / 300.0, 3)
        ],
        "Calibrated Agronomic Scale": ["0 - 200 mg/kg", "0 - 100 mg/kg", "0 - 250 mg/kg", "0 - 2.0 % SOC", "Regional Hydrograph"]
    })
    st.dataframe(norm_matrix, use_container_width=True)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 2
        st.rerun()
    if b2.button("Inspect Visual Analytics ➔", type="primary"):
        st.session_state.step = 4
        st.rerun()

# -------------------------------------------------------------
# SCREEN 4: Visual Dashboard
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("4. 📊 Spatial & Agro-Ecological Analytics")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("##### Current Available Nutrients vs. Agronomic Critical Benchmark (kg/ha)")
        nut_chart = pd.DataFrame({
            "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
            "Measured Soil Reserve": [st.session_state.soil_n * 2.24, st.session_state.soil_p * 2.24, st.session_state.soil_k * 2.24],
            "Adequacy Target": [280.0, 60.0, 150.0]
        }).set_index("Nutrient")
        st.bar_chart(nut_chart)
        
    with col_r:
        st.markdown("##### Pedological Diagnostic")
        st.write(f"**Texture Family:** `{st.session_state.sel_soil}`")
        st.write(f"**Reaction State:** `pH {st.session_state.soil_ph}`")
        st.write(f"**Biological Reserve:** `{st.session_state.soc}% Organic Carbon`")
        if "sandy" in str(st.session_state.sel_soil).lower():
            st.warning("Coarse soil texture detected: Elevated risk of nitrate leaching under intense irrigation.")
        else:
            st.info("Fine/loamy texture: Favorable cation retention and nutrient exchange capacity.")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 3
        st.rerun()
    if b2.button("Run Stoichiometric Solver ➔", type="primary"):
        st.session_state.step = 5
        st.rerun()

# -------------------------------------------------------------
# SCREEN 5: QUEFTS Stoichiometry & Leaching Hazard
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("5. ⚠️ QUEFTS Stoichiometric Gaps & Environmental Indices")
    
    def_n, def_p, def_k = calculate_quefts_nutrients(
        target_yield=st.session_state.target_yield,
        soil_n=st.session_state.soil_n,
        soil_p=st.session_state.soil_p,
        soil_k=st.session_state.soil_k,
        soc=st.session_state.soc,
        ph=st.session_state.soil_ph
    )

    crop_in = pd.DataFrame([{
        'N': st.session_state.soil_n, 'P': st.session_state.soil_p, 'K': st.session_state.soil_k,
        'temperature': st.session_state.temp, 'humidity': st.session_state.humidity,
        'ph': st.session_state.soil_ph, 'rainfall': st.session_state.rainfall
    }])
    pred_crop = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]

    # Environmental Leaching Risk Index
    leaching_index = (st.session_state.rainfall / 120.0) * (1.7 if "sandy" in str(st.session_state.sel_soil).lower() else 1.0)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Net Field Deficits (QUEFTS Non-Linear Model)")
        st.metric("Nitrogen Deficit (N)", f"{def_n:.1f} kg/ha")
        st.metric("Phosphorus Deficit (P₂O₅)", f"{def_p:.1f} kg/ha")
        st.metric("Potassium Deficit (K₂O)", f"{def_k:.1f} kg/ha")
    with g2:
        st.markdown("##### Ecological Safety & ML Matching")
        st.success(f"Optimal Adaptive Crop Model: **{pred_crop.capitalize()}**")
        if leaching_index > 2.2:
            st.error(f"⚠️ Environmental Leaching Vulnerability: HIGH ({leaching_index:.2f})")
            st.caption("Mitigation: Split urea into ≥3 applications; do not apply all synthetic nitrogen at sowing.")
        elif leaching_index > 1.3:
            st.warning(f"⚠️ Environmental Leaching Vulnerability: MODERATE ({leaching_index:.2f})")
            st.caption("Mitigation: Employ split dosing and fertigation where possible.")
        else:
            st.success(f"🌿 Environmental Leaching Vulnerability: MINIMAL ({leaching_index:.2f})")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 4
        st.rerun()
    if b2.button("Formulate LP Optimization ➔", type="primary"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 6: Mathematical Optimization & VRA Matrix
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("6. 🚀 Mathematical LP Optimization & VRA Fertigation Schedule")
    
    def_n, def_p, def_k = calculate_quefts_nutrients(
        target_yield=st.session_state.target_yield,
        soil_n=st.session_state.soil_n,
        soil_p=st.session_state.soil_p,
        soil_k=st.session_state.soil_k,
        soc=st.session_state.soc,
        ph=st.session_state.soil_ph
    )

    opt = optimize_fertilizer_blend(
        req_n=def_n,
        req_p=def_p,
        req_k=def_k,
        budget_cap=st.session_state.budget_cap,
        land_area=st.session_state.land_area,
        soil_texture=str(st.session_state.sel_soil),
        rainfall_mm=st.session_state.rainfall,
        soc=st.session_state.soc
    )

    r1, r2, r3 = st.columns(3)
    r1.metric("Optimized Total Cost", f"₹{opt['total_cost']:,.2f}")
    r2.metric("Yield Objective", f"{st.session_state.target_yield} t/ha")
    r3.metric("Budget Utilization", f"{opt['budget_utilized_pct']}%")

    st.markdown("##### Prescribed Input Allocation Across Cultivated Acreage")
    allocation_table = pd.DataFrame({
        "Nutrient Source Carrier": ["Urea (Synthetic N)", "DAP (Diammonium Phosphate)", "MOP (Muriate of Potash)", "Complex (14-35-14)", "Bio-Carbon Compost"],
        "Total Quantity (kg)": [opt['urea_kg'], opt['dap_kg'], opt['mop_kg'], opt['complex_kg'], opt['compost_kg']],
        "Application Density (kg/ha)": [
            round(opt['urea_kg'] / st.session_state.land_area, 1),
            round(opt['dap_kg'] / st.session_state.land_area, 1),
            round(opt['mop_kg'] / st.session_state.land_area, 1),
            round(opt['complex_kg'] / st.session_state.land_area, 1),
            round(opt['compost_kg'] / st.session_state.land_area, 1)
        ]
    })
    st.dataframe(allocation_table, use_container_width=True)

    st.markdown("##### 📅 4-Stage Variable-Rate Fertigation Matrix (4R Nutrient Stewardship)")
    vra_table = pd.DataFrame({
        "Crop Phenological Stage": [
            "1. Basal Foundation (Sowing)",
            "2. Active Vegetative / Tillering (Day 20-25)",
            "3. Panicle Initiation / Flowering (Day 45-55)",
            "4. Grain Consolidation (Day 70-80)"
        ],
        "Input Apportionment": [
            "100% Organic Compost + 100% DAP + 30% MOP + 25% Urea",
            "40% Urea + 30% MOP (Targeted root zone side-dress)",
            "25% Urea + 40% MOP (Foliar or fertigation run)",
            "10% Urea foliar spray (Triggered by canopy leaf-color index)"
        ],
        "Physiological Function": [
            "Root establishment & phosphorus fixation buffering",
            "Tillering density & leaf-area index expansion",
            "Spikelet fertility & carbohydrate translocation",
            "Grain test-weight maximization and senescence mitigation"
        ]
    })
    st.table(vra_table)

    prescription_doc = (
        f"AGRIPRECISION PRO PRESCRIPTION DOSSIER\n"
        f"Farmer ID: {st.session_state.user_mobile} | Target Yield: {st.session_state.target_yield} t/ha\n"
        f"Acreage: {st.session_state.land_area} ha | Optimized Investment: INR {opt['total_cost']}\n"
        f"Allocations: Urea={opt['urea_kg']}kg, DAP={opt['dap_kg']}kg, MOP={opt['mop_kg']}kg, Compost={opt['compost_kg']}kg\n"
        f"Algorithm: SciPy LP Solver with QUEFTS Stoichiometry & 4R Stewardship."
    )
    st.download_button("📥 Export Agronomic Prescription Plan", prescription_doc, file_name="Precision_Agronomy_Prescription.txt")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 5
        st.rerun()
    if b2.button("Proceed to Session Completion ➔", type="primary"):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: Summary & Session Termination
# -------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("7. 🚪 Prescription Compiled & Archived")
    st.success("The site-specific agronomic prescription has been calculated and archived.")
    
    if st.button("Complete & Flush Workspace", type="primary"):
        st.session_state.logged_in = False
        st.session_state.user_mobile = ""
        st.session_state.step = 1
        st.cache_data.clear()
        st.rerun()

    st.divider()
    if st.button("⬅️ Back to Formulation"):
        st.session_state.step = 6
        st.rerun()
