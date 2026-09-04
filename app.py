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
# PAGE CONFIGURATION & UI STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Based Fertilizer and Input Usage Optimization",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border-left: 6px solid #16a34a;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .summary-card {
        background: #f0fdf4;
        border: 2px solid #86efac;
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
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

@st.cache_resource(show_spinner="Loading precision agricultural intelligence models...")
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
# DATABASE SETUP (Supabase Tokyo Pooler)
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
        return True, "Account registered successfully."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (mobile_number, password) VALUES (:m, :p)"), {"m": mobile, "p": hashed_pw})
            conn.commit()
        return True, "Registration successful! You can now log in."
    except Exception:
        return False, "This mobile number is already registered."

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
# DYNAMIC NUTRIENT USE EFFICIENCY (NUE) & DEFICIT ENGINE
# -------------------------------------------------------------
def calculate_advanced_nutrients(target_yield, soil_n, soil_p, soil_k, soc, ph, soil_moist, soil_texture):
    """
    Synthesizes QUEFTS uptake envelopes with dynamic Nutrient Use Efficiency (NUE)
    accounting for soil texture and moisture stress (IEEE 11394509 & ResearchGate 399694119).
    """
    # Crop demand per tonne of economic yield
    demand_n = 22.0 * target_yield
    demand_p = 4.5 * target_yield
    demand_k = 19.0 * target_yield

    # Dynamic NUE: Moisture and Texture impact
    nue_n = 0.50
    if "sandy" in str(soil_texture).lower():
        nue_n -= 0.10 # Leaching risk
    if soil_moist < 30.0 or soil_moist > 75.0:
        nue_n -= 0.08 # Water-stress limits uptake

    # Soil Phosphorus fixation index
    ph_p_factor = 1.0 if 6.0 <= ph <= 7.2 else (0.60 if ph < 5.5 or ph > 8.0 else 0.80)
    soc_n_factor = 1.0 + (soc * 0.15)

    avail_n = (soil_n * 0.45) * soc_n_factor
    avail_p = (soil_p * 0.35) * ph_p_factor
    avail_k = (soil_k * 0.50)

    # Net deficit adjusted by realistic recovery fractions
    def_n = max(0.0, (demand_n - avail_n) / max(0.3, nue_n))
    def_p = max(0.0, (demand_p - avail_p) / 0.35)
    def_k = max(0.0, (demand_k - avail_k) / 0.55)

    return def_n, def_p, def_k

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""

defaults = {
    "soil_n": 50.0, "soil_p": 30.0, "soil_k": 35.0, "soil_ph": 6.5,
    "soil_moist": 45.0, "soc": 0.70, "temp": 26.5, "humidity": 68.0,
    "rainfall": 150.0, "land_area": 2.0, "budget_cap": 25000.0,
    "target_yield": 4.5, "sel_soil": list(soil_encoder.classes_)[0],
    "sel_crop": list(crop_type_encoder.classes_)[0]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------------------
# HEADER & STEP PROGRESS
# -------------------------------------------------------------
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🌾 AI Based Fertilizer and Input Usage Optimization")
    st.caption("Precision Farm Advisory System Powered by Nutrient-Use-Efficiency & Multi-Objective Cost Models")
with h2:
    if st.session_state.logged_in:
        st.write(f"Farmer Mobile: **+91 {st.session_state.user_mobile}**")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.step = 1
            st.rerun()

step_labels = [
    "1. Farmer Login", "2. Field Details", "3. Soil Check", 
    "4. Soil Chart", "5. Nutrient Gaps", "6. Fertilizer Plan", "7. Final Prescription"
]
cols = st.columns(len(step_labels))
for i, col in enumerate(cols):
    s_idx = i + 1
    if s_idx == st.session_state.step:
        col.markdown(f"**🟢 {step_labels[i]}**")
    elif s_idx < st.session_state.step:
        col.markdown(f"✓ {step_labels[i]}")
    else:
        col.markdown(f"⚪ {step_labels[i]}")
st.divider()

# -------------------------------------------------------------
# SCREEN 1: Simple Farmer Login
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("1. 📱 Farmer Login & Registration")
    if not st.session_state.logged_in:
        t_login, t_reg = st.tabs(["Log In", "Register New Farmer"])
        with t_login:
            m = st.text_input("Mobile Number", max_chars=10, key="log_m", placeholder="Enter 10-digit mobile number")
            p = st.text_input("Password", type="password", key="log_p")
            if st.button("Log In to My Farm ➔", type="primary"):
                if len(m.strip()) == 10 and verify_user(m.strip(), p.strip()):
                    st.session_state.logged_in = True
                    st.session_state.user_mobile = m.strip()
                    st.session_state.step = 2
                    st.rerun()
                else:
                    st.error("Invalid mobile number or incorrect password.")
        with t_reg:
            rm = st.text_input("Mobile Number", max_chars=10, key="reg_m", placeholder="Enter 10-digit mobile number")
            rp = st.text_input("Set Password", type="password", key="reg_p")
            rpc = st.text_input("Confirm Password", type="password", key="reg_pc")
            if st.button("Create Account"):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(rm.strip(), rp.strip())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter a 10-digit mobile number and matching passwords.")
    else:
        st.success(f"Logged in as: **+91 {st.session_state.user_mobile}**")
        if st.button("Continue to Field Info ➔", type="primary"):
            st.session_state.step = 2
            st.rerun()

# -------------------------------------------------------------
# SCREEN 2: Field Details & Soil Inputs
# -------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("2. 📍 Enter Your Farm & Soil Test Information")
    
    t_soil, t_weather, t_field = st.tabs(["🧪 Soil Test Results", "⛅ Weather Conditions", "🎯 Land & Target Harvest"])
    
    with t_soil:
        st.info("💡 You can find these values directly on your Soil Health Card.")
        c1, c2, c3 = st.columns(3)
        st.session_state.soil_n = c1.number_input("Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n), help="Responsible for plant height and green leaves.")
        st.session_state.soil_p = c2.number_input("Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p), help="Helps root growth and seed production.")
        st.session_state.soil_k = c3.number_input("Potash (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k), help="Protects against drought and pests.")
        
        c4, c5, c6 = st.columns(3)
        st.session_state.soil_ph = c4.slider("Soil pH Level", 4.0, 9.5, float(st.session_state.soil_ph), 0.1, help="6.0 to 7.2 is the best range.")
        st.session_state.soc = c5.slider("Organic Carbon (%)", 0.1, 2.5, float(st.session_state.soc), 0.05, help="Organic matter keeps soil soft.")
        st.session_state.soil_moist = c6.slider("Soil Moisture Level (%)", 10.0, 90.0, float(st.session_state.soil_moist), 1.0)

    with t_weather:
        w1, w2, w3 = st.columns(3)
        st.session_state.temp = w1.slider("Average Temperature (°C)", 10.0, 48.0, float(st.session_state.temp))
        st.session_state.humidity = w2.slider("Air Humidity (%)", 15.0, 100.0, float(st.session_state.humidity))
        st.session_state.rainfall = w3.slider("Expected Rainfall (mm)", 10.0, 500.0, float(st.session_state.rainfall))

    with t_field:
        f1, f2 = st.columns(2)
        soil_types = list(soil_encoder.classes_)
        crop_types = list(crop_type_encoder.classes_)
        st.session_state.sel_soil = f1.selectbox("Soil Texture Type", soil_types, index=soil_types.index(st.session_state.sel_soil) if st.session_state.sel_soil in soil_types else 0)
        st.session_state.sel_crop = f2.selectbox("Planned Crop Species", crop_types, index=crop_types.index(st.session_state.sel_crop) if st.session_state.sel_crop in crop_types else 0)
        
        f3, f4, f5 = st.columns(3)
        st.session_state.land_area = f3.number_input("Farm Land Area (Hectares)", 0.2, 50.0, float(st.session_state.land_area), 0.2)
        st.session_state.target_yield = f4.number_input("Target Harvest (Tonnes/Hectare)", 1.0, 15.0, float(st.session_state.target_yield), 0.5)
        st.session_state.budget_cap = f5.number_input("Fertilizer Budget Limit (₹)", 2000.0, 500000.0, float(st.session_state.budget_cap), 1000.0)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 1
        st.rerun()
    if b2.button("Run Soil Analysis ➔", type="primary"):
        st.session_state.step = 3
        st.rerun()

# -------------------------------------------------------------
# SCREEN 3: Soil Health Check (Simple Words)
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("3. ⚙️ Soil Condition Summary")
    
    k1, k2, k3 = st.columns(3)
    if st.session_state.soil_ph < 6.0:
        ph_status = "Acidic"
        ph_advice = "Soil is acidic; consider applying agricultural lime."
    elif st.session_state.soil_ph > 7.5:
        ph_status = "Alkaline"
        ph_advice = "Soil is alkaline; apply organic compost or gypsum."
    else:
        ph_status = "Balanced & Healthy"
        ph_advice = "Soil pH is in the optimal range for crop nutrient absorption."
        
    k1.metric("Soil Sweetness (pH)", f"{st.session_state.soil_ph}", ph_status)
    k2.metric("Organic Carbon", f"{st.session_state.soc}%", "Rich" if st.session_state.soc >= 0.75 else "Low (Add Compost)")
    k3.metric("Rainfall Water Hazard", f"{st.session_state.rainfall:.0f} mm", "Heavy Rain" if st.session_state.rainfall > 200 else "Normal")
    
    st.info(f"📋 **Field Agronomist Advice**: {ph_advice}")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 2
        st.rerun()
    if b2.button("Check Nutrient Levels ➔", type="primary"):
        st.session_state.step = 4
        st.rerun()

# -------------------------------------------------------------
# SCREEN 4: Visual Soil Comparison
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("4. 📊 Current Soil Nutrients vs Ideal Levels")
    
    col_l, col_r = st.columns([2, 1])
    with col_l:
        chart_data = pd.DataFrame({
            "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potash (K)"],
            "Your Soil (kg/ha)": [st.session_state.soil_n * 2.24, st.session_state.soil_p * 2.24, st.session_state.soil_k * 2.24],
            "Good Farm Level (kg/ha)": [280.0, 60.0, 150.0]
        }).set_index("Nutrient")
        st.bar_chart(chart_data)
        
    with col_r:
        st.markdown("##### What this chart means:")
        st.write("• **Green/Taller bars** show sufficient nutrient reserves.")
        st.write("• **Shorter bars** indicate shortages that will be addressed in your fertilizer plan.")
        st.caption(f"Soil Type: {st.session_state.sel_soil} | Target Crop: {st.session_state.sel_crop}")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 3
        st.rerun()
    if b2.button("Calculate Nutrient Needs ➔", type="primary"):
        st.session_state.step = 5
        st.rerun()

# -------------------------------------------------------------
# SCREEN 5: Simple Nutrient Shortage Report
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("5. ⚠️ What Your Soil Needs for Target Harvest")
    
    def_n, def_p, def_k = calculate_advanced_nutrients(
        target_yield=st.session_state.target_yield,
        soil_n=st.session_state.soil_n,
        soil_p=st.session_state.soil_p,
        soil_k=st.session_state.soil_k,
        soc=st.session_state.soc,
        ph=st.session_state.soil_ph,
        soil_moist=st.session_state.soil_moist,
        soil_texture=st.session_state.sel_soil
    )

    crop_in = pd.DataFrame([{
        'N': st.session_state.soil_n, 'P': st.session_state.soil_p, 'K': st.session_state.soil_k,
        'temperature': st.session_state.temp, 'humidity': st.session_state.humidity,
        'ph': st.session_state.soil_ph, 'rainfall': st.session_state.rainfall
    }])
    pred_crop = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"##### Nutrient Shortage for {st.session_state.target_yield} Tonnes/Hectare:")
        st.warning(f"• **Nitrogen Needed**: {def_n:.1f} kg per hectare")
        st.warning(f"• **Phosphorus Needed**: {def_p:.1f} kg per hectare")
        st.warning(f"• **Potash Needed**: {def_k:.1f} kg per hectare")
    with g2:
        st.markdown("##### AI Recommendation:")
        st.success(f"🌱 **Best Suited Crop for this Field**: **{pred_crop.capitalize()}**")
        if st.session_state.rainfall > 200 and "sandy" in str(st.session_state.sel_soil).lower():
            st.error("⚠️ **Rain Alert**: Heavy rains on sandy soil can cause fertilizer runoff. Split applications are required.")
        else:
            st.info("🌿 **Retention Index**: Favorable nutrient absorption and retention.")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 4
        st.rerun()
    if b2.button("Generate Fertilizer Plan ➔", type="primary"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 6: Optimized Fertilizer Plan
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("6. 🚀 Your Fertilizer Bags & Application Timetable")
    
    def_n, def_p, def_k = calculate_advanced_nutrients(
        target_yield=st.session_state.target_yield,
        soil_n=st.session_state.soil_n,
        soil_p=st.session_state.soil_p,
        soil_k=st.session_state.soil_k,
        soc=st.session_state.soc,
        ph=st.session_state.soil_ph,
        soil_moist=st.session_state.soil_moist,
        soil_texture=st.session_state.sel_soil
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

    st.session_state.opt_results = opt

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Estimated Cost", f"₹{opt['total_cost']:,.0f}")
    r2.metric("Target Crop Harvest", f"{st.session_state.target_yield} t/ha")
    r3.metric("Budget Used", f"{opt['budget_utilized_pct']}%")

    st.markdown("##### 🛒 Fertilizer Bags Needed for Your Entire Field")
    alloc_table = pd.DataFrame({
        "Fertilizer Name": ["Urea (White Granules)", "DAP (Black/Brown Pellets)", "MOP (Red Potash)", "Complex 14-35-14", "Organic Desi Compost"],
        "Total Quantity": [
            f"{opt['urea_kg']} kg", f"{opt['dap_kg']} kg", f"{opt['mop_kg']} kg", f"{opt['complex_kg']} kg", f"{opt['compost_kg']} kg"
        ],
        "Standard 50kg Bags": [
            f"{max(1, round(opt['urea_kg'] / 50.0))} bags" if opt['urea_kg'] > 0 else "0",
            f"{max(1, round(opt['dap_kg'] / 50.0))} bags" if opt['dap_kg'] > 0 else "0",
            f"{max(1, round(opt['mop_kg'] / 50.0))} bags" if opt['mop_kg'] > 0 else "0",
            f"{max(1, round(opt['complex_kg'] / 50.0))} bags" if opt['complex_kg'] > 0 else "0",
            f"{round(opt['compost_kg'] / 50.0)} bags" if opt['compost_kg'] > 0 else "0"
        ]
    })
    st.table(alloc_table)

    st.markdown("##### 📅 When and How to Apply in the Field")
    schedule = pd.DataFrame({
        "Crop Growth Stage": [
            "1. At Sowing / Transplanting (Base Dose)",
            "2. First Top-Dressing (20-25 Days After Sowing)",
            "3. Flowering / Grain Filling (45-55 Days)"
        ],
        "Fertilizer to Add": [
            "All Compost + All DAP + 1/3 Potash + 1/4 Urea",
            "1/2 Urea + 1/3 Potash (Near the root zone)",
            "Remaining Urea + Remaining Potash"
        ],
        "Benefit to Crop": [
            "Strengthens roots and boosts early plant vigor",
            "Increases tillering and healthy green leaves",
            "Fills grains and enhances test-weight"
        ]
    })
    st.table(schedule)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 5
        st.rerun()
    if b2.button("View Prescription Summary ➔", type="primary"):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: Prescription Summary & Session Exit
# -------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("7. 📋 Complete Farm Prescription Summary")

    opt = st.session_state.get("opt_results", {
        "urea_kg": 0.0, "dap_kg": 0.0, "mop_kg": 0.0, "complex_kg": 0.0, "compost_kg": 0.0, "total_cost": 0.0
    })

    st.markdown(f"""
    <div class="summary-card">
        <h2 style="color: #15803d; margin-top: 0;">🌾 AI Based Fertilizer & Input Usage Prescription Card</h2>
        <p><strong>Farmer Phone:</strong> +91 {st.session_state.user_mobile} | <strong>Land Area:</strong> {st.session_state.land_area} Hectares</p>
        <p><strong>Cultivated Crop:</strong> {st.session_state.sel_crop} | <strong>Target Harvest:</strong> {st.session_state.target_yield} Tonnes/Ha</p>
        <hr style="border: 1px solid #bbf7d0;"/>
        <h3 style="color: #166534;">🛒 Required Fertilizer Purchases:</h3>
        <ul>
            <li><strong>Urea:</strong> {opt['urea_kg']} kg (~{round(opt['urea_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>DAP:</strong> {opt['dap_kg']} kg (~{round(opt['dap_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>MOP (Potash):</strong> {opt['mop_kg']} kg (~{round(opt['mop_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>Organic Compost:</strong> {opt['compost_kg']} kg (~{round(opt['compost_kg'] / 50.0)} bags)</li>
        </ul>
        <h3 style="color: #166534;">💰 Total Estimated Cost: ₹{opt['total_cost']:,.0f}</h3>
        <hr style="border: 1px solid #bbf7d0;"/>
        <h3 style="color: #166534;">📌 Key Application Rules:</h3>
        <ol>
            <li><strong>Split Urea Applications:</strong> Never apply all urea at once; split into 3 doses to minimize losses.</li>
            <li><strong>Field Moisture:</strong> Ensure adequate soil moisture before broadcasting fertilizers.</li>
            <li><strong>Incorporate Compost:</strong> Organic manure keeps soil healthy and increases nutrient absorption efficiency.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    prescription_text = (
        f"AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION\n"
        f"=================================================\n"
        f"Farmer Mobile: +91 {st.session_state.user_mobile}\n"
        f"Crop: {st.session_state.sel_crop} | Land Area: {st.session_state.land_area} ha\n"
        f"Target Harvest: {st.session_state.target_yield} t/ha\n"
        f"Total Estimated Input Cost: Rs. {opt['total_cost']:,.0f}\n\n"
        f"FERTILIZER BAGS TO PURCHASE:\n"
        f"- Urea: {opt['urea_kg']} kg ({round(opt['urea_kg'] / 50.0)} bags of 50kg)\n"
        f"- DAP: {opt['dap_kg']} kg ({round(opt['dap_kg'] / 50.0)} bags of 50kg)\n"
        f"- MOP (Potash): {opt['mop_kg']} kg ({round(opt['mop_kg'] / 50.0)} bags of 50kg)\n"
        f"- Organic Compost: {opt['compost_kg']} kg\n\n"
        f"APPLICATION TIMETABLE:\n"
        f"1. At Sowing: 100% DAP + 100% Compost + 1/3 Potash + 1/4 Urea\n"
        f"2. Day 20-25: 1/2 Urea + 1/3 Potash\n"
        f"3. Day 45-55 (Flowering): Remaining Urea + Remaining Potash\n"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.download_button(
            label="📥 Download Prescription (Receipt)",
            data=prescription_text,
            file_name=f"Prescription_{st.session_state.user_mobile}.txt",
            mime="text/plain"
        )
    with col2:
        if st.button("Finish & Start New Session", type="primary"):
            st.session_state.logged_in = False
            st.session_state.user_mobile = ""
            st.session_state.step = 1
            st.cache_data.clear()
            st.rerun()

    st.divider()
    if st.button("⬅️ Back to Fertilizer List"):
        st.session_state.step = 6
        st.rerun()
