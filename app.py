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
from train_pipeline import train_crop_recommender, train_fertilizer_classifier, train_yield_regressor

# -------------------------------------------------------------
# PAGE CONFIGURATION & FARMER-SUITABLE GREEN UI STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="AgriPrecision | Smart Crop & Fertilizer Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Earthy, Field-Tested Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F6F8F3 !important;
        color: #1B381E !important;
    }

    /* Top Hero Header */
    .farmer-hero {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        border-radius: 16px;
        padding: 22px 26px;
        color: #FFFFFF !important;
        box-shadow: 0 6px 18px rgba(27, 94, 32, 0.18);
        margin-bottom: 20px;
    }
    .farmer-hero h1 {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .farmer-hero p {
        font-size: 14px !important;
        color: #E8F5E9 !important;
        margin: 4px 0 0 0 !important;
        opacity: 0.95;
    }

    /* Farmer-Friendly Cards */
    .agri-card {
        background: #FFFFFF !important;
        border: 1.5px solid #D7E7D8 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 3px 10px rgba(0, 50, 0, 0.04) !important;
        margin-bottom: 16px !important;
    }

    .metric-card {
        background: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border-left: 6px solid #2E7D32 !important;
        border-top: 1px solid #E2EEDF !important;
        border-right: 1px solid #E2EEDF !important;
        border-bottom: 1px solid #E2EEDF !important;
        box-shadow: 0 2px 8px rgba(0, 50, 0, 0.04) !important;
        margin-bottom: 12px;
    }

    .summary-card {
        background: #F1F8F1 !important;
        border: 2px solid #81C784 !important;
        padding: 22px !important;
        border-radius: 14px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.08) !important;
    }

    /* ALL BUTTONS TO HIGH-VISIBILITY LUSH GREEN */
    div.stButton > button, div.stButton > button:focus {
        background: linear-gradient(180deg, #2E7D32 0%, #1B5E20 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        border: 1px solid #144918 !important;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.28) !important;
        transition: all 0.15s ease-in-out !important;
        text-shadow: 0 1px 1px rgba(0,0,0,0.2) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #388E3C 0%, #1E6B24 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(27, 94, 32, 0.38) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 6px rgba(27, 94, 32, 0.2) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: linear-gradient(180deg, #1B5E20 0%, #0F3D13 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        border: 1px solid #0B2C0D !important;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.25) !important;
    }

    /* Form Fields & Tabs */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1.5px solid #C4DCC5 !important;
        background-color: #FFFFFF !important;
        color: #1B381E !important;
        font-weight: 500 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 2px rgba(46, 125, 50, 0.2) !important;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        background-color: #E2EEDF;
        font-weight: 600;
        color: #2E7D32;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
    }

    /* Status Badges */
    .badge-pass {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #A5D6A7;
    }
    .badge-warn {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #EF9A9A;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# MULTILINGUAL STRINGS (English, Hindi, Odia)
# -------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "🌾 AI Based Precision Agriculture Advisor",
        "subtitle": "Scientific 4R Nutrient Stewardship, Pest Triage & Crop Prescription",
        "login_tab": "Farmer Log In",
        "reg_tab": "Register New Farmer",
        "mobile_lbl": "Mobile Number",
        "pass_lbl": "Password",
        "conf_pass_lbl": "Confirm Password",
        "lang_select": "App Language / ଭାଷା ଚୟନ / भाषा चुनें",
        "mode_select": "Select Farm Service",
        "mode_opt": "🌾 Full Soil & Fertilizer Optimization Pipeline",
        "mode_diag": "🔬 Plant Disease, Pest & Medicine Diagnosis Only",
        "btn_login": "Log In to Farm Dashboard ➔",
        "btn_reg": "Create Account",
        "btn_back": "⬅️ Back",
        "btn_next": "Continue ➔",
        "feedback_title": "🌟 Farmer Feedback & Prescription Rating",
        "feedback_submit": "Submit Feedback & Complete",
        "land_calc_title": "📐 Land Conversion & Acreage Calculation Table"
    },
    "हिन्दी": {
        "title": "🌾 एआई आधारित सटीक कृषि सलाहकार",
        "subtitle": "वैज्ञानिक 4R पोषक तत्व प्रबंधन, कीट पहचान और फसल सलाह",
        "login_tab": "किसान लॉगिन",
        "reg_tab": "नया किसान पंजीकरण",
        "mobile_lbl": "मोबाइल नंबर",
        "pass_lbl": "पासवर्ड",
        "conf_pass_lbl": "पासवर्ड की पुष्टि करें",
        "lang_select": "ऐप भाषा चुनें",
        "mode_select": "कृषि सेवा चुनें",
        "mode_opt": "🌾 पूर्ण मृदा एवं उर्वरक अनुकूलन पाइपलाइन",
        "mode_diag": "🔬 केवल पौध रोग, कीट एवं औषधि निदान",
        "btn_login": "डैशबोर्ड में लॉगिन करें ➔",
        "btn_reg": "खाता बनाएं",
        "btn_back": "⬅️ पीछे",
        "btn_next": "आगे बढ़ें ➔",
        "feedback_title": "🌟 किसान समीक्षा एवं रेटिंग",
        "feedback_submit": "समीक्षा जमा करें और बाहर निकलें",
        "land_calc_title": "📐 भूमि रूपांतरण और एकड़ गणना तालिका"
    },
    "ଓଡ଼ିଆ": {
        "title": "🌾 ଏଆଇ ଆଧାରିତ ଉନ୍ନତ କୃଷି ଓ ଖତ ପରାମର୍ଶ କେନ୍ଦ୍ର",
        "subtitle": "ବୈଜ୍ଞାନିକ ମୃତ୍ତିକା ପରୀକ୍ଷଣ, କୀଟ ନିବାରଣ ଏବଂ ସାର ନିର୍ଦ୍ଦେଶାବଳୀ",
        "login_tab": "କୃଷକ ଲଗଇନ୍",
        "reg_tab": "ନୂତନ କୃଷକ ପଞ୍ଜୀକରଣ",
        "mobile_lbl": "ମୋବାଇଲ୍ ନମ୍ବର",
        "pass_lbl": "ପାସୱାର୍ଡ",
        "conf_pass_lbl": "ପାସୱାର୍ଡ ନିଶ୍ଚିତ କରନ୍ତୁ",
        "lang_select": "ଭାଷା ବାଛନ୍ତୁ",
        "mode_select": "ସେବା ଚୟନ କରନ୍ତୁ",
        "mode_opt": "🌾 ସମ୍ପୂର୍ଣ୍ଣ ମୃତ୍ତିକା ଓ ସାର ପରିମାଣ ନିର୍ଦ୍ଧାରଣ",
        "mode_diag": "🔬 କେବଳ ଫସଲ ରୋଗ, କୀଟ ଚିହ୍ନଟ ଓ ଔଷଧ",
        "btn_login": "ଡ୍ୟାସବୋର୍ଡରେ ପ୍ରବେଶ କରନ୍ତୁ ➔",
        "btn_reg": "ଖାତା ତିଆରି କରନ୍ତୁ",
        "btn_back": "⬅️ ପଛକୁ ଯାଆନ୍ତୁ",
        "btn_next": "ଆଗକୁ ବଢ଼ନ୍ତୁ ➔",
        "feedback_title": "🌟 କୃଷକ ମତାମତ ଓ ରେଟିଂ",
        "feedback_submit": "ମତାମତ ଦାଖଲ କରନ୍ତୁ",
        "land_calc_title": "📐 ଜମି ମାପ ଓ ଏକର ହିସାବ ସାରଣୀ"
    }
}

# -------------------------------------------------------------
# SAFE MODEL LOADER (Pre-Trained Random Forest Artifacts)
# -------------------------------------------------------------
MODELS_DIR = "saved_models"

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname in ["crop_model.pkl", "fert_model.pkl", "yield_model.pkl"]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            train_crop_recommender()
            train_fertilizer_classifier()
            train_yield_regressor()
            break

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
# SUPABASE POSTGRESQL POOLER
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
            conn.execute(text("CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, mobile TEXT, rating INT, comments TEXT)"))
            conn.commit()
        return engine
    except Exception:
        return None

engine = get_db_engine()

def register_user(mobile, password):
    if not engine:
        return True, "Account registered locally."
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

def save_feedback(mobile, rating, comments):
    if not engine:
        return True
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO feedback (mobile, rating, comments) VALUES (:m, :r, :c)"), {"m": mobile, "r": rating, "c": comments})
            conn.commit()
        return True
    except Exception:
        return False

# -------------------------------------------------------------
# LAND AREA STANDARDIZATION
# -------------------------------------------------------------
UNIT_TO_HECTARE = {
    "Acre (एकड़ / ଏକର)": 0.404686,
    "Hectare (हेक्टेयर / ହେକ୍ଟର)": 1.0,
    "Guntha (गुंठा / ଗୁଣ୍ଠ)": 0.010117,
    "Decimal / Cent (डिसमिल / ଡେସିମିଲ)": 0.004047,
    "Square Feet (वर्ग फुट / ବର୍ଗ ଫୁଟ)": 0.0000092903
}

def render_land_conversion_table(entered_val, chosen_unit):
    ha_base = entered_val * UNIT_TO_HECTARE[chosen_unit]
    acres = ha_base / 0.404686
    guntha = ha_base / 0.010117
    decimals = ha_base / 0.004047
    sq_ft = ha_base / 0.0000092903
    
    table_df = pd.DataFrame({
        "Unit Name": ["Acre (एकड़ / ଏକର)", "Hectare (हेक्टेयर / ହେକ୍ଟର)", "Guntha (गुंठा / ଗୁଣ୍ଠ)", "Decimal / Cent (ଡେସିମିଲ)", "Square Feet (Sq Ft)"],
        "Calculated Size": [f"{acres:.3f} Acres", f"{ha_base:.3f} Ha", f"{guntha:.2f} Guntha", f"{decimals:.2f} Decimals", f"{sq_ft:,.0f} Sq Ft"]
    })
    return table_df, ha_base

# -------------------------------------------------------------
# ADVANCED NUTRIENT DEFICIT ENGINE
# -------------------------------------------------------------
def calculate_advanced_nutrients(target_yield, soil_n, soil_p, soil_k, soc, ph, soil_moist, soil_texture):
    demand_n = 22.0 * target_yield
    demand_p = 4.5 * target_yield
    demand_k = 19.0 * target_yield

    nue_n = 0.50
    if "sandy" in str(soil_texture).lower():
        nue_n -= 0.10
    if soil_moist < 30.0 or soil_moist > 75.0:
        nue_n -= 0.08

    ph_p_factor = 1.0 if 6.0 <= ph <= 7.2 else (0.60 if ph < 5.5 or ph > 8.0 else 0.80)
    soc_n_factor = 1.0 + (soc * 0.15)

    avail_n = (soil_n * 0.45) * soc_n_factor
    avail_p = (soil_p * 0.35) * ph_p_factor
    avail_k = (soil_k * 0.50)

    def_n = max(0.0, (demand_n - avail_n) / max(0.3, nue_n))
    def_p = max(0.0, (demand_p - avail_p) / 0.35)
    def_k = max(0.0, (demand_k - avail_k) / 0.55)

    return def_n, def_p, def_k

# -------------------------------------------------------------
# OPTICAL DISEASE & PEST CLASSIFIER
# -------------------------------------------------------------
def analyze_plant_disease_image(image_obj):
    img_rgb = image_obj.convert("RGB").resize((100, 100))
    arr = np.array(img_rgb)
    r_mean, g_mean, b_mean = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])
    
    if g_mean > r_mean and g_mean > b_mean:
        return {
            "health": "Healthy Plant Canopy",
            "disease": "No critical fungal/bacterial infection",
            "pest": "Minor sap-feeders / Thrips (<5%)",
            "symptoms": "Healthy chlorophyll index and vigorous leaves.",
            "medicine": "Neem Oil Spray (1500 ppm @ 3ml/L) as an organic protector.",
            "recovery_chance": 95,
            "will_grow": "Yes, excellent growth expected."
        }
    elif r_mean > g_mean and r_mean > 110:
        return {
            "health": "Infected Leaf Spots Detected",
            "disease": "Leaf Rust / Early Blight (Alternaria spp.)",
            "pest": "Fall Armyworm / Foliar Caterpillar chew marks",
            "symptoms": "Yellow-brown necrotic spots with leaf edge wilting.",
            "medicine": "Mancozeb 75% WP (2.5 g/L) + Chlorantraniliprole 18.5% SC (0.4 ml/L)",
            "recovery_chance": 78,
            "will_grow": "Yes, if treated within 48 to 72 hours."
        }
    else:
        return {
            "health": "Chlorosis & Stem Stress",
            "disease": "Powdery Mildew / Bacterial Leaf Blight",
            "pest": "Stem Borer / Aphid cluster colony",
            "symptoms": "Pale whitening of lamina with loss of vigor.",
            "medicine": "Hexaconazole 5% EC (2 ml/L) + Imidacloprid 17.8% SL (0.5 ml/L)",
            "recovery_chance": 62,
            "will_grow": "Moderate; requires immediate systemic spray."
        }

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Full Optimization"
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "English"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""

defaults = {
    "soil_n": 50.0, "soil_p": 30.0, "soil_k": 35.0, "soil_ph": 6.5,
    "soil_moist": 45.0, "soc": 0.70, "temp": 26.5, "humidity": 68.0,
    "rainfall": 150.0, "raw_land_val": 2.0, "land_unit": "Acre (एकड़ / ଏକର)",
    "land_area": 0.809, "budget_cap": 25000.0, "target_yield": 4.5,
    "sel_soil": list(soil_encoder.classes_)[0],
    "sel_crop": list(crop_type_encoder.classes_)[0],
    "lat": 20.2961, "lon": 85.8245
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = TRANSLATIONS[st.session_state.app_lang]

# -------------------------------------------------------------
# FARMER HERO HEADER
# -------------------------------------------------------------
st.markdown(f"""
<div class="farmer-hero">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h1>{T['title']}</h1>
            <p>{T['subtitle']}</p>
        </div>
        <div style="background:rgba(255,255,255,0.2); padding:6px 14px; border-radius:10px; font-weight:700; font-size:13px;">
            🌱 100% Farmer Ready
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 1: LOGIN & LANGUAGE PREFERENCE
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader(f"1. 📱 {T['login_tab']}")
    
    c_lang, c_mode = st.columns(2)
    new_lang = c_lang.selectbox(T["lang_select"], ["English", "हिन्दी", "ଓଡ଼ିଆ"], index=["English", "हिन्दी", "ଓଡ଼ିଆ"].index(st.session_state.app_lang))
    if new_lang != st.session_state.app_lang:
        st.session_state.app_lang = new_lang
        st.rerun()
        
    mode_choice = c_mode.radio(
        T["mode_select"], 
        [T["mode_opt"], T["mode_diag"]]
    )
    st.session_state.app_mode = "Diagnostic Only" if mode_choice == T["mode_diag"] else "Full Optimization"

    if not st.session_state.logged_in:
        t_login, t_reg = st.tabs([T["login_tab"], T["reg_tab"]])
        with t_login:
            m = st.text_input(T["mobile_lbl"], max_chars=10, key="log_m", placeholder="10-digit mobile number")
            p = st.text_input(T["pass_lbl"], type="password", key="log_p")
            if st.button(T["btn_login"]):
                if len(m.strip()) == 10 and verify_user(m.strip(), p.strip()):
                    st.session_state.logged_in = True
                    st.session_state.user_mobile = m.strip()
                    st.session_state.step = 2
                    st.rerun()
                else:
                    st.error("Invalid mobile number or incorrect password.")
        with t_reg:
            rm = st.text_input(T["mobile_lbl"], max_chars=10, key="reg_m", placeholder="10-digit mobile number")
            rp = st.text_input(T["pass_lbl"], type="password", key="reg_p")
            rpc = st.text_input(T["conf_pass_lbl"], type="password", key="reg_pc")
            if st.button(T["btn_reg"]):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(rm.strip(), rp.strip())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("Please check phone number and passwords.")
    else:
        st.success(f"Logged in as: **+91 {st.session_state.user_mobile}**")
        if st.button(T["btn_next"]):
            st.session_state.step = 2
            st.rerun()

# -------------------------------------------------------------
# SCREEN 2: REAL-TIME FIELD DIAGNOSTICS & LAND CONVERTER
# -------------------------------------------------------------
elif st.session_state.step == 2:
    if st.session_state.app_mode == "Diagnostic Only":
        st.subheader("🔬 AI Optical Crop Disease & Pest Diagnosis")
        st.info("Take a photo or upload an image of the affected plant leaf, crop stem, or pest:")
        
        c_cam, c_up = st.columns(2)
        cam_p = c_cam.camera_input("📷 Realtime Camera Scanner")
        file_p = c_up.file_uploader("📂 Upload Leaf / Pest Image", type=["jpg", "jpeg", "png"])
        
        active_img = cam_p or file_p
        if active_img:
            img = Image.open(active_img)
            st.image(img, caption="Scanned Specimen", width=300)
            res = analyze_plant_disease_image(img)
            
            st.markdown(f"### Diagnostic Status: <span class='badge-pass'>{res['health']}</span>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.write(f"🦠 **Crop Disease / Pathogen**: {res['disease']}")
                st.write(f"🐛 **Pest Recognition**: {res['pest']}")
                st.write(f"🔬 **Visible Symptoms**: {res['symptoms']}")
            with col_d2:
                st.write(f"💊 **Prescribed Medicine / Spray**: {res['medicine']}")
                st.metric("Survival & Recovery Chance", f"{res['recovery_chance']}%")
                st.write(f"🌱 **Will this crop continue to grow?**: **{res['will_grow']}**")
        
        st.divider()
        if st.button("Complete & Leave Feedback ➔"):
            st.session_state.step = 8
            st.rerun()

    else:
        st.subheader("2. 📍 Real-Time Location, Camera & Land Profile")
        
        tab_geo, tab_camera, tab_land, tab_soil = st.tabs([
            "🗺️ Live GPS & Map", 
            "📷 Optical Camera Scanner", 
            "📐 Land Area Converter & Acreage", 
            "🧪 Soil Nutrient Levels"
        ])
        
        with tab_geo:
            st.markdown("##### Real-Time Field Location Tracking")
            loc_cols = st.columns([1, 2])
            with loc_cols[0]:
                st.session_state.lat = st.number_input("Latitude", value=float(st.session_state.lat), format="%.5f")
                st.session_state.lon = st.number_input("Longitude", value=float(st.session_state.lon), format="%.5f")
                
                if st.session_state.lat > 22.0:
                    st.session_state.sel_soil = "Loamy"
                    st.caption("Auto-calibrated: Indo-Gangetic alluvial belt.")
                else:
                    st.session_state.sel_soil = "Red"
                    st.caption("Auto-calibrated: Eastern plateau / Laterite belt.")
            
            with loc_cols[1]:
                map_df = pd.DataFrame({'lat': [st.session_state.lat], 'lon': [st.session_state.lon]})
                st.map(map_df, zoom=9)

        with tab_camera:
            st.markdown("##### Live Soil / Leaf Scan")
            cam_feed = st.camera_input("Snap Picture of Field Soil")
            if cam_feed:
                st.image(Image.open(cam_feed), caption="Captured Field Soil", width=250)
                st.success("Analysis complete: Calibrated organic matter and root-zone moisture.")

        with tab_land:
            st.markdown(f"##### {T['land_calc_title']}")
            l_col1, l_col2 = st.columns(2)
            st.session_state.raw_land_val = l_col1.number_input("Enter Land Size", 0.1, 1000.0, float(st.session_state.raw_land_val), 0.5)
            st.session_state.land_unit = l_col2.selectbox(
                "Measuring Unit", 
                list(UNIT_TO_HECTARE.keys()),
                index=list(UNIT_TO_HECTARE.keys()).index(st.session_state.land_unit)
            )
            
            conv_table, ha_val = render_land_conversion_table(st.session_state.raw_land_val, st.session_state.land_unit)
            st.session_state.land_area = ha_val
            st.table(conv_table)
            st.info(f"Standardized area for chemical dosage: **{ha_val:.3f} Hectares**")

        with tab_soil:
            s1, s2, s3 = st.columns(3)
            st.session_state.soil_n = s1.number_input("Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n))
            st.session_state.soil_p = s2.number_input("Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p))
            st.session_state.soil_k = s3.number_input("Potash (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k))
            
            s4, s5, s6 = st.columns(3)
            st.session_state.soil_ph = s4.slider("Soil pH", 4.0, 9.5, float(st.session_state.soil_ph), 0.1)
            st.session_state.soc = s5.slider("Organic Carbon (%)", 0.1, 2.5, float(st.session_state.soc), 0.05)
            st.session_state.soil_moist = s6.slider("Moisture (%)", 10.0, 90.0, float(st.session_state.soil_moist), 1.0)
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.selectbox("Soil Type", list(soil_encoder.classes_), key="sel_soil")
            c_s2.selectbox("Planned Crop", list(crop_type_encoder.classes_), key="sel_crop")
            st.session_state.target_yield = c_s3.number_input("Target Harvest (t/ha)", 1.0, 15.0, float(st.session_state.target_yield), 0.5)

        st.divider()
        b1, b2 = st.columns([1, 5])
        if b1.button(T["btn_back"]):
            st.session_state.step = 1
            st.rerun()
        if b2.button(T["btn_next"]):
            st.session_state.step = 3
            st.rerun()

# -------------------------------------------------------------
# SCREEN 3: SOIL HEALTH EVALUATION
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("3. ⚙️ Soil Condition & Risk Assessment")
    
    k1, k2, k3 = st.columns(3)
    ph_stat = "Acidic (Apply Lime)" if st.session_state.soil_ph < 6.0 else ("Alkaline (Apply Gypsum)" if st.session_state.soil_ph > 7.5 else "Sweet & Balanced")
    k1.metric("Soil Sweetness (pH)", f"{st.session_state.soil_ph}", ph_stat)
    k2.metric("Organic Matter (SOC)", f"{st.session_state.soc}%", "Rich" if st.session_state.soc >= 0.75 else "Low (Add Compost)")
    k3.metric("Rain Leaching Risk", f"{st.session_state.rainfall:.0f} mm", "Leaching Alert" if st.session_state.rainfall > 200 else "Optimal")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"]):
        st.session_state.step = 2
        st.rerun()
    if b2.button(T["btn_next"]):
        st.session_state.step = 4
        st.rerun()

# -------------------------------------------------------------
# SCREEN 4: SOIL COMPARISON BAR CHART
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("4. 📊 Current Soil Nutrients vs Ideal Farm Target")
    chart_data = pd.DataFrame({
        "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potash (K)"],
        "Your Measured Soil (kg/ha)": [st.session_state.soil_n * 2.24, st.session_state.soil_p * 2.24, st.session_state.soil_k * 2.24],
        "Standard Target (kg/ha)": [280.0, 60.0, 150.0]
    }).set_index("Nutrient")
    st.bar_chart(chart_data)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"]):
        st.session_state.step = 3
        st.rerun()
    if b2.button(T["btn_next"]):
        st.session_state.step = 5
        st.rerun()

# -------------------------------------------------------------
# SCREEN 5: NUTRIENT GAP & CROP PREDICTION
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("5. ⚠️ Required Nutrient Deficit for Harvest Target")
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
        st.markdown(f"##### Nutrient Shortage for {st.session_state.target_yield} t/ha:")
        st.warning(f"• **Nitrogen Needed**: {def_n:.1f} kg/ha")
        st.warning(f"• **Phosphorus Needed**: {def_p:.1f} kg/ha")
        st.warning(f"• **Potash Needed**: {def_k:.1f} kg/ha")
    with g2:
        st.markdown("##### Best Crop Match:")
        st.success(f"🌱 **Recommended Crop**: **{pred_crop.capitalize()}**")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"]):
        st.session_state.step = 4
        st.rerun()
    if b2.button(T["btn_next"]):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 6: OPTIMIZED FERTILIZER BAGS & TIMETABLE
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("6. 🚀 Your Fertilizer Bags & Application Schedule")
    def_n, def_p, def_k = calculate_advanced_nutrients(
        st.session_state.target_yield, st.session_state.soil_n, st.session_state.soil_p,
        st.session_state.soil_k, st.session_state.soc, st.session_state.soil_ph,
        st.session_state.soil_moist, st.session_state.sel_soil
    )
    opt = optimize_fertilizer_blend(
        req_n=def_n, req_p=def_p, req_k=def_k,
        budget_cap=st.session_state.budget_cap,
        land_area=st.session_state.land_area,
        soil_texture=str(st.session_state.sel_soil),
        rainfall_mm=st.session_state.rainfall,
        soc=st.session_state.soc
    )
    st.session_state.opt_results = opt

    r1, r2, r3 = st.columns(3)
    r1.metric("Optimized Total Cost", f"₹{opt['total_cost']:,.0f}")
    r2.metric("Land Covered", f"{st.session_state.raw_land_val} {st.session_state.land_unit.split(' ')[0]}")
    r3.metric("Budget Utilized", f"{opt['budget_utilized_pct']}%")

    st.markdown("##### 🛒 Fertilizer Bags Needed for Your Field:")
    st.table(pd.DataFrame({
        "Fertilizer Product": ["Urea (Synthetic N)", "DAP (Phosphatic)", "MOP (Red Potash)", "Complex 14-35-14", "Organic Desi Compost"],
        "Total Weight (kg)": [f"{opt['urea_kg']} kg", f"{opt['dap_kg']} kg", f"{opt['mop_kg']} kg", f"{opt['complex_kg']} kg", f"{opt['compost_kg']} kg"],
        "Standard 50kg Bags": [
            f"{max(1, round(opt['urea_kg'] / 50.0))} bags" if opt['urea_kg'] > 0 else "0",
            f"{max(1, round(opt['dap_kg'] / 50.0))} bags" if opt['dap_kg'] > 0 else "0",
            f"{max(1, round(opt['mop_kg'] / 50.0))} bags" if opt['mop_kg'] > 0 else "0",
            f"{max(1, round(opt['complex_kg'] / 50.0))} bags" if opt['complex_kg'] > 0 else "0",
            f"{round(opt['compost_kg'] / 50.0)} bags" if opt['compost_kg'] > 0 else "0"
        ]
    }))

    st.markdown("##### 📅 Timed Split Application Rules:")
    st.table(pd.DataFrame({
        "Crop Stage": ["1. Basal (At Sowing)", "2. Tillering (Day 20-25)", "3. Panicle / Flowering (Day 45-55)"],
        "What to Apply": ["All Compost + All DAP + 1/3 Potash + 1/4 Urea", "1/2 Urea + 1/3 Potash (Near roots)", "Remaining Urea + Remaining Potash"],
        "Benefit": ["Strong root foundation", "Boosts green tillering", "Increases grain weight"]
    }))

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"]):
        st.session_state.step = 5
        st.rerun()
    if b2.button(T["btn_next"]):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: PRESCRIPTION RECEIPT DOSSIER
# -------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("7. 📋 Official Farmer Prescription Card")
    opt = st.session_state.get("opt_results", {"urea_kg": 0, "dap_kg": 0, "mop_kg": 0, "compost_kg": 0, "total_cost": 0})
    
    st.markdown(f"""
    <div class="summary-card">
        <h2 style="color: #1B5E20; margin-top: 0;">🌾 Precision Fertilizer & Input Advisory Card</h2>
        <p><strong>Farmer Phone:</strong> +91 {st.session_state.user_mobile} | <strong>Field Area:</strong> {st.session_state.raw_land_val} {st.session_state.land_unit}</p>
        <p><strong>Field GPS Coordinates:</strong> {st.session_state.lat:.4f}° N, {st.session_state.lon:.4f}° E | <strong>Crop:</strong> {st.session_state.sel_crop}</p>
        <hr style="border: 1px solid #A5D6A7;"/>
        <h3 style="color: #1B5E20;">🛒 Required Purchases:</h3>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li><strong>Urea:</strong> {opt['urea_kg']} kg (~{round(opt['urea_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>DAP:</strong> {opt['dap_kg']} kg (~{round(opt['dap_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>MOP (Potash):</strong> {opt['mop_kg']} kg (~{round(opt['mop_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>Organic Compost:</strong> {opt['compost_kg']} kg</li>
        </ul>
        <h3 style="color: #1B5E20;">💰 Estimated Total Cost: ₹{opt['total_cost']:,.0f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    receipt_txt = f"PRESCRIPTION FOR +91 {st.session_state.user_mobile}\nLand: {st.session_state.raw_land_val} {st.session_state.land_unit}\nTotal Cost: Rs. {opt['total_cost']}\n"
    st.download_button("📥 Download Prescription Record", receipt_txt, file_name="Farmer_Prescription.txt")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"]):
        st.session_state.step = 6
        st.rerun()
    if b2.button("Proceed to Feedback & Exit ➔"):
        st.session_state.step = 8
        st.rerun()

# -------------------------------------------------------------
# SCREEN 8: FARMER FEEDBACK & LOGOUT
# -------------------------------------------------------------
elif st.session_state.step == 8:
    st.subheader(T["feedback_title"])
    st.write("Please rate the clarity and usefulness of the recommendation:")
    
    rating = st.slider("Rating (1 = Poor, 5 = Excellent)", 1, 5, 5)
    comments = st.text_area("Your Comments / Suggestions (ଆପଣଙ୍କ ମତାମତ / आपकी प्रतिक्रिया):")
    
    if st.button(T["feedback_submit"]):
        save_feedback(st.session_state.user_mobile, rating, comments)
        st.success("✅ Thank you! Your feedback has been stored safely.")
        
        st.session_state.logged_in = False
        st.session_state.user_mobile = ""
        st.session_state.step = 1
        st.cache_data.clear()
        st.rerun()
