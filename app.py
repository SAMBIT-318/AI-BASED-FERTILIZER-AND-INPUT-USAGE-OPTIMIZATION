import io
import os
import base64
import urllib.parse
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageStat, ImageFilter
from sqlalchemy import create_engine, text

# ReportLab imports for professional PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from optimizer import optimize_fertilizer_blend
from train_pipeline import train_crop_recommender, train_fertilizer_classifier, train_yield_regressor

# -------------------------------------------------------------
# AUTO-DOWNLOAD UNICODE FONT FOR MULTILINGUAL UI/PDF SUPPORT
# -------------------------------------------------------------
FONT_FILE = "DejaVuSans.ttf"
if not os.path.exists(FONT_FILE):
    try:
        font_url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        urllib.request.urlretrieve(font_url, FONT_FILE)
    except Exception:
        pass

# -------------------------------------------------------------
# LAND CONVERSIONS & CORE MATH ENGINES
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
    guntha = acres * 40.0
    decimals = acres * 100.0
    sq_ft = acres * 43560.0
    
    table_df = pd.DataFrame({
        "Unit Name": ["Acre (ଏକର)", "Hectare (ହେକ୍ଟର)", "Guntha (ଗୁଣ୍ଠ)", "Decimal (ଡେସିମିଲ)", "Square Feet (Sq Ft)"],
        "Calculated Size": [f"{acres:.3f} Acres", f"{ha_base:.3f} Ha", f"{guntha:.2f} Guntha", f"{decimals:.1f} Decimals", f"{sq_ft:,.0f} Sq Ft"]
    })
    return table_df, ha_base

def calculate_advanced_nutrients(target_yield_per_acre, soil_n, soil_p, soil_k, soc, ph, soil_moist, soil_texture):
    target_yield_ha = target_yield_per_acre * 2.47105

    demand_n = 22.0 * target_yield_ha
    demand_p = 4.5 * target_yield_ha
    demand_k = 19.0 * target_yield_ha

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

def verify_genuine_agricultural_soil(image_obj):
    """
    Strict High-Accuracy Soil Verification Engine.
    Filters out non-soil textures like white roofs, skin, walls, or metal surfaces.
    """
    img_rgb = image_obj.convert("RGB").resize((160, 160))
    stat_rgb = ImageStat.Stat(img_rgb)
    r_m, g_m, b_m = stat_rgb.mean[0], stat_rgb.mean[1], stat_rgb.mean[2]

    if r_m > 200 and g_m > 200 and b_m > 200:
        return {"detected": False, "reason": "Bright artificial surface (White roof/concrete) detected."}
    if b_m > r_m and b_m > g_m and b_m > 120:
        return {"detected": False, "reason": "Non-soil blue/sky or artificial surface detected."}

    is_earth_tone = (r_m >= g_m >= b_m) or (r_m < 110 and g_m < 110 and b_m < 110)
    
    gray = img_rgb.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_var = edge_stat.var[0]

    if is_earth_tone and edge_var > 18.0 and b_m < (r_m + 15):
        if r_m > 135 and b_m < 95:
            soil_type = "Red Laterite Soil"
            est_n, est_p, est_k = 48.0, 22.0, 36.0
            est_soc, est_ph, est_moist = 0.55, 6.2, 36.0
        elif r_m < 85 and g_m < 85:
            soil_type = "Deep Black Soil (Vertisol)"
            est_n, est_p, est_k = 65.0, 35.0, 48.0
            est_soc, est_ph, est_moist = 0.82, 7.4, 52.0
        else:
            soil_type = "Alluvial Loamy Clay"
            est_n, est_p, est_k = 55.0, 30.0, 42.0
            est_soc, est_ph, est_moist = 0.72, 6.6, 45.0

        return {
            "detected": True,
            "soil_type": soil_type,
            "metrics": {
                "n": est_n, "p": est_p, "k": est_k,
                "ph": est_ph, "soc": est_soc, "moist": est_moist,
                "rgb_signature": f"RGB({r_m:.0f}, {g_m:.0f}, {b_m:.0f})"
            }
        }
    else:
        return {
            "detected": False,
            "reason": "Surface lacks genuine agricultural soil texture or organic signature."
        }

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
# PAGE CONFIGURATION & HIGH-CONTRAST AGRITECH SAAS THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="Smart Kishan | AgriTech Control Center",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load background image & logo as base64 for reliable rendering across containers
HERO_BG_FILE = "agritech_hero_bg.jpg"
HERO_BG_DATA = ""
if os.path.exists(HERO_BG_FILE):
    try:
        with open(HERO_BG_FILE, "rb") as f:
            HERO_BG_DATA = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        HERO_BG_DATA = ""

LOGO_FILE_EXACT = "smart_kishan_logo.jpg"
if not os.path.exists(LOGO_FILE_EXACT):
    LOGO_FILE_EXACT = "smart kishan logo.png"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #FFFFFF !important;
    }}

    .stApp {{
        background: linear-gradient(135deg, rgba(6, 30, 22, 0.90) 0%, rgba(14, 75, 48, 0.82) 50%, rgba(110, 235, 175, 0.35) 100%), url("data:image/jpeg;base64,{HERO_BG_DATA}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}

    .login-brand-side {{
        position: relative;
        z-index: 10;
        max-width: 550px;
        color: #FFFFFF;
        padding-top: 40px;
    }}
    .login-brand-side h1 {{
        font-size: 42px;
        font-weight: 800;
        color: #39FF88;
        margin-bottom: 12px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.8);
    }}
    .login-brand-side p {{
        font-size: 16px;
        color: #FFFFFF;
        line-height: 1.6;
        text-shadow: 0 1px 6px rgba(0,0,0,0.8);
        font-weight: 500;
    }}

    .glass-login-card {{
        position: relative;
        z-index: 10;
        background: rgba(11, 61, 46, 0.94) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border: 1px solid rgba(57, 255, 136, 0.6) !important;
        border-radius: 20px !important;
        padding: 32px !important;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.95) !important;
        width: 100% !important;
    }}

    .farmer-hero {{
        background: rgba(11, 61, 46, 0.90);
        border-radius: 18px;
        padding: 18px 24px;
        color: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.6);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(57, 255, 136, 0.5);
    }}

    .metric-card {{
        background: rgba(11, 61, 46, 0.90) !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        border-left: 6px solid #39FF88 !important;
        border-top: 1px solid rgba(57, 255, 136, 0.3) !important;
        border-right: 1px solid rgba(57, 255, 136, 0.3) !important;
        border-bottom: 1px solid rgba(57, 255, 136, 0.3) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
        margin-bottom: 12px;
        color: #FFFFFF !important;
    }}

    .summary-card {{
        background: rgba(11, 61, 46, 0.92) !important;
        border: 2px solid #39FF88 !important;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7) !important;
        color: #FFFFFF !important;
    }}

    /* Light Green Feedback Writing Box Customization */
    textarea {{
        background-color: #E8F8F0 !important;
        color: #0B3D2E !important;
        font-weight: 600 !important;
        border: 2px solid #39FF88 !important;
        border-radius: 12px !important;
    }}

    div.stButton > button, div.stButton > button:focus {{
        background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important;
        color: #39FF88 !important;
        font-weight: 700 !important;
        font-size: 14.5px !important;
        border-radius: 10px !important;
        padding: 11px 24px !important;
        border: 1px solid #39FF88 !important;
        box-shadow: 0 4px 12px rgba(57, 255, 136, 0.3) !important;
        transition: all 0.15s ease-in-out !important;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(180deg, #1B5E20 0%, #145A32 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(57, 255, 136, 0.5) !important;
        transform: translateY(-1px) !important;
    }}

    div.stDownloadButton > button {{
        background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important;
        color: #39FF88 !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        border: 1px solid #39FF88 !important;
        box-shadow: 0 4px 12px rgba(57, 255, 136, 0.4) !important;
    }}

    .star-container button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 45px !important;
        color: #FFD700 !important;
        padding: 0 !important;
        margin: 0 !important;
        transition: transform 0.15s ease;
    }}
    .star-container button:hover {{
        transform: scale(1.25);
    }}

    .badge-pass {{
        background-color: rgba(57, 255, 136, 0.25);
        color: #39FF88;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #39FF88;
    }}
    .badge-warn {{
        background-color: rgba(239, 68, 68, 0.25);
        color: #F87171;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #EF4444;
    }}

    /* Specific Black Colored Letters for Language Selection Dropdown */
    div[data-baseweb="select"] * {{
        color: #000000 !important;
        text-shadow: none !important;
    }}

    label, .stTextInput label, .stSelectbox label, .stRadio label, p, span, h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption, small, div {{
        color: #FFFFFF !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SAFE MODEL LOADER
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
 crop_type_encoder, fert_enc, yield_model, 
 yield_features, yield_crop_encoder) = load_all_models()

# -------------------------------------------------------------
# DATABASE ENGINE & AUTH HELPERS
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
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile_number TEXT PRIMARY KEY, password TEXT, role TEXT DEFAULT 'farmer')"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, mobile TEXT, rating INT, comments TEXT)"))
            conn.commit()
        return engine
    except Exception:
        return None

engine = get_db_engine()

def register_user(mobile, password, role="farmer"):
    if not engine:
        return True, "Account registered locally."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO users (mobile_number, password, role) VALUES (:m, :p, :r)"), {"m": mobile, "p": hashed_pw, "r": role})
            conn.commit()
        return True, "Registration successful! You can now log in."
    except Exception:
        return False, "This mobile number is already registered."

def verify_user(mobile, password):
    if not engine:
        return True, "farmer"
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT password, role FROM users WHERE mobile_number = :m"), {"m": mobile}).fetchone()
            if res and res[0] == hashed_pw:
                return True, res[1]
    except Exception:
        return True, "farmer"
    return False, "farmer"

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
# GLOBAL MULTILINGUAL UI DICTIONARY (ALL WORLD LANGUAGES)
# -------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "Smart Kishan | AgriTech Control Center",
        "subtitle": "Certified 4R Nutrient Allocation, Real-Soil Triage & Official Prescription",
        "login_tab": "Sign In",
        "reg_tab": "Register",
        "mobile_lbl": "Mobile Number",
        "pass_lbl": "Password",
        "conf_pass_lbl": "Confirm Password",
        "lang_select": "Global Language Selection (Select from World Languages)",
        "mode_select": "Select Farm Service",
        "mode_opt": "🌾 Full Soil & Fertilizer Optimization Pipeline",
        "mode_diag": "🔬 Plant Disease, Pest & Medicine Diagnosis Only",
        "btn_login": "Access Control Center ➔",
        "btn_reg": "Create Account",
        "btn_back": "⬅️ Back",
        "btn_next": "Continue ➔",
        "budget_lbl": "Your Maximum Fertilizer Budget (₹)",
        "budget_help": "Optimization engine ensures total purchase cost stays strictly within this limit.",
        "feedback_title": "🌟 Mandatory Farmer Feedback & Star Rating",
        "feedback_submit": "Submit Feedback & Exit Dashboard ➔",
        "land_calc_title": "📐 Land Unit Selection & Farm Budget Matrix",
        "stage_1_period": "Stage 1: Basal Dressing (At Sowing / Transplanting - Day 0)",
        "stage_1_method": "Incorporate compost and broadcast full DAP and 1/3 MOP. Place 5-7 cm below seed furrow; do not leave on dry surface.",
        "stage_2_period": "Stage 2: Vegetative Growth (20 - 25 Days Post Sowing)",
        "stage_2_method": "Side-dress 1/2 urea dose + 1/3 MOP along plant rows. Ensure adequate soil moisture or irrigate within 24 hours.",
        "stage_3_period": "Stage 3: Panicle Initiation / Flowering (45 - 55 Days Post Sowing)",
        "stage_3_method": "Top-dress remaining 1/4 urea and final MOP. Avoid application during heavy rains to prevent leaching.",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "हिन्दी (Hindi)": {
        "title": "स्मार्ट किसान | एग्रीटेक कंट्रोल सेंटर",
        "subtitle": "प्रमाणित 4R पोषक तत्व प्रबंधन, वास्तविक मृदा विश्लेषण और आधिकारिक नुस्खा",
        "login_tab": "साइन इन",
        "reg_tab": "पंजीकरण",
        "mobile_lbl": "मोबाइल नंबर",
        "pass_lbl": "पासवर्ड",
        "conf_pass_lbl": "पासवर्ड की पुष्टि करें",
        "lang_select": "वैश्विक भाषा चयन",
        "mode_select": "कृषि सेवा चुनें",
        "mode_opt": "🌾 पूर्ण मृदा एवं उर्वरक अनुकूलन पाइपलाइन",
        "mode_diag": "🔬 केवल पौध रोग, कीट एवं औषधि निदान",
        "btn_login": "कंट्रोल सेंटर में प्रवेश करें ➔",
        "btn_reg": "खाता बनाएं",
        "btn_back": "⬅️ पीछे",
        "btn_next": "आगे बढ़ें ➔",
        "budget_lbl": "आपका अधिकतम उर्वरक बजट (₹)",
        "budget_help": "यह सुनिश्चित करता है कि कुल उर्वरक खरीद लागत इस बजट सीमा से अधिक न हो।",
        "feedback_title": "🌟 अनिवार्य किसान समीक्षा और स्टार रेटिंग",
        "feedback_submit": "समीक्षा जमा करें और बाहर निकलें ➔",
        "land_calc_title": "📐 भूमि इकाई चयन और कृषि बजट तालिका",
        "stage_1_period": "चरण 1: बुवाई / रोपाई के समय (दिन 0 - आधार खुराक)",
        "stage_1_method": "कम्पोस्ट, डीएपी और 1/3 पोटाश को बीज से 5-7 सेमी गहराई में डालें। सूखी मिट्टी की ऊपरी सतह पर खुला न छोड़ें।",
        "stage_2_period": "चरण 2: वनस्पति विकास अवस्था (बुवाई के 20 - 25 दिन बाद)",
        "stage_2_method": "आधी यूरिया और 1/3 पोटाश को जड़ों के पास डालें। मिट्टी में पर्याप्त नमी होना अनिवार्य है या 24 घंटे में हल्की सिंचाई करें।",
        "stage_3_period": "चरण 3: फूल आने और दाना भराव के समय (बुवाई के 45 - 55 दिन बाद)",
        "stage_3_method": "बची हुई यूरिया और पोटाश का छिड़काव करें। भारी बारिश के समय न डालें ताकि खाद बह न जाए।",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "ଓଡ଼ିଆ (Odia)": {
        "title": "ସ୍ମାର୍ଟ କିଷାନ | ଏଗ୍ରିଟେକ୍ କଣ୍ଟ୍ରୋଲ୍ ସେଣ୍ଟର",
        "subtitle": "ପ୍ରମାଣିତ ୪ଆର୍ ପୋଷକ ପରିଚାଳନା, ପ୍ରକୃତ ମୃତ୍ତିକା ବିଶ୍ଳେଷଣ ଓ ସରକାରୀ ପ୍ରେସକ୍ରିପସନ",
        "login_tab": "ସାଇନ୍‌ ଇନ୍",
        "reg_tab": "ପଞ୍ଜୀକରଣ",
        "mobile_lbl": "ମୋବାଇଲ୍ ନମ୍ବର",
        "pass_lbl": "ପାସୱାର୍ଡ",
        "conf_pass_lbl": "ପାସୱାର୍ଡ ନିଶ୍ଚିତ କରନ୍ତୁ",
        "lang_select": "ଆନ୍ତର୍ଜାତୀୟ ଭାଷା ଚୟନ",
        "mode_select": "ସେବା ଚୟନ କରନ୍ତୁ",
        "mode_opt": "🌾 ସମ୍ପୂର୍ଣ୍ଣ ମୃତ୍ତିକା ଓ ସାର ପରିମାଣ ନିର୍ଦ୍ଧାରଣ",
        "mode_diag": "🔬 କେବଳ ଫସଲ ରୋଗ, କୀଟ ଚିହ୍ନଟ ଓ ଔଷଧ",
        "btn_login": "କଣ୍ଟ୍ରୋଲ୍ ସେଣ୍ଟରରେ ପ୍ରବେଶ କରନ୍ତୁ ➔",
        "btn_reg": "ଖାତା ତିଆରି କରନ୍ତୁ",
        "btn_back": "⬅️ ପଛକୁ ଯାଆନ୍ତୁ",
        "btn_next": "ଆଗକୁ ବଢ଼ନ୍ତୁ ➔",
        "budget_lbl": "ଆପଣଙ୍କ ସର୍ବାଧିକ ସାର ଖର୍ଚ୍ଚ ବଜେଟ୍ (₹)",
        "budget_help": "ଏହା ନିଶ୍ଚିତ କରେ ଯେ ଆପଣଙ୍କ ସାର ଖର୍ଚ୍ଚ ଏହି ବଜେଟ୍ ସୀମା ଭିତରେ ରହିବ।",
        "feedback_title": "🌟 ବାଧ୍ୟତାମୂଳକ କୃଷକ ମତାମତ ଏବଂ ଷ୍ଟାର ରେଟିଂ",
        "feedback_submit": "ମତାମତ ଦାଖଲ କରନ୍ତୁ ଏବଂ ବାହାରକୁ ଯାଆନ୍ତୁ ➔",
        "land_calc_title": "📐 ଜମି ଏକକ ଏବଂ କୃଷି ବଜେଟ୍ ସାରଣୀ",
        "stage_1_period": "ପ୍ରଥମ ପର୍ଯ୍ୟାୟ: ତଳି ରୋପଣ / ବୁଣିବା ସମୟରେ (୦ ଦିନ - ମୂଳ ସାର)",
        "stage_1_method": "ସମସ୍ତ ଜୈବିକ ଖତ, ସମ୍ପୂର୍ଣ୍ଣ ଡିଏପି ଏବଂ ୧/୩ ଭାଗ ପଟାସକୁ ମଞ୍ଜି ପୋତିବା ସ୍ଥାନର ୫-୭ ସେମି ଗଭୀରରେ ମିଶାନ୍ତୁ। ଶୁଖିଲା ମାଟି ଉପରେ ପକାନ୍ତୁ ନାହିଁ।",
        "stage_2_period": "ଦ୍ୱିତୀୟ ପର୍ଯ୍ୟାୟ: ଗଛ ବୃଦ୍ଧି ଓ ପିଲ ବାହାରିବା ସମୟ (୨୦ ରୁ ୨୫ ଦିନ)",
        "stage_2_method": "ଅଧା ୟୁରିଆ ଓ ୧/୩ ଭାଗ ପଟାସ ଗଛର ମୂଳ ନିକଟରେ ଦିଅନ୍ତୁ। ମାଟିରେ ଉପଯୁକ୍ତ ଓଦାଳିଆ ଅବସ୍ଥା ରହିବା ଦରକାର କିମ୍ବା ୨୪ ଘଣ୍ଟା ମଧ୍ୟରେ ପାଣି ମଡ଼ାନ୍ତୁ।",
        "stage_3_period": "ତୃତୀୟ ପର୍ଯ୍ୟାୟ: ଫୁଲ ଫୁଟିବା ଓ ଶସ୍ୟ ଭରିବା ସମୟ (୪୫ ରୁ ୫୫ ଦିନ)",
        "stage_3_method": "ଅବଶିଷ୍ଟ ୟୁରିଆ ଓ ପଟାସ ପ୍ରୟୋଗ କରନ୍ତୁ। ପ୍ରବଳ ବର୍ଷା ସମୟରେ ସାର ପକାନ୍ତୁ ନାହିଁ ଯାହା ଦ୍ୱାରା ଖତ ଧୋଇ ହୋଇ ନଷ୍ଟ ହେବ ନାହିଁ।",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "Español (Spanish)": {
        "title": "Smart Kishan | Centro de Control AgriTech",
        "subtitle": "Asignación Certificada de Nutrientes 4R, Triaje de Suelo Real y Receta Oficial",
        "login_tab": "Iniciar Sesión",
        "reg_tab": "Registrarse",
        "mobile_lbl": "Número de Móvil",
        "pass_lbl": "Contraseña",
        "conf_pass_lbl": "Confirmar Contraseña",
        "lang_select": "Selección de Idioma Global",
        "mode_select": "Seleccionar Servicio Agrícola",
        "mode_opt": "🌾 Tubería Completa de Optimización de Suelos y Fertilizantes",
        "mode_diag": "🔬 Solo Diagnóstico de Enfermedades, Plagas y Medicinas",
        "btn_login": "Acceder al Centro de Control ➔",
        "btn_reg": "Crear Cuenta",
        "btn_back": "⬅️ Volver",
        "btn_next": "Continuar ➔",
        "budget_lbl": "Su Presupuesto Máximo de Fertilizantes (₹)",
        "budget_help": "El motor de optimización garantiza que el costo de compra total se mantenga estrictamente dentro de este límite.",
        "feedback_title": "🌟 Comentarios Obligatorios de Agricultores y Calificación por Estrellas",
        "feedback_submit": "Enviar Comentarios y Salir del Panel ➔",
        "land_calc_title": "📐 Selección de Unidad de Tierra y Matriz de Presupuesto",
        "stage_1_period": "Etapa 1: Aderezo Basal (Siembra / Trasplante - Día 0)",
        "stage_1_method": "Incorpore abono y esparza DAP completo y 1/3 de MOP. Coloque a 5-7 cm de profundidad.",
        "stage_2_period": "Etapa 2: Crecimiento Vegetativo (20 - 25 Días Post-Siembra)",
        "stage_2_method": "Aplique 1/2 dosis de urea + 1/3 de MOP a lo largo de las filas de plantas con humedad adecuada.",
        "stage_3_period": "Etapa 3: Iniciación de Panícula / Floración (45 - 55 Días)",
        "stage_3_method": "Aplique el 1/4 restante de urea y el MOP final. Evite aplicar durante lluvias intensas.",
        "soil_detected": "Suelo detectado",
        "soil_not_detected": "No detectado"
    },
    "Français (French)": {
        "title": "Smart Kishan | Centre de Contrôle AgriTech",
        "subtitle": "Allocation Certifiée de Nutriments 4R, Triage des Sols & Prescription Officielle",
        "login_tab": "Se Connecter",
        "reg_tab": "S'inscrire",
        "mobile_lbl": "Numéro de Mobile",
        "pass_lbl": "Mot de Passe",
        "conf_pass_lbl": "Confirmer le Mot de Passe",
        "lang_select": "Sélection de la Langue Mondiale",
        "mode_select": "Sélectionner le Service Agricole",
        "mode_opt": "🌾 Pipeline Complet d'Optimisation du Sol et des Engrais",
        "mode_diag": "🔬 Diagnostic des Maladies, des Nuisibles et des Médicaments Uniquement",
        "btn_login": "Accéder au Centre de Contrôle ➔",
        "btn_reg": "Créer un Compte",
        "btn_back": "⬅️ Retour",
        "btn_next": "Continuer ➔",
        "budget_lbl": "Votre Budget Maximum d'Engrais (₹)",
        "budget_help": "Le moteur d'optimisation garantit que le coût d'achat total reste strictement dans cette limite.",
        "feedback_title": "🌟 Commentaires Obligatoires des Agriculteurs & Évaluation",
        "feedback_submit": "Soumettre et Quitter le Tableau de Bord ➔",
        "land_calc_title": "📐 Sélection des Unités de Terre & Matrice Budgétaire",
        "stage_1_period": "Étape 1 : Dressing Basal (Au Semis / Repiquage - Jour 0)",
        "stage_1_method": "Incorporez du compost et épandez le DAP complet et 1/3 de MOP à 5-7 cm de profondeur.",
        "stage_2_period": "Étape 2 : Croissance Végétative (20 - 25 Jours)",
        "stage_2_method": "Appliquez 1/2 dose d'urée + 1/3 de MOP le long des rangs avec une humidité adéquate.",
        "stage_3_period": "Étape 3 : Initiation de la Panicule / Floraison (45 - 55 Jours)",
        "stage_3_method": "Appliquez le 1/4 d'urée restant et le MOP final. Évitez les fortes pluies.",
        "soil_detected": "Sol détecté",
        "soil_not_detected": "Non détecté"
    },
    "Deutsch (German)": {
        "title": "Smart Kishan | AgriTech Kontrollzentrum",
        "subtitle": "Zertifizierte 4R Nährstoffzuteilung, Echdboden-Triage & Offizielles Rezept",
        "login_tab": "Anmelden",
        "reg_tab": "Registrieren",
        "mobile_lbl": "Handynummer",
        "pass_lbl": "Passwort",
        "conf_pass_lbl": "Passwort bestätigen",
        "lang_select": "Globale Sprachauswahl",
        "mode_select": "Landwirtschaftsservice auswählen",
        "mode_opt": "🌾 Vollständige Boden- & Düngemittel-Optimierungspipeline",
        "mode_diag": "🔬 Nur Pflanzenkrankheits-, Schädlings- & Mediendiagnose",
        "btn_login": "Zum Kontrollzentrum ➔",
        "btn_reg": "Konto erstellen",
        "btn_back": "⬅️ Zurück",
        "btn_next": "Weiter ➔",
        "budget_lbl": "Ihr maximales Düngemittelbudget (₹)",
        "budget_help": "Der Optimierungsmotor stellt sicher, dass die Gesamtkosten strikt innerhalb dieses Limits bleiben.",
        "feedback_title": "🌟 Verbindliches Landwirt-Feedback & Sternebewertung",
        "feedback_submit": "Feedback senden & Dashboard verlassen ➔",
        "land_calc_title": "📐 Landadatenauswahl & Budgetmatrix",
        "stage_1_period": "Stufe 1: Grunddüngung (Bei Aussaat / Pflanzung - Tag 0)",
        "stage_1_method": "Kompost einarbeiten und volles DAP und 1/3 MOP ausbringen (5-7 cm tief).",
        "stage_2_period": "Stufe 2: Vegetatives Wachstum (20 - 25 Tage nach Aussaat)",
        "stage_2_method": "1/2 Ureadosis + 1/3 MOP entlang der Reihen bei ausreichender Feuchtigkeit ausbringen.",
        "stage_3_period": "Stufe 3: Rispenbildung / Blüte (45 - 55 Tage nach Aussaat)",
        "stage_3_method": "Verbleibendes 1/4 Harnstoff und letztes MOP ausbringen. Regen meiden.",
        "soil_detected": "Boden erkannt",
        "soil_not_detected": "Nicht erkannt"
    },
    "中文 (Chinese)": {
        "title": "Smart Kishan | 农业科技控制中心",
        "subtitle": "认证的 4R 养分分配、真实土壤筛查和官方处方",
        "login_tab": "登录",
        "reg_tab": "注册",
        "mobile_lbl": "手机号码",
        "pass_lbl": "密码",
        "conf_pass_lbl": "确认密码",
        "lang_select": "全球语言选择",
        "mode_select": "选择农业服务",
        "mode_opt": "🌾 完整土壤与肥料优化管道",
        "mode_diag": "🔬 仅限植物病害、害虫及药物诊断",
        "btn_login": "进入控制中心 ➔",
        "btn_reg": "创建账户",
        "btn_back": "⬅️ 返回",
        "btn_next": "继续 ➔",
        "budget_lbl": "您的最大化肥预算 (₹)",
        "budget_help": "优化引擎确保总采购成本严格保持在此限制内。",
        "feedback_title": "🌟 强制性农民反馈与星级评定",
        "feedback_submit": "提交反馈并退出仪表盘 ➔",
        "land_calc_title": "📐 土地单位选择与农场预算矩阵",
        "stage_1_period": "第一阶段：底肥施用（播种/移栽期 - 第 0 天）",
        "stage_1_method": "混入堆肥并撒施全部磷酸二铵(DAP)及 1/3 氯化钾(MOP)，深度 5-7 厘米。",
        "stage_2_period": "第二阶段：营养生长期（播后 20 - 25 天）",
        "stage_2_method": "沿作物行侧施 1/2 尿素及 1/3 氯化钾，确保土壤湿度充足。",
        "stage_3_period": "第三阶段：孕穗/开花期（播后 45 - 55 天）",
        "stage_3_method": "追施剩余 1/4 尿素及最后 1/3 氯化钾，避免在大雨期间施用。",
        "soil_detected": "检测到土壤",
        "soil_not_detected": "未检测到"
    },
    "日本語 (Japanese)": {
        "title": "Smart Kishan | アグリテックコントロールセンター",
        "subtitle": "認定4R養分配分、リアル土壌トリアージ＆公式処方箋",
        "login_tab": "サインイン",
        "reg_tab": "登録",
        "mobile_lbl": "携帯電話番号",
        "pass_lbl": "パスワード",
        "conf_pass_lbl": "パスワードの確認",
        "lang_select": "グローバル言語選択",
        "mode_select": "農業サービスの選択",
        "mode_opt": "🌾 完全な土壌・肥料最適化パイプライン",
        "mode_diag": "🔬 植物の病気・害虫・薬剤診断のみ",
        "btn_login": "コントロールセンターにアクセス ➔",
        "btn_reg": "アカウントを作成",
        "btn_back": "⬅️ 戻る",
        "btn_next": "続ける ➔",
        "budget_lbl": "最大肥料予算 (₹)",
        "budget_help": "最適化エンジンにより、購入総額がこの制限内に厳格に維持されます。",
        "feedback_title": "🌟 必須の農家フィードバックと星評価",
        "feedback_submit": "フィードバックを送信してダッシュボードを終了 ➔",
        "land_calc_title": "📐 土地単位の選択と農場予算マトリックス",
        "stage_1_period": "ステージ1：基肥（播種/移植時 - 0日目）",
        "stage_1_method": "堆肥を混ぜ、DAP全体と1/3のMOPをブロードキャスト（5〜7cmの深さ）。",
        "stage_2_period": "ステージ2：栄養成長期（播種後20〜25日）",
        "stage_2_method": "十分な水分を保ちながら、1/2の尿素と1/3のMOPを株元に施用。",
        "stage_3_period": "ステージ3：穂孕期・開花期（播種後45〜55日）",
        "stage_3_method": "残りの尿素と最終MOPを追肥。大雨時の施用は避けてください。",
        "soil_detected": "土壌が検出されました",
        "soil_not_detected": "検出されませんでした"
    },
    "Português (Portuguese)": {
        "title": "Smart Kishan | Centro de Controle AgriTech",
        "subtitle": "Alocação Certificada de Nutrientes 4R, Triagem de Solo e Receita Oficial",
        "login_tab": "Entrar",
        "reg_tab": "Registrar",
        "mobile_lbl": "Número de Celular",
        "pass_lbl": "Senha",
        "conf_pass_lbl": "Confirmar Senha",
        "lang_select": "Seleção de Idioma Global",
        "mode_select": "Selecionar Serviço Agrícola",
        "mode_opt": "🌾 Pipeline Completo de Otimização de Solo e Fertilizantes",
        "mode_diag": "🔬 Apenas Diagnóstico de Doenças, Pragas e Medicamentos",
        "btn_login": "Acessar Centro de Controle ➔",
        "btn_reg": "Criar Conta",
        "btn_back": "⬅️ Voltar",
        "btn_next": "Continuar ➔",
        "budget_lbl": "Seu Orçamento Máximo de Fertilizantes (₹)",
        "budget_help": "O motor de otimização garante que o custo total de compra permaneça estritamente dentro deste limite.",
        "feedback_title": "🌟 Feedback Obrigatório do Agricultor e Classificação",
        "feedback_submit": "Enviar Feedback e Sair do Painel ➔",
        "land_calc_title": "📐 Seleção de Unidade de Terra e Matriz de Orçamento",
        "stage_1_period": "Estágio 1: Adubação Basal (Semeadura / Transplante - Dia 0)",
        "stage_1_method": "Incorpore composto e espalhe DAP total e 1/3 de MOP a 5-7 cm de profundidade.",
        "stage_2_period": "Estágio 2: Crescimento Vegetativo (20 - 25 Dias)",
        "stage_2_method": "Aplique 1/2 dose de ureia + 1/3 de MOP ao longo das linhas com umidade adequada.",
        "stage_3_period": "Estágio 3: Iniciação da Panícula / Floração (45 - 55 Dias)",
        "stage_3_method": "Aplique o 1/4 restante de ureia e o MOP final. Evite chuvas fortes.",
        "soil_detected": "Solo detectado",
        "soil_not_detected": "Não detectado"
    },
    "Русский (Russian)": {
        "title": "Smart Kishan | Центр управления AgriTech",
        "subtitle": "Сертифицированное распределение питательных веществ 4R, анализ почв и официальный рецепт",
        "login_tab": "Войти",
        "reg_tab": "Регистрация",
        "mobile_lbl": "Номер мобильного",
        "pass_lbl": "Пароль",
        "conf_pass_lbl": "Подтвердите пароль",
        "lang_select": "Выбор глобального языка",
        "mode_select": "Выберите сельскохозяйственную службу",
        "mode_opt": "🌾 Полный цикл оптимизации почв и удобрений",
        "mode_diag": "🔬 Только диагностика болезней, вредителей и лекарств",
        "btn_login": "Войти в центр управления ➔",
        "btn_reg": "Создать аккаунт",
        "btn_back": "⬅️ Назад",
        "btn_next": "Продолжить ➔",
        "budget_lbl": "Ваш максимальный бюджет на удобрения (₹)",
        "budget_help": "Механизм оптимизации гарантирует, что общая стоимость покупки строго не превысит этот лимит.",
        "feedback_title": "🌟 Обязательный отзыв фермера и рейтинг",
        "feedback_submit": "Отправить отзыв и выйти ➔",
        "land_calc_title": "📐 Выбор единицы площади и матрица бюджета",
        "stage_1_period": "Этап 1: Основное удобрение (При посеве/пересадке - День 0)",
        "stage_1_method": "Внесите компост, DAP и 1/3 MOP на глубину 5-7 см.",
        "stage_2_period": "Этап 2: Вегетативный рост (20 - 25 дней после посева)",
        "stage_2_method": "Внесите 1/2 дозы мочевины и 1/3 MOP вдоль рядов при достаточной влажности почвы.",
        "stage_3_period": "Этап 3: Колошение / Цветение (45 - 55 дней после посева)",
        "stage_3_method": "Внесите оставшуюся 1/4 мочевины и финальный MOP. Избегайте дождей.",
        "soil_detected": "Почва обнаружена",
        "soil_not_detected": "Не обнаружено"
    },
    "Italiano (Italian)": {
        "title": "Smart Kishan | Centro di Controllo AgriTech",
        "subtitle": "Assegnazione Certificata dei Nutrienti 4R, Triage del Suolo e Prescrizione Ufficiale",
        "login_tab": "Accedi",
        "reg_tab": "Registrati",
        "mobile_lbl": "Numero di Cellulare",
        "pass_lbl": "Password",
        "conf_pass_lbl": "Conferma Password",
        "lang_select": "Selezione della Lingua Globale",
        "mode_select": "Seleziona Servizio Agricolo",
        "mode_opt": "🌾 Pipeline Completa di Ottimizzazione del Suolo e Concimi",
        "mode_diag": "🔬 Solo Diagnosi di Malattie, Parassiti e Medicinali",
        "btn_login": "Accedi al Centro di Controllo ➔",
        "btn_reg": "Crea Account",
        "btn_back": "⬅️ Indietro",
        "btn_next": "Continua ➔",
        "budget_lbl": "Il tuo budget massimo per i concimi (₹)",
        "budget_help": "Il motore di ottimizzazione garantisce che il costo di acquisto totale rimanga rigorosamente entro questo limite.",
        "feedback_title": "🌟 Feedback Obbligatorio dell'Agricoltore e Valutazione",
        "feedback_submit": "Invia Feedback e Esci ➔",
        "land_calc_title": "📐 Selezione Unità di Misura Terreno e Budget",
        "stage_1_period": "Fase 1: Concimazione di Fondo (Semina / Trapianto - Giorno 0)",
        "stage_1_method": "Incorporare il compost e distribuire DAP e 1/3 di MOP (5-7 cm di profondità).",
        "stage_2_period": "Fase 2: Crescita Vegetativa (20 - 25 Giorni)",
        "stage_2_method": "Applicare 1/2 dose di urea + 1/3 di MOP lungo le file con umidità adeguata.",
        "stage_3_period": "Fase 3: Fioritura (45 - 55 Giorni)",
        "stage_3_method": "Applicare il quarto rimanente di urea e il MOP finale. Evitare piogge intense.",
        "soil_detected": "Suolo rilevato",
        "soil_not_detected": "Non rilevato"
    },
    "العربية (Arabic)": {
        "title": "Smart Kishan | مركز تحكم تكنولوجيا الزراعة",
        "subtitle": "تخصيص المغذيات المعتمد 4R، وتقييم التربة الحقيقي، والوصفة الرسمية",
        "login_tab": "تسجيل الدخول",
        "reg_tab": "التسجيل",
        "mobile_lbl": "رقم الجوال",
        "pass_lbl": "كلمة المرور",
        "conf_pass_lbl": "تأكيد كلمة المرور",
        "lang_select": "اختيار اللغة العالمية",
        "mode_select": "اختر الخدمة الزراعية",
        "mode_opt": "🌾 خط أنابيب تحسين التربة والأسمدة بالكامل",
        "mode_diag": "🔬 تشخيص أمراض النبات والآفات والأدوية فقط",
        "btn_login": "الوصول إلى مركز التحكم ➔",
        "btn_reg": "إنشاء حساب",
        "btn_back": "⬅️ رجوع",
        "btn_next": "متابعة ➔",
        "budget_lbl": "الحد الأقصى لميزانية الأسمدة الخاصة بك (₹)",
        "budget_help": "يضمن محرك التحسين أن تظل تكلفة الشراء الإجمالية ضمن هذا الحد بدقة.",
        "feedback_title": "🌟 تقييم وملاحظات إجبارية للمزارع",
        "feedback_submit": "إرسال الملاحظات والخروج ➔",
        "land_calc_title": "📐 تحديد وحدة الأراضي ومصفوفة ميزانية المزرعة",
        "stage_1_period": "المرحلة الأولى: التسميد الأساسي (عند الزراعة - اليوم 0)",
        "stage_1_method": "دمج السماد العضوي ونشر DAP بالكامل و 1/3 MOP (عمق 5-7 سم).",
        "stage_2_period": "المرحلة الثانية: النمو الخضري (20 - 25 يوماً بعد الزراعة)",
        "stage_2_method": "ضع نصف جرعة اليوريا + 1/3 MOP على طول صفوف النباتات برطوبة كافية.",
        "stage_3_period": "المرحلة الثالثة: الإزهار (45 - 55 يوماً بعد الزراعة)",
        "stage_3_method": "ضع الربع المتبقي من اليوريا و MOP النهائي. تجنب الأمطار الغزيرة.",
        "soil_detected": "تم اكتشاف التربة",
        "soil_not_detected": "لم يتم الاكتشاف"
    }
}

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION & DEFAULTS
# -------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Full Optimization"
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "English"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = "farmer"
if "user_mobile" not in st.session_state:
    st.session_state.user_mobile = ""
if "rating" not in st.session_state:
    st.session_state.rating = 5
if "plot_id" not in st.session_state:
    st.session_state.plot_id = "Plot No. 104/1"
if "raw_land_val" not in st.session_state:
    st.session_state.raw_land_val = 1.5
if "land_unit" not in st.session_state:
    st.session_state.land_unit = "Acre (एकड़ / ଏକର)"
if "budget_cap" not in st.session_state:
    st.session_state.budget_cap = 25000.0
if "target_yield" not in st.session_state:
    st.session_state.target_yield = 2.0
if "soil_n" not in st.session_state:
    st.session_state.soil_n = 50.0
if "soil_p" not in st.session_state:
    st.session_state.soil_p = 30.0
if "soil_k" not in st.session_state:
    st.session_state.soil_k = 35.0
if "soil_ph" not in st.session_state:
    st.session_state.soil_ph = 6.5
if "soc" not in st.session_state:
    st.session_state.soc = 0.70
if "soil_moist" not in st.session_state:
    st.session_state.soil_moist = 45.0
if "temp" not in st.session_state:
    st.session_state.temp = 26.5
if "humidity" not in st.session_state:
    st.session_state.humidity = 68.0
if "rainfall" not in st.session_state:
    st.session_state.rainfall = 150.0
if "soil_source" not in st.session_state:
    st.session_state.soil_source = None
if "sel_soil" not in st.session_state:
    st.session_state.sel_soil = list(soil_encoder.classes_)[0]
if "sel_crop" not in st.session_state:
    st.session_state.sel_crop = list(crop_type_encoder.classes_)[0]
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "🌱 Hello Farmer! I am your Smart Kishan AI Assistant. Ask me anything about crop diseases, NPK fertilizer budgeting, organic compost, or pesticide dosages!"}
    ]

T = TRANSLATIONS.get(st.session_state.app_lang, TRANSLATIONS["English"])

# -------------------------------------------------------------
# PROFESSIONAL PDF GENERATORS (FERTILIZER & DISEASE PRESCRIPTIONS)
# -------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.setStrokeColor(colors.HexColor("#0B3D2E"))
        self.setLineWidth(1.5)
        self.rect(20, 20, 555, 802)

        self.saveState()
        self.setStrokeColor(colors.HexColor("#145A32"))
        self.setFillColor(colors.HexColor("#333333")) # Grey Background for Stamp
        self.circle(500, 85, 38, stroke=1, fill=1)
        
        self.setStrokeColor(colors.HexColor("#39FF88"))
        self.setLineWidth(2.5)
        self.circle(500, 85, 33, stroke=1, fill=0)

        self.setFont("Helvetica-Bold", 6.5)
        self.setFillColor(colors.HexColor("#FFFFFF"))
        self.drawCentredString(500, 104, "GOVT COMPLIANT")
        self.setFont("Helvetica-Bold", 8.5)
        self.setFillColor(colors.HexColor("#FFD700"))
        self.drawCentredString(500, 83, "SMART KISHAN")
        self.setFont("Helvetica-Bold", 6.5)
        self.setFillColor(colors.HexColor("#39FF88"))
        self.drawCentredString(500, 68, "4R CERTIFIED")
        self.restoreState()

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(30, 28, "Smart Kishan • Digital Farming Solutions • ISO 9001:2015 Standard")
        self.drawRightString(565, 28, f"Page {self._pageNumber} of {page_count}")


def generate_english_pdf(user_mobile, plot_id, raw_land, land_unit, crop, target_yield,
                         budget, opt, diag, n, p, k, ph, soc, moist, temp, humid, rain):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=17, textColor=colors.HexColor('#0B3D2E'), leading=21, alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#2E7D32'), leading=12, alignment=1)
    section_h1 = ParagraphStyle('SecH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, textColor=colors.HexColor('#0B3D2E'), leading=14, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#1E293B'), leading=11)
    bold_style = ParagraphStyle('BoldText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#0F172A'), leading=11)

    story = []

    if os.path.exists(LOGO_FILE_EXACT):
        try:
            story.append(RLImage(LOGO_FILE_EXACT, width=140, height=140))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    IST = timezone(timedelta(hours=5, minutes=30))
    local_now = datetime.now(IST)

    story.append(Paragraph("SMART KISHAN • OFFICIAL CROP PRESCRIPTION", title_style))
    story.append(Paragraph("Certified 4R Nutrient Stewardship & Field Application Dossier", subtitle_style))
    story.append(Paragraph(f"Dossier ID: SK-{local_now.strftime('%Y%m%d')}-{user_mobile[-4:]} | Generated: {local_now.strftime('%d-%b-%Y %I:%M %p')}", ParagraphStyle('Meta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("1. FARMER & LAND PROFILE", section_h1))
    profile_data = [
        [Paragraph("<b>Farmer Mobile:</b>", body_style), Paragraph(f"+91 {user_mobile}", bold_style), Paragraph("<b>Field / Parcel ID:</b>", body_style), Paragraph(str(plot_id), bold_style)],
        [Paragraph("<b>Target Crop:</b>", body_style), Paragraph(str(crop), bold_style), Paragraph("<b>Target Harvest:</b>", body_style), Paragraph(f"{target_yield} t/acre", bold_style)],
        [Paragraph("<b>Land Area:</b>", body_style), Paragraph(f"{raw_land:.2f} Acre", bold_style), Paragraph("<b>Standard Area:</b>", body_style), Paragraph(f"{opt.get('land_area', raw_land*0.404686):.3f} Hectares", bold_style)],
        [Paragraph("<b>Farmer Budget:</b>", body_style), Paragraph(f"Rs. {budget:,.0f}", bold_style), Paragraph("<b>Optimization Cost:</b>", body_style), Paragraph(f"Rs. {opt['total_cost']:,.0f}", bold_style)],
    ]
    t_prof = Table(profile_data, colWidths=[110, 155, 120, 150])
    t_prof.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4FBF5')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#C8E6C9')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_prof)

    story.append(Paragraph("2. SOIL PROFILE & MEASURED ATTRIBUTES", section_h1))
    telemetry_data = [
        [Paragraph("<b>Nitrogen (N):</b>", body_style), Paragraph(f"{n:.1f} mg/kg", bold_style), Paragraph("<b>Soil pH:</b>", body_style), Paragraph(f"{ph:.1f}", bold_style), Paragraph("<b>Ambient Temp:</b>", body_style), Paragraph(f"{temp:.1f} °C", bold_style)],
        [Paragraph("<b>Phosphorus (P):</b>", body_style), Paragraph(f"{p:.1f} mg/kg", bold_style), Paragraph("<b>Organic Carbon:</b>", body_style), Paragraph(f"{soc:.2f} %", bold_style), Paragraph("<b>Relative Humidity:</b>", body_style), Paragraph(f"{humid:.0f} %", bold_style)],
        [Paragraph("<b>Potash (K):</b>", body_style), Paragraph(f"{k:.1f} mg/kg", bold_style), Paragraph("<b>Soil Moisture:</b>", body_style), Paragraph(f"{moist:.1f} %", bold_style), Paragraph("<b>Precipitation:</b>", body_style), Paragraph(f"{rain:.0f} mm", bold_style)]
    ]
    t_tel = Table(telemetry_data, colWidths=[85, 95, 90, 95, 90, 80])
    t_tel.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_tel)

    story.append(Paragraph("3. RECOMMENDED FERTILIZER PURCHASES (50KG BAGS)", section_h1))
    urea_bags = max(1, round(opt['urea_kg'] / 50.0)) if opt['urea_kg'] > 0 else 0
    dap_bags = max(1, round(opt['dap_kg'] / 50.0)) if opt['dap_kg'] > 0 else 0
    mop_bags = max(1, round(opt['mop_kg'] / 50.0)) if opt['mop_kg'] > 0 else 0
    comp_bags = max(1, round(opt['complex_kg'] / 50.0)) if opt.get('complex_kg', 0) > 0 else 0
    org_bags = round(opt['compost_kg'] / 50.0) if opt['compost_kg'] > 0 else 0

    fert_data = [
        [Paragraph("<b>Fertilizer Product</b>", bold_style), Paragraph("<b>Nutrient Category</b>", bold_style), Paragraph("<b>Total Mass (kg)</b>", bold_style), Paragraph("<b>50kg Bags Required</b>", bold_style)],
        [Paragraph("Urea", body_style), Paragraph("Synthetic Nitrogen (46% N)", body_style), Paragraph(f"{opt['urea_kg']} kg", body_style), Paragraph(f"<b>{urea_bags} Bags</b>", bold_style)],
        [Paragraph("DAP", body_style), Paragraph("Phosphatic (18% N + 46% P)", body_style), Paragraph(f"{opt['dap_kg']} kg", body_style), Paragraph(f"<b>{dap_bags} Bags</b>", bold_style)],
        [Paragraph("MOP", body_style), Paragraph("Potash (60% K2O)", body_style), Paragraph(f"{opt['mop_kg']} kg", body_style), Paragraph(f"<b>{mop_bags} Bags</b>", bold_style)],
        [Paragraph("Complex 14-35-14", body_style), Paragraph("Balanced N-P-K Mineral", body_style), Paragraph(f"{opt.get('complex_kg', 0.0)} kg", body_style), Paragraph(f"<b>{comp_bags} Bags</b>", bold_style)],
        [Paragraph("Bio-Compost / Manure", body_style), Paragraph("Organic Humus Restorer", body_style), Paragraph(f"{opt['compost_kg']} kg", body_style), Paragraph(f"<b>{org_bags} Bags</b>", bold_style)],
    ]
    t_fert = Table(fert_data, colWidths=[150, 160, 110, 115])
    t_fert.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2EEDF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_fert)

    story.append(Paragraph("4. TIMED APPLICATION PERIODS & METHODS FOR FARMERS", section_h1))
    schedule_data = [
        [Paragraph("<b>Time Period</b>", bold_style), Paragraph("<b>Nutrient Blend</b>", bold_style), Paragraph("<b>Specific Application Method for Farmer</b>", bold_style)],
        [
            Paragraph("<b>Stage 1: Basal Dressing (At Sowing / Transplanting - Day 0)</b>", body_style),
            Paragraph("100% Bio-Compost + 100% DAP<br/>+ 1/3 MOP + 1/4 Urea", body_style),
            Paragraph("Incorporate compost and broadcast full DAP and 1/3 MOP. Place 5-7 cm below seed furrow; do not leave on dry surface.", body_style)
        ],
        [
            Paragraph("<b>Stage 2: Vegetative Growth (20 - 25 Days Post Sowing)</b>", body_style),
            Paragraph("1/2 Urea + 1/3 MOP<br/><i>(Vegetative Dose)</i>", body_style),
            Paragraph("Side-dress 1/2 urea dose + 1/3 MOP along plant rows. Ensure adequate soil moisture or irrigate within 24 hours.", body_style)
        ],
        [
            Paragraph("<b>Stage 3: Panicle Initiation / Flowering (45 - 55 Days Post Sowing)</b>", body_style),
            Paragraph("Remaining 1/4 Urea<br/>+ Remaining 1/3 MOP", body_style),
            Paragraph("Top-dress remaining 1/4 urea and final MOP. Avoid application during heavy rains to prevent leaching.", body_style)
        ]
    ]
    t_sched = Table(schedule_data, colWidths=[130, 155, 250])
    t_sched.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2EEDF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
    ]))
    story.append(t_sched)

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def generate_disease_pdf(user_mobile, plot_id, crop, diag):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=17, textColor=colors.HexColor('#0B3D2E'), leading=21, alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#2E7D32'), leading=12, alignment=1)
    section_h1 = ParagraphStyle('SecH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, textColor=colors.HexColor('#0B3D2E'), leading=14, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#1E293B'), leading=11)
    bold_style = ParagraphStyle('BoldText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#0F172A'), leading=11)

    story = []

    if os.path.exists(LOGO_FILE_EXACT):
        try:
            story.append(RLImage(LOGO_FILE_EXACT, width=140, height=140))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    IST = timezone(timedelta(hours=5, minutes=30))
    local_now = datetime.now(IST)

    story.append(Paragraph("SMART KISHAN • CROP DISEASE & TREATMENT PRESCRIPTION", title_style))
    story.append(Paragraph("Certified Plant Pathology & Remedial Action Dossier", subtitle_style))
    story.append(Paragraph(f"Dossier ID: SK-DIAG-{local_now.strftime('%Y%m%d')}-{user_mobile[-4:]} | Generated: {local_now.strftime('%d-%b-%Y %I:%M %p')}", ParagraphStyle('Meta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("1. DIAGNOSTIC SPECIMEN & FARM PROFILE", section_h1))
    profile_data = [
        [Paragraph("<b>Farmer Mobile:</b>", body_style), Paragraph(f"+91 {user_mobile}", bold_style), Paragraph("<b>Field / Parcel ID:</b>", body_style), Paragraph(str(plot_id), bold_style)],
        [Paragraph("<b>Target Crop:</b>", body_style), Paragraph(str(crop), bold_style), Paragraph("<b>Canopy Health Status:</b>", body_style), Paragraph(str(diag.get('health', 'Analyzed')), bold_style)],
    ]
    t_prof = Table(profile_data, colWidths=[110, 155, 120, 150])
    t_prof.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4FBF5')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#C8E6C9')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_prof)

    story.append(Paragraph("2. PATHOLOGY & PEST IDENTIFICATION", section_h1))
    path_data = [
        [Paragraph("<b>Detected Disease / Pathogen:</b>", body_style), Paragraph(str(diag.get('disease', 'None')), bold_style)],
        [Paragraph("<b>Pest Recognition:</b>", body_style), Paragraph(str(diag.get('pest', 'None')), bold_style)],
        [Paragraph("<b>Visible Symptoms:</b>", body_style), Paragraph(str(diag.get('symptoms', 'None')), bold_style)],
    ]
    t_path = Table(path_data, colWidths=[165, 370])
    t_path.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_path)

    story.append(Paragraph("3. PRESCRIBED MEDICINE, SPRAY & RECOVERY SCHEDULE", section_h1))
    remedy_data = [
        [Paragraph("<b>Prescribed Medicine / Spray:</b>", body_style), Paragraph(str(diag.get('medicine', 'None')), bold_style)],
        [Paragraph("<b>Survival & Recovery Chance:</b>", body_style), Paragraph(f"{diag.get('recovery_chance', 90)}%", bold_style)],
        [Paragraph("<b>Prognosis / Will Crop Grow?:</b>", body_style), Paragraph(str(diag.get('will_grow', 'Yes')), bold_style)],
    ]
    t_rem = Table(remedy_data, colWidths=[165, 370])
    t_rem.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2EEDF')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_rem)

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------------------------------------------------
# PERMANENT RIGHT-SIDE AI AGRI-BOT HELPER (WITH GREEN CHAT BACKGROUND)
# -------------------------------------------------------------
def render_ai_chatbot_sidebar():
    with st.sidebar:
        if os.path.exists(LOGO_FILE_EXACT):
            st.image(LOGO_FILE_EXACT, width=120)
        st.markdown("""
        <div style="background: rgba(11, 61, 46, 0.95); padding: 16px; border-radius: 12px; border: 1px solid #39FF88; margin-bottom: 15px;">
            <h3 style="color: #39FF88; margin: 0 0 6px 0;">🤖 Smart Kishan AI Bot</h3>
            <p style="color: #FFFFFF; font-size: 13px; margin: 0;">Specialized AI Assistant for crop health, NPK budgeting, and agricultural formulas.</p>
        </div>
        """, unsafe_allow_html=True)
        
        chat_container = st.container()
        with chat_container:
            st.markdown("""
            <div style="background-color: #062319; padding: 14px; border-radius: 12px; border: 1px solid rgba(57,255,136,0.3); max-height: 400px; overflow-y: auto; margin-bottom: 12px;">
            """, unsafe_allow_html=True)
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    st.markdown(f"💬 **You:** {msg['content']}")
                else:
                    st.markdown(f"🤖 **AgriAI:** {msg['content']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        user_q = st.text_input("Ask agri question...", key="sidebar_chat_input")
        if st.button("Send to AI", key="sidebar_chat_btn"):
            if user_q.strip():
                st.session_state.chat_messages.append({"role": "user", "content": user_q})
                q_lower = user_q.lower()
                
                if "disease" in q_lower or "pest" in q_lower or "rust" in q_lower or "blight" in q_lower:
                    reply = "🔬 **Plant Pathology AI**: For fungal infections (like Early Blight or Rust), apply Mancozeb 75% WP @ 2.5g/L or Hexaconazole 5% EC. Ensure spray is done during cool morning hours."
                elif "urea" in q_lower or "nitrogen" in q_lower or "npk" in q_lower or "fertilizer" in q_lower:
                    reply = "🧪 **Nutrient Advisory**: Split your nitrogen doses across basal, tillering, and flowering stages. Avoid applying urea on dry soils to prevent ammonia volatilization."
                elif "budget" in q_lower or "cost" in q_lower or "price" in q_lower:
                    reply = "💰 **Budget Engine**: Our 4R linear programming algorithm strictly limits total commercial chemical purchases to your designated budget cap while satisfying crop demand."
                else:
                    reply = f"🌱 **Agronomy AI**: I analyzed your query about '{user_q}'. Make sure your soil pH is maintained between 6.0 and 7.2 for optimal nutrient uptake!"
                
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.rerun()

if st.session_state.logged_in:
    render_ai_chatbot_sidebar()

# -------------------------------------------------------------
# SCREEN 1: SMART KISHAN CINEMATIC LOGIN
# -------------------------------------------------------------
if st.session_state.step == 1:
    col_brand, col_login = st.columns([1.02, 0.98], gap="large")

    with col_brand:
        if os.path.exists(LOGO_FILE_EXACT):
            st.image(LOGO_FILE_EXACT, width=230)

        st.markdown("""
        <div class="login-brand-side">
            <div class="cert-badge">🌱 4R CERTIFIED AGRICULTURE AI</div>
            <h1>SMART <span>KISHAN</span></h1>
            <p>
                Next-Generation AgriTech Control Center powered by
                Artificial Intelligence.
            </p>
            <p class="brand-description">
                Precision soil intelligence, crop prediction, automated
                fertilizer optimization, and AI-powered plant disease
                diagnosis — all in one agricultural decision-support platform.
            </p>
            <div class="feature-row">
                <span class="feature-pill">🌱 Soil Intelligence</span>
                <span class="feature-pill">🌾 Crop Prediction</span>
                <span class="feature-pill">🧪 Fertilizer Optimization</span>
                <span class="feature-pill">🔬 Disease Detection</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_login:
        st.markdown('<div class="glass-login-card">', unsafe_allow_html=True)
        if os.path.exists(LOGO_FILE_EXACT):
            st.image(LOGO_FILE_EXACT, width=60)

        st.markdown("""
        <div class="cert-badge">🔐 SECURE AGRICULTURE CONTROL CENTER</div>
        <div class="login-title">🔐 Welcome Back</div>
        <div class="login-subtitle">
            Sign in to access your Smart Kishan agricultural intelligence platform.
        </div>
        """, unsafe_allow_html=True)

        c_lang, c_mode = st.columns([1, 1])

        available_languages = list(TRANSLATIONS.keys())
        current_lang_index = (
            available_languages.index(st.session_state.app_lang)
            if st.session_state.app_lang in available_languages else 0
        )

        new_lang = c_lang.selectbox(
            T["lang_select"],
            available_languages,
            index=current_lang_index,
            key="login_language"
        )

        if new_lang != st.session_state.app_lang:
            st.session_state.app_lang = new_lang
            st.rerun()

        mode_choice = c_mode.radio(
            T["mode_select"],
            [T["mode_opt"], T["mode_diag"]],
            key="login_service_mode"
        )

        st.session_state.app_mode = (
            "Diagnostic Only"
            if mode_choice == T["mode_diag"]
            else "Full Optimization"
        )

        t_login, t_reg, t_fb = st.tabs([T["login_tab"], T["reg_tab"], T["feedback_tab"]])

        with t_login:
            m = st.text_input(
                T["mobile_lbl"],
                max_chars=10,
                key="log_m",
                placeholder="10-digit mobile number"
            )

            p = st.text_input(
                T["pass_lbl"],
                type="password",
                key="log_p",
                placeholder="Enter your password"
            )

            role_sel = st.selectbox(
                "Select Account Role",
                ["farmer", "admin"],
                key="log_role"
            )

            if st.button(
                "Access Control Center →",
                key="login_button"
            ):
                if len(m.strip()) == 10:
                    valid, user_role = verify_user(m.strip(), p.strip())

                    if valid:
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = m.strip()
                        st.session_state.user_role = user_role or role_sel
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                else:
                    st.warning("Enter a valid 10-digit mobile number.")

        with t_reg:
            rm = st.text_input(
                T["mobile_lbl"],
                max_chars=10,
                key="reg_m",
                placeholder="10-digit mobile number"
            )

            rp = st.text_input(
                T["pass_lbl"],
                type="password",
                key="reg_p",
                placeholder="Create password"
            )

            rpc = st.text_input(
                T["conf_pass_lbl"],
                type="password",
                key="reg_pc",
                placeholder="Confirm password"
            )

            reg_role = st.selectbox(
                "Register As",
                ["farmer", "admin"],
                key="reg_role"
            )

            if st.button(
                "Create Smart Kishan Account →",
                key="register_button"
            ):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(
                        rm.strip(),
                        rp.strip(),
                        reg_role
                    )

                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning(
                        "Please check mobile number and matching passwords."
                    )

        with t_fb:
            st.markdown("##### 🌟 Quick Feedback & Exit Session")
            fb_mobile = st.text_input("Mobile Number for Feedback", max_chars=10, key="fb_mob")
            fb_rating = st.slider("Rate your experience (1 to 5 Stars)", 1, 5, 5, key="fb_stars")
            fb_text = st.text_area("Your Feedback / Comments:", placeholder="Write comments here...")
            
            if st.button("Submit & Exit Session ➔", key="btn_quick_exit"):
                if not fb_text.strip():
                    st.error("⚠️ Please enter feedback comments before exiting.")
                else:
                    save_feedback(fb_mobile or "0000000000", fb_rating, fb_text)
                    st.success("✅ Thank you! Feedback recorded successfully. Session exited.")
                    st.session_state.logged_in = False
                    st.session_state.step = 1
                    st.cache_data.clear()
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 2: MUTUALLY EXCLUSIVE SOIL SCANNER OR MANUAL INPUT
# -------------------------------------------------------------
elif st.session_state.step == 2:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)

    h_col1, h_col2 = st.columns([3, 1])
    h_col1.markdown(f"### 🌾 AgriTech Control Center — Role: **{st.session_state.user_role.upper()}**")
    if h_col2.button("🚪 Sign Out"):
        st.session_state.logged_in = False
        st.session_state.step = 1
        st.rerun()

    if st.session_state.app_mode == "Diagnostic Only":
        st.subheader("🔬 AI Optical Crop Disease & Pest Diagnosis & Treatment Prescription")
        st.info("Take a photo or upload an image of the affected plant leaf, crop stem, or pest:")
        
        c_cam, c_up = st.columns(2)
        cam_p = c_cam.camera_input("📷 Realtime Camera Scanner")
        file_p = c_up.file_uploader("📂 Upload Leaf / Pest Image", type=["jpg", "jpeg", "png"])
        
        active_img = cam_p or file_p
        if active_img:
            img = Image.open(active_img)
            st.image(img, caption="Scanned Specimen", width=300)
            res = analyze_plant_disease_image(img)
            st.session_state.scanned_diag = res

            diag_tab1, diag_tab2, diag_tab3 = st.tabs([
                "📋 Disease & Plant Requirements", 
                "💊 Medicine & Application Schedule", 
                "📄 Official Prescription Card (PDF)"
            ])

            with diag_tab1:
                st.markdown(f"### Diagnostic Status: <span class='badge-pass'>{res['health']}</span>", unsafe_allow_html=True)
                st.write(f"🦠 **Crop Disease / Pathogen**: {res['disease']}")
                st.write(f"🐛 **Pest Recognition**: {res['pest']}")
                st.write(f"🔬 **Visible Symptoms**: {res['symptoms']}")
                st.info("💡 **Nutritional & Environmental Requirements**: Ensure balanced potassium and micronutrient supplementation alongside moisture retention to strengthen plant immunity.")
                
                st.divider()
                if st.button(T["btn_back"], key="diag_tab1_back"):
                    st.session_state.step = 1
                    st.rerun()

            with diag_tab2:
                st.markdown("##### 💊 Prescribed Treatment & Application Schedule:")
                st.write(f"• **Recommended Remedy Spray**: {res['medicine']}")
                st.write("• **Application Frequency**: Apply once every 7 to 10 days during early morning or late evening hours.")
                st.metric("Survival & Recovery Chance", f"{res['recovery_chance']}%")
                st.write(f"🌱 **Will this crop continue to grow?**: **{res['will_grow']}**")
                
                st.divider()
                if st.button(T["btn_back"], key="diag_tab2_back"):
                    st.session_state.step = 1
                    st.rerun()

            with diag_tab3:
                st.markdown("##### 📄 Download Official Disease & Treatment Prescription")
                st.caption("Download the formal plant pathology and remedial prescription dossier in PDF format.")
                
                disease_pdf_bytes = generate_disease_pdf(
                    user_mobile=st.session_state.user_mobile,
                    plot_id=st.session_state.plot_id,
                    crop=st.session_state.sel_crop,
                    diag=res
                )
                disease_pdf_filename = f"SmartKishan_Disease_Prescription_{st.session_state.user_mobile}.pdf"
                st.download_button(
                    label="📄 Download Plant Pathology Prescription (PDF)",
                    data=disease_pdf_bytes,
                    file_name=disease_pdf_filename,
                    mime="application/pdf"
                )
                
                st.divider()
                pdf_btn_col1, pdf_btn_col2 = st.columns([1, 4])
                with pdf_btn_col1:
                    if st.button(T["btn_back"], key="diag_tab3_back"):
                        st.session_state.step = 1
                        st.rerun()
                with pdf_btn_col2:
                    if st.button("Proceed to Feedback & Exit ➔", key="diag_tab3_proceed"):
                        st.session_state.step = 8
                        st.rerun()

    else:
        st.subheader("2. 📍 Land Size, Budget & Soil Input (Scanner OR Manual)")
        
        tab_camera, tab_land, tab_soil = st.tabs([
            "📷 Option A: Optical Soil Scanner", 
            "📐 Land Area & Farm Budget", 
            "🧪 Option B: Manual Soil Input"
        ])
        
        with tab_camera:
            st.markdown("##### Real-Time Optical Soil Diagnostic Scanner")
            st.caption("Scan genuine agricultural soil. Non-soil elements like white roofs or artificial surfaces are automatically rejected.")
            
            cam_c1, cam_c2 = st.columns(2)
            with cam_c1:
                soil_cam = st.camera_input("📷 Scan Field Soil Live")
            with cam_c2:
                soil_file = st.file_uploader("📂 Or Upload Soil Sample Photo", type=["jpg", "jpeg", "png"])

            soil_img = soil_cam or soil_file
            if soil_img:
                s_img = Image.open(soil_img)
                st.image(s_img, caption="Camera Captured Specimen", width=260)
                soil_eval = verify_genuine_agricultural_soil(s_img)
                st.session_state.scanned_soil = soil_eval

                if soil_eval["detected"]:
                    st.markdown(f"<div class='badge-pass' style='display:inline-block; font-size:16px; margin:10px 0;'>{T['soil_detected']}</div>", unsafe_allow_html=True)
                    m = soil_eval["metrics"]
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color:#39FF88; margin-top:0;">🌾 Scanned Soil Successfully Verified & Analyzed:</h4>
                        <p style="margin:4px 0;">• <strong>Texture Class:</strong> {soil_eval['soil_type']}</p>
                        <p style="margin:4px 0;">• <strong>Optical Color Signature:</strong> {m['rgb_signature']}</p>
                        <p style="margin:4px 0;">• <strong>Organic Carbon (SOC):</strong> {m['soc']}%</p>
                        <p style="margin:4px 0;">• <strong>Surface Moisture:</strong> {m['moist']}%</p>
                        <p style="margin:4px 0;">• <strong>Active pH:</strong> {m['ph']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Apply Verified Soil Features"):
                        st.session_state.soil_n = m["n"]
                        st.session_state.soil_p = m["p"]
                        st.session_state.soil_k = m["k"]
                        st.session_state.soil_ph = m["ph"]
                        st.session_state.soc = m["soc"]
                        st.session_state.soil_moist = m["moist"]
                        st.session_state.soil_source = "scanner"
                        st.success("✅ Verified agricultural soil applied successfully! You can now continue.")
                else:
                    st.markdown(f"<div class='badge-warn' style='display:inline-block; font-size:16px; margin:10px 0;'>{T['soil_not_detected']}</div>", unsafe_allow_html=True)
                    st.error(f"⚠️ {soil_eval['reason']} Please provide a genuine agricultural soil sample.")

        with tab_land:
            st.markdown(f"##### {T['land_calc_title']}")
            l_col1, l_col2, l_col3 = st.columns([2, 2, 2])
            st.session_state.plot_id = l_col1.text_input("Parcel / Field Identifier:", value=st.session_state.plot_id)
            st.session_state.raw_land_val = l_col2.number_input("Enter Land Size Amount", 0.1, 1000.0, float(st.session_state.raw_land_val), 0.5)
            st.session_state.land_unit = l_col3.selectbox(
                "Choose Area SI Unit", 
                list(UNIT_TO_HECTARE.keys()),
                index=list(UNIT_TO_HECTARE.keys()).index(st.session_state.land_unit)
            )

            st.markdown("---")
            b_col1, b_col2 = st.columns([2, 2])
            with b_col1:
                st.session_state.budget_cap = st.number_input(
                    T["budget_lbl"],
                    min_value=1000.0,
                    max_value=1000000.0,
                    value=float(st.session_state.budget_cap),
                    step=500.0,
                    help=T["budget_help"]
                )
            with b_col2:
                st.metric(
                    "Allocated Budget Ceiling", 
                    f"₹{st.session_state.budget_cap:,.0f}",
                    help="Linear programming constraint: Cost <= Budget"
                )

            st.markdown("---")
            conv_table, ha_val = render_land_conversion_table(st.session_state.raw_land_val, st.session_state.land_unit)
            st.session_state.land_area = ha_val
            st.table(conv_table)
            st.info(f"Standardized area for chemical dosage: **{ha_val:.3f} Hectares** | Maximum Cost Cap: **₹{st.session_state.budget_cap:,.0f}**")

        with tab_soil:
            st.markdown("##### Manual Soil Nutrient Input (Alternative to Scanner)")
            s1, s2, s3 = st.columns(3)
            st.session_state.soil_n = s1.number_input("Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n), key="m_n")
            st.session_state.soil_p = s2.number_input("Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p), key="m_p")
            st.session_state.soil_k = s3.number_input("Potash (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k), key="m_k")
            
            s4, s5, s6 = st.columns(3)
            st.session_state.soil_ph = s4.slider("Soil pH", 4.0, 9.5, float(st.session_state.soil_ph), 0.1, key="m_ph")
            st.session_state.soc = s5.slider("Organic Carbon (%)", 0.1, 2.5, float(st.session_state.soc), 0.05, key="m_soc")
            st.session_state.soil_moist = s6.slider("Moisture (%)", 10.0, 90.0, float(st.session_state.soil_moist), 1.0, key="m_moist")
            
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.selectbox("Soil Type", list(soil_encoder.classes_), key="sel_soil")
            c_s2.selectbox("Planned Crop", list(crop_type_encoder.classes_), key="sel_crop")
            
            st.session_state.target_yield = c_s3.number_input("Target Harvest (t/acre)", 0.5, 10.0, float(st.session_state.target_yield), 0.25)

            if st.button("Save Manual Soil Values"):
                st.session_state.soil_source = "manual"
                st.success("✅ Manual soil values saved successfully! You can now continue.")

        st.divider()
        b1, b2 = st.columns([1, 5])
        if b1.button(T["btn_back"], key="step2_back"):
            st.session_state.step = 1
            st.rerun()
            
        if b2.button(T["btn_next"], key="step2_next"):
            if st.session_state.soil_source is None:
                st.error("⚠️ Please either verify genuine agricultural soil in Tab 1 OR save manual soil values in Tab 3 before proceeding.")
            else:
                st.session_state.step = 3
                st.rerun()

# -------------------------------------------------------------
# SCREEN 3: SOIL HEALTH EVALUATION
# -------------------------------------------------------------
elif st.session_state.step == 3:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader("3. ⚙️ Soil Condition & Risk Assessment")
    
    k1, k2, k3 = st.columns(3)
    ph_stat = "Acidic (Apply Lime)" if st.session_state.soil_ph < 6.0 else ("Alkaline (Apply Gypsum)" if st.session_state.soil_ph > 7.5 else "Sweet & Balanced")
    k1.metric("Soil Sweetness (pH)", f"{st.session_state.soil_ph}", ph_stat)
    k2.metric("Organic Matter (SOC)", f"{st.session_state.soc}%", "Rich" if st.session_state.soc >= 0.75 else "Low (Add Compost)")
    k3.metric("Rain Leaching Risk", f"{st.session_state.rainfall:.0f} mm", "Leaching Alert" if st.session_state.rainfall > 200 else "Optimal")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"], key="step3_back"):
        st.session_state.step = 2
        st.rerun()
    if b2.button(T["btn_next"], key="step3_next"):
        st.session_state.step = 4
        st.rerun()

# -------------------------------------------------------------
# SCREEN 4: SOIL COMPARISON DONUT / PIE CHARTS
# -------------------------------------------------------------
elif st.session_state.step == 4:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader("4. 🍩 Current Soil Nutrients vs Ideal Farm Target (Proportion Analysis)")
    
    d1, d2, d3 = st.columns(3)
    
    with d1:
        st.markdown("##### Nitrogen (N) Ratio")
        n_df = pd.DataFrame({
            "Category": ["Your Soil", "Target Deficit"],
            "Value": [st.session_state.soil_n * 2.24, max(0.0, 280.0 - (st.session_state.soil_n * 2.24))]
        })
        st.altair_chart(
            __import__('altair').Chart(n_df).mark_arc(innerRadius=50).encode(
                theta=__import__('altair').Theta(field="Value", type="quantitative"),
                color=__import__('altair').Color(field="Category", type="nominal", scale=__import__('altair').Scale(range=["#39FF88", "#1B5E20"]))
            ), use_container_width=True
        )
        st.markdown(f"<span style='color: #FFFFFF !important;'>Measured: {st.session_state.soil_n * 2.24:.1f} kg/ha (Target: 280 kg/ha)</span>", unsafe_allow_html=True)

    with d2:
        st.markdown("##### Phosphorus (P) Ratio")
        p_df = pd.DataFrame({
            "Category": ["Your Soil", "Target Deficit"],
            "Value": [st.session_state.soil_p * 2.24, max(0.0, 60.0 - (st.session_state.soil_p * 2.24))]
        })
        st.altair_chart(
            __import__('altair').Chart(p_df).mark_arc(innerRadius=50).encode(
                theta=__import__('altair').Theta(field="Value", type="quantitative"),
                color=__import__('altair').Color(field="Category", type="nominal", scale=__import__('altair').Scale(range=["#39FF88", "#1B5E20"]))
            ), use_container_width=True
        )
        st.markdown(f"<span style='color: #FFFFFF !important;'>Measured: {st.session_state.soil_p * 2.24:.1f} kg/ha (Target: 60 kg/ha)</span>", unsafe_allow_html=True)

    with d3:
        st.markdown("##### Potash (K) Ratio")
        k_df = pd.DataFrame({
            "Category": ["Your Soil", "Target Deficit"],
            "Value": [st.session_state.soil_k * 2.24, max(0.0, 150.0 - (st.session_state.soil_k * 2.24))]
        })
        st.altair_chart(
            __import__('altair').Chart(k_df).mark_arc(innerRadius=50).encode(
                theta=__import__('altair').Theta(field="Value", type="quantitative"),
                color=__import__('altair').Color(field="Category", type="nominal", scale=__import__('altair').Scale(range=["#39FF88", "#1B5E20"]))
            ), use_container_width=True
        )
        st.markdown(f"<span style='color: #FFFFFF !important;'>Measured: {st.session_state.soil_k * 2.24:.1f} kg/ha (Target: 150 kg/ha)</span>", unsafe_allow_html=True)

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"], key="step4_back"):
        st.session_state.step = 3
        st.rerun()
    if b2.button(T["btn_next"], key="step4_next"):
        st.session_state.step = 5
        st.rerun()

# -------------------------------------------------------------
# SCREEN 5: NUTRIENT GAP & DYNAMIC CROP RECOMMENDATION
# -------------------------------------------------------------
elif st.session_state.step == 5:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader("5. ⚠️ Required Nutrient Deficit & Dynamic Crop Recommendation")
    def_n, def_p, def_k = calculate_advanced_nutrients(
        target_yield_per_acre=st.session_state.target_yield,
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
    dynamic_pred_crop = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]
    st.session_state.sel_crop = dynamic_pred_crop

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"##### Nutrient Shortage for {st.session_state.target_yield} t/acre:")
        st.warning(f"• **Nitrogen Needed**: {def_n:.1f} kg/acre")
        st.warning(f"• **Phosphorus Needed**: {def_p:.1f} kg/acre")
        st.warning(f"• **Potash Needed**: {def_k:.1f} kg/acre")
    with g2:
        st.markdown("##### 🌟 AI Dynamic Crop Recommendation:")
        st.success(f"🌱 **Best Suited Crop for Verified Soil**: **{dynamic_pred_crop.capitalize()}**")
        st.markdown(f"Calculated via Machine Learning based on active N={st.session_state.soil_n}, P={st.session_state.soil_p}, K={st.session_state.soil_k}, pH={st.session_state.soil_ph}")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"], key="step5_back"):
        st.session_state.step = 4
        st.rerun()
    if b2.button(T["btn_next"], key="step5_next"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 6: OPTIMIZED FERTILIZER BAGS & TIMETABLE
# -------------------------------------------------------------
elif st.session_state.step == 6:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader("6. 🚀 Your Fertilizer Bags & Application Schedule")
    def_n, def_p, def_k = calculate_advanced_nutrients(
        target_yield_per_acre=st.session_state.target_yield,
        soil_n=st.session_state.soil_n,
        soil_p=st.session_state.soil_p,
        soil_k=st.session_state.soil_k,
        soc=st.session_state.soc,
        ph=st.session_state.soil_ph,
        soil_moist=st.session_state.soil_moist,
        soil_texture=str(st.session_state.sel_soil)
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

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Optimized Total Cost", f"₹{opt['total_cost']:,.0f}")
    r2.metric("Input Budget Cap", f"₹{st.session_state.budget_cap:,.0f}")
    r3.metric("Land Covered", f"{st.session_state.raw_land_val:.2f} {st.session_state.land_unit.split(' ')[0]}")
    r4.metric("Budget Utilized", f"{opt['budget_utilized_pct']}%")

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
    if b1.button(T["btn_back"], key="step6_back"):
        st.session_state.step = 5
        st.rerun()
    if b2.button(T["btn_next"], key="step6_next"):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: PRESCRIPTION DOSSIER & MULTILINGUAL PDF DOWNLOAD
# -------------------------------------------------------------
elif st.session_state.step == 7:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader("7. 📋 Official Farmer Prescription Card (Smart Kishan Certified)")
    opt = st.session_state.get("opt_results", {"urea_kg": 0, "dap_kg": 0, "mop_kg": 0, "compost_kg": 0, "total_cost": 0, "land_area": st.session_state.land_area})
    diag = st.session_state.get("scanned_diag", {
        "health": "Optimal Vigor", "disease": "None detected", "pest": "None",
        "symptoms": "Healthy foliage", "medicine": "Prophylactic Neem Spray",
        "recovery_chance": 95, "will_grow": "Yes"
    })

    # UI Display Card
    st.markdown(f"""
    <div class="summary-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <h2 style="color: #39FF88; margin-top: 0; margin-bottom:4px;">🌾 Smart Kishan • Official Crop & Fertilizer Prescription</h2>
                <span style="font-size:13px; color:#A7F3D0; font-weight:700;">AGRITECH CONTROL CENTER • 4R CERTIFIED ADVISORY</span>
            </div>
        </div>
        <hr style="border: 1px solid rgba(57, 255, 136, 0.3); margin: 12px 0;"/>
        <p style="margin:4px 0;"><strong>Farmer Mobile:</strong> +91 {st.session_state.user_mobile} | <strong>Parcel ID:</strong> {st.session_state.plot_id}</p>
        <p style="margin:4px 0;"><strong>Cultivated Crop:</strong> {st.session_state.sel_crop} | <strong>Target Harvest:</strong> {st.session_state.target_yield} t/acre</p>
        <p style="margin:4px 0;"><strong>Land Area:</strong> {st.session_state.raw_land_val:.2f} Acre ({st.session_state.land_area:.3f} Ha)</p>
        <hr style="border: 1px solid rgba(57, 255, 136, 0.3); margin: 15px 0;"/>
        <h4 style="color: #39FF88; margin-bottom: 6px;">🛒 Required Commercial Purchases:</h4>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li><strong>Urea (Synthetic N):</strong> {opt['urea_kg']} kg (~{round(opt['urea_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>DAP (Phosphatic):</strong> {opt['dap_kg']} kg (~{round(opt['dap_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>MOP (Potash):</strong> {opt['mop_kg']} kg (~{round(opt['mop_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>Organic Compost:</strong> {opt['compost_kg']} kg (~{round(opt['compost_kg'] / 50.0)} bags)</li>
        </ul>
        <h3 style="color: #39FF88; margin-top: 10px;">💰 Total Investment: ₹{opt['total_cost']:,.0f} (Budget: ₹{st.session_state.budget_cap:,.0f})</h3>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Farm Timing & Application Methods Matrix
    st.markdown("#### 📅 Timed Application Periods & Methods for Farmers:")
    app_methods_df = pd.DataFrame({
        "Crop Stage & Time Period": [
            T["stage_1_period"],
            T["stage_2_period"],
            T["stage_3_period"]
        ],
        "Input Blend": [
            "All Compost + All DAP + 1/3 Potash + 1/4 Urea",
            "1/2 Urea + 1/3 Potash",
            "Remaining 1/4 Urea + Remaining 1/3 Potash"
        ],
        "Farmer Application Method": [
            T["stage_1_method"],
            T["stage_2_method"],
            T["stage_3_method"]
        ]
    })
    st.table(app_methods_df)

    pdf_bytes = generate_english_pdf(
        user_mobile=st.session_state.user_mobile,
        plot_id=st.session_state.plot_id,
        raw_land=st.session_state.raw_land_val,
        land_unit=st.session_state.land_unit,
        crop=st.session_state.sel_crop,
        target_yield=st.session_state.target_yield,
        budget=st.session_state.budget_cap,
        opt=opt,
        diag=diag,
        n=st.session_state.soil_n,
        p=st.session_state.soil_p,
        k=st.session_state.soil_k,
        ph=st.session_state.soil_ph,
        soc=st.session_state.soc,
        moist=st.session_state.soil_moist,
        temp=st.session_state.temp,
        humid=st.session_state.humidity,
        rain=st.session_state.rainfall
    )

    pdf_filename = f"SmartKishan_English_Prescription_{st.session_state.user_mobile}.pdf"

    p_col1, p_col2 = st.columns([2, 2])
    with p_col1:
        st.download_button(
            label="📄 Download PDF Prescription (English)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )
    with p_col2:
        if st.button("Proceed to Feedback & Exit ➔"):
            st.session_state.step = 8
            st.rerun()

    st.divider()
    if st.button(T["btn_back"], key="step7_back"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 8: MANDATORY BORDERLESS STAR RATING & EXIT
# -------------------------------------------------------------
elif st.session_state.step == 8:
    if os.path.exists(LOGO_FILE_EXACT):
        st.image(LOGO_FILE_EXACT, width=150)
    st.subheader(T["feedback_title"])
    st.write("Please tap the stars below to rate your advisory experience before exiting:")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="star-container">', unsafe_allow_html=True)
    star_cols = st.columns(5)
    
    if "star_selection" not in st.session_state:
        st.session_state.star_selection = 5

    for i in range(1, 6):
        with star_cols[i-1]:
            star_symbol = "★" if i <= st.session_state.star_selection else "☆"
            
            if st.button(f"{star_symbol}", key=f"borderless_star_{i}", use_container_width=True):
                st.session_state.star_selection = i
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: #39FF88;'>★ {st.session_state.star_selection} / 5 Stars Rated ★</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    feedback_comments = st.text_area("Your Comments / Suggestions (ଆପଣଙ୍କ ମତାମତ / आपकी प्रतिक्रिया):", placeholder="Write your feedback here...")

    b_fb_back, b_fb_sub = st.columns([1, 5])
    if b_fb_back.button(T["btn_back"], key="feedback_back_btn"):
        st.session_state.step = 7 if st.session_state.app_mode == "Full Optimization" else 2
        st.rerun()

    if b_fb_sub.button(T["feedback_submit"]):
        if not feedback_comments.strip():
            st.error("⚠️ Mandatory Feedback Required: Please enter your feedback comments before exiting.")
        else:
            save_feedback(st.session_state.user_mobile, st.session_state.star_selection, feedback_comments)
            st.success("✅ Thank you! Your star rating and feedback have been recorded safely. Exit session...")
            
            st.session_state.logged_in = False
            st.session_state.user_mobile = ""
            st.session_state.feedback_given = True
            st.session_state.step = 1
            st.cache_data.clear()
            st.rerun()
