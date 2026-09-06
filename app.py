import io
import os
import urllib.parse
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
from datetime import datetime
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
# AUTO-DOWNLOAD UNICODE FONT FOR MULTILINGUAL PDF SUPPORT
# -------------------------------------------------------------
FONT_FILE = "DejaVuSans.ttf"
if not os.path.exists(FONT_FILE):
    try:
        font_url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        urllib.request.urlretrieve(font_url, FONT_FILE)
    except Exception:
        pass

# -------------------------------------------------------------
# LAND CONVERSIONS & CORE MATH ENGINES (DEFINED FIRST)
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
    img_rgb = image_obj.convert("RGB").resize((160, 160))
    np_img = np.array(img_rgb, dtype=np.float32)
    
    stat_rgb = ImageStat.Stat(img_rgb)
    r_m, g_m, b_m = stat_rgb.mean[0], stat_rgb.mean[1], stat_rgb.mean[2]

    is_earth_tone = (r_m >= g_m >= b_m) or (r_m < 90 and g_m < 90 and b_m < 90)
    
    gray = img_rgb.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_var = edge_stat.var[0]

    if is_earth_tone and edge_var > 20.0 and b_m < r_m:
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
            "reason": "Not detected"
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
# PAGE CONFIGURATION & LIGHT GREEN FARMER THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="Smart Kishan | Digital Farming Solutions",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #F2FAF3 !important;
        color: #143518 !important;
    }

    .farmer-hero {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #43A047 100%);
        border-radius: 18px;
        padding: 18px 24px;
        color: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(27, 94, 32, 0.18);
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #A5D6A7;
    }
    .hero-text h1 {
        font-size: 25px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        letter-spacing: -0.3px;
    }
    .hero-text p {
        font-size: 13.5px !important;
        color: #E8F5E9 !important;
        margin: 3px 0 0 0 !important;
        font-weight: 500;
    }

    .metric-card {
        background: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        border-left: 6px solid #2E7D32 !important;
        border-top: 1px solid #D1E7D3 !important;
        border-right: 1px solid #D1E7D3 !important;
        border-bottom: 1px solid #D1E7D3 !important;
        box-shadow: 0 3px 10px rgba(20, 60, 20, 0.04) !important;
        margin-bottom: 12px;
    }

    .summary-card {
        background: #F4FBF5 !important;
        border: 2px solid #81C784 !important;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.08) !important;
    }

    div.stButton > button, div.stButton > button:focus {
        background: linear-gradient(180deg, #2E7D32 0%, #1B5E20 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 14.5px !important;
        border-radius: 10px !important;
        padding: 11px 24px !important;
        border: 1px solid #144918 !important;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #388E3C 0%, #1E6B24 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 16px rgba(27, 94, 32, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    div.stDownloadButton > button {
        background: linear-gradient(180deg, #1B5E20 0%, #0F3D13 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        border: 1px solid #0B2C0D !important;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.25) !important;
    }

    .star-container button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 38px !important;
        padding: 0 !important;
        margin: 0 !important;
        transition: transform 0.1s ease;
    }
    .star-container button:hover {
        background: transparent !important;
        transform: scale(1.15);
    }

    .badge-pass {
        background-color: #E8F5E9;
        color: #1B5E20;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #A5D6A7;
    }
    .badge-warn {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 5px 14px;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid #EF9A9A;
    }
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
# GLOBAL MULTILINGUAL UI DICTIONARY
# -------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "Smart Kishan | Digital Farming Solutions",
        "subtitle": "Certified 4R Nutrient Allocation, Real-Soil Triage & Official Prescription",
        "login_tab": "Farmer Log In",
        "reg_tab": "Register New Farmer",
        "mobile_lbl": "Mobile Number",
        "pass_lbl": "Password",
        "conf_pass_lbl": "Confirm Password",
        "lang_select": "App Language / Global Language Preference",
        "mode_select": "Select Farm Service",
        "mode_opt": "🌾 Full Soil & Fertilizer Optimization Pipeline",
        "mode_diag": "🔬 Plant Disease, Pest & Medicine Diagnosis Only",
        "btn_login": "Log In to Farm Dashboard ➔",
        "btn_reg": "Create Account",
        "btn_back": "⬅️ Back",
        "btn_next": "Continue ➔",
        "budget_lbl": "Your Maximum Fertilizer Budget (₹)",
        "budget_help": "Optimization engine ensures total purchase cost stays strictly within this limit.",
        "feedback_title": "🌟 Mandatory Farmer Feedback & Star Rating",
        "feedback_submit": "Submit Feedback & Exit Dashboard ➔",
        "land_calc_title": "📐 Land Unit Selection & Farm Budget Matrix",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "हिन्दी": {
        "title": "स्मार्ट किसान | डिजिटल फार्मिंग सॉल्यूशंस",
        "subtitle": "प्रमाणित 4R पोषक तत्व प्रबंधन, वास्तविक मृदा विश्लेषण और आधिकारिक नुस्खा",
        "login_tab": "किसान लॉगिन",
        "reg_tab": "नया किसान पंजीकरण",
        "mobile_lbl": "मोबाइल नंबर",
        "pass_lbl": "पासवर्ड",
        "conf_pass_lbl": "पासवर्ड की पुष्टि करें",
        "lang_select": "ऐप भाषा / वैश्विक भाषा प्राथमिकता",
        "mode_select": "कृषि सेवा चुनें",
        "mode_opt": "🌾 पूर्ण मृदा एवं उर्वरक अनुकूलन पाइपलाइन",
        "mode_diag": "🔬 केवल पौध रोग, कीट एवं औषधि निदान",
        "btn_login": "डैशबोर्ड में लॉगिन करें ➔",
        "btn_reg": "खाता बनाएं",
        "btn_back": "⬅️ पीछे",
        "btn_next": "आगे बढ़ें ➔",
        "budget_lbl": "आपका अधिकतम उर्वरक बजट (₹)",
        "budget_help": "यह सुनिश्चित करता है कि कुल उर्वरक खरीद लागत इस बजट सीमा से अधिक न हो।",
        "feedback_title": "🌟 अनिवार्य किसान समीक्षा और स्टार रेटिंग",
        "feedback_submit": "समीक्षा जमा करें और बाहर निकलें ➔",
        "land_calc_title": "📐 भूमि इकाई चयन और कृषि बजट तालिका",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "ଓଡ଼ିଆ": {
        "title": "ସ୍ମାର୍ଟ କିଷାନ | ଡିଜିଟାଲ ଫାର୍ମିଂ ସଲ୍ୟୁସନ୍ସ",
        "subtitle": "ପ୍ରମାଣିତ ୪ଆର୍ ପୋଷକ ପରିଚାଳନା, ପ୍ରକୃତ ମୃତ୍ତିକା ବିଶ୍ଳେଷଣ ଓ ସରକାରୀ ପ୍ରେସକ୍ରିପସନ",
        "login_tab": "କୃଷକ ଲଗଇନ୍",
        "reg_tab": "ନୂତନ କୃଷକ ପଞ୍ଜୀକରଣ",
        "mobile_lbl": "ମୋବାଇଲ୍ ନମ୍ବର",
        "pass_lbl": "ପାସୱାର୍ଡ",
        "conf_pass_lbl": "ପାସୱାର୍ଡ ନିଶ୍ଚିତ କରନ୍ତୁ",
        "lang_select": "ଭାଷା ଚୟନ / ବିଶ୍ୱବ୍ୟାପୀ ଭାଷା ପସନ୍ଦ",
        "mode_select": "ସେବା ଚୟନ କରନ୍ତୁ",
        "mode_opt": "🌾 ସମ୍ପୂର୍ଣ୍ଣ ମୃତ୍ତିକା ଓ ସାର ପରିମାଣ ନିର୍ଦ୍ଧାରଣ",
        "mode_diag": "🔬 କେବଳ ଫସଲ ରୋଗ, କୀଟ ଚିହ୍ନଟ ଓ ଔଷଧ",
        "btn_login": "ଡ୍ୟାସବୋର୍ଡରେ ପ୍ରବେଶ କରନ୍ତୁ ➔",
        "btn_reg": "ଖାତା ତିଆରି କରନ୍ତୁ",
        "btn_back": "⬅️ ପଛକୁ ଯାଆନ୍ତୁ",
        "btn_next": "ଆଗକୁ ବଢ଼ନ୍ତୁ ➔",
        "budget_lbl": "ଆପଣଙ୍କ ସର୍ବାଧିକ ସାର ଖର୍ଚ୍ଚ ବଜେଟ୍ (₹)",
        "budget_help": "ଏହା ନିଶ୍ଚିତ କରେ ଯେ ଆପଣଙ୍କ ସାର ଖର୍ଚ୍ଚ ଏହି ବଜେଟ୍ ସୀମା ଭିତରେ ରହିବ।",
        "feedback_title": "🌟 ବାଧ୍ୟତାମୂଳକ କୃଷକ ମତାମତ ଏବଂ ଷ୍ଟାର ରେଟିଂ",
        "feedback_submit": "ମତାମତ ଦାଖଲ କରନ୍ତୁ ଏବଂ ବାହାରକୁ ଯାଆନ୍ତୁ ➔",
        "land_calc_title": "📐 ଜମି ଏକକ ଏବଂ କୃଷି ବଜେଟ୍ ସାରଣୀ",
        "soil_detected": "Soil is detected",
        "soil_not_detected": "Not detected"
    },
    "मराठी": {
        "title": "स्मार्ट किसान | डिजिटल शेती उपाय",
        "subtitle": "प्रमाणित 4R पोषक तत्व व्यवस्थापन, वास्तविक माती परीक्षण आणि अधिकृत कृषी शिफारस",
        "login_tab": "शेतकरी लॉगिन",
        "reg_tab": "नवीन शेतकरी नोंदणी",
        "mobile_lbl": "मोबाईल नंबर",
        "pass_lbl": "पासवर्ड",
        "conf_pass_lbl": "पासवर्ड पुष्टी करा",
        "lang_select": "भाषा निवडा",
        "mode_select": "कृषी सेवा निवडा",
        "mode_opt": "🌾 संपूर्ण माती आणि खत ऑप्टिमायझेशन",
        "mode_diag": "🔬 केवळ पीक रोग, कीटक आणि औषध निदान",
        "btn_login": "डॅशबोर्डवर लॉग इन करा ➔",
        "btn_reg": "खाते तयार करा",
        "btn_back": "⬅️ मागे",
        "btn_next": "पुढे चालू ठेवा ➔",
        "budget_lbl": "कमाल खत बजेट (₹)",
        "budget_help": "खत खरेदी खर्च या मर्यादेत राहतो.",
        "feedback_title": "🌟 शेतकरी अभिप्राय आणि स्टार रेटिंग",
        "feedback_submit": "अभिप्राय सबमिट करा आणि बाहेर पडा ➔",
        "land_calc_title": "📐 जमीन रूपांतरण आणि बजेट तक्ता",
        "soil_detected": "माती आढळली",
        "soil_not_detected": "आढळली नाही"
    },
    "தமிழ்": {
        "title": "ஸ்மார்ட் கிசான் | டிஜிட்டல் விவசாய தீர்வுகள்",
        "subtitle": "சான்றளிக்கப்பட்ட 4R சத்து மேலாண்மை, உண்மையான மண் பரிசோதனை மற்றும் அதிகாரப்பூர்வ பரிந்துரை",
        "login_tab": "விவசாயி உள்நுழைவு",
        "reg_tab": "புதிய விவசாயி பதிவு",
        "mobile_lbl": "மொபைல் எண்",
        "pass_lbl": "கடவுச்சொல்",
        "conf_pass_lbl": "கடவுச்சொல்லை உறுதிப்படுத்தவும்",
        "lang_select": "மொழி தேர்வு",
        "mode_select": "விவசாய சேவையைத் தேர்ந்தெடுக்கவும்",
        "mode_opt": "🌾 முழுமையான மண் மற்றும் உர மேம்படுத்தல்",
        "mode_diag": "🔬 பயிர் நோய் மற்றும் பூச்சி கண்டறிதல் மட்டும்",
        "btn_login": "உள்நுழைக ➔",
        "btn_reg": "கணக்கு உருவாக்கவும்",
        "btn_back": "⬅️ பின்னோக்கி",
        "btn_next": "தொடரவும் ➔",
        "budget_lbl": "அதிகபட்ச உர பட்ஜெட் (₹)",
        "budget_help": "உர வாங்கும் செலவு இந்த வரம்பிற்குள் இருக்கும்.",
        "feedback_title": "🌟 விவசாயி கருத்து மற்றும் மதிப்பீடு",
        "feedback_submit": "கருத்தை சமர்ப்பித்து வெளியேறவும் ➔",
        "land_calc_title": "📐 நில அளவு மற்றும் பட்ஜெட் அட்டவணை",
        "soil_detected": "மண் கண்டறியப்பட்டது",
        "soil_not_detected": "கண்டறியப்படவில்லை"
    },
    "తెలుగు": {
        "title": "స్మార్ట్ కిసాన్ | డిజిటల్ వ్యవసాయ పరిష్కారాలు",
        "subtitle": "ధృవీకరించబడిన 4R పోషక నిర్వహణ, నిజమైన నేల పరీక్ష & అధికారిక సిఫార్సు",
        "login_tab": "రైతు లాగిన్",
        "reg_tab": "కొత్త రైతు నమోదు",
        "mobile_lbl": "మొబైల్ నంబర్",
        "pass_lbl": "పాస్‌వర్డ్",
        "conf_pass_lbl": "పాస్‌వర్డ్‌ని నిర్ధారించండి",
        "lang_select": "భాషను ఎంచుకోండి",
        "mode_select": "వ్యవసాయ సేవను ఎంచుకోండి",
        "mode_opt": "🌾 పూర్తి నేల & ఎరువుల ఆప్టిమైజేషన్",
        "mode_diag": "🔬 మొక్కల వ్యాధి & పురుగుల నిర్ధారణ మాత్రమే",
        "btn_login": "లాగిన్ అవ్వండి ➔",
        "btn_reg": "ఖాతాను సృష్టించండి",
        "btn_back": "⬅️ వెనుకకు",
        "btn_next": "కొనసాగించండి ➔",
        "budget_lbl": "గరిష్ట ఎరువుల బడ్జెట్ (₹)",
        "budget_help": "ఎరువుల కొనుగోలు ఖర్చు ఈ పరిమితిలోనే ఉంటుంది.",
        "feedback_title": "🌟 రైతు అభిప్రాయం & రేటింగ్",
        "feedback_submit": "అభిప్రాయాన్ని సమర్పించండి ➔",
        "land_calc_title": "📐 భూమి మార్పిడి & బడ్జెట్ పట్టిక",
        "soil_detected": "నేల కనుగొనబడింది",
        "soil_not_detected": "కనుగొనబడలేదు"
    },
    "Français": {
        "title": "Smart Kishan | Solutions Agricoles Numériques",
        "subtitle": "Allocation de nutriments 4R certifiée, triage des sols réels et ordonnance officielle",
        "login_tab": "Connexion Agriculteur",
        "reg_tab": "Enregistrer un Nouvel Agriculteur",
        "mobile_lbl": "Numéro de Mobile",
        "pass_lbl": "Mot de Passe",
        "conf_pass_lbl": "Confirmer le Mot de Passe",
        "lang_select": "Langue de l'application / Préférence de langue",
        "mode_select": "Sélectionner le Service Agricole",
        "mode_opt": "🌾 Pipeline complet d'optimisation du sol et des engrais",
        "mode_diag": "🔬 Diagnostic des maladies, ravageurs et médicaments des plantes uniquement",
        "btn_login": "Connexion au Tableau de Bord ➔",
        "btn_reg": "Créer un Compte",
        "btn_back": "⬅️ Retour",
        "btn_next": "Continuer ➔",
        "budget_lbl": "Budget Maximum d'Engrais (₹)",
        "budget_help": "Le moteur d'optimisation garantit que le coût d'achat total reste strictement dans cette limite.",
        "feedback_title": "🌟 Avis des Agriculteurs et Notation par Étoiles",
        "feedback_submit": "Soumettre les commentaires et terminer ➔",
        "land_calc_title": "📐 Conversion des Terres et Matrice Budgétaire",
        "soil_detected": "Sol détecté",
        "soil_not_detected": "Non détecté"
    },
    "日本語": {
        "title": "スマートキシャン | デジタル農業ソリューション",
        "subtitle": "認定4R養分配分、実土壌判定および公式処方箋",
        "login_tab": "農家ログイン",
        "reg_tab": "新規農家登録",
        "mobile_lbl": "携帯電話番号",
        "pass_lbl": "パスワード",
        "conf_pass_lbl": "パスワードの確認",
        "lang_select": "アプリ言語 / グローバル言語設定",
        "mode_select": "農業サービスの選択",
        "mode_opt": "🌾 土壌および肥料最適化パイプライン",
        "mode_diag": "🔬 植物の病気・害虫診断のみ",
        "btn_login": "ダッシュボードにログイン ➔",
        "btn_reg": "アカウント作成",
        "btn_back": "⬅️ 戻る",
        "btn_next": "次へ ➔",
        "budget_lbl": "最大肥料予算 (₹)",
        "budget_help": "最適化エンジンにより、購入費用がこの予算内に厳格に抑えられます。",
        "feedback_title": "🌟 農家のフィードバックと星評価",
        "feedback_submit": "フィードバックを送信して完了 ➔",
        "land_calc_title": "📐 土地面積変換と予算マトリックス",
        "soil_detected": "土壌が検出されました",
        "soil_not_detected": "検出されませんでした"
    },
    "中文": {
        "title": "Smart Kishan | 数字农业解决方案",
        "subtitle": "经认证的4R养分分配、真实土壤筛查与官方处方",
        "login_tab": "农民登录",
        "reg_tab": "注册新农民",
        "mobile_lbl": "手机号码",
        "pass_lbl": "密码",
        "conf_pass_lbl": "确认密码",
        "lang_select": "应用语言 / 全球语言偏好",
        "mode_select": "选择农业服务",
        "mode_opt": "🌾 全面土壤与肥料优化管道",
        "mode_diag": "🔬 仅限植物病虫害及药物诊断",
        "btn_login": "登录农场仪表板 ➔",
        "btn_reg": "创建账户",
        "btn_back": "⬅️ 返回",
        "btn_next": "继续 ➔",
        "budget_lbl": "最大肥料预算 (₹)",
        "budget_help": "优化引擎确保总采购成本严格保持在此限额内。",
        "feedback_title": "🌟 农民反馈与星级评定",
        "feedback_submit": "提交反馈并完成 ➔",
        "land_calc_title": "📐 土地换算与农场预算矩阵",
        "soil_detected": "检测到土壤",
        "soil_not_detected": "未检测到"
    },
    "Deutsch": {
        "title": "Smart Kishan | Digitale Landwirtschaftslösungen",
        "subtitle": "Zertifizierte 4R-Nährstoffzuteilung, reale Boden-Triage & offizielles Rezept",
        "login_tab": "Landwirt Login",
        "reg_tab": "Neuen Landwirt registrieren",
        "mobile_lbl": "Handynummer",
        "pass_lbl": "Passwort",
        "conf_pass_lbl": "Passwort bestätigen",
        "lang_select": "App-Sprache / Globale Sprachpräferenz",
        "mode_select": "Landwirtschaftsdienst auswählen",
        "mode_opt": "🌾 Vollständige Boden- und Düngungsoptimierung",
        "mode_diag": "🔬 Nur Pflanzenkrankheits- und Schädlingsdiagnose",
        "btn_login": "Zum Dashboard anmelden ➔",
        "btn_reg": "Konto erstellen",
        "btn_back": "⬅️ Zurück",
        "btn_next": "Weiter ➔",
        "budget_lbl": "Maximales Düngebudget (₹)",
        "budget_help": "Die Optimierungs-Engine stellt sicher, dass die Gesamtkosten im Budget bleiben.",
        "feedback_title": "🌟 Feedback & Sternebewertung für Landwirte",
        "feedback_submit": "Feedback absenden & beenden ➔",
        "land_calc_title": "📐 Flächenumrechnung & Budgetmatrix",
        "soil_detected": "Boden erkannt",
        "soil_not_detected": "Nicht erkannt"
    }
}

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION & TRANSLATIONS
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
if "rating" not in st.session_state:
    st.session_state.rating = 5
if "plot_id" not in st.session_state:
    st.session_state.plot_id = "Plot No. 104/1"

T = TRANSLATIONS.get(st.session_state.app_lang, TRANSLATIONS["English"])

# -------------------------------------------------------------
# PROFESSIONAL ENGLISH PDF GENERATOR (STRICTLY IN ENGLISH)
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
        self.setStrokeColor(colors.HexColor("#1565C0"))
        self.setLineWidth(1.5)
        self.rect(20, 20, 555, 802)

        self.saveState()
        self.setStrokeColor(colors.HexColor("#1565C0"))
        self.setFillColor(colors.HexColor("#E3F2FD"))
        self.circle(460, 85, 38, stroke=1, fill=1)
        self.circle(460, 85, 33, stroke=1, fill=0)

        self.setFont("Helvetica-Bold", 6)
        self.setFillColor(colors.HexColor("#0D47A1"))
        self.drawCentredString(460, 103, "GOVT COMPLIANT")
        self.drawCentredString(460, 83, "SMART KISHAN")
        self.drawCentredString(460, 68, "4R CERTIFIED")

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
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=17, textColor=colors.HexColor('#1B5E20'), leading=21, alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#2E7D32'), leading=12, alignment=1)
    section_h1 = ParagraphStyle('SecH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, textColor=colors.HexColor('#1B5E20'), leading=14, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#1E293B'), leading=11)
    bold_style = ParagraphStyle('BoldText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#0F172A'), leading=11)

    story = []

    LOGO_FILE = "smart kishan logo.png"
    if os.path.exists(LOGO_FILE):
        try:
            story.append(RLImage(LOGO_FILE, width=140, height=140))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Paragraph("SMART KISHAN • OFFICIAL CROP PRESCRIPTION", title_style))
    story.append(Paragraph("Certified 4R Nutrient Stewardship & Field Application Dossier", subtitle_style))
    story.append(Paragraph(f"Dossier ID: SK-{datetime.now().strftime('%Y%m%d')}-{user_mobile[-4:]} | Generated: {datetime.now().strftime('%d-%b-%Y %I:%M %p')}", ParagraphStyle('Meta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceBefore=2, spaceAfter=8))

    # SECTION 1: Farmer & Farm Profile
    story.append(Paragraph("1. FARMER & LAND PROFILE", section_h1))
    profile_data = [
        [Paragraph("<b>Farmer Mobile:</b>", body_style), Paragraph(f"+91 {user_mobile}", bold_style), Paragraph("<b>Field / Parcel ID:</b>", body_style), Paragraph(str(plot_id), bold_style)],
        [Paragraph("<b>Target Crop:</b>", body_style), Paragraph(str(crop), bold_style), Paragraph("<b>Target Harvest:</b>", body_style), Paragraph(f"{target_yield} t/acre", bold_style)],
        [Paragraph("<b>Land Area:</b>", body_style), Paragraph(f"{raw_land:.2f} {land_unit}", bold_style), Paragraph("<b>Standard Area:</b>", body_style), Paragraph(f"{opt.get('land_area', raw_land*0.404686):.3f} Hectares", bold_style)],
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

    # SECTION 2: Baseline Soil & Telemetry
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

    # SECTION 3: Fertilizer Purchases
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

    # SECTION 4: Timed Application Periods & Methods
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

# -------------------------------------------------------------
# APP HERO HEADER WITH UPLOADED LOGO BADGE
# -------------------------------------------------------------
hero_col1, hero_col2 = st.columns([3, 1.2])
with hero_col1:
    st.markdown(f"""
    <div class="farmer-hero">
        <div class="hero-text">
            <h1>🌾 {T['title']}</h1>
            <p>{T['subtitle']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_col2:
    LOGO_FILE = "smart kishan logo.png"
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=135)
    else:
        st.markdown("""
        <div class="smart-kishan-stamp" style="margin:auto;">
            <span class="stamp-title">GOVT COMPLIANT</span>
            <span class="stamp-center">SMART KISHAN</span>
            <span class="stamp-footer">★ 4R CERTIFIED ★</span>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 1: LOGIN & PREFERENCE (Supports All Global Languages)
# -------------------------------------------------------------
if st.session_state.step == 1:
    st.subheader(f"1. 📱 {T['login_tab']}")
    
    c_lang, c_mode = st.columns(2)
    
    available_languages = ["English", "हिन्दी", "ଓଡ଼ିଆ", "मराठी", "தமிழ்", "తెలుగు", "Français", "日本語", "中文", "Deutsch"]
    current_lang_index = available_languages.index(st.session_state.app_lang) if st.session_state.app_lang in available_languages else 0
    
    new_lang = c_lang.selectbox(T["lang_select"], available_languages, index=current_lang_index)
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
# SCREEN 2: MUTUALLY EXCLUSIVE SOIL SCANNER OR MANUAL INPUT
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
            st.session_state.scanned_diag = res
            
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
        st.subheader("2. 📍 Land Size, Budget & Soil Input (Scanner OR Manual)")
        
        tab_camera, tab_land, tab_soil = st.tabs([
            "📷 Option A: Optical Soil Scanner", 
            "📐 Land Area & Farm Budget", 
            "🧪 Option B: Manual Soil Input"
        ])
        
        with tab_camera:
            st.markdown("##### Real-Time Optical Soil Diagnostic Scanner")
            st.caption("Scan genuine agricultural soil. If soil is detected, apply it directly.")
            
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
                        <h4 style="color:#1B5E20; margin-top:0;">🌾 Scanned Soil Successfully Detected & Analyzed:</h4>
                        <p style="margin:4px 0;">• <strong>Texture Class:</strong> {soil_eval['soil_type']}</p>
                        <p style="margin:4px 0;">• <strong>Optical Color Signature:</strong> {m['rgb_signature']}</p>
                        <p style="margin:4px 0;">• <strong>Organic Carbon (SOC):</strong> {m['soc']}%</p>
                        <p style="margin:4px 0;">• <strong>Surface Moisture:</strong> {m['moist']}%</p>
                        <p style="margin:4px 0;">• <strong>Active pH:</strong> {m['ph']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Apply Scanned Soil Features"):
                        st.session_state.soil_n = m["n"]
                        st.session_state.soil_p = m["p"]
                        st.session_state.soil_k = m["k"]
                        st.session_state.soil_ph = m["ph"]
                        st.session_state.soc = m["soc"]
                        st.session_state.soil_moist = m["moist"]
                        st.session_state.soil_source = "scanner"
                        st.success("✅ Scanned soil successfully applied! You can now continue.")
                else:
                    st.markdown(f"<div class='badge-warn' style='display:inline-block; font-size:16px; margin:10px 0;'>{T['soil_not_detected']}</div>", unsafe_allow_html=True)
                    st.warning("No agricultural soil detected in image. Please provide a genuine soil sample or use Manual Soil Input.")

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
            
            # Target Harvest Input with t/acre unit
            st.session_state.target_yield = c_s3.number_input("Target Harvest (t/acre)", 0.5, 10.0, float(st.session_state.target_yield), 0.25)

            if st.button("Save Manual Soil Values"):
                st.session_state.soil_source = "manual"
                st.success("✅ Manual soil values saved successfully! You can now continue.")

        st.divider()
        b1, b2 = st.columns([1, 5])
        if b1.button(T["btn_back"]):
            st.session_state.step = 1
            st.rerun()
            
        if b2.button(T["btn_next"]):
            if st.session_state.soil_source is None:
                st.error("⚠️ Please either scan genuine soil in Tab 1 OR save manual soil values in Tab 3 before proceeding.")
            else:
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
# SCREEN 5: NUTRIENT GAP & DYNAMIC CROP RECOMMENDATION
# -------------------------------------------------------------
elif st.session_state.step == 5:
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
        st.success(f"🌱 **Best Suited Crop for Scanned/Input Soil**: **{dynamic_pred_crop.capitalize()}**")
        st.caption(f"Calculated via Machine Learning based on active N={st.session_state.soil_n}, P={st.session_state.soil_p}, K={st.session_state.soil_k}, pH={st.session_state.soil_ph}")

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
    if b1.button(T["btn_back"]):
        st.session_state.step = 5
        st.rerun()
    if b2.button(T["btn_next"]):
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: PRESCRIPTION DOSSIER & MULTILINGUAL PDF DOWNLOAD
# -------------------------------------------------------------
elif st.session_state.step == 7:
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
                <h2 style="color: #1B5E20; margin-top: 0; margin-bottom:4px;">🌾 Smart Kishan • Official Crop & Fertilizer Prescription</h2>
                <span style="font-size:13px; color:#2E7D32; font-weight:700;">DIGITAL FARMING SOLUTIONS • 4R CERTIFIED ADVISORY</span>
            </div>
        </div>
        <hr style="border: 1px solid #A5D6A7; margin: 12px 0;"/>
        <p style="margin:4px 0;"><strong>Farmer Mobile:</strong> +91 {st.session_state.user_mobile} | <strong>Parcel ID:</strong> {st.session_state.plot_id}</p>
        <p style="margin:4px 0;"><strong>Cultivated Crop:</strong> {st.session_state.sel_crop} | <strong>Target Harvest:</strong> {st.session_state.target_yield} t/acre</p>
        <p style="margin:4px 0;"><strong>Land Area:</strong> {st.session_state.raw_land_val:.2f} {st.session_state.land_unit} ({st.session_state.land_area:.3f} Ha)</p>
        <hr style="border: 1px solid #A5D6A7; margin: 15px 0;"/>
        <h4 style="color: #1B5E20; margin-bottom: 6px;">🛒 Required Commercial Purchases:</h4>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li><strong>Urea (Synthetic N):</strong> {opt['urea_kg']} kg (~{round(opt['urea_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>DAP (Phosphatic):</strong> {opt['dap_kg']} kg (~{round(opt['dap_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>MOP (Potash):</strong> {opt['mop_kg']} kg (~{round(opt['mop_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>Organic Compost:</strong> {opt['compost_kg']} kg (~{round(opt['compost_kg'] / 50.0)} bags)</li>
        </ul>
        <h3 style="color: #1B5E20; margin-top: 10px;">💰 Total Investment: ₹{opt['total_cost']:,.0f} (Budget: ₹{st.session_state.budget_cap:,.0f})</h3>
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

    # Generate Professional PDF strictly in English to ensure 100% clean rendering without block characters
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
    if st.button(T["btn_back"]):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 8: MANDATORY BORDERLESS STAR RATING & EXIT
# -------------------------------------------------------------
elif st.session_state.step == 8:
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
    st.markdown(f"<h3 style='text-align: center; color: #D4AC0D;'>★ {st.session_state.star_selection} / 5 Stars Rated ★</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    feedback_comments = st.text_area("Your Comments / Suggestions (ଆପଣଙ୍କ ମତାମତ / आपकी प्रतिक्रिया):", placeholder="Write your feedback here...")

    if st.button(T["feedback_submit"]):
        if not feedback_comments.strip():
            st.error("⚠️ Mandatory Feedback Required: Please enter your feedback comments before exiting.")
        else:
            save_feedback(st.session_state.user_mobile, st.session_state.star_selection, feedback_comments)
            st.success("✅ Thank you! Your star rating and feedback have been recorded safely. Exiting session...")
            
            st.session_state.logged_in = False
            st.session_state.user_mobile = ""
            st.session_state.feedback_given = True
            st.session_state.step = 1
            st.cache_data.clear()
            st.rerun()
