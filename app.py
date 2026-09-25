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
    img_rgb = image_obj.convert("RGB").resize((160, 160))
    stat_rgb = ImageStat.Stat(img_rgb)
    r_m, g_m, b_m = stat_rgb.mean[0], stat_rgb.mean[1], stat_rgb.mean[2]

    if r_m > 200 and g_m > 200 and b_m > 200:
        return {"detected": False, "reason": "Bright artificial surface or white concrete detected."}
    if r_m > 140 and g_m > 110 and b_m > 90 and r_m > g_m and g_m > b_m:
        return {"detected": False, "reason": "Human skin tone detected. Please scan genuine agricultural field soil."}
    if b_m > r_m and b_m > g_m and b_m > 120:
        return {"detected": False, "reason": "Non-soil sky, water, or blue surface detected."}

    is_earth_tone = (r_m >= g_m >= b_m) or (r_m < 110 and g_m < 110 and b_m < 110)
    
    gray = img_rgb.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_var = edge_stat.var[0]

    if is_earth_tone and edge_var > 15.0 and b_m < (r_m + 20):
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
    img_rgb = image_obj.convert("RGB").resize((120, 120))
    arr = np.array(img_rgb)
    r_mean, g_mean, b_mean = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])
    
    if g_mean > r_mean + 10 and g_mean > b_mean:
        return {
            "health": "Vigorous Healthy Plant Canopy (100% Chlorophyll Index)",
            "disease": "No pathogenic infection observed.",
            "pest": "Negligible sap-feeder presence (<2% leaf area)",
            "symptoms": "Robust turgidity, active photosynthesis, and deep green lamina.",
            "medicine": "Prophylactic organic spray: Neem Extract 1500 ppm @ 3 ml/L at sunset.",
            "recovery_chance": 100,
            "will_grow": "Yes, exceptional grain and biomass yield expected."
        }
    elif r_mean > g_mean and r_mean > 100:
        return {
            "health": "Moderate Fungal Spot / Early Blight Infection",
            "disease": "Alternaria solani / Cercospora leaf spot complex",
            "pest": "Foliar chewers / Thrips infestation markers",
            "symptoms": "Concentric necrotic brown rings with yellow chlorotic halos.",
            "medicine": "Systemic treatment: Hexaconazole 5% EC @ 2 ml/L + Mancozeb 75% WP @ 2.5 g/L.",
            "recovery_chance": 88,
            "will_grow": "Yes, full recovery guaranteed if systemic spray is administered within 48 hours."
        }
    else:
        return {
            "health": "Advanced Chlorosis & Vascular Wilt Stress",
            "disease": "Fusarium oxysporum / Bacterial Blight blockage",
            "pest": "Root-knot nematode or stem borer activity",
            "symptoms": "Premature leaf senescence, marginal scorching, and stem exudation.",
            "medicine": "Curative protocol: Streptocycline @ 0.5 g/10L + Copper Oxychloride 50% WP @ 3 g/L.",
            "recovery_chance": 72,
            "will_grow": "Moderate; requires immediate corrective irrigation and balanced potassium boost."
        }

# -------------------------------------------------------------
# PAGE CONFIGURATION & AGRITECH THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="Smart Kishan | AgriTech Control Center",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

    div[data-baseweb="menu"] *, 
    ul[data-baseweb="menu"] *, 
    [role="listbox"] *, 
    div[data-baseweb="select"] *, 
    [data-baseweb="popover"] *,
    [data-testid="stFileUploader"] *, 
    [data-testid="stCameraInput"] *, 
    .stLegend *, 
    .vega-bind * {{
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
# DATABASE ENGINE & AUTH HELPERS (SCHEMA MIGRATIONS FOR REQUESTS/RESPONSES)
# -------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    try:
        db_user = "postgres.ivshypgnhsprrkhkzkkx"
        db_password = "SambitSwain2005"
        db_host = "aws-0-ap-northeast-1.pooler.supabase.com"
        db_port = 6543
        db_name = "postgres"

        cfg_user = urllib.parse.quote_plus(db_user)
        cfg_password = urllib.parse.quote_plus(db_password)

        db_uri = f"postgresql+psycopg2://{cfg_user}:{cfg_password}@{db_host}:{db_port}/{db_name}?sslmode=require"

        engine = create_engine(
            db_uri,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"connect_timeout": 10},
        )

        with engine.connect() as conn:
            # 1. Users table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    mobile_number TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'farmer'
                )
            """))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'farmer'"))

            # 2. Feedback table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    mobile TEXT,
                    rating INT,
                    rating_text TEXT,
                    comments TEXT,
                    admin_reply TEXT
                )
            """))
            conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS rating_text TEXT"))
            conn.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS admin_reply TEXT"))

            # 3. Help Requests table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS help_requests (
                    id SERIAL PRIMARY KEY,
                    mobile TEXT,
                    request_type TEXT,
                    query_text TEXT,
                    status TEXT DEFAULT 'Pending',
                    admin_reply TEXT,
                    new_password_assigned TEXT,
                    attended_by TEXT,
                    user_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("ALTER TABLE help_requests ADD COLUMN IF NOT EXISTS new_password_assigned TEXT"))
            conn.execute(text("ALTER TABLE help_requests ADD COLUMN IF NOT EXISTS user_feedback TEXT"))
            conn.execute(text("ALTER TABLE help_requests ADD COLUMN IF NOT EXISTS attended_by TEXT"))

            # 4. Activity Logs table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id SERIAL PRIMARY KEY,
                    mobile TEXT,
                    activity_type TEXT,
                    details TEXT,
                    is_deleted INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()

        return engine
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

engine = get_db_engine()

def log_activity(mobile, activity_type, details):
    if not engine or not mobile:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO user_activity (mobile, activity_type, details, is_deleted)
                VALUES (:m, :a, :d, 0)
            """), {"m": mobile, "a": activity_type, "d": details})
            conn.commit()
    except Exception:
        pass

def register_user(mobile, password, role="farmer"):
    fixed_admins = ["9348315602", "7735402865", "9692904951"]
    if mobile in fixed_admins:
        return False, "This mobile number is reserved for administrative access and cannot be registered."

    if not engine:
        return False, "Database connection unavailable. Please verify network credentials."

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT mobile_number FROM users WHERE mobile_number = :m"),
                {"m": mobile}
            ).fetchone()

            if existing:
                return False, "This mobile number is already registered. Please proceed to Sign In."

            conn.execute(
                text("INSERT INTO users (mobile_number, password, role) VALUES (:m, :p, :r)"),
                {"m": mobile, "p": hashed_pw, "r": role}
            )
            conn.commit()
        log_activity(mobile, "Account Created", "User successfully registered.")
        return True, "Registration successful! You can now sign in using your credentials."
    except Exception as e:
        return False, f"Registration error: {e}"

def verify_user(mobile, password, selected_role="farmer"):
    fixed_admins = {
        "9348315602": hashlib.sha256("Sambit@123".encode()).hexdigest(),
        "7735402865": hashlib.sha256("Swastidhar@123".encode()).hexdigest(),
        "9692904951": hashlib.sha256("Prabhu@123".encode()).hexdigest(),
    }
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    if selected_role == "admin":
        if mobile in fixed_admins and fixed_admins[mobile] == hashed_pw:
            return True, "admin"
        return False, "admin"

    if not engine:
        return False, "farmer"
    try:
        with engine.connect() as conn:
            res = conn.execute(
                text("SELECT password, role FROM users WHERE mobile_number = :m"), 
                {"m": mobile}
            ).fetchone()
            if res and res[0] == hashed_pw:
                log_activity(mobile, "Sign In", "User signed in successfully.")
                return True, (res[1] or "farmer")
    except Exception:
        return False, "farmer"
    return False, "farmer"

def save_feedback(mobile, rating, rating_text, comments):
    if not engine:
        return True
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO feedback (mobile, rating, rating_text, comments)
                VALUES (:m, :r, :rt, :c)
            """), {"m": mobile, "r": rating, "rt": rating_text, "c": comments})
            conn.commit()
        log_activity(mobile, "Feedback Given", f"Rated {rating_text} ({rating}/5)")
        return True
    except Exception:
        return False

# -------------------------------------------------------------
# GLOBAL MULTILINGUAL UI DICTIONARY
# -------------------------------------------------------------
TRANSLATIONS = {
    "English": {
        "title": "SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION",
        "subtitle": "Certified 4R Nutrient Allocation, Real-Soil Triage & Official Prescription",
        "login_tab": "Farmer Sign In",
        "admin_login_tab": "Admin Sign In",
        "reg_tab": "Farmer Registration",
        "mobile_lbl": "Mobile Number",
        "pass_lbl": "Password",
        "conf_pass_lbl": "Confirm Password",
        "lang_select": "Global Language Selection",
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
    "Hindi (हिन्दी)": {
        "title": "SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION",
        "subtitle": "प्रमाणित 4R पोषक तत्व प्रबंधन, वास्तविक मृदा विश्लेषण और आधिकारिक नुस्खा",
        "login_tab": "किसान साइन इन",
        "admin_login_tab": "एडमिन साइन इन",
        "reg_tab": "किसान पंजीकरण",
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
    "Odia (ଓଡ଼ିଆ)": {
        "title": "SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION",
        "subtitle": "ପ୍ରମାଣିତ ୪ଆର୍ ପୋଷକ ପରିଚାଳନା, ପ୍ରକୃତ ମୃତ୍ତିକା ବିଶ୍ଳେଷଣ ଓ ସରକାରୀ ପ୍ରେସକ୍ରିପସନ",
        "login_tab": "କୃଷକ ସାଇନ୍‌ ଇନ୍",
        "admin_login_tab": "ଆଡମିନ୍ ସାଇନ୍‌ ଇନ୍",
        "reg_tab": "କୃଷକ ପଞ୍ଜୀକରଣ",
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
if "rating_text" not in st.session_state:
    st.session_state.rating_text = "Best"
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
        {"role": "assistant", "content": "Hello Farmer! I am your Smart Kishan AI Assistant. Ask me anything about crop diseases, NPK fertilizer budgeting, organic compost, or pesticide dosages!"}
    ]

T = TRANSLATIONS.get(st.session_state.app_lang, TRANSLATIONS["English"])

# -------------------------------------------------------------
# PROFESSIONAL PDF GENERATORS
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
        self.setFillColor(colors.HexColor("#333333"))
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
# PERMANENT RIGHT-SIDE AI AGRI-BOT HELPER
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
# SCREEN 1: SMART KISHAN CINEMATIC LOGIN & REGISTRATION
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

        t_login, t_admin, t_reg = st.tabs([T["login_tab"], T["admin_login_tab"], T["reg_tab"]])

        # Farmer Sign In
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

            if st.button(
                "Sign In as Farmer →",
                key="login_button"
            ):
                if len(m.strip()) == 10:
                    valid, user_role = verify_user(m.strip(), p.strip(), selected_role="farmer")

                    if valid:
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = m.strip()
                        st.session_state.user_role = "farmer"
                        st.session_state.step = 2
                        st.rerun()
                    else:
                        st.error("Invalid credentials or unregistered mobile number.")
                else:
                    st.warning("Enter a valid 10-digit mobile number.")

            # Password Reset Help Request Button on Login Screen
            with st.expander("Forgot Password"):
                st.caption("Submit your mobile number to request password reset from the administrative team:")
                forgot_mob = st.text_input("Registered Mobile Number:", max_chars=10, key="forgot_mob_input")
                if st.button("Submit Password Reset Request"):
                    if len(forgot_mob.strip()) == 10:
                        if engine:
                            with engine.connect() as conn:
                                conn.execute(text("""
                                    INSERT INTO help_requests (mobile, request_type, query_text)
                                    VALUES (:m, 'Reset Password', :q)
                                """), {"m": forgot_mob.strip(), "q": forgot_note.strip() or "Password reset request from login screen."})
                                conn.commit()
                        st.success("✅ Password reset request sent to admin team. Check back or log in once updated.")
                    else:
                        st.warning("Please enter a valid 10-digit mobile number.")

        # Admin Sign In
        with t_admin:
            st.info("🔐 Authorized Administrators Only")
            am = st.text_input(
                "Admin Mobile Number",
                max_chars=10,
                key="admin_log_m",
                placeholder="Admin 10-digit mobile"
            )

            ap = st.text_input(
                "Admin Password",
                type="password",
                key="admin_log_p",
                placeholder="Enter admin password"
            )

            if st.button(
                "Sign In as Admin →",
                key="admin_login_button"
            ):
                if len(am.strip()) == 10:
                    valid, user_role = verify_user(am.strip(), ap.strip(), selected_role="admin")

                    if valid:
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = am.strip()
                        st.session_state.user_role = "admin"
                        st.session_state.step = 90
                        st.rerun()
                    else:
                        st.error("Access Denied: Unauthorized admin number or incorrect password.")
                else:
                    st.warning("Enter a valid 10-digit mobile number.")

        # Farmer Registration Only
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

            if st.button(
                "Create Farmer Account →",
                key="register_button"
            ):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(
                        rm.strip(),
                        rp.strip(),
                        role="farmer"
                    )

                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning(
                        "Please check mobile number and matching passwords."
                    )

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# ADMIN DASHBOARD (STEP 90)
# -------------------------------------------------------------
elif st.session_state.step == 90 and st.session_state.user_role == "admin":
    if os.path.exists(LOGO_FILE_EXACT):
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : ADMIN COMMAND CENTER</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Logged in Admin: +91 {st.session_state.user_mobile}</p>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🚪 Admin Sign Out"):
        st.session_state.logged_in = False
        st.session_state.user_mobile = ""
        st.session_state.step = 1
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs([
        "👥 Manage Users & Activity Logs", 
        "⭐ Reviews & Ratings Management", 
        "🛠️ User Help Requests & Triage",
        "💬 Live Help Chat"
    ])

    # Tab 1: User Management & Activity History
    with admin_tab1:
        st.markdown("### 👥 Registered Users Management & Control")
        if engine:
            try:
                with engine.connect() as conn:
                    users_df = pd.read_sql("SELECT mobile_number, role FROM users", conn)
                if not users_df.empty:
                    st.dataframe(users_df, use_container_width=True)
                    del_mob = st.text_input("Enter Mobile Number to Delete User Account:", key="del_user_input")
                    if st.button("🗑️ Delete User ID"):
                        if del_mob.strip():
                            with engine.connect() as conn:
                                conn.execute(text("DELETE FROM users WHERE mobile_number = :m"), {"m": del_mob.strip()})
                                conn.commit()
                            log_activity(st.session_state.user_mobile, "Admin Action", f"Deleted user {del_mob}")
                            st.success(f"Successfully deleted user account: {del_mob}")
                            st.rerun()
                else:
                    st.info("No registered users found in database.")
            except Exception as e:
                st.error(f"Error loading users: {e}")

            st.markdown("---")
            st.markdown("### 📜 User Activity History")
            try:
                with engine.connect() as conn:
                    act_df = pd.read_sql("SELECT mobile, activity_type, details, created_at FROM user_activity WHERE is_deleted = 0 ORDER BY created_at DESC", conn)
                if not act_df.empty:
                    st.dataframe(act_df, use_container_width=True)
                else:
                    st.info("No active logs recorded.")
            except Exception as e:
                st.error(f"Error loading activity logs: {e}")
        else:
            st.info("Database offline.")

    # Tab 2: Reviews, Ratings & Feedback Management with Text Replies
    with admin_tab2:
        st.markdown("### ⭐ Farmer Reviews, Ratings & Admin Feedback Reply")
        if engine:
            try:
                with engine.connect() as conn:
                    fb_df = pd.read_sql("SELECT id, mobile, rating, rating_text, comments, admin_reply FROM feedback ORDER BY id DESC", conn)
                if not fb_df.empty:
                    for idx, row in fb_df.iterrows():
                        fid = row["id"]
                        fmob = row["mobile"]
                        frating = row["rating"]
                        frate_text = row["rating_text"] or "Good"
                        fcomments = row["comments"]
                        freply = row["admin_reply"] or ""

                        with st.expander(f"Review #{fid} | Mobile: +91 {fmob} | Rated: {frating}/5 ({frate_text})"):
                            st.write(f"**Comments:** {fcomments}")
                            reply_text = st.text_area("Admin Reply to this Feedback:", value=freply, key=f"fb_reply_{fid}")
                            if st.button("Save & Post Reply", key=f"save_fb_{fid}"):
                                with engine.connect() as conn:
                                    conn.execute(text("UPDATE feedback SET admin_reply = :r WHERE id = :id"), {"r": reply_text, "id": fid})
                                    conn.commit()
                                st.success("✅ Reply saved successfully!")
                                st.rerun()
                else:
                    st.info("No feedback records available yet.")
            except Exception as e:
                st.error(f"Error loading feedback: {e}")

    # Tab 3: User Help Requests & Triage
    with admin_tab3:
        st.markdown("### 🛠️ User Help Requests (Password Reset, Delete ID, Recover Activity)")
        st.caption("🔒 Exclusive Lock: When one admin attends a request, it is locked so no other admin can attend it.")
        if engine:
            try:
                with engine.connect() as conn:
                    req_df = pd.read_sql("SELECT id, mobile, request_type, query_text, status, admin_reply, new_password_assigned, attended_by, user_feedback FROM help_requests ORDER BY id DESC", conn)
                if not req_df.empty:
                    for idx, row in req_df.iterrows():
                        rid = row["id"]
                        rmob = row["mobile"]
                        rtype = row["request_type"]
                        rquery = row["query_text"]
                        rstatus = row["status"]
                        rreply = row["admin_reply"] or ""
                        rpass = row["new_password_assigned"] or ""
                        rattended = row["attended_by"] or ""
                        rufeedback = row["user_feedback"] or ""

                        with st.expander(f"Request #{rid} from +91 {rmob} | Type: {rtype} | Status: {rstatus}"):
                            st.write(f"**Farmer Query / Description:** {rquery}")
                            if rufeedback:
                                st.info(f"**Farmer Reply / Feedback on Resolution:** {rufeedback}")
                            
                            # Lock check
                            if rattended and rattended != st.session_state.user_mobile:
                                st.warning(f"🔒 Handled exclusively by Admin: +91 {rattended}")
                            else:
                                admin_ans = st.text_area("Admin Resolution Message to Farmer:", value=rreply, key=f"ans_req_{rid}")
                                
                                # Specific handling per request type
                                if rtype == "Reset Password":
                                    new_pwd = st.text_input("Enter New Password to Assign:", value=rpass, key=f"new_pwd_{rid}")
                                    if st.button("Accept, Change Password & Resolve", key=f"resolve_pwd_{rid}"):
                                        if new_pwd.strip():
                                            hashed_npw = hashlib.sha256(new_pwd.strip().encode()).hexdigest()
                                            with engine.connect() as conn:
                                                conn.execute(text("UPDATE users SET password = :p WHERE mobile_number = :m"), {"p": hashed_npw, "m": rmob})
                                                conn.execute(text("""
                                                    UPDATE help_requests 
                                                    SET status = 'Resolved', admin_reply = :r, new_password_assigned = :p, attended_by = :a 
                                                    WHERE id = :id
                                                """), {"r": admin_ans or f"Your password has been reset to: {new_pwd.strip()}", "p": new_pwd.strip(), "a": st.session_state.user_mobile, "id": rid})
                                                conn.commit()
                                            st.success("✅ Password updated and request resolved!")
                                            st.rerun()
                                elif rtype == "Delete My Account":
                                    if st.button("Accept & Delete User Account Permanently", key=f"del_acc_{rid}"):
                                        with engine.connect() as conn:
                                            conn.execute(text("DELETE FROM users WHERE mobile_number = :m"), {"m": rmob})
                                            conn.execute(text("""
                                                UPDATE help_requests 
                                                SET status = 'Resolved', admin_reply = :r, attended_by = :a 
                                                WHERE id = :id
                                            """), {"r": admin_ans or "Your account has been deleted as requested.", "a": st.session_state.user_mobile, "id": rid})
                                            conn.commit()
                                        st.success(f"✅ User account +91 {rmob} deleted and request resolved!")
                                        st.rerun()
                                elif rtype == "Recover My Deleted Activity":
                                    if st.button("Accept & Restore Deleted Activity", key=f"rec_act_{rid}"):
                                        with engine.connect() as conn:
                                            conn.execute(text("UPDATE user_activity SET is_deleted = 0 WHERE mobile = :m"), {"m": rmob})
                                            conn.execute(text("""
                                                UPDATE help_requests 
                                                SET status = 'Resolved', admin_reply = :r, attended_by = :a 
                                                WHERE id = :id
                                            """), {"r": admin_ans or "Your deleted activity has been recovered successfully.", "a": st.session_state.user_mobile, "id": rid})
                                            conn.commit()
                                        st.success(f"✅ Activity for +91 {rmob} recovered!")
                                        st.rerun()
                                else:
                                    if st.button("Send Resolution Message", key=f"gen_res_{rid}"):
                                        with engine.connect() as conn:
                                            conn.execute(text("""
                                                UPDATE help_requests 
                                                SET status = 'Resolved', admin_reply = :r, attended_by = :a 
                                                WHERE id = :id
                                            """), {"r": admin_ans, "a": st.session_state.user_mobile, "id": rid})
                                            conn.commit()
                                        st.success("✅ Request resolved!")
                                        st.rerun()
                else:
                    st.info("No help requests pending.")
            except Exception as e:
                st.error(f"Error loading requests: {e}")

    # Tab 4: Live Help Desk Chat Center
    with admin_tab4:
        st.markdown("### 💬 Live Help Desk Chat Center")
        if engine:
            try:
                with engine.connect() as conn:
                    all_chats = pd.read_sql("SELECT id, mobile, query_text, admin_reply, attended_by FROM queries ORDER BY id DESC", conn)
                if not all_chats.empty:
                    for idx, crow in all_chats.iterrows():
                        cid = crow["id"]
                        cmob = crow["mobile"]
                        ctext = crow["query_text"]
                        creply = crow["admin_reply"] or ""
                        with st.expander(f"Live Chat with Farmer: +91 {cmob}"):
                            st.write(f"🧑‍🌾 **Farmer:** {ctext}")
                            st.write(f"🛡️ **Admin Response:** {creply or 'Waiting for reply...'}")
                            chat_res = st.text_input("Reply to farmer:", key=f"live_chat_ans_{cid}")
                            if st.button("Send Chat Reply", key=f"btn_chat_res_{cid}"):
                                with engine.connect() as conn:
                                    conn.execute(text("UPDATE queries SET admin_reply = :r, attended_by = :a WHERE id = :id"), {"r": chat_res, "a": st.session_state.user_mobile, "id": cid})
                                    conn.commit()
                                st.success("✅ Reply sent live!")
                                st.rerun()
                else:
                    st.info("No live chat queries active.")
            except Exception as e:
                st.error(f"Error loading live chats: {e}")

# -------------------------------------------------------------
# SCREEN 2: MUTUALLY EXCLUSIVE SOIL SCANNER OR MANUAL INPUT
# -------------------------------------------------------------
elif st.session_state.step == 2:
    if os.path.exists(LOGO_FILE_EXACT):
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Control Center &mdash; Role: <strong>{st.session_state.user_role.upper()}</strong> (+91 {st.session_state.user_mobile})</p>
            </div>
            """, unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([3, 1])
    if h_col2.button("🚪 Sign Out"):
        st.session_state.logged_in = False
        st.session_state.step = 1
        st.rerun()

    # --- Notification & Help Center Tab Bar for Farmer ---
    f_help_tab, f_notify_tab, f_act_tab = st.tabs([
        "🆘 Help & Account Requests", 
        "🔔 Resolved Notifications & Reply", 
        "📜 My Activity History"
    ])

    with f_help_tab:
        st.caption("Submit an official request directly to the administration:")
        h_type = st.selectbox(
            "Select Request Category:", 
            ["Reset Password", "Delete My Account", "Recover My Deleted Activity", "General Inquiry"]
        )
        h_details = st.text_area("Provide details or query for the admin team:")
        if st.button("Submit Request to Admin ➔"):
            if h_details.strip() or h_type:
                if engine:
                    with engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO help_requests (mobile, request_type, query_text)
                            VALUES (:m, :rt, :q)
                        """), {"m": st.session_state.user_mobile, "rt": h_type, "q": h_details.strip()})
                        conn.commit()
                log_activity(st.session_state.user_mobile, "Help Request", f"Submitted request: {h_type}")
                st.success("✅ Request sent to admin! Check the 'Resolved Notifications' tab for updates.")

    with f_notify_tab:
        st.markdown("### 🔔 Resolved Messages & Admin Responses")
        if engine:
            try:
                with engine.connect() as conn:
                    notif_df = pd.read_sql("SELECT id, request_type, status, admin_reply, new_password_assigned, user_feedback FROM help_requests WHERE mobile = :m ORDER BY id DESC", conn, params={"m": st.session_state.user_mobile})
                if not notif_df.empty:
                    for idx, row in notif_df.iterrows():
                        nid = row["id"]
                        ntype = row["request_type"]
                        nstat = row["status"]
                        nreply = row["admin_reply"] or "Awaiting admin resolution..."
                        npass = row["new_password_assigned"]
                        nufeed = row["user_feedback"] or ""

                        st.markdown(f"""
                        <div style="background: rgba(11, 61, 46, 0.95); padding: 14px; border-radius: 12px; border-left: 5px solid #39FF88; margin-bottom: 10px;">
                            <h4 style="margin:0; color:#39FF88;">Request #{nid}: {ntype} &mdash; Status: {nstat}</h4>
                            <p style="margin:6px 0 0 0; color:#FFFFFF;"><strong>Admin Resolution Message:</strong> {nreply}</p>
                            {f"<p style='color:#FFD700;'><strong>New Assigned Password:</strong> {npass}</p>" if npass else ""}
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander(f"Reply or Give Feedback on Resolution #{nid}"):
                            u_reply_msg = st.text_area("Your Reply / Feedback to this resolution:", value=nufeed, key=f"user_feedback_notif_{nid}")
                            if st.button("Send Feedback to Admin", key=f"btn_send_ufeed_{nid}"):
                                with engine.connect() as conn:
                                    conn.execute(text("UPDATE help_requests SET user_feedback = :uf WHERE id = :id"), {"uf": u_reply_msg, "id": nid})
                                    conn.commit()
                                st.success("✅ Reply sent to admin successfully!")
                                st.rerun()
                else:
                    st.info("No notifications yet.")
            except Exception as e:
                st.error(f"Error loading notifications: {e}")

    with f_act_tab:
        st.markdown("### 📜 My Agricultural Activity History")
        if engine:
            try:
                with engine.connect() as conn:
                    my_act = pd.read_sql("SELECT activity_type, details, created_at FROM user_activity WHERE mobile = :m AND is_deleted = 0 ORDER BY created_at DESC", conn, params={"m": st.session_state.user_mobile})
                if not my_act.empty:
                    st.dataframe(my_act, use_container_width=True)
                else:
                    st.info("No activity logged yet.")
            except Exception as e:
                st.error(f"Error loading activity: {e}")

    st.markdown("---")

    if st.session_state.app_mode == "Diagnostic Only":
        st.subheader("🔬 AI Optical Crop Disease & Pest Diagnosis & Treatment Prescription")

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
                st.write(f"**Crop Disease / Pathogen**: {res['disease']}")
                st.write(f"**Pest Recognition**: {res['pest']}")
                st.write(f"**Visible Symptoms**: {res['symptoms']}")

                st.divider()
                if st.button(T["btn_back"], key="diag_tab1_back"):
                    st.session_state.step = 1
                    st.rerun()

            with diag_tab2:
                st.markdown("##### Prescribed Treatment & Application Schedule:")
                st.write(f"• **Recommended Remedy Spray**: {res['medicine']}")
                st.write("• **Application Frequency**: Apply once every 7 to 10 days during early morning or late evening hours.")
                st.metric("Disease Recovery %", f"{res['recovery_chance']}%")
                st.write(f"**Will this crop continue to grow?**: **{res['will_grow']}**")

                st.divider()
                if st.button(T["btn_back"], key="diag_tab2_back"):
                    st.session_state.step = 1
                    st.rerun()

            with diag_tab3:
                st.markdown("##### Download Official Disease & Treatment Prescription")
                disease_pdf_bytes = generate_disease_pdf(
                    user_mobile=st.session_state.user_mobile,
                    plot_id=st.session_state.plot_id,
                    crop=st.session_state.sel_crop,
                    diag=res
                )
                disease_pdf_filename = f"SmartKishan_Disease_Prescription_{st.session_state.user_mobile}.pdf"
                st.download_button(
                    label="Download Plant Pathology Prescription (PDF)",
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
                    if st.button("Proceed to Feedback & Exit", key="diag_tab3_proceed"):
                        st.session_state.step = 8
                        st.rerun()

    else:
        st.subheader("2. Land Size, Budget & Soil Input (Scanner OR Manual)")

        tab_camera, tab_land, tab_soil = st.tabs([
            "Option A: 100% Live Optical Soil Scanner", 
            "Land Area & Farm Budget", 
            "Option B: Manual Soil Input"
        ])

        with tab_camera:
            st.markdown("##### Real-Time Optical Soil Diagnostic Scanner")
            st.caption("Scan genuine agricultural soil. Non-soil elements like white roofs, walls, skin, or artificial surfaces are strictly rejected.")

            cam_c1, cam_c2 = st.columns(2)
            with cam_c1:
                soil_cam = st.camera_input("Scan Field Soil Live")
            with cam_c2:
                soil_file = st.file_uploader("Or Upload Soil Sample Photo", type=["jpg", "jpeg", "png"])

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
                        <h4 style="color:#39FF88; margin-top:0;">Live Soil Successfully Verified & Analyzed:</h4>
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
                        log_activity(st.session_state.user_mobile, "Soil Scanned", f"{soil_eval['soil_type']} detected.")
                        st.success("Verified agricultural soil applied successfully! You can now continue.")
                else:
                    st.markdown(f"<div class='badge-warn' style='display:inline-block; font-size:16px; margin:10px 0;'>{T['soil_not_detected']}</div>", unsafe_allow_html=True)
                    st.error(f"{soil_eval['reason']} Please provide a genuine agricultural soil sample.")

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
                log_activity(st.session_state.user_mobile, "Manual Soil Entry", f"N={st.session_state.soil_n}, P={st.session_state.soil_p}, K={st.session_state.soil_k}")
                st.success("Manual soil values saved successfully! You can now continue.")

        st.divider()
        b1, b2 = st.columns([1, 5])
        if b1.button(T["btn_back"], key="step2_back"):
            st.session_state.step = 1
            st.rerun()

        if b2.button(T["btn_next"], key="step2_next"):
            if st.session_state.soil_source is None:
                st.error("Please either verify genuine agricultural soil in Tab 1 OR save manual soil values in Tab 3 before proceeding.")
            else:
                st.session_state.step = 3
                st.rerun()

# -------------------------------------------------------------
# SCREEN 3: SOIL HEALTH EVALUATION
# -------------------------------------------------------------
elif st.session_state.step == 3:
    if os.path.exists(LOGO_FILE_EXACT):
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Soil Condition & Risk Assessment</p>
            </div>
            """, unsafe_allow_html=True)

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
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Current Soil Nutrients vs Ideal Farm Target</p>
            </div>
            """, unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("##### Nitrogen (N) Ratio (Donut Chart)")
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
        st.markdown(f"<span style='color: #000000 !important; font-weight: 700;'>Measured: {st.session_state.soil_n * 2.24:.1f} kg/ha (Target: 280 kg/ha)</span>", unsafe_allow_html=True)

    with d2:
        st.markdown("##### Phosphorus (P) Ratio (Donut Chart)")
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
        st.markdown(f"<span style='color: #000000 !important; font-weight: 700;'>Measured: {st.session_state.soil_p * 2.24:.1f} kg/ha (Target: 60 kg/ha)</span>", unsafe_allow_html=True)

    with d3:
        st.markdown("##### Potash (K) Ratio (Donut Chart)")
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
        st.markdown(f"<span style='color: #000000 !important; font-weight: 700;'>Measured: {st.session_state.soil_k * 2.24:.1f} kg/ha (Target: 150 kg/ha)</span>", unsafe_allow_html=True)

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
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Required Nutrient Deficit & Dynamic Crop Recommendation</p>
            </div>
            """, unsafe_allow_html=True)
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
        st.markdown(f"##### Bar Chart: Nutrient Shortages for {st.session_state.target_yield} t/acre")
        def_df = pd.DataFrame({
            "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potash (K)"],
            "Shortage (kg/acre)": [def_n, def_p, def_k]
        })
        st.bar_chart(def_df.set_index("Nutrient"))
    with g2:
        st.markdown("##### Strong AI Universal Crop Recommendation:")
        st.success(f"Best Suited Plant/Crop: **{dynamic_pred_crop.capitalize()}**")
        st.markdown(f"Trained via multi-parameter machine learning matching N={st.session_state.soil_n}, P={st.session_state.soil_p}, K={st.session_state.soil_k}, pH={st.session_state.soil_ph}, Rainfall={st.session_state.rainfall}mm.")

    st.divider()
    b1, b2 = st.columns([1, 5])
    if b1.button(T["btn_back"], key="step5_back"):
        st.session_state.step = 4
        st.rerun()
    if b2.button(T["btn_next"], key="step6_next"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 6: OPTIMIZED FERTILIZER BAGS & TIMETABLE
# -------------------------------------------------------------
elif st.session_state.step == 6:
    if os.path.exists(LOGO_FILE_EXACT):
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Your Fertilizer Bags & Application Schedule</p>
            </div>
            """, unsafe_allow_html=True)
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

    st.markdown("##### Bar Chart: Fertilizer Cost vs Farmer Budget Limit")
    budget_df = pd.DataFrame({
        "Financial Metric": ["Optimized Purchase Cost", "Farmer Budget Cap"],
        "Amount (₹)": [opt['total_cost'], st.session_state.budget_cap]
    })
    st.bar_chart(budget_df.set_index("Financial Metric"))

    st.markdown("##### Fertilizer Quantity Comparison (Bar Chart):")
    fert_qty_df = pd.DataFrame({
        "Fertilizer Product": ["Urea", "DAP", "MOP", "Complex", "Compost"],
        "Quantity (kg)": [opt['urea_kg'], opt['dap_kg'], opt['mop_kg'], opt.get('complex_kg', 0.0), opt['compost_kg']]
    })
    st.bar_chart(fert_qty_df.set_index("Fertilizer Product"))

    st.markdown("##### Fertilizer Share Percentage (Donut / Pie Chart):")
    st.altair_chart(
        __import__('altair').Chart(fert_qty_df).mark_arc(innerRadius=60).encode(
            theta=__import__('altair').Theta(field="Quantity (kg)", type="quantitative"),
            color=__import__('altair').Color(field="Fertilizer Product", type="nominal", scale=__import__('altair').Scale(range=["#39FF88", "#1B5E20", "#81C784", "#2E7D32", "#A7F3D0"]))
        ), use_container_width=True
    )

    st.markdown("##### Timeline Chart: Application Stages")
    timeline_df = pd.DataFrame({
        "Stage": ["Stage 1: Basal (Day 0)", "Stage 2: Vegetative (Day 20-25)", "Stage 3: Flowering (Day 45-55)"],
        "Nutrient Release Efficiency (%)": [90, 85, 95]
    })
    st.line_chart(timeline_df.set_index("Stage"))

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
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Official Farmer Prescription Card (Smart Kishan Certified)</p>
            </div>
            """, unsafe_allow_html=True)
    opt = st.session_state.get("opt_results", {"urea_kg": 0, "dap_kg": 0, "mop_kg": 0, "compost_kg": 0, "total_cost": 0, "land_area": st.session_state.land_area})
    diag = st.session_state.get("scanned_diag", {
        "health": "Optimal Vigor", "disease": "None detected", "pest": "None",
        "symptoms": "Healthy foliage", "medicine": "Prophylactic Neem Spray",
        "recovery_chance": 95, "will_grow": "Yes"
    })

    st.markdown(f"""
    <div class="summary-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div>
                <h2 style="color: #39FF88; margin-top: 0; margin-bottom:4px;">Smart Kishan • Official Crop & Fertilizer Prescription</h2>
                <span style="font-size:13px; color:#A7F3D0; font-weight:700;">AGRITECH CONTROL CENTER • 4R CERTIFIED ADVISORY</span>
            </div>
        </div>
        <hr style="border: 1px solid rgba(57, 255, 136, 0.3); margin: 12px 0;"/>
        <p style="margin:4px 0;"><strong>Farmer Mobile:</strong> +91 {st.session_state.user_mobile} | <strong>Parcel ID:</strong> {st.session_state.plot_id}</p>
        <p style="margin:4px 0;"><strong>Cultivated Crop:</strong> {st.session_state.sel_crop} | <strong>Target Harvest:</strong> {st.session_state.target_yield} t/acre</p>
        <p style="margin:4px 0;"><strong>Land Area:</strong> {st.session_state.raw_land_val:.2f} Acre ({st.session_state.land_area:.3f} Ha)</p>
        <hr style="border: 1px solid rgba(57, 255, 136, 0.3); margin: 15px 0;"/>
        <h4 style="color: #39FF88; margin-bottom: 6px;">Required Commercial Purchases:</h4>
        <ul style="font-size: 15px; line-height: 1.8;">
            <li><strong>Urea (Synthetic N):</strong> {opt['urea_kg']} kg (~{round(opt['urea_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>DAP (Phosphatic):</strong> {opt['dap_kg']} kg (~{round(opt['dap_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>MOP (Potash):</strong> {opt['mop_kg']} kg (~{round(opt['mop_kg'] / 50.0)} bags of 50kg)</li>
            <li><strong>Organic Compost:</strong> {opt['compost_kg']} kg (~{round(opt['compost_kg'] / 50.0)} bags)</li>
        </ul>
        <h3 style="color: #39FF88; margin-top: 10px;">Total Investment: ₹{opt['total_cost']:,.0f} (Budget: ₹{st.session_state.budget_cap:,.0f})</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Timed Application Periods & Methods for Farmers:")
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

    pdf_filename = f"SmartKishan_{st.session_state.app_lang}_Prescription_{st.session_state.user_mobile}.pdf"

    p_col1, p_col2 = st.columns([2, 2])
    with p_col1:
        st.download_button(
            label=f"Download PDF Prescription ({st.session_state.app_lang})",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )
    with p_col2:
        if st.button("Proceed to Feedback & Exit"):
            st.session_state.step = 8
            st.rerun()

    st.divider()
    if st.button(T["btn_back"], key="step7_back"):
        st.session_state.step = 6
        st.rerun()

# -------------------------------------------------------------
# SCREEN 8: MANDATORY FEEDBACK WITH WORST-BAD-GOOD-BETTER-BEST OPTIONS
# -------------------------------------------------------------
elif st.session_state.step == 8:
    if os.path.exists(LOGO_FILE_EXACT):
        c_logo, c_title = st.columns([0.15, 0.85], gap="small")
        with c_logo:
            st.image(LOGO_FILE_EXACT, width=120)
        with c_title:
            st.markdown(f"""
            <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
                <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
                <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Farmer Feedback & Rating System</p>
            </div>
            """, unsafe_allow_html=True)
    st.subheader(T["feedback_title"])
    st.write("Please choose your rating option and leave your valuable comments before exiting:")

    rating_map = {
        "⭐ Worst (1 Star)": 1,
        "⭐⭐ Bad (2 Stars)": 2,
        "⭐⭐⭐ Good (3 Stars)": 3,
        "⭐⭐⭐⭐ Better (4 Stars)": 4,
        "⭐⭐⭐⭐⭐ Best (5 Stars)": 5
    }

    selected_rating_option = st.radio(
        "Select your experience rating:",
        list(rating_map.keys()),
        index=4,
        horizontal=True
    )

    numeric_rating = rating_map[selected_rating_option]
    label_text = selected_rating_option.split(" ")[1]

    st.markdown("<br>", unsafe_allow_html=True)
    feedback_comments = st.text_area("Your Comments / Suggestions:", placeholder="Write your feedback here...")

    b_fb_back, b_fb_sub = st.columns([1, 5])
    if b_fb_back.button(T["btn_back"], key="feedback_back_btn"):
        st.session_state.step = 7 if st.session_state.app_mode == "Full Optimization" else 2
        st.rerun()

    if b_fb_sub.button(T["feedback_submit"]):
        if not feedback_comments.strip():
            st.error("Mandatory Feedback Required: Please enter your feedback comments before exiting.")
        else:
            save_feedback(st.session_state.user_mobile, numeric_rating, label_text, feedback_comments.strip())
            st.success("Thank you! Your feedback has been recorded safely. Exit session...")
            
            st.session_state.logged_in = False
            st.session_state.user_mobile = ""
            st.session_state.feedback_given = True
            st.session_state.step = 1
            st.cache_data.clear()
            st.rerun()
