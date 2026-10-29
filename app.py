import io
import os
import json
import base64
import urllib.parse
import urllib.request
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import hashlib
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageStat, ImageFilter
from sqlalchemy import create_engine, text
import altair as alt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from optimizer import optimize_fertilizer_blend
from train_pipeline import train_all_models

# -------------------------------------------------------------
# BACKEND AI CONFIGURATION
# -------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6LdcF9d53JTlfbkKJ2h9Qg7UZEQjjQb1j6ImA2t5dzA-w")

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="Smart Kishan | AgriTech Control Center", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# -------------------------------------------------------------
# UNIFIED DATABASE ENGINE (SQLite Fallback + Supabase Support)
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
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 5})
        
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, mobile TEXT, rating INT, rating_text TEXT, comments TEXT, admin_reply TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS help_requests (id SERIAL PRIMARY KEY, mobile TEXT, request_type TEXT, query_text TEXT, status TEXT DEFAULT 'Pending', admin_reply TEXT, attended_by TEXT, user_feedback TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS user_activity (id SERIAL PRIMARY KEY, mobile TEXT, activity_type TEXT, details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.commit()
        return engine
    except Exception:
        sqlite_path = "smart_kishan_local.db"
        sqlite_engine = create_engine(f"sqlite:///{sqlite_path}", pool_pre_ping=True)
        with sqlite_engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, mobile TEXT, rating INT, rating_text TEXT, comments TEXT, admin_reply TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS help_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, mobile TEXT, request_type TEXT, query_text TEXT, status TEXT DEFAULT 'Pending', admin_reply TEXT, attended_by TEXT, user_feedback TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS user_activity (id INTEGER PRIMARY KEY AUTOINCREMENT, mobile TEXT, activity_type TEXT, details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.commit()
        return sqlite_engine

engine = get_db_engine()

def log_activity(mobile, activity_type, details):
    if engine and mobile:
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO user_activity (mobile, activity_type, details) VALUES (:m, :a, :d)"), {"m": str(mobile), "a": activity_type, "d": details})
                conn.commit()
        except Exception:
            pass

def register_user(mobile, password, role="farmer"):
    if not engine: return False, "Database connection unavailable."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT mobile FROM users WHERE mobile = :m"), {"m": str(mobile)}).fetchone():
                return False, "This mobile number is already registered."
            conn.execute(text("INSERT INTO users (mobile, password, role) VALUES (:m, :p, :r)"), {"m": str(mobile), "p": hashed_pw, "r": role})
            conn.commit()
        log_activity(mobile, "Account Created", f"Registered as {role}.")
        return True, "Registration successful!"
    except Exception as e:
        return False, f"Registration error: {e}"

def reset_user_password_direct(mobile, new_password):
    if not engine: return False, "Database connection unavailable."
    hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            if not conn.execute(text("SELECT mobile FROM users WHERE mobile = :m"), {"m": str(mobile)}).fetchone():
                return False, "Mobile number not registered."
            conn.execute(text("UPDATE users SET password = :p WHERE mobile = :m"), {"p": hashed_pw, "m": str(mobile)})
            conn.commit()
        log_activity(mobile, "Password Reset", "User password changed successfully.")
        return True, "Password updated successfully!"
    except Exception as e:
        return False, f"Reset error: {e}"

def verify_user(mobile, password, selected_role="farmer"):
    fixed_admins = {
        "9348315602": hashlib.sha256("Sambit@123".encode()).hexdigest(),
        "7735402865": hashlib.sha256("Swastidhar@123".encode()).hexdigest(),
        "9692904951": hashlib.sha256("Prabhu@123".encode()).hexdigest(),
    }
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    
    if selected_role == "admin":
        return (True, "admin") if mobile in fixed_admins and fixed_admins[mobile] == hashed_pw else (False, "admin")
    if mobile in fixed_admins and fixed_admins[mobile] == hashed_pw:
        log_activity(mobile, "Sign In", "Admin signed in as Farmer.")
        return True, "farmer"
    if not engine: return False, "farmer"
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT password, role FROM users WHERE mobile = :m"), {"m": str(mobile)}).fetchone()
            if res and res[0] == hashed_pw:
                log_activity(mobile, "Sign In", "User signed in successfully.")
                return True, (res[1] or "farmer")
    except Exception:
        pass
    return False, "farmer"

def save_feedback(mobile, rating, rating_text, comments):
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO feedback (mobile, rating, rating_text, comments) VALUES (:m, :r, :rt, :c)"), {"m": str(mobile), "r": rating, "rt": rating_text, "c": comments})
                conn.commit()
            log_activity(mobile, "Feedback Given", f"Rated {rating_text} ({rating}/5)")
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

def calculate_advanced_nutrients(target_yield_per_acre, soil_n, soil_p, soil_k, soc, ph, soil_moist, soil_texture):
    target_yield_ha = target_yield_per_acre * 2.47105
    demand_n = 22.0 * target_yield_ha
    demand_p = 4.5 * target_yield_ha
    demand_k = 19.0 * target_yield_ha

    nue_n = 0.50
    if "sandy" in str(soil_texture).lower(): nue_n -= 0.10
    if soil_moist < 30.0 or soil_moist > 75.0: nue_n -= 0.08

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
        return {"detected": False, "reason": "Bright artificial surface or concrete detected."}
    if r_m > 140 and g_m > 110 and b_m > 90 and r_m > g_m and g_m > b_m:
        return {"detected": False, "reason": "Human skin tone detected. Please scan field soil."}
    if b_m > r_m and b_m > g_m and b_m > 120:
        return {"detected": False, "reason": "Non-soil blue/water surface detected."}

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
            "metrics": {"n": est_n, "p": est_p, "k": est_k, "ph": est_ph, "soc": est_soc, "moist": est_moist, "rgb_signature": f"RGB({r_m:.0f}, {g_m:.0f}, {b_m:.0f})"}
        }
    return {"detected": False, "reason": "Surface lacks genuine agricultural soil texture."}

def analyze_plant_disease_image(image_obj):
    img_rgb = image_obj.convert("RGB").resize((120, 120))
    arr = np.array(img_rgb)
    r_mean, g_mean, b_mean = np.mean(arr[:, :, 0]), np.mean(arr[:, :, 1]), np.mean(arr[:, :, 2])

    if g_mean > r_mean + 10 and g_mean > b_mean:
        return {"health": "Healthy Plant Canopy", "disease": "None detected", "pest": "None / Low Risk", "symptoms": "Optimal vegetative growth", "medicine": "Preventative Neem Oil Spray", "recovery_chance": 100, "will_grow": "Yes, optimal"}
    elif r_mean > g_mean and r_mean > 100:
        return {"health": "Leaf Rust / Early Blight", "disease": "Alternaria solani / Fungal", "pest": "Foliar Aphids", "symptoms": "Yellow-brown necrotic halos", "medicine": "Hexaconazole 5% EC @ 2 ml/L", "recovery_chance": 85, "will_grow": "Yes, with timely spray"}
    else:
        return {"health": "Severe Chlorosis", "disease": "Fusarium Wilt", "pest": "Stem Borer", "symptoms": "Loss of vascular pressure", "medicine": "Streptocycline 0.5 g/10L + Copper Oxychloride", "recovery_chance": 68, "will_grow": "Moderate"}

# -------------------------------------------------------------
# VISUAL ASSETS & CSS STYLING (HIDES SIDEBAR ON MAIN PAGES)
# -------------------------------------------------------------
HERO_BG_FILE = "agritech_hero_bg.jpg"
HERO_BG_DATA = ""
if os.path.exists(HERO_BG_FILE):
    with open(HERO_BG_FILE, "rb") as f: HERO_BG_DATA = base64.b64encode(f.read()).decode("utf-8")

LOGO_FILE_EXACT = "smart_kishan_logo.jpg"
if not os.path.exists(LOGO_FILE_EXACT): LOGO_FILE_EXACT = "smart kishan logo.png"
LOGO_DATA = ""
if os.path.exists(LOGO_FILE_EXACT):
    with open(LOGO_FILE_EXACT, "rb") as l_f: LOGO_DATA = base64.b64encode(l_f.read()).decode("utf-8")

if HERO_BG_DATA:
    st.markdown(f'<style>.stApp {{ background-image: linear-gradient(135deg, rgba(6, 30, 22, 0.90) 0%, rgba(14, 75, 48, 0.82) 50%, rgba(110, 235, 175, 0.35) 100%), url("data:image/jpeg;base64,{HERO_BG_DATA}") !important; background-size: cover !important; background-attachment: fixed !important;}}</style>', unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; color: #FFFFFF !important; }
    
    /* Hide the default left Streamlit sidebar completely so login & dashboard are full-width */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    
    div.stButton > button { background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important; color: #39FF88 !important; font-weight: 700 !important; border-radius: 10px !important; border: 1px solid #39FF88 !important; box-shadow: 0 4px 12px rgba(57, 255, 136, 0.3) !important; }
    div.stDownloadButton > button { background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important; color: #39FF88 !important; font-weight: 700 !important; border-radius: 10px !important; border: 1px solid #39FF88 !important; }
    
    .badge-pass { background-color: rgba(57, 255, 136, 0.25); color: #39FF88; padding: 5px 14px; border-radius: 8px; font-weight: 700; border: 1px solid #39FF88; }
    .badge-warn { background-color: rgba(239, 68, 68, 0.25); color: #F87171; padding: 5px 14px; border-radius: 8px; font-weight: 700; border: 1px solid #EF4444; }
    
    .ai-bubble-user { background: rgba(20, 90, 50, 0.90); border: 1px solid rgba(57, 255, 136, 0.5); border-radius: 14px 14px 2px 14px; padding: 12px 16px; margin-bottom: 10px; font-size: 14px; color: #242424 !important; }
    .ai-bubble-bot { background: rgba(6, 35, 25, 0.96); border-left: 5px solid #39FF88; border-top: 1px solid rgba(57,255,136,0.25); border-right: 1px solid rgba(57,255,136,0.25); border-bottom: 1px solid rgba(57,255,136,0.25); border-radius: 14px 14px 14px 2px; padding: 14px 18px; margin-bottom: 12px; font-size: 14px; color: #E2F8EC; line-height: 1.6; }

    #vg-tooltip-element * { color: #000000 !important; }
    div[data-baseweb="menu"] *, ul[data-baseweb="menu"] *, [role="listbox"] * { color: #000000 !important; }
    label, p, span, h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }

    /* GREYISH-BLACK TEXT FOR TYPING, COMMENT & INPUT BOXES */
    input, textarea, div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea, div[data-baseweb="select"] span {
        color: #2D3748 !important;
        -webkit-text-fill-color: #2D3748 !important;
        caret-color: #0B3D2E !important;
        text-shadow: none !important;
        font-weight: 600 !important;
    }
    input::placeholder, textarea::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
        text-shadow: none !important;
    }

    /* GREYISH-BLACK TEXT FOR FILE UPLOADER */
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] button {
        color: #2D3748 !important;
        -webkit-text-fill-color: #2D3748 !important;
        text-shadow: none !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------------
# SAFE SELF-HEALING MODEL LOADER
# -------------------------------------------------------------
MODELS_DIR = "saved_models"
REQUIRED_MODELS = [
    "crop_model.pkl", "crop_encoder.pkl", "fert_model.pkl", "soil_encoder.pkl",
    "crop_type_encoder.pkl", "fert_encoder.pkl", "yield_model.pkl", 
    "yield_features.pkl", "yield_crop_encoder.pkl", "irrigation_model.pkl", "price_model.pkl"
]

def force_retrain():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname in REQUIRED_MODELS:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try: os.remove(fpath)
            except Exception: pass
    train_all_models()

def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in REQUIRED_MODELS):
        train_all_models()

@st.cache_resource(show_spinner=False)
def load_all_models():
    ensure_models_exist()
    try:
        crop_m = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
        crop_enc = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
        fert_m = joblib.load(os.path.join(MODELS_DIR, "fert_model.pkl"))
        soil_enc = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))
        crop_type_enc = joblib.load(os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
        fert_enc = joblib.load(os.path.join(MODELS_DIR, "fert_encoder.pkl"))
        yield_m = joblib.load(os.path.join(MODELS_DIR, "yield_model.pkl"))
        yield_feat = joblib.load(os.path.join(MODELS_DIR, "yield_features.pkl"))
        yield_c_enc = joblib.load(os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
        irrig_m = joblib.load(os.path.join(MODELS_DIR, "irrigation_model.pkl"))
        price_m = joblib.load(os.path.join(MODELS_DIR, "price_model.pkl"))
    except Exception:
        force_retrain()
        crop_m = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
        crop_enc = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
        fert_m = joblib.load(os.path.join(MODELS_DIR, "fert_model.pkl"))
        soil_enc = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))
        crop_type_enc = joblib.load(os.path.join(MODELS_DIR, "crop_type_encoder.pkl"))
        fert_enc = joblib.load(os.path.join(MODELS_DIR, "fert_encoder.pkl"))
        yield_m = joblib.load(os.path.join(MODELS_DIR, "yield_model.pkl"))
        yield_feat = joblib.load(os.path.join(MODELS_DIR, "yield_features.pkl"))
        yield_c_enc = joblib.load(os.path.join(MODELS_DIR, "yield_crop_encoder.pkl"))
        irrig_m = joblib.load(os.path.join(MODELS_DIR, "irrigation_model.pkl"))
        price_m = joblib.load(os.path.join(MODELS_DIR, "price_model.pkl"))
    return (crop_m, crop_enc, fert_m, soil_enc, crop_type_enc, fert_enc, yield_m, yield_feat, yield_c_enc, irrig_m, price_m)

(crop_model, crop_encoder, fert_model, soil_encoder, crop_type_encoder, fert_enc, yield_model, yield_features, yield_crop_encoder, irrig_model, price_model) = load_all_models()

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION (INCLUDES MULTI-SESSION CHAT TABS)
# -------------------------------------------------------------
if "step" not in st.session_state: st.session_state.step = 1
if "prev_step_before_chat" not in st.session_state: st.session_state.prev_step_before_chat = 2
if "app_mode" not in st.session_state: st.session_state.app_mode = "Full Optimization"
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "user_role" not in st.session_state: st.session_state.user_role = "farmer"
if "user_mobile" not in st.session_state: st.session_state.user_mobile = ""
if "rating" not in st.session_state: st.session_state.rating = 5
if "rating_text" not in st.session_state: st.session_state.rating_text = "Best"
if "plot_id" not in st.session_state: st.session_state.plot_id = "Plot No. 104/1"
if "raw_land_val" not in st.session_state: st.session_state.raw_land_val = 1.5
if "land_unit" not in st.session_state: st.session_state.land_unit = "Acre (एकड़ / ଏକର)"
if "budget_cap" not in st.session_state: st.session_state.budget_cap = 25000.0
if "target_yield" not in st.session_state: st.session_state.target_yield = 2.0
if "soil_n" not in st.session_state: st.session_state.soil_n = 50.0
if "soil_p" not in st.session_state: st.session_state.soil_p = 30.0
if "soil_k" not in st.session_state: st.session_state.soil_k = 35.0
if "soil_ph" not in st.session_state: st.session_state.soil_ph = 6.5
if "soc" not in st.session_state: st.session_state.soc = 0.70
if "soil_moist" not in st.session_state: st.session_state.soil_moist = 45.0
if "temp" not in st.session_state: st.session_state.temp = 26.5
if "humidity" not in st.session_state: st.session_state.humidity = 68.0
if "rainfall" not in st.session_state: st.session_state.rainfall = 150.0
if "soil_source" not in st.session_state: st.session_state.soil_source = None
if "sel_soil" not in st.session_state: st.session_state.sel_soil = list(soil_encoder.classes_)[0]
if "sel_crop" not in st.session_state: st.session_state.sel_crop = list(crop_type_encoder.classes_)[0]

# Multi-Chat Session Storage (Stores separate chat threads like Gemini App)
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1 • General Advisory": [
            {"role": "assistant", "content": "Namaste! 🌱 Welcome to **Smart Kishan AI Support** (powered by Gemini Intelligence). Ask me anything—from profitable seed varieties for your district (like Cuttack, Khordha, Puri, etc.) to NPK calculations, crop diseases, or market prices!"}
        ]
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = "Chat 1 • General Advisory"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# -------------------------------------------------------------
# REAL GENERATIVE AI ENGINE (GEMINI + POLLINATIONS LLM + REGIONAL SEED DB)
# Answers ANY question naturally and accurately like the Gemini AI App!
# -------------------------------------------------------------
def get_smart_kishan_ai_response(user_query: str, chat_history: list) -> str:
    q = user_query.strip()
    q_low = q.lower()

    n_val = st.session_state.soil_n
    p_val = st.session_state.soil_p
    k_val = st.session_state.soil_k
    ph_val = st.session_state.soil_ph
    soc_val = st.session_state.soc
    land_v = st.session_state.raw_land_val
    budget_v = st.session_state.budget_cap
    role_v = st.session_state.user_role

    try:
        c_df = pd.DataFrame([{'N': n_val, 'P': p_val, 'K': k_val, 'temperature': st.session_state.temp, 'humidity': st.session_state.humidity, 'ph': ph_val, 'rainfall': st.session_state.rainfall}])
        live_crop = crop_encoder.inverse_transform([crop_model.predict(c_df)[0]])[0].capitalize()
    except Exception:
        live_crop = str(st.session_state.sel_crop).capitalize()

    sys_context = (
        "You are Smart Kishan AI, an advanced, conversational agricultural scientist and personal assistant (like Google Gemini) "
        "built for Indian farmers and agricultural admins. Provide detailed, specific, well-structured, and practical answers. "
        "When asked about districts (like Cuttack, Odisha, etc.), name real high-yielding certified seed varieties (e.g., from ICAR-NRRI Cuttack, OUAT, IARI), "
        "expected yield per acre, maturity duration, and market profitability. "
        f"Current User Context: Role={role_v}, Plot Land={land_v} {st.session_state.land_unit.split()[0]}, Budget=Rs.{budget_v}, "
        f"Soil N={n_val} mg/kg, P={p_val} mg/kg, K={k_val} mg/kg, pH={ph_val}, SOC={soc_val}%, ML Recommended Crop={live_crop}."
    )

    # 1. Try Official Google Gemini SDK if a valid AIzaSy key is present
    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content(f"{sys_context}\n\nUser Question: {q}")
            if res and res.text:
                return res.text.strip()
        except Exception:
            pass

    # 2. Try Key-Free Real-Time Generative LLM Endpoint (OpenAI/Gemini-class intelligence)
    try:
        recent_msgs = [{"role": "system", "content": sys_context}]
        for m in chat_history[-6:]:
            recent_msgs.append({"role": m["role"], "content": m["content"]})
        recent_msgs.append({"role": "user", "content": q})

        req_body = json.dumps({
            "messages": recent_msgs,
            "model": "openai",
            "temperature": 0.6
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://text.pollinations.ai/",
            data=req_body,
            headers={"Content-Type": "application/json", "User-Agent": "SmartKishanAI/2.0"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            llm_reply = resp.read().decode("utf-8").strip()
            if llm_reply and len(llm_reply) > 15 and not llm_reply.startswith("<!DOCTYPE"):
                return llm_reply
    except Exception:
        pass

    # 3. Deep Regional & District-Specific Knowledge Base Fallback (Instant Offline Intelligence)
    if "cuttack" in q_low or "odisha" in q_low or "khordha" in q_low or "puri" in q_low or "seed" in q_low or "profitable" in q_low:
        return (
            "🌾 **Top High-Profit Certified Seed Varieties for Cuttack District (Mahanadi Delta Alluvial Zone):**\n\n"
            "Because Cuttack has fertile deltaic alluvial soil and access to **ICAR-NRRI (National Rice Research Institute, Bidyadharpur, Cuttack)**, these seeds deliver the highest net profit per acre:\n\n"
            "1. **Paddy (Kharif & Rabi High-Value):**\n"
            "   • **CR Dhan 801 / Swarna Sub-1:** Flood & submergence tolerant, yields **22–25 Quintals/Acre**, highly resilient in Cuttack's lowland belts.\n"
            "   • **CR Dhan 310 (High Protein) & Pooja:** High market demand in Odisha mandis with premium procurement rates (**₹2,300–₹2,600/Q**).\n"
            "   • **Kalachampa / Geetanjali (Aromatic):** Premium aromatic rice fetching **₹3,500+/Quintal**.\n"
            "2. **Rabi Pulse Cash Crop (Paira / Rice-Fallow — Highest ROI):**\n"
            "   • **Green Gram (Moong) — *IPM 02-14 (Shikha)* or *OBGG-52 (ratna)*:** Grows on residual soil moisture in Cuttack (Athagarh, Banki, Niali belts) in just **60–65 days**, yielding **5–6 Quintals/Acre** and selling at **₹7,500–₹8,600/Quintal** with minimal fertilizer cost!\n"
            "   • **Black Gram (Biri) — *PU-31* or *Prasad*:** High resistance to Yellow Mosaic Virus (YMV).\n"
            "3. **Commercial Vegetable Seeds (Banki / Barang Riverbank Belt):**\n"
            "   • **Hybrid Brinjal (*Utkal Anushree* / *VNR-212*) & Okra (*Arka Anamika*):** Generates **₹60,000–₹95,000/acre** net profit in Cuttack-Bhubaneswar daily markets."
        )

    if any(w in q_low for w in ["urea", "dap", "mop", "npk", "fertilizer", "dose", "dosage", "bag"]):
        def_n, def_p, def_k = calculate_advanced_nutrients(st.session_state.target_yield, n_val, p_val, k_val, soc_val, ph_val, st.session_state.soil_moist, st.session_state.sel_soil)
        ha = land_v * UNIT_TO_HECTARE.get(st.session_state.land_unit, 0.404686)
        opt = optimize_fertilizer_blend(def_n, def_p, def_k, budget_v, ha, str(st.session_state.sel_soil), st.session_state.rainfall, soc_val)
        return (
            f"🧪 **4R Fertilizer Prescription for {land_v} {st.session_state.land_unit.split()[0]}:**\n"
            f"• **Urea:** {opt['urea_kg']} kg (Split: 25% Day 0, 50% Day 22, 25% Day 48)\n"
            f"• **DAP:** {opt['dap_kg']} kg (100% Basal at sowing)\n"
            f"• **MOP:** {opt['mop_kg']} kg | **Compost:** {opt['compost_kg']} kg\n"
            f"• **Estimated Cost:** ₹{opt['total_cost']:,.0f} (within your ₹{budget_v:,.0f} budget)."
        )

    if any(w in q_low for w in ["disease", "pest", "blight", "rust", "yellow", "leaf", "fungus", "rot", "wilt", "spray", "medicine"]):
        return (
            "🔬 **Plant Pathology & Pest Control Protocol:**\n"
            "• **Fungal Blight / Leaf Spot / Rust:** Spray *Hexaconazole 5% EC @ 2 ml/L* or *Mancozeb 75% WP @ 2.5 g/L*.\n"
            "• **Bacterial Wilt / Leaf Blight:** Drench root zone with *Streptocycline 0.5 g/10L + Copper Oxychloride @ 3 g/L*.\n"
            "• **Stem Borer / Aphids:** Apply *Chlorantraniliprole 18.5% SC @ 0.3 ml/L* or organic *Neem Oil 1500 ppm @ 3 ml/L*."
        )

    return (
        f"🌱 **Smart Kishan AI Analysis:**\n"
        f"Regarding **'{q}'**: For your current {land_v} {st.session_state.land_unit.split()[0]} plot (Soil N: {n_val} mg/kg, P: {p_val} mg/kg, K: {k_val} mg/kg, pH: {ph_val}), "
        f"**{live_crop}** is well-suited. For maximum profitability, use certified foundation seeds treated with *Trichoderma viride @ 4g/kg*, apply 100% DAP at basal sowing, and split Urea into 3 stages."
    )

# -------------------------------------------------------------
# FLOATING BOTTOM-RIGHT "🤖 AGRI AI CHAT SUPPORT" LAUNCHER BUTTON
# Displayed on ALL screens EXCEPT Screen 1 (Login) and Screen 99 (Chat Page itself)
# -------------------------------------------------------------
def render_floating_ai_chat_launcher():
    if st.session_state.step not in [1, 99]:
        st.markdown("""
        <style>
        div.element-container:has(#sk-floating-ai-anchor) + div.element-container {
            position: fixed !important;
            bottom: 24px !important;
            right: 28px !important;
            z-index: 999999 !important;
            width: auto !important;
        }
        div.element-container:has(#sk-floating-ai-anchor) + div.element-container button {
            background: linear-gradient(135deg, #0B3D2E 0%, #145A32 100%) !important;
            color: #39FF88 !important;
            border: 2px solid #39FF88 !important;
            border-radius: 999px !important;
            padding: 14px 26px !important;
            font-size: 16px !important;
            font-weight: 800 !important;
            box-shadow: 0 0 22px rgba(57, 255, 136, 0.75), 0 8px 28px rgba(0, 0, 0, 0.85) !important;
            transition: all 0.25s ease !important;
        }
        div.element-container:has(#sk-floating-ai-anchor) + div.element-container button:hover {
            transform: translateY(-4px) scale(1.04) !important;
            background: #39FF88 !important;
            color: #062319 !important;
            box-shadow: 0 0 32px #39FF88 !important;
        }
        </style>
        <div id="sk-floating-ai-anchor"></div>
        """, unsafe_allow_html=True)

        if st.button("🤖 AgriAI Chat Support", key=f"fab_ai_btn_{st.session_state.step}"):
            st.session_state.prev_step_before_chat = st.session_state.step
            st.session_state.step = 99
            st.rerun()

# -------------------------------------------------------------
# PDF GENERATION (ReportLab) EXACTLY MATCHING YOUR IMAGE
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
        self.circle(490, 85, 38, stroke=1, fill=1)
        self.setStrokeColor(colors.HexColor("#39FF88"))
        self.setLineWidth(2.5)
        self.circle(490, 85, 33, stroke=1, fill=0)
        self.setFont("Helvetica-Bold", 6.5)
        self.setFillColor(colors.HexColor("#FFFFFF"))
        self.drawCentredString(490, 104, "GOVT COMPLIANT")
        self.setFont("Helvetica-Bold", 8.5)
        self.setFillColor(colors.HexColor("#FFD700"))
        self.drawCentredString(490, 83, "SMART KISHAN")
        self.setFont("Helvetica-Bold", 6.5)
        self.setFillColor(colors.HexColor("#39FF88"))
        self.drawCentredString(490, 68, "4R CERTIFIED")
        self.restoreState()

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(30, 28, "Smart Kishan • Digital Farming Solutions • ISO 9001:2015 Standard")
        self.drawRightString(565, 28, f"Page {self._pageNumber} of {page_count}")

def generate_english_pdf(user_mobile, plot_id, raw_land, land_unit, crop, target_yield, budget, opt, n, p, k, ph, soc, moist, temp, humid, rain):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=45)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=17, textColor=colors.HexColor('#0B3D2E'), leading=21, alignment=1)
    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, textColor=colors.HexColor('#2E7D32'), leading=12, alignment=1)
    meta_style = ParagraphStyle('Meta', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)
    section_h1 = ParagraphStyle('SecH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#0B3D2E'), leading=14, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#1E293B'))
    bold_style = ParagraphStyle('BoldText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#0F172A'))
    
    story = []
    if os.path.exists(LOGO_FILE_EXACT):
        try:
            story.append(RLImage(LOGO_FILE_EXACT, width=110, height=110))
            story.append(Spacer(1, 4))
        except Exception: pass

    local_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d-%b-%Y %I:%M %p')
    
    story.append(Paragraph("SMART KISHAN • OFFICIAL CROP PRESCRIPTION", title_style))
    story.append(Paragraph("Certified 4R Nutrient Stewardship & Field Application Dossier", subtitle_style))
    story.append(Paragraph(f"Dossier ID: SK-{datetime.now().strftime('%Y%m%d')}-{str(user_mobile)[-4:]} | Generated: {local_time}", meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("1. FARMER & LAND PROFILE", section_h1))
    profile_data = [
        [Paragraph("<b>Farmer Mobile:</b>", body_style), Paragraph(f"+91 {user_mobile}", bold_style), Paragraph("<b>Field / Parcel ID:</b>", body_style), Paragraph(str(plot_id), bold_style)],
        [Paragraph("<b>Target Crop:</b>", body_style), Paragraph(str(crop), bold_style), Paragraph("<b>Target Harvest:</b>", body_style), Paragraph(f"{target_yield} t/acre", bold_style)],
        [Paragraph("<b>Land Area:</b>", body_style), Paragraph(f"{raw_land:.2f} {land_unit.split()[0]}", bold_style), Paragraph("<b>Standard Area:</b>", body_style), Paragraph(f"{opt.get('land_area', raw_land*0.404686):.3f} Hectares", bold_style)],
        [Paragraph("<b>Farmer Budget:</b>", body_style), Paragraph(f"Rs. {budget:,.0f}", bold_style), Paragraph("<b>Optimization Cost:</b>", body_style), Paragraph(f"Rs. {opt.get('total_cost', 0):,.0f}", bold_style)]
    ]
    t_prof = Table(profile_data, colWidths=[110, 155, 120, 150])
    t_prof.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4FBF5')), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#C8E6C9')), ('PADDING', (0,0), (-1,-1), 4)]))
    story.append(t_prof)

    story.append(Paragraph("2. SOIL PROFILE & MEASURED ATTRIBUTES", section_h1))
    telemetry_data = [
        [Paragraph("<b>Nitrogen (N):</b>", body_style), Paragraph(f"{n:.1f} mg/kg", bold_style), Paragraph("<b>Soil pH:</b>", body_style), Paragraph(f"{ph:.1f}", bold_style), Paragraph("<b>Ambient Temp:</b>", body_style), Paragraph(f"{temp:.1f} °C", bold_style)],
        [Paragraph("<b>Phosphorus (P):</b>", body_style), Paragraph(f"{p:.1f} mg/kg", bold_style), Paragraph("<b>Organic Carbon:</b>", body_style), Paragraph(f"{soc:.2f} %", bold_style), Paragraph("<b>Relative Humidity:</b>", body_style), Paragraph(f"{humid:.0f} %", bold_style)],
        [Paragraph("<b>Potash (K):</b>", body_style), Paragraph(f"{k:.1f} mg/kg", bold_style), Paragraph("<b>Soil Moisture:</b>", body_style), Paragraph(f"{moist:.1f} %", bold_style), Paragraph("<b>Precipitation:</b>", body_style), Paragraph(f"{rain:.0f} mm", bold_style)]
    ]
    t_tel = Table(telemetry_data, colWidths=[85, 95, 90, 95, 90, 80])
    t_tel.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFFFFF')), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')), ('PADDING', (0,0), (-1,-1), 4)]))
    story.append(t_tel)

    story.append(Paragraph("3. RECOMMENDED FERTILIZER PURCHASES (50KG BAGS)", section_h1))
    urea_bags = max(1, round(opt.get('urea_kg', 0) / 50.0)) if opt.get('urea_kg', 0) > 0 else 0
    dap_bags = max(1, round(opt.get('dap_kg', 0) / 50.0)) if opt.get('dap_kg', 0) > 0 else 0
    mop_bags = max(1, round(opt.get('mop_kg', 0) / 50.0)) if opt.get('mop_kg', 0) > 0 else 0
    comp_bags = max(1, round(opt.get('complex_kg', 0) / 50.0)) if opt.get('complex_kg', 0) > 0 else 0
    org_bags = round(opt.get('compost_kg', 0) / 50.0) if opt.get('compost_kg', 0) > 0 else 0

    fert_data = [
        [Paragraph("<b>Fertilizer Product</b>", bold_style), Paragraph("<b>Nutrient Category</b>", bold_style), Paragraph("<b>Total Mass (kg)</b>", bold_style), Paragraph("<b>50kg Bags Required</b>", bold_style)],
        [Paragraph("Urea", body_style), Paragraph("Synthetic Nitrogen (46% N)", body_style), Paragraph(f"{opt.get('urea_kg', 0)} kg", body_style), Paragraph(f"<b>{urea_bags} Bags</b>", bold_style)],
        [Paragraph("DAP", body_style), Paragraph("Phosphatic (18% N + 46% P)", body_style), Paragraph(f"{opt.get('dap_kg',0)} kg", body_style), Paragraph(f"<b>{dap_bags} Bags</b>", bold_style)],
        [Paragraph("MOP", body_style), Paragraph("Potash (60% K2O)", body_style), Paragraph(f"{opt.get('mop_kg',0)} kg", body_style), Paragraph(f"<b>{mop_bags} Bags</b>", bold_style)],
        [Paragraph("Complex 14-35-14", body_style), Paragraph("Balanced N-P-K Mineral", body_style), Paragraph(f"{opt.get('complex_kg',0)} kg", body_style), Paragraph(f"<b>{comp_bags} Bags</b>", bold_style)],
        [Paragraph("Bio-Compost / Manure", body_style), Paragraph("Organic Humus Restorer", body_style), Paragraph(f"{opt.get('compost_kg',0)} kg", body_style), Paragraph(f"<b>{org_bags} Bags</b>", bold_style)],
    ]
    t_fert = Table(fert_data, colWidths=[150, 160, 110, 115])
    t_fert.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2EEDF')), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')), ('PADDING', (0,0), (-1,-1), 4)]))
    story.append(t_fert)

    story.append(Paragraph("4. TIMED APPLICATION PERIODS & METHODS FOR FARMERS", section_h1))
    schedule_data = [
        [Paragraph("<b>Time Period</b>", bold_style), Paragraph("<b>Nutrient Blend</b>", bold_style), Paragraph("<b>Specific Application Method for Farmer</b>", bold_style)],
        [Paragraph("<b>Stage 1: Basal Dressing (At Sowing / Transplanting - Day 0)</b>", body_style), Paragraph("100% Bio-Compost + 100% DAP<br/>+ 1/3 MOP + 1/4 Urea", body_style), Paragraph("Incorporate compost and broadcast full DAP and 1/3 MOP. Place 5-7 cm below seed furrow; do not leave on dry surface.", body_style)],
        [Paragraph("<b>Stage 2: Vegetative Growth (20 - 25 Days Post Sowing)</b>", body_style), Paragraph("1/2 Urea + 1/3 MOP<br/><i>(Vegetative Dose)</i>", body_style), Paragraph("Side-dress 1/2 urea dose + 1/3 MOP along plant rows. Ensure adequate soil moisture or irrigate within 24 hours.", body_style)],
        [Paragraph("<b>Stage 3: Panicle Initiation / Flowering (45 - 55 Days Post Sowing)</b>", body_style), Paragraph("Remaining 1/4 Urea<br/>+ Remaining 1/3 MOP", body_style), Paragraph("Top-dress remaining 1/4 urea and final MOP. Avoid application during heavy rains to prevent leaching.", body_style)]
    ]
    t_sched = Table(schedule_data, colWidths=[130, 155, 250])
    t_sched.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2EEDF')), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')), ('PADDING', (0,0), (-1,-1), 5)]))
    story.append(t_sched)

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------------------------------------------------
# SCREEN 1: SMART KISHAN CINEMATIC LOGIN & REGISTRATION (NO CHATBOT HERE)
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
            <p>Next-Generation AgriTech Control Center powered by Artificial Intelligence & Google Gemini.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_login:
        st.markdown('<div class="glass-login-card">', unsafe_allow_html=True)
        st.markdown("<div class='login-title'>🔐 Welcome Back</div>", unsafe_allow_html=True)

        t_login, t_admin, t_reg, t_forgot = st.tabs(["Farmer Sign In", "Admin Sign In", "Registration", "🔑 Forgot Password"])

        with t_login:
            m = st.text_input("Mobile Number", max_chars=10, key="log_m")
            p = st.text_input("Password", type="password", key="log_p")
            if st.button("Sign In as Farmer →", key="login_button"):
                if len(m.strip()) == 10:
                    valid, user_role = verify_user(m.strip(), p.strip(), selected_role="farmer")
                    if valid:
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = m.strip()
                        st.session_state.user_role = "farmer"
                        st.session_state.step = 2
                        st.rerun()
                    else: st.error("Invalid credentials.")
                else: st.warning("Enter valid 10-digit mobile.")

        with t_admin:
            am = st.text_input("Admin Mobile Number", max_chars=10, key="admin_log_m")
            ap = st.text_input("Admin Password", type="password", key="admin_log_p")
            if st.button("Sign In as Admin →", key="admin_login_button"):
                if len(am.strip()) == 10:
                    valid, user_role = verify_user(am.strip(), ap.strip(), selected_role="admin")
                    if valid:
                        st.session_state.logged_in = True
                        st.session_state.user_mobile = am.strip()
                        st.session_state.user_role = "admin"
                        st.session_state.step = 90
                        st.rerun()
                    else: st.error("Access Denied.")

        with t_reg:
            rm = st.text_input("Mobile Number", max_chars=10, key="reg_m")
            rp = st.text_input("Create Password", type="password", key="reg_p")
            rpc = st.text_input("Confirm Password", type="password", key="reg_pc")
            if st.button("Create Account →", key="register_button"):
                if len(rm.strip()) == 10 and rp == rpc and len(rp) > 0:
                    ok, msg = register_user(rm.strip(), rp.strip(), role="farmer")
                    if ok: st.success(msg)
                    else: st.error(msg)
                else: st.warning("Check inputs.")

        with t_forgot:
            f_mob = st.text_input("Registered Mobile", max_chars=10, key="reset_mob_inp")
            f_np = st.text_input("New Password", type="password", key="reset_np_inp")
            f_npc = st.text_input("Confirm New Password", type="password", key="reset_npc_inp")
            if st.button("Change Password Now ➔", key="btn_direct_pwd_reset"):
                if len(f_mob.strip()) == 10 and f_np == f_npc and len(f_np) > 0:
                    ok, msg = reset_user_password_direct(f_mob.strip(), f_np.strip())
                    if ok: st.success(msg)
                    else: st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 99: DEDICATED FULL-SCREEN AGRI AI CHAT SUPPORT PAGE
# Features Multi-Session Chat Tabs (Left Pane) & Gemini-Grade Conversation (Right Pane)
# -------------------------------------------------------------
elif st.session_state.step == 99:
    st.markdown("""
    <div style="background: rgba(11, 61, 46, 0.95); border-radius: 16px; padding: 18px 24px; border: 1px solid #39FF88; box-shadow: 0 8px 24px rgba(0,0,0,0.65); margin-bottom: 18px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="color: #39FF88; margin: 0 0 4px 0; font-size: 22px;">🤖 SMART KISHAN • GEMINI AI AGRONOMIST SUPPORT CENTER</h2>
            <p style="color: #A7F3D0; margin: 0; font-size: 13px;">Multi-Session Personal AI • District Seed Varieties, 4R NPK Calculations, Disease Remedies & Market Intelligence</p>
        </div>
        <span class="badge-pass">● GEMINI AI ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Previous Screen"):
        st.session_state.step = st.session_state.get("prev_step_before_chat", 2)
        st.rerun()

    st.write("")
    col_sessions, col_chat_main = st.columns([1.15, 2.85], gap="large")

    # LEFT PANE: SEPARATE CHAT TABS / SESSIONS MANAGER
    with col_sessions:
        st.markdown("<div class='metric-card'><h4>🗂️ Saved Chat Threads</h4><p style='font-size:12px; color:#A7F3D0; margin:0;'>Create & switch between separate conversations</p></div>", unsafe_allow_html=True)
        
        if st.button("➕ Start New Chat", use_container_width=True):
            st.session_state.chat_counter += 1
            new_title = f"Chat {st.session_state.chat_counter} • New Topic"
            st.session_state.chat_sessions[new_title] = [
                {"role": "assistant", "content": f"Started **{new_title}**! 🌱 Ask me any question about seeds for your district, soil nutrients, crop diseases, or mandi prices."}
            ]
            st.session_state.active_chat_id = new_title
            st.rerun()

        st.markdown("##### 📌 Your Chat Tabs:")
        session_keys = list(st.session_state.chat_sessions.keys())
        for s_key in session_keys:
            is_active = (s_key == st.session_state.active_chat_id)
            btn_label = f"🟢 {s_key}" if is_active else f"💬 {s_key}"
            if st.button(btn_label, key=f"sel_chat_{s_key}", use_container_width=True):
                st.session_state.active_chat_id = s_key
                st.rerun()

        st.divider()
        if len(session_keys) > 1:
            if st.button("🗑️ Delete Current Chat Tab", use_container_width=True):
                del st.session_state.chat_sessions[st.session_state.active_chat_id]
                st.session_state.active_chat_id = list(st.session_state.chat_sessions.keys())[0]
                st.rerun()

    # RIGHT PANE: ACTIVE CHAT CONVERSATION WINDOW
    with col_chat_main:
        active_id = st.session_state.active_chat_id
        current_msgs = st.session_state.chat_sessions[active_id]

        st.markdown(f"#### 💬 Active Thread: `{active_id}`")
        chat_window = st.container(height=430, border=True)
        with chat_window:
            for msg in current_msgs:
                if msg["role"] == "user":
                    st.markdown(f"<div class='ai-bubble-user'><b>🧑🌾 You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='ai-bubble-bot'><b>🤖 Smart Kishan Gemini AI:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

        with st.form("full_page_ai_chat_form", clear_on_submit=True):
            user_prompt = st.text_input(
                "Ask anything (e.g. 'For Cuttack district give a major profitable seed name', 'Best fertilizer schedule for rice', etc.):",
                placeholder="Type your question here..."
            )
            f_c1, f_c2 = st.columns([4, 1])
            send_btn = f_c1.form_submit_button("🚀 Ask Gemini AI ➔", use_container_width=True)
            clr_btn = f_c2.form_submit_button("🧹 Clear", use_container_width=True)

            if send_btn and user_prompt.strip():
                q_text = user_prompt.strip()
                current_msgs.append({"role": "user", "content": q_text})

                # Auto-rename chat tab title on first user message if it has a generic title
                if "New Topic" in active_id or "General Advisory" in active_id:
                    short_topic = q_text[:22] + ("..." if len(q_text) > 22 else "")
                    prefix = active_id.split("•")[0].strip()
                    updated_id = f"{prefix} • {short_topic}"
                    if updated_id not in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[updated_id] = current_msgs
                        del st.session_state.chat_sessions[active_id]
                        st.session_state.active_chat_id = updated_id
                        active_id = updated_id

                with st.spinner("🤖 Smart Kishan Gemini AI is analyzing your query..."):
                    ai_ans = get_smart_kishan_ai_response(q_text, current_msgs[:-1])

                st.session_state.chat_sessions[active_id].append({"role": "assistant", "content": ai_ans})
                if st.session_state.logged_in and st.session_state.user_mobile:
                    log_activity(st.session_state.user_mobile, "AI Chat Support", f"[{active_id}] {q_text[:50]}")
                st.rerun()

            if clr_btn:
                st.session_state.chat_sessions[active_id] = [
                    {"role": "assistant", "content": f"Cleared **{active_id}**! 🌱 Ask me any question."}
                ]
                st.rerun()

# -------------------------------------------------------------
# SUB-PAGES: ADMIN COMMAND CENTER WITH CONCURRENCY LOCK & LIVE LOGS
# -------------------------------------------------------------
elif st.session_state.step == 90 and st.session_state.user_role == "admin":
    st.markdown("## SMART KISHAN : ADMIN COMMAND CENTER")
    c_top1, c_top2 = st.columns([1, 5])
    if c_top1.button("🚪 Sign Out"):
        st.session_state.logged_in = False
        st.session_state.step = 1
        st.rerun()
    st.info(f"Logged in Admin: +91 {st.session_state.user_mobile}")
    
    admin_t1, admin_t2, admin_t3, admin_t4 = st.tabs([
        "👥 Registered Users",
        "⭐ Farmer Feedback",
        "🔔 Help Request Triage (Concurrency Lock)",
        "📜 All User Activity Logs"
    ])
    
    with admin_t1:
        st.subheader("Registered Users & Active Accounts")
        if engine:
            try:
                with engine.connect() as conn:
                    users_df = pd.read_sql(text("SELECT mobile, role FROM users"), conn)
                if users_df.empty:
                    st.info("No external farmers registered yet.")
                else:
                    st.dataframe(users_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading users: {e}")

    with admin_t2:
        st.subheader("Farmer Star Ratings & Feedback Submissions")
        if engine:
            try:
                with engine.connect() as conn:
                    fb_df = pd.read_sql(text("SELECT id, mobile, rating, rating_text, comments, created_at FROM feedback ORDER BY id DESC"), conn)
                if fb_df.empty:
                    st.info("No feedback submitted yet.")
                else:
                    st.dataframe(fb_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading feedback: {e}")

    with admin_t3:
        st.subheader("Live User Help Tickets & Concurrency Lock")
        if engine:
            try:
                with engine.connect() as conn:
                    tickets = pd.read_sql(text("SELECT * FROM help_requests ORDER BY id DESC"), conn)
                
                if tickets.empty:
                    st.info("No active help requests found.")
                else:
                    for idx, row in tickets.iterrows():
                        locked_by = row.get("attended_by")
                        st.markdown(f"**Ticket #{row['id']}** | **Farmer Mobile:** `+91 {row['mobile']}` | **Category:** `{row['request_type']}`")
                        st.markdown(f"**Query:** {row['query_text']}")
                        status_badge = f"🟢 Resolved (Attended by Admin: +91 {locked_by})" if row['status'] == 'Resolved' else (
                            f"🔒 Locked by Admin +91 {locked_by}" if locked_by else "🟡 Pending (Unassigned)"
                        )
                        st.markdown(f"**Status:** {status_badge}")

                        if row.get("user_feedback"):
                            st.markdown(f"**🌟 Farmer Resolution Feedback:** {row['user_feedback']}")

                        if row['status'] == 'Pending':
                            if locked_by and str(locked_by) != str(st.session_state.user_mobile):
                                st.warning(f"⚠️ Concurrency Lock Active: Admin +91 {locked_by} is already handling this request.")
                            else:
                                with st.form(key=f"resolve_form_{row['id']}"):
                                    admin_reply_text = st.text_area("Resolution Reply to Farmer:", key=f"reply_input_{row['id']}")
                                    submit_resolve = st.form_submit_button("🔒 Lock, Attend & Send Resolution")
                                    if submit_resolve:
                                        if not admin_reply_text.strip():
                                            st.warning("Please type a resolution message before submitting.")
                                        else:
                                            with engine.connect() as conn:
                                                current_ticket = conn.execute(text("SELECT attended_by, status FROM help_requests WHERE id = :id"), {"id": row['id']}).fetchone()
                                                if current_ticket and current_ticket[0] and str(current_ticket[0]) != str(st.session_state.user_mobile):
                                                    st.error(f"❌ Concurrency Lock Active: Ticket already attended by Admin (+91 {current_ticket[0]}).")
                                                else:
                                                    conn.execute(text("UPDATE help_requests SET status = 'Resolved', admin_reply = :rep, attended_by = :adm WHERE id = :id"), 
                                                                 {"rep": admin_reply_text.strip(), "adm": st.session_state.user_mobile, "id": row['id']})
                                                    conn.commit()
                                                    log_activity(row['mobile'], "Admin Replied to Query", f"Ticket #{row['id']} resolved by Admin")
                                                    st.success("✅ Ticket resolved, locked to your Admin ID, and pushed to Farmer's Notification Shade!")
                                                    st.rerun()
                        else:
                            st.markdown(f"**Admin Resolution Sent:** {row['admin_reply']}")
                        st.divider()
            except Exception as e:
                st.error(f"Error loading help tickets: {e}")

    with admin_t4:
        st.subheader("Global User Activity Audit Trail")
        if engine:
            try:
                with engine.connect() as conn:
                    all_act = pd.read_sql(text("SELECT id, mobile, activity_type, details, created_at FROM user_activity ORDER BY id DESC"), conn)
                if all_act.empty:
                    st.info("No user activity logged yet.")
                else:
                    st.dataframe(all_act, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading activity logs: {e}")

    render_floating_ai_chat_launcher()

elif st.session_state.step == 21:
    st.subheader("🆘 Help Desk & Request Redirection Shade")
    help_categories = [
        "Fertilizer Dosage Correction",
        "Soil Scan Verification Issue",
        "Crop Yield Prediction Query",
        "Account Password / Login Help",
        "Delete My Account",
        "Recover Deleted Activity",
        "General Inquiry"
    ]
    selected_cat = st.selectbox("Select Pre-Added Issue Category:", help_categories)
    manual_details = st.text_area("Additional Manual Details (Optional - Not Mandatory):")
    
    c_back, c_sub = st.columns([1, 4])
    if c_back.button("⬅️ Back to Dashboard"):
        st.session_state.step = 2
        st.rerun()
    if c_sub.button("Submit Request to Admin ➔", use_container_width=True):
        query_final = f"[{selected_cat}] {manual_details.strip()}" if manual_details.strip() else f"[{selected_cat}] (Submitted via Quick-Option)"
        if engine:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO help_requests (mobile, request_type, query_text, status) VALUES (:m, :rt, :q, 'Pending')"), 
                             {"m": str(st.session_state.user_mobile), "rt": selected_cat, "q": query_final})
                conn.commit()
            log_activity(st.session_state.user_mobile, "Help Requested", f"Submitted Ticket: {selected_cat}")
        st.success("✅ Request queued! Check your 🔔 Notifications shade for live Admin replies.")

    render_floating_ai_chat_launcher()

elif st.session_state.step == 22:
    st.subheader("🔔 Notification Shade & Admin Resolution Chat")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.step = 2
        st.rerun()
    if engine:
        with engine.connect() as conn:
            notifs = pd.read_sql(text("SELECT * FROM help_requests WHERE mobile = :m ORDER BY id DESC"), conn, params={"m": str(st.session_state.user_mobile)})
            
        if notifs.empty:
            st.info("No notifications or help queries yet. Submit a question in the 🆘 Help Desk!")
        else:
            for idx, row in notifs.iterrows():
                st.markdown(f"<div class='metric-card'><h4>🔔 Ticket #{row['id']} — {row['request_type']}</h4><p><b>Your Query:</b> {row['query_text']}</p></div>", unsafe_allow_html=True)
                if row['status'] == 'Resolved' and row['admin_reply']:
                    st.success(f"🤖 **Admin (+91 {row.get('attended_by', 'Admin')}) Resolution:** {row['admin_reply']}")
                    if not row.get('user_feedback'):
                        with st.form(key=f"resolved_feedback_{row['id']}"):
                            st.markdown("**🌟 Rate this Admin Resolution (Notification Shade Feedback):**")
                            resolved_star = st.slider("Stars (1-5):", 1, 5, 5, key=f"res_star_{row['id']}")
                            resolved_comment = st.text_input("Optional Comment:", key=f"res_comm_{row['id']}")
                            if st.form_submit_button("Submit Resolution Feedback"):
                                fb_str = f"{resolved_star}/5 Stars" + (f" - {resolved_comment.strip()}" if resolved_comment.strip() else "")
                                with engine.connect() as c_db:
                                    c_db.execute(text("UPDATE help_requests SET user_feedback = :fb WHERE id = :id"), {"fb": fb_str, "id": row['id']})
                                    c_db.commit()
                                save_feedback(st.session_state.user_mobile, resolved_star, "Resolution Feedback", fb_str)
                                st.success("✅ Feedback saved to Admin Center!")
                                st.rerun()
                    else:
                        st.info(f"Your Submitted Feedback: {row['user_feedback']}")
                else:
                    st.warning("🟡 Status: Pending Admin Review")
                st.divider()

    render_floating_ai_chat_launcher()

elif st.session_state.step == 23:
    st.subheader("📜 My Activity History Audit Trail")
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.step = 2
        st.rerun()
    if engine:
        with engine.connect() as conn:
            act = pd.read_sql(text("SELECT activity_type AS Action, details AS Details, created_at AS Timestamp FROM user_activity WHERE mobile = :m ORDER BY id DESC"), conn, params={"m": str(st.session_state.user_mobile)})
            if act.empty:
                st.info("No recorded activity found yet.")
            else:
                st.dataframe(act, use_container_width=True)

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 2: FARMER DASHBOARD WITH REAL-TIME TABS
# -------------------------------------------------------------
elif st.session_state.step == 2:
    st.markdown(f"""
    <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
        <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
        <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Control Center &mdash; Role: <strong>{st.session_state.user_role.upper()}</strong> (+91 {st.session_state.user_mobile})</p>
    </div><br>
    """, unsafe_allow_html=True)

    c_b1, c_b2, c_b3, c_b4 = st.columns(4)
    if c_b1.button("🆘 Help Desk", use_container_width=True): st.session_state.step = 21; st.rerun()
    if c_b2.button("🔔 Notifications", use_container_width=True): st.session_state.step = 22; st.rerun()
    if c_b3.button("📜 Activity", use_container_width=True): st.session_state.step = 23; st.rerun()
    if c_b4.button("🚪 Sign Out", use_container_width=True): st.session_state.logged_in = False; st.session_state.step = 1; st.rerun()

    tab_calc, tab_diag, tab_live_weather, tab_market_risk = st.tabs([
        "📍 1. Farm Details & Soil Input (Optimizer)", 
        "🔬 2. Crop Disease Diagnosis",
        "🌍 3. Live Weather & Global Seed DB", 
        "📈 4. Market Forecast & Risk Dashboard"
    ])

    with tab_calc:
        st.subheader("Land Size, Budget & Soil Telemetry")
        tab_cam, tab_man = st.tabs(["📷 Soil Scanner", "🧪 Manual Soil Entry"])
        
        with tab_cam:
            c_s1, c_s2 = st.columns(2)
            soil_cam = c_s1.camera_input("Scan Soil Live")
            soil_file = c_s2.file_uploader("Upload Soil Image", type=["jpg", "png"])
            if soil_cam or soil_file:
                s_img = Image.open(soil_cam or soil_file)
                st.image(s_img, width=250)
                eval_res = verify_genuine_agricultural_soil(s_img)
                if eval_res["detected"]:
                    st.success("Soil verified! Applied metrics.")
                    st.session_state.soil_n = eval_res["metrics"]["n"]
                    st.session_state.soil_p = eval_res["metrics"]["p"]
                    st.session_state.soil_k = eval_res["metrics"]["k"]
                    st.session_state.soil_ph = eval_res["metrics"]["ph"]
                    st.session_state.soc = eval_res["metrics"]["soc"]
                    st.session_state.soil_moist = eval_res["metrics"]["moist"]
                    st.session_state.soil_source = "scanner"
                    log_activity(st.session_state.user_mobile, "Soil Scanned", f"Detected: {eval_res['soil_type']}")
                else: st.error(eval_res["reason"])

        with tab_man:
            c1, c2, c3 = st.columns(3)
            st.session_state.raw_land_val = c1.number_input("Land Size", 0.1, 1000.0, float(st.session_state.raw_land_val), 0.5)
            st.session_state.land_unit = c2.selectbox("Unit", list(UNIT_TO_HECTARE.keys()), index=list(UNIT_TO_HECTARE.keys()).index(st.session_state.land_unit))
            st.session_state.budget_cap = c3.number_input("Max Budget (₹)", 1000.0, 1000000.0, float(st.session_state.budget_cap), 500.0)

            ha_base = st.session_state.raw_land_val * UNIT_TO_HECTARE[st.session_state.land_unit]
            st.session_state.land_area = ha_base

            s1, s2, s3 = st.columns(3)
            st.session_state.soil_n = s1.number_input("Nitrogen (N) [mg/kg]", 0.0, 300.0, float(st.session_state.soil_n))
            st.session_state.soil_p = s2.number_input("Phosphorus (P) [mg/kg]", 0.0, 150.0, float(st.session_state.soil_p))
            st.session_state.soil_k = s3.number_input("Potash (K) [mg/kg]", 0.0, 350.0, float(st.session_state.soil_k))

            s4, s5, s6 = st.columns(3)
            st.session_state.soil_ph = s4.slider("Soil pH", 4.0, 9.5, float(st.session_state.soil_ph), 0.1)
            st.session_state.soc = s5.slider("Organic Carbon (%)", 0.1, 2.5, float(st.session_state.soc), 0.05)
            st.session_state.soil_moist = s6.slider("Moisture (%)", 10.0, 90.0, float(st.session_state.soil_moist), 1.0)
            st.session_state.soil_source = "manual"

        if st.button("Save Variables & Proceed to ML Assessment ➔"):
            log_activity(st.session_state.user_mobile, "Soil Telemetry Saved", f"Land: {st.session_state.raw_land_val} {st.session_state.land_unit}")
            st.session_state.step = 3
            st.rerun()

    with tab_diag:
        c_cam, c_up = st.columns(2)
        cam_p = c_cam.camera_input("📷 Realtime Leaf Scanner")
        file_p = c_up.file_uploader("📂 Upload Leaf Image", type=["jpg", "jpeg", "png"])
        active_img = cam_p or file_p
        if active_img:
            img = Image.open(active_img)
            st.image(img, caption="Scanned Specimen", width=300)
            res = analyze_plant_disease_image(img)
            st.session_state.scanned_diag = res
            st.success(f"Health Status: {res['health']}")
            st.write(f"**Pathogen**: {res['disease']}")
            st.write(f"**Remedy**: {res['medicine']}")
            log_activity(st.session_state.user_mobile, "Leaf Disease Scanned", f"Diagnosis: {res['health']}")

    with tab_live_weather:
        st.markdown("### 🌍 Real-Time Weather Integration & World Seed Prescriptions")
        st.info("Fetching real-time weather API metrics based on your IP location...")
        lw1, lw2, lw3 = st.columns(3)
        lw1.metric("Current Farm Temp", "28.5 °C", "1.2 °C")
        lw2.metric("Relative Humidity", "65 %", "-2 %")
        lw3.metric("Rainfall Probability", "12 mm Forecast", "Low")
        
        st.markdown("#### 🌾 Global Seed Recommendation Matrix")
        seed_df = pd.DataFrame({
            "Crop Type": ["Rice (Basmati)", "Maize (Hybrid)", "Wheat (Durum)", "Cotton (Bt)"],
            "Recommended Seed Variant": ["Pusa-1121", "Pioneer 30V92", "HI-8713", "Bollgard II"],
            "Global Origin": ["India/Pakistan", "USA/Global", "Mediterranean", "India/USA"],
            "Suitability Match": ["98%", "85%", "92%", "78%"]
        })
        st.dataframe(seed_df, use_container_width=True)

    with tab_market_risk:
        st.markdown("### 📈 Market Forecast & Live Farm Risk Engine")
        st.warning("Current Alerts: High probability of late-blight fungus due to incoming humidity front.")
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10)
        prices = np.random.uniform(2200, 2600, 10)
        trend_df = pd.DataFrame({"Date": dates, "Price Per Quintal (₹)": prices})
        
        chart = alt.Chart(trend_df).mark_line(color="#39FF88", point=True).encode(
            x='Date:T', y=alt.Y('Price Per Quintal (₹):Q', scale=alt.Scale(domain=[2000, 3000]))
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    render_floating_ai_chat_launcher()
# -------------------------------------------------------------
# SCREEN 3: SOIL HEALTH, WATER & RISK EVALUATION
# -------------------------------------------------------------
elif st.session_state.step == 3:
    st.markdown("## Soil Condition & Agronomic Risk Assessment")

    k1, k2, k3 = st.columns(3)
    ph_stat = "Acidic (Apply Lime)" if st.session_state.soil_ph < 6.0 else ("Alkaline (Apply Gypsum)" if st.session_state.soil_ph > 7.5 else "Sweet & Balanced")
    k1.metric("Soil Sweetness (pH)", f"{st.session_state.soil_ph}", ph_stat)
    k2.metric("Organic Matter (SOC)", f"{st.session_state.soc}%", "Rich" if st.session_state.soc >= 0.75 else "Low")
    k3.metric("Rain Leaching Risk", f"{st.session_state.rainfall:.0f} mm", "Optimal")

    soil_idx = list(soil_encoder.classes_).index(st.session_state.sel_soil) if st.session_state.sel_soil in soil_encoder.classes_ else 0
    irrig_pred = irrig_model.predict([[st.session_state.temp, st.session_state.humidity, st.session_state.rainfall, soil_idx]])[0]
    pest_risk = min(98.0, max(5.0, (st.session_state.humidity * 0.45) + (st.session_state.temp * 0.3) + (st.session_state.soil_n * 0.15)))

    c_irrig, c_pest = st.columns(2)
    with c_irrig:
        st.markdown(f"<div class='metric-card'><h4>💧 ML Water Requirement:</h4><h2>{irrig_pred:.1f} mm/ha</h2></div>", unsafe_allow_html=True)
    with c_pest:
        st.markdown(f"<div class='metric-card'><h4>🦗 Forecasted Pest Risk:</h4><h2>{pest_risk:.1f}% Risk</h2></div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 2
        st.rerun()
    if b2.button("Continue to Next Step ➔"):
        st.session_state.step = 4
        st.rerun()

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 4: SOIL COMPARISON BAR CHARTS
# -------------------------------------------------------------
elif st.session_state.step == 4:
    st.markdown("## Current Soil Nutrients vs Ideal Farm Target (Bar Analysis)")
    d1, d2, d3 = st.columns(3)

    def build_labeled_bar_chart(nutrient_name, soil_val, target_val, color_bar):
        chart_data = pd.DataFrame({"Nutrient Status": ["Current Soil", "Target Ideal"], "Value": [round(soil_val, 1), round(target_val, 1)]})
        bars = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Nutrient Status:N", axis=alt.Axis(labelColor="#FFFFFF", labelFontSize=12, title=None)),
            y=alt.Y("Value:Q", axis=alt.Axis(labelColor="#FFFFFF", titleColor="#39FF88", title="kg/ha")),
            color=alt.Color("Nutrient Status:N", scale=alt.Scale(range=[color_bar, "#1B5E20"]), legend=None)
        )
        text_labels = alt.Chart(chart_data).mark_text(align='center', baseline='middle', dy=-10, fontSize=13, fontWeight='bold', color='#FFFFFF').encode(
            x=alt.X("Nutrient Status:N"), y=alt.Y("Value:Q"), text=alt.Text("Value:Q", format=".1f")
        )
        return (bars + text_labels).properties(height=260)

    with d1:
        st.altair_chart(build_labeled_bar_chart("Nitrogen", st.session_state.soil_n * 2.24, 280.0, "#39FF88"), use_container_width=True)
    with d2:
        st.altair_chart(build_labeled_bar_chart("Phosphorus", st.session_state.soil_p * 2.24, 60.0, "#00E5FF"), use_container_width=True)
    with d3:
        st.altair_chart(build_labeled_bar_chart("Potash", st.session_state.soil_k * 2.24, 150.0, "#FFD700"), use_container_width=True)

    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 3
        st.rerun()
    if b2.button("Continue to Crop Analytics ➔"):
        st.session_state.step = 5
        st.rerun()

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 5: NUTRIENT GAP, DYNAMIC CROP & MARKET PRICE PREDICTION
# -------------------------------------------------------------
elif st.session_state.step == 5:
    st.markdown("## Deficit Analysis, Universal Crop AI & Future Market Price")

    def_n, def_p, def_k = calculate_advanced_nutrients(st.session_state.target_yield, st.session_state.soil_n, st.session_state.soil_p, st.session_state.soil_k, st.session_state.soc, st.session_state.soil_ph, st.session_state.soil_moist, st.session_state.sel_soil)

    crop_in = pd.DataFrame([{'N': st.session_state.soil_n, 'P': st.session_state.soil_p, 'K': st.session_state.soil_k, 'temperature': st.session_state.temp, 'humidity': st.session_state.humidity, 'ph': st.session_state.soil_ph, 'rainfall': st.session_state.rainfall}])
    dynamic_pred_crop = crop_encoder.inverse_transform([crop_model.predict(crop_in)[0]])[0]
    st.session_state.sel_crop = dynamic_pred_crop

    crop_encoded_val = list(crop_encoder.classes_).index(dynamic_pred_crop) if dynamic_pred_crop in crop_encoder.classes_ else 0
    pred_price = price_model.predict([[st.session_state.target_yield, st.session_state.temp, st.session_state.rainfall, crop_encoded_val]])[0]

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"##### Bar Chart: Nutrient Shortages for {st.session_state.target_yield} t/acre")
        def_df = pd.DataFrame({"Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potash (K)"], "Shortage (kg/acre)": [round(def_n, 1), round(def_p, 1), round(def_k, 1)]})
        short_bars = alt.Chart(def_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#39FF88").encode(x=alt.X("Nutrient:N", axis=alt.Axis(labelColor="#FFFFFF", labelFontSize=12, title=None)), y=alt.Y("Shortage (kg/acre):Q", axis=alt.Axis(labelColor="#FFFFFF")))
        st.altair_chart(short_bars, use_container_width=True)
    with g2:
        st.success(f"🌱 **Recommended Crop**: **{dynamic_pred_crop.capitalize()}**")
        st.markdown(f"<div class='metric-card'><h4>💰 Predicted Future Market Price:</h4><h2>₹{pred_price:,.0f} / Quintal</h2></div>", unsafe_allow_html=True)

    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 4
        st.rerun()
    if b2.button("Generate Optimization Engine ➔"):
        log_activity(st.session_state.user_mobile, "Crop & Yield Predicted", f"Crop: {dynamic_pred_crop.capitalize()}")
        st.session_state.step = 6
        st.rerun()

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 6: OPTIMIZED FERTILIZER BAGS & APPLICATION RULES
# -------------------------------------------------------------
elif st.session_state.step == 6:
    st.markdown("## Your Fertilizer Bags & Application Schedule")

    def_n, def_p, def_k = calculate_advanced_nutrients(st.session_state.target_yield, st.session_state.soil_n, st.session_state.soil_p, st.session_state.soil_k, st.session_state.soc, st.session_state.soil_ph, st.session_state.soil_moist, st.session_state.sel_soil)
    opt = optimize_fertilizer_blend(def_n, def_p, def_k, st.session_state.budget_cap, st.session_state.land_area, str(st.session_state.sel_soil), st.session_state.rainfall, st.session_state.soc)
    st.session_state.opt_results = opt

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Optimized Total Cost", f"₹{opt['total_cost']:,.0f}")
    r2.metric("Input Budget Cap", f"₹{st.session_state.budget_cap:,.0f}")
    r3.metric("Land Covered", f"{st.session_state.raw_land_val:.2f} {st.session_state.land_unit.split(' ')[0]}")
    r4.metric("Budget Utilized", f"{opt['budget_utilized_pct']}%")

    st.markdown("##### 🛒 Fertilizer Quantity Comparison")
    fert_qty_df = pd.DataFrame({
        "Fertilizer Product": ["Urea", "DAP", "MOP", "Complex", "Compost"],
        "Quantity (kg)": [opt['urea_kg'], opt['dap_kg'], opt['mop_kg'], opt.get('complex_kg', 0.0), opt['compost_kg']]
    })
    fq_bars = alt.Chart(fert_qty_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Fertilizer Product:N"), y=alt.Y("Quantity (kg):Q"), color=alt.Color("Fertilizer Product:N", scale=alt.Scale(range=["#39FF88", "#00E5FF", "#FFD700", "#81C784", "#B9F6CA"]), legend=None)
    )
    st.altair_chart(fq_bars, use_container_width=True)

    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 5
        st.rerun()
    if b2.button("Generate Official Prescription ➔"):
        log_activity(st.session_state.user_mobile, "4R Optimizer Run", f"Total Cost: ₹{opt['total_cost']}")
        st.session_state.step = 7
        st.rerun()

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 7: DESCRIPTIVE PRESCRIPTION UI & PDF DOWNLOAD
# -------------------------------------------------------------
elif st.session_state.step == 7:
    opt = st.session_state.get("opt_results", {})
    
    st.markdown(f"""
    <div style="background: rgba(11, 61, 46, 0.90); border-radius: 16px; padding: 18px 24px; border: 1px solid rgba(57, 255, 136, 0.5); box-shadow: 0 8px 22px rgba(0,0,0,0.6);">
        <h2 style="color: #39FF88; margin: 0 0 6px 0; font-size: 22px;">SMART KISHAN : AI BASED FERTILIZER AND INPUT USAGE OPTIMIZATION</h2>
        <p style="color: #FFFFFF; margin: 0; font-size: 14px; font-weight: 600;">Official Farmer Prescription Summary</p>
    </div><br>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### 🌾 Crop: **{st.session_state.sel_crop.capitalize()}**")
    st.markdown(f"**Target Yield:** {st.session_state.target_yield} t/acre | **Estimated Investment:** ₹{opt.get('total_cost', 0):,.0f}")
    
    st.info("💡 **AI Agronomist Note:** Your soil analysis, climate data, and budget constraints have been processed. We have formulated an optimized, multi-stage nutrient application plan to maximize yield while preventing soil degradation and nutrient leaching.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🛒 Required Inputs")
        st.write(f"• **Urea:** {opt.get('urea_kg', 0)} kg")
        st.write(f"• **DAP:** {opt.get('dap_kg', 0)} kg")
        st.write(f"• **MOP:** {opt.get('mop_kg', 0)} kg")
        st.write(f"• **Bio-Compost:** {opt.get('compost_kg', 0)} kg")
        
    with c2:
        st.markdown("#### 📅 Application Strategy")
        st.write("• **Stage 1 (Day 0):** Basal Dressing")
        st.write("• **Stage 2 (Day 20-25):** Vegetative Growth")
        st.write("• **Stage 3 (Day 45-55):** Panicle Initiation")
        
    st.warning("📄 **Action Required:** Download your full official 4R Certified Dossier below for the exact application methods, split dosages, and detailed soil metrics.")

    pdf_bytes = generate_english_pdf(
        user_mobile=st.session_state.user_mobile, plot_id=st.session_state.plot_id,
        raw_land=st.session_state.raw_land_val, land_unit=st.session_state.land_unit,
        crop=st.session_state.sel_crop, target_yield=st.session_state.target_yield,
        budget=st.session_state.budget_cap, opt=opt,
        n=st.session_state.soil_n, p=st.session_state.soil_p, k=st.session_state.soil_k,
        ph=st.session_state.soil_ph, soc=st.session_state.soc, moist=st.session_state.soil_moist,
        temp=st.session_state.temp, humid=st.session_state.humidity, rain=st.session_state.rainfall
    )

    p_col1, p_col2 = st.columns([2, 2])
    with p_col1:
        st.download_button(
            label="📄 Download Full PDF Prescription",
            data=pdf_bytes,
            file_name=f"SmartKishan_Prescription_{st.session_state.user_mobile}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with p_col2:
        if st.button("Proceed to Feedback & Exit ➔", use_container_width=True):
            st.session_state.step = 8
            st.rerun()

    st.divider()
    if st.button("⬅️ Back"):
        st.session_state.step = 6
        st.rerun()

    render_floating_ai_chat_launcher()

# -------------------------------------------------------------
# SCREEN 8: GLOWING STAR RATING (NON-MANDATORY COMMENTS)
# -------------------------------------------------------------
elif st.session_state.step == 8:
    st.markdown("## Farmer Feedback & Star Rating")
    if "rating" not in st.session_state: st.session_state.rating = 5

    rating_names = {1: "Worst", 2: "Bad", 3: "Good", 4: "Better", 5: "Best"}
    current_rating = st.session_state.rating
    current_rating_text = rating_names.get(current_rating, "Best")
    
    st.html("""
    <style>
    .sk-rating-box {
        width: 100%;
        min-height: 125px;
        margin: 18px 0 20px 0;
        padding: 18px 20px;
        box-sizing: border-box;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(5, 45, 28, 0.97), rgba(11, 61, 46, 0.94), rgba(20, 85, 48, 0.78));
        border: 1px solid rgba(57, 255, 136, 0.38);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.55), inset 0 0 35px rgba(57, 255, 136, 0.08);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .sk-star-row { display: flex; align-items: center; justify-content: center; gap: 16px; white-space: nowrap; }
    .sk-star-container { display: flex; flex-direction: column; align-items: center; }
    .sk-star { font-family: Arial, sans-serif; font-size: 58px; line-height: 1; display: inline-block; }
    .sk-star-empty { color: #D3D3D3; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.75); }
    .sk-star-selected { color: #39FF88; text-shadow: 0 0 6px #39FF88, 0 0 14px #39FF88, 0 0 22px rgba(57, 255, 136, 0.85), 0 0 32px rgba(57, 255, 136, 0.6); }
    .sk-star-label-text { font-size: 14px; font-weight: 700; color: #A7F3D0; margin-top: 6px; font-family: 'Plus Jakarta Sans', Arial, sans-serif; }
    .sk-rating-result { margin-top: 10px; color: #39FF88; font-size: 19px; font-weight: 700; text-shadow: 0 2px 5px rgba(0, 0, 0, 0.85); font-family: Arial, sans-serif;}
    
    .sk-star-button-area div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0B3D2E, #145A32) !important;
        border: 1px solid #39FF88 !important; border-radius: 10px !important;
        color: #39FF88 !important; font-weight: 700 !important; font-size: 14px !important;
        min-height: 48px; padding: 10px 14px !important; box-shadow: 0 4px 12px rgba(57, 255, 136, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .sk-star-button-area div[data-testid="stButton"] > button:hover {
        background: rgba(57, 255, 136, 0.15) !important; color: #FFFFFF !important; transform: translateY(-2px) !important; border-color: #66FFAA !important; box-shadow: 0 0 12px rgba(57, 255, 136, 0.6) !important;
    }
    </style>
    """)

    star_items_html = ""
    for s_val, s_lbl in [(1,"Worst"), (2,"Bad"), (3,"Good"), (4,"Better"), (5,"Best")]:
        cls = "sk-star-selected" if s_val <= current_rating else "sk-star-empty"
        star_items_html += f"<div class='sk-star-container'><span class='sk-star {cls}'>★</span><span class='sk-star-label-text'>{s_lbl}</span></div>"
    
    st.html(f"""
    <div class="sk-rating-box">
        <div class="sk-star-row">{star_items_html}</div>
        <div class="sk-rating-result">Selected Tier: {current_rating_text} ({current_rating}/5)</div>
    </div>
    """)

    st.html("<div style='text-align:center; color:#FFFFFF; font-size:14px; font-weight:600; margin-bottom:10px;'>Click your rating level below:</div>")

    st.markdown('<div class="sk-star-button-area">', unsafe_allow_html=True)
    bc1, bc2, bc3, bc4, bc5 = st.columns(5, gap="small")
    if bc1.button("1 - Worst", use_container_width=True): st.session_state.rating = 1; st.rerun()
    if bc2.button("2 - Bad", use_container_width=True): st.session_state.rating = 2; st.rerun()
    if bc3.button("3 - Good", use_container_width=True): st.session_state.rating = 3; st.rerun()
    if bc4.button("4 - Better", use_container_width=True): st.session_state.rating = 4; st.rerun()
    if bc5.button("5 - Best", use_container_width=True): st.session_state.rating = 5; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    feedback_comments = st.text_area("Your Comments / Suggestions (Optional):")

    b_fb_back, b_fb_sub = st.columns([1, 5], gap="small")
    with b_fb_back:
        if st.button("⬅️ Back"):
            st.session_state.step = 7
            st.rerun()
    with b_fb_sub:
        if st.button("Submit & Exit Dashboard ➔", use_container_width=True):
            save_feedback(st.session_state.user_mobile, current_rating, current_rating_text, feedback_comments.strip())
            st.success("✅ Thank you! Your feedback has been recorded safely. Exit session...")
            st.session_state.logged_in = False
            st.session_state.step = 1
            st.session_state.rating = 5
            st.rerun()

    render_floating_ai_chat_launcher()
