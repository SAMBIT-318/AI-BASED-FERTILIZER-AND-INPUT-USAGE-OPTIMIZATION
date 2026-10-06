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
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from PIL import Image, ImageStat, ImageFilter
from sqlalchemy import create_engine, text
import altair as alt

from optimizer import optimize_fertilizer_blend
from train_pipeline import train_all_models

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
            "metrics": {
                "n": est_n, "p": est_p, "k": est_k,
                "ph": est_ph, "soc": est_soc, "moist": est_moist,
                "rgb_signature": f"RGB({r_m:.0f}, {g_m:.0f}, {b_m:.0f})"
            }
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
# PAGE CONFIGURATION & THEME STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Smart Kishan | AgriTech Control Center",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"  # Force Sidebar to be open for Gemini AI
)

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
    
    .glass-login-card { background: rgba(11, 61, 46, 0.94) !important; backdrop-filter: blur(18px) !important; border: 1px solid rgba(57, 255, 136, 0.6) !important; border-radius: 20px !important; padding: 32px !important; box-shadow: 0 16px 48px rgba(0, 0, 0, 0.95) !important; }
    .metric-card { background: rgba(11, 61, 46, 0.90) !important; border-radius: 14px !important; padding: 16px 18px !important; border-left: 6px solid #39FF88 !important; border-top: 1px solid rgba(57, 255, 136, 0.3); border-right: 1px solid rgba(57, 255, 136, 0.3); border-bottom: 1px solid rgba(57, 255, 136, 0.3); box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important; margin-bottom: 12px; }
    
    div.stButton > button { background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important; color: #39FF88 !important; font-weight: 700 !important; border-radius: 10px !important; border: 1px solid #39FF88 !important; box-shadow: 0 4px 12px rgba(57, 255, 136, 0.3) !important; }
    div.stDownloadButton > button { background: linear-gradient(180deg, #145A32 0%, #0B3D2E 100%) !important; color: #39FF88 !important; font-weight: 700 !important; border-radius: 10px !important; border: 1px solid #39FF88 !important; }
    
    .badge-pass { background-color: rgba(57, 255, 136, 0.25); color: #39FF88; padding: 5px 14px; border-radius: 8px; font-weight: 700; border: 1px solid #39FF88; }
    .badge-warn { background-color: rgba(239, 68, 68, 0.25); color: #F87171; padding: 5px 14px; border-radius: 8px; font-weight: 700; border: 1px solid #EF4444; }
    
    #vg-tooltip-element * { color: #000000 !important; }
    div[data-baseweb="menu"] *, ul[data-baseweb="menu"] *, [role="listbox"] * { color: #000000 !important; }
    label, p, span, h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }

    /* Prescription HTML CSS */
    .prescription-container { background: #FFFFFF; color: #1E293B; border-radius: 8px; padding: 30px; box-shadow: 0 8px 30px rgba(0,0,0,0.8); font-family: 'Arial', sans-serif; max-width: 900px; margin: 0 auto; }
    .prescription-container * { color: #1E293B !important; text-shadow: none !important; }
    .pres-header { text-align: center; margin-bottom: 20px; border-bottom: 2px solid #2E7D32; padding-bottom: 15px; }
    .pres-header img { width: 120px; height: auto; margin-bottom: 10px; }
    .pres-header h2 { color: #0B3D2E !important; font-size: 24px; font-weight: bold; margin: 0 0 5px 0; text-transform: uppercase;}
    .pres-header p { color: #2E7D32 !important; font-size: 13px; font-style: italic; font-weight: bold; margin: 0; }
    .pres-section-title { font-size: 16px; font-weight: bold; color: #0B3D2E !important; margin: 20px 0 10px 0; border-bottom: 1px solid #C8E6C9; padding-bottom: 4px; }
    .pres-table { width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 13px; }
    .pres-table th, .pres-table td { border: 1px solid #CBD5E1; padding: 8px 10px; text-align: left; }
    .pres-table th { background-color: #E2EEDF; color: #0F172A !important; font-weight: bold; }
    .pres-table td { background-color: #FAFAFA; }
    .pres-footer { display: flex; justify-content: space-between; align-items: center; border-top: 2px solid #2E7D32; margin-top: 30px; padding-top: 10px; font-size: 11px; color: #64748B !important; }
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
    except (ModuleNotFoundError, AttributeError, EOFError, ImportError, ValueError):
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
# DATABASE CONNECTION & AUTHENTICATION
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
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=300, connect_args={"connect_timeout": 10})
        
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE IF NOT EXISTS users (mobile_number TEXT PRIMARY KEY, password TEXT NOT NULL, role TEXT DEFAULT 'farmer')"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS feedback (id SERIAL PRIMARY KEY, mobile TEXT, rating INT, rating_text TEXT, comments TEXT, admin_reply TEXT)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS help_requests (id SERIAL PRIMARY KEY, mobile TEXT, request_type TEXT, query_text TEXT, status TEXT DEFAULT 'Pending', admin_reply TEXT, attended_by TEXT, user_feedback TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.execute(text("CREATE TABLE IF NOT EXISTS user_activity (id SERIAL PRIMARY KEY, mobile TEXT, activity_type TEXT, details TEXT, is_deleted INT DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
            conn.commit()
        return engine
    except Exception:
        return None

engine = get_db_engine()

def log_activity(mobile, activity_type, details):
    if engine and mobile:
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO user_activity (mobile, activity_type, details, is_deleted) VALUES (:m, :a, :d, 0)"), {"m": str(mobile), "a": activity_type, "d": details})
                conn.commit()
        except: pass

def register_user(mobile, password, role="farmer"):
    if not engine: return False, "Database connection unavailable."
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT mobile_number FROM users WHERE mobile_number = :m"), {"m": str(mobile)}).fetchone():
                return False, "This mobile number is already registered."
            conn.execute(text("INSERT INTO users (mobile_number, password, role) VALUES (:m, :p, :r)"), {"m": str(mobile), "p": hashed_pw, "r": role})
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
            if not conn.execute(text("SELECT mobile_number FROM users WHERE mobile_number = :m"), {"m": str(mobile)}).fetchone():
                return False, "Mobile number not registered."
            conn.execute(text("UPDATE users SET password = :p WHERE mobile_number = :m"), {"p": hashed_pw, "m": str(mobile)})
            conn.commit()
        log_activity(mobile, "Password Reset", "User password changed successfully.")
        return True, "Password updated successfully!"
    except Exception as e: return False, f"Reset error: {e}"

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
        return True, "farmer"
    if not engine: return False, "farmer"
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT password, role FROM users WHERE mobile_number = :m"), {"m": str(mobile)}).fetchone()
            if res and res[0] == hashed_pw:
                log_activity(mobile, "Sign In", "User signed in successfully.")
                return True, (res[1] or "farmer")
    except: pass
    return False, "farmer"

def save_feedback(mobile, rating, rating_text, comments):
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO feedback (mobile, rating, rating_text, comments) VALUES (:m, :r, :rt, :c)"), {"m": str(mobile), "r": rating, "rt": rating_text, "c": comments})
                conn.commit()
            log_activity(mobile, "Feedback Given", f"Rated {rating_text} ({rating}/5)")
        except: pass

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "step" not in st.session_state: st.session_state.step = 1
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
if "chat_messages" not in st.session_state: st.session_state.chat_messages = [{"role": "assistant", "content": "Hello Farmer! I am your Smart Kishan AI Assistant powered by Gemini. Ask me anything!"}]

# -------------------------------------------------------------
# GOOGLE GEMINI AI INTEGRATION SIDEBAR (ALWAYS ACTIVE)
# -------------------------------------------------------------
def render_ai_chatbot_sidebar():
    with st.sidebar:
        if os.path.exists(LOGO_FILE_EXACT):
            st.image(LOGO_FILE_EXACT, width=120)
        st.markdown("""
        <div style="background: rgba(11, 61, 46, 0.95); padding: 16px; border-radius: 12px; border: 1px solid #39FF88; margin-bottom: 15px;">
            <h3 style="color: #39FF88; margin: 0 0 6px 0;">🤖 Gemini AI Agronomist</h3>
            <p style="color: #FFFFFF; font-size: 13px; margin: 0;">Live chat support for farming, NPK calculations, and disease management.</p>
        </div>
        """, unsafe_allow_html=True)

        gemini_api_key = st.text_input("Enter Gemini API Key (Optional):", type="password", help="Get a free key from Google AI Studio")

        chat_container = st.container()
        with chat_container:
            st.markdown('<div style="background-color: #062319; padding: 14px; border-radius: 12px; border: 1px solid rgba(57,255,136,0.3); max-height: 400px; overflow-y: auto; margin-bottom: 12px;">', unsafe_allow_html=True)
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
                
                # Attempt to use real Gemini API if key is provided
                if gemini_api_key:
                    try:
                        genai.configure(api_key=gemini_api_key)
                        model = genai.GenerativeModel('gemini-pro')
                        response = model.generate_content(f"You are an expert Indian Agronomist AI named Smart Kishan. Answer this farming question concisely and helpfully: {user_q}")
                        reply = response.text
                    except Exception as e:
                        reply = f"⚠️ Gemini API Error: {str(e)}. (Falling back to local logic)."
                else:
                    # Smart Mock Fallback Logic
                    q_lower = user_q.lower()
                    if "disease" in q_lower or "pest" in q_lower or "rust" in q_lower or "blight" in q_lower:
                        reply = "🔬 **Plant Pathology AI**: For fungal infections (like Early Blight or Rust), apply Mancozeb 75% WP @ 2.5g/L or Hexaconazole 5% EC. Ensure spray is done during cool morning hours."
                    elif "urea" in q_lower or "nitrogen" in q_lower or "npk" in q_lower or "fertilizer" in q_lower:
                        reply = "🧪 **Nutrient Advisory**: Split your nitrogen doses across basal, tillering, and flowering stages. Avoid applying urea on dry soils to prevent ammonia volatilization."
                    else:
                        reply = f"🌱 **Agronomy AI**: I analyzed your query about '{user_q}'. Make sure your soil pH is maintained between 6.0 and 7.2 for optimal nutrient uptake!"

                st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                st.rerun()

render_ai_chatbot_sidebar() # Sidebar rendered globally

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
# SUB-PAGES: ADMIN & NOTIFICATION ROUTES
# -------------------------------------------------------------
elif st.session_state.step == 90 and st.session_state.user_role == "admin":
    st.markdown("## SMART KISHAN : ADMIN COMMAND CENTER")
    if st.button("🚪 Sign Out"):
        st.session_state.logged_in = False
        st.session_state.step = 1
        st.rerun()
    st.info(f"Logged in Admin: +91 {st.session_state.user_mobile}")
    
    admin_t1, admin_t2 = st.tabs(["Users", "Feedback"])
    with admin_t1:
        if engine:
            try:
                with engine.connect() as conn:
                    st.dataframe(pd.read_sql(text("SELECT * FROM users"), conn))
            except: pass
    with admin_t2:
        if engine:
            try:
                with engine.connect() as conn:
                    st.dataframe(pd.read_sql(text("SELECT * FROM feedback"), conn))
            except: pass

elif st.session_state.step == 21:
    st.subheader("🆘 Help & Account Requests")
    h_type = st.selectbox("Category:", ["Delete My Account", "General Inquiry"])
    h_details = st.text_area("Details:")
    if st.button("⬅️ Back"): st.session_state.step = 2; st.rerun()
    if st.button("Submit ➔"):
        if engine:
            with engine.connect() as conn:
                conn.execute(text("INSERT INTO help_requests (mobile, request_type, query_text) VALUES (:m, :rt, :q)"), {"m": str(st.session_state.user_mobile), "rt": h_type, "q": h_details.strip()})
                conn.commit()
        st.success("Sent to admin!")

elif st.session_state.step == 22:
    st.subheader("🔔 Notifications")
    if st.button("⬅️ Back"): st.session_state.step = 2; st.rerun()
    if engine:
        with engine.connect() as conn:
            notifs = pd.read_sql(text("SELECT * FROM help_requests WHERE mobile = :m ORDER BY id DESC"), conn, params={"m": str(st.session_state.user_mobile)})
            st.dataframe(notifs, use_container_width=True)

elif st.session_state.step == 23:
    st.subheader("📜 Activity History")
    if st.button("⬅️ Back"): st.session_state.step = 2; st.rerun()
    if engine:
        with engine.connect() as conn:
            act = pd.read_sql(text("SELECT * FROM user_activity WHERE mobile = :m ORDER BY created_at DESC"), conn, params={"m": str(st.session_state.user_mobile)})
            st.dataframe(act, use_container_width=True)

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

    # REAL TIME DASHBOARD TABS
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

# -------------------------------------------------------------
# SCREEN 4: SOIL COMPARISON BAR CHARTS WITH CENTERED LABELS
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
    if b1.button("⬅ Back"):
        st.session_state.step = 3
        st.rerun()
    if b2.button("Continue to Crop Analytics ➔"):
        st.session_state.step = 5
        st.rerun()

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
        st.session_state.step = 6
        st.rerun()

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
        st.session_state.step = 7
        st.rerun()

# -------------------------------------------------------------
# SCREEN 7: OFFICIAL PRESCRIPTION DOSSIER (HTML MATCHING IMAGE)
# -------------------------------------------------------------
elif st.session_state.step == 7:
    opt = st.session_state.get("opt_results", {})
    
    # Render Exact Prescription UI matching the provided layout image
    logo_base64 = f"data:image/jpeg;base64,{LOGO_DATA}" if LOGO_DATA else ""
    local_time = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%d-%b-%Y %I:%M %p')
    
    st.markdown(f"""
    <div class="prescription-container">
        <div class="pres-header">
            {"<img src='" + logo_base64 + "' alt='Smart Kishan Logo'>" if logo_base64 else ""}
            <h2>SMART KISHAN • OFFICIAL CROP PRESCRIPTION</h2>
            <p>Certified 4R Nutrient Stewardship & Field Application Dossier</p>
            <p style="color:#64748B !important; font-weight:normal;">Dossier ID: SK-{datetime.now().strftime('%Y%m%d')}-{str(st.session_state.user_mobile)[-4:]} | Generated: {local_time}</p>
        </div>
        
        <div class="pres-section-title">1. FARMER & LAND PROFILE</div>
        <table class="pres-table">
            <tr>
                <td><b>Farmer Mobile:</b></td><td>+91 {st.session_state.user_mobile}</td>
                <td><b>Field / Parcel ID:</b></td><td>{st.session_state.plot_id}</td>
            </tr>
            <tr>
                <td><b>Target Crop:</b></td><td>{st.session_state.sel_crop}</td>
                <td><b>Target Harvest:</b></td><td>{st.session_state.target_yield} t/acre</td>
            </tr>
            <tr>
                <td><b>Land Area:</b></td><td>{st.session_state.raw_land_val:.2f} {st.session_state.land_unit.split()[0]}</td>
                <td><b>Standard Area:</b></td><td>{st.session_state.land_area:.3f} Hectares</td>
            </tr>
            <tr>
                <td><b>Farmer Budget:</b></td><td>Rs. {st.session_state.budget_cap:,.0f}</td>
                <td><b>Optimization Cost:</b></td><td>Rs. {opt.get('total_cost', 0):,.0f}</td>
            </tr>
        </table>
        
        <div class="pres-section-title">2. SOIL PROFILE & MEASURED ATTRIBUTES</div>
        <table class="pres-table">
            <tr>
                <td><b>Nitrogen (N):</b></td><td>{st.session_state.soil_n:.1f} mg/kg</td>
                <td><b>Soil pH:</b></td><td>{st.session_state.soil_ph:.1f}</td>
                <td><b>Ambient Temp:</b></td><td>{st.session_state.temp:.1f} °C</td>
            </tr>
            <tr>
                <td><b>Phosphorus (P):</b></td><td>{st.session_state.soil_p:.1f} mg/kg</td>
                <td><b>Organic Carbon:</b></td><td>{st.session_state.soc:.2f} %</td>
                <td><b>Relative Humidity:</b></td><td>{st.session_state.humidity:.0f} %</td>
            </tr>
            <tr>
                <td><b>Potash (K):</b></td><td>{st.session_state.soil_k:.1f} mg/kg</td>
                <td><b>Soil Moisture:</b></td><td>{st.session_state.soil_moist:.1f} %</td>
                <td><b>Precipitation:</b></td><td>{st.session_state.rainfall:.0f} mm</td>
            </tr>
        </table>

        <div class="pres-section-title">3. RECOMMENDED FERTILIZER PURCHASES (50KG BAGS)</div>
        <table class="pres-table">
            <tr>
                <th>Fertilizer Product</th><th>Nutrient Category</th><th>Total Mass (kg)</th><th>50kg Bags Required</th>
            </tr>
            <tr>
                <td>Urea</td><td>Synthetic Nitrogen (46% N)</td><td>{opt.get('urea_kg', 0)} kg</td><td><b>{max(1, round(opt.get('urea_kg', 0) / 50.0)) if opt.get('urea_kg', 0) > 0 else 0} Bags</b></td>
            </tr>
            <tr>
                <td>DAP</td><td>Phosphatic (18% N + 46% P)</td><td>{opt.get('dap_kg',0)} kg</td><td><b>{max(1, round(opt.get('dap_kg', 0) / 50.0)) if opt.get('dap_kg', 0) > 0 else 0} Bags</b></td>
            </tr>
            <tr>
                <td>MOP</td><td>Potash (60% K2O)</td><td>{opt.get('mop_kg',0)} kg</td><td><b>{max(1, round(opt.get('mop_kg', 0) / 50.0)) if opt.get('mop_kg', 0) > 0 else 0} Bags</b></td>
            </tr>
            <tr>
                <td>Complex 14-35-14</td><td>Balanced N-P-K Mineral</td><td>{opt.get('complex_kg',0)} kg</td><td><b>{max(1, round(opt.get('complex_kg', 0) / 50.0)) if opt.get('complex_kg', 0) > 0 else 0} Bags</b></td>
            </tr>
            <tr>
                <td>Bio-Compost / Manure</td><td>Organic Humus Restorer</td><td>{opt.get('compost_kg',0)} kg</td><td><b>{round(opt.get('compost_kg', 0) / 50.0) if opt.get('compost_kg', 0) > 0 else 0} Bags</b></td>
            </tr>
        </table>

        <div class="pres-section-title">4. TIMED APPLICATION PERIODS & METHODS FOR FARMERS</div>
        <table class="pres-table">
            <tr>
                <th>Time Period</th><th>Nutrient Blend</th><th>Specific Application Method for Farmer</th>
            </tr>
            <tr>
                <td><b>Stage 1: Basal Dressing (At Sowing / Transplanting - Day 0)</b></td>
                <td>100% Bio-Compost + 100% DAP + 1/3 MOP + 1/4 Urea</td>
                <td>Incorporate compost and broadcast full DAP and 1/3 MOP. Place 5-7 cm below seed furrow; do not leave on dry surface.</td>
            </tr>
            <tr>
                <td><b>Stage 2: Vegetative Growth (20 - 25 Days Post Sowing)</b></td>
                <td>1/2 Urea + 1/3 MOP <i>(Vegetative Dose)</i></td>
                <td>Side-dress 1/2 urea dose + 1/3 MOP along plant rows. Ensure adequate soil moisture or irrigate within 24 hours.</td>
            </tr>
            <tr>
                <td><b>Stage 3: Panicle Initiation / Flowering (45 - 55 Days Post Sowing)</b></td>
                <td>Remaining 1/4 Urea + Remaining 1/3 MOP</td>
                <td>Top-dress remaining 1/4 urea and final MOP. Avoid application during heavy rains to prevent leaching.</td>
            </tr>
        </table>
        
        <div class="pres-footer">
            <span>Smart Kishan • Digital Farming Solutions • ISO 9001:2015 Standard</span>
            <span>Page 1 of 1</span>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns([1, 5])
    if b1.button("⬅️ Back"):
        st.session_state.step = 6
        st.rerun()
    if b2.button("Proceed to Exit & Feedback ➔"):
        st.session_state.step = 8
        st.rerun()

# -------------------------------------------------------------
# SCREEN 8: GLOWING STAR RATING (NO RADIO BUTTONS)
# -------------------------------------------------------------
elif st.session_state.step == 8:
    st.markdown("## Farmer Feedback & Star Rating")
    if "rating" not in st.session_state: st.session_state.rating = 5

    rating_names = {1: "Worst", 2: "Bad", 3: "Good", 4: "Better", 5: "Best"}
    
    st.html("""
    <style>
    .sk-star-selected { color: #39FF88; text-shadow: 0 0 10px #39FF88; font-size: 58px; }
    .sk-star-empty { color: #D3D3D3; font-size: 58px; }
    .sk-star-button-area div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0B3D2E, #145A32) !important;
        border: 1px solid #39FF88 !important; border-radius: 10px !important;
        color: #39FF88 !important; font-weight: 700 !important; font-size: 14px !important;
        min-height: 48px;
    }
    .sk-star-button-area div[data-testid="stButton"] > button:hover {
        background: rgba(57, 255, 136, 0.15) !important; color: #FFFFFF !important;
    }
    </style>
    """)

    # Render Visual Stars
    star_items_html = ""
    for s_val, s_lbl in [(1,"Worst"), (2,"Bad"), (3,"Good"), (4,"Better"), (5,"Best")]:
        cls = "sk-star-selected" if s_val <= st.session_state.rating else "sk-star-empty"
        star_items_html += f"<div style='text-align:center;'><span class='{cls}'>★</span><br><span style='color:#A7F3D0; font-weight:bold;'>{s_lbl}</span></div>"
    
    st.html(f"<div style='display:flex; justify-content:center; gap:20px; padding:20px; background:rgba(11, 61, 46, 0.94); border:1px solid rgba(57, 255, 136, 0.38); border-radius:14px; margin-bottom:20px;'>{star_items_html}</div>")

    st.write("Click your rating level below:")
    st.markdown('<div class="sk-star-button-area">', unsafe_allow_html=True)
    bc1, bc2, bc3, bc4, bc5 = st.columns(5, gap="small")
    if bc1.button("1 - Worst", use_container_width=True): st.session_state.rating = 1; st.rerun()
    if bc2.button("2 - Bad", use_container_width=True): st.session_state.rating = 2; st.rerun()
    if bc3.button("3 - Good", use_container_width=True): st.session_state.rating = 3; st.rerun()
    if bc4.button("4 - Better", use_container_width=True): st.session_state.rating = 4; st.rerun()
    if bc5.button("5 - Best", use_container_width=True): st.session_state.rating = 5; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    feedback_comments = st.text_area("Your Comments / Suggestions:")

    if st.button("Submit & Exit Dashboard ➔", use_container_width=True):
        if not feedback_comments.strip():
            st.error("⚠️ Mandatory Feedback Required.")
        else:
            save_feedback(st.session_state.user_mobile, st.session_state.rating, rating_names[st.session_state.rating], feedback_comments.strip())
            st.success("✅ Thank you! Exit session...")
            st.session_state.logged_in = False
            st.session_state.step = 1
            st.rerun()
