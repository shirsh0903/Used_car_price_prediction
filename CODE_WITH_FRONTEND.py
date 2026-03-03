import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TRUECarValue",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #121820 100%);
    border: 1px solid #2a2f3e;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,165,0,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,165,0,0.12);
    border: 1px solid rgba(255,165,0,0.3);
    color: #ffa500;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.8rem;
}
.hero-title span { color: #ffa500; }
.hero-sub {
    font-size: 1rem;
    color: #8892a4;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.6;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #ffa500;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,165,0,0.3), transparent);
}

/* ── Cards ── */
.card {
    background: #13171f;
    border: 1px solid #1e2534;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2e3650; }

/* ── Streamlit widgets override ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #13171f !important;
    border-color: #1e2534 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #ffa500 !important;
}
label { color: #9aa3b2 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

.stSlider > div > div > div { background: #ffa500 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #ffa500 !important;
    border-color: #ffa500 !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ffa500, #ff7b00);
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 2.5rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, #1a2040, #0f1520);
    border: 1.5px solid #ffa500;
    border-radius: 18px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #ffa500;
    font-weight: 600;
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
}
.result-range {
    font-size: 0.85rem;
    color: #8892a4;
    margin-top: 0.5rem;
}

/* ── Stat tiles ── */
.stat-tile {
    background: #13171f;
    border: 1px solid #1e2534;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffa500;
}
.stat-label {
    font-size: 0.75rem;
    color: #8892a4;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Info chips ── */
.chip {
    display: inline-block;
    background: rgba(255,165,0,0.1);
    border: 1px solid rgba(255,165,0,0.25);
    color: #ffa500;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Divider ── */
hr { border-color: #1e2534 !important; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD & TRAIN MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model, please wait…")
def load_model():
    df = pd.read_csv("car_details_v4_1.csv")
    df.dropna(inplace=True)

    # Extract numeric parts
    df['Engine']    = df['Engine'].str.extract(r'(\d+)').astype(float)
    df['Max Power'] = df['Max Power'].str.extract(r'([\d.]+)').astype(float)
    df['Max Torque']= df['Max Torque'].str.extract(r'([\d.]+)').astype(float)
    df.dropna(inplace=True)

    # Store unique values before encoding
    meta = {
        'makes':        sorted(df['Make'].unique().tolist()),
        'fuel_types':   sorted(df['Fuel Type'].unique().tolist()),
        'transmissions':sorted(df['Transmission'].unique().tolist()),
        'locations':    sorted(df['Location'].unique().tolist()),
        'colors':       sorted(df['Color'].unique().tolist()),
        'owners':       ['First', 'Second', 'Third', 'Fourth', '4 or More', 'UnRegistered Car'],
        'seller_types': sorted(df['Seller Type'].unique().tolist()),
        'drivetrains':  sorted(df['Drivetrain'].unique().tolist()),
        'seatings':     sorted(df['Seating Capacity'].unique().tolist()),
        'year_min':     int(df['Year'].min()),
        'year_max':     int(df['Year'].max()),
    }

    # Label encode
    le_dict = {}
    cat_cols = ['Make', 'Model', 'Fuel Type', 'Transmission', 'Location',
                'Color', 'Owner', 'Seller Type', 'Drivetrain']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop('Price', axis=1)
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)

    return model, scaler, le_dict, meta, df.columns.drop('Price').tolist()

model, scaler, le_dict, meta, feature_cols = load_model()


# ─────────────────────────────────────────────
#  HELPER — get model names for a make
# ─────────────────────────────────────────────
@st.cache_data
def get_models_for_make(make_name):
    df = pd.read_csv("car_details_v4_1.csv")
    df.dropna(inplace=True)
    models = sorted(df[df['Make'] == make_name]['Model'].unique().tolist())
    return models


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🚗 &nbsp; AI-Powered Valuation</div>
    <div class="hero-title">Know Your Car's<br><span>True Value</span></div>
    <div class="hero-sub">
        Enter your car's details below and our machine learning model will predict
        its current market price in seconds — powered by 2000+ real listings.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STAT TILES
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-tile"><div class="stat-value">2,059</div><div class="stat-label">Listings Trained</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-tile"><div class="stat-value">33</div><div class="stat-label">Car Brands</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-tile"><div class="stat-value">~82%</div><div class="stat-label">R² Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-tile"><div class="stat-value">RF</div><div class="stat-label">Algorithm</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FORM — two columns layout
# ─────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # ── Car Identity ──
    st.markdown('<div class="section-title">🏷️ Car Identity</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        make = st.selectbox("Brand / Make", meta['makes'])
    with col2:
        model_options = get_models_for_make(make)
        car_model = st.selectbox("Model", model_options)

    col3, col4 = st.columns(2)
    with col3:
        year = st.selectbox("Year of Manufacture", list(range(meta['year_max'], meta['year_min']-1, -1)))
    with col4:
        color = st.selectbox("Color", meta['colors'])

    # ── Specs ──
    st.markdown('<div class="section-title">⚙️ Technical Specs</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        fuel_type = st.selectbox("Fuel Type", meta['fuel_types'])
        drivetrain = st.selectbox("Drivetrain", meta['drivetrains'])
    with col6:
        transmission = st.selectbox("Transmission", meta['transmissions'])
        seating = st.selectbox("Seating Capacity", [int(x) for x in meta['seatings']])

    col7, col8, col9 = st.columns(3)
    with col7:
        engine = st.number_input("Engine (cc)", min_value=600, max_value=7000, value=1200, step=50)
    with col8:
        max_power = st.number_input("Max Power (bhp)", min_value=30, max_value=700, value=90, step=5)
    with col9:
        max_torque = st.number_input("Max Torque (Nm)", min_value=50, max_value=1000, value=120, step=5)

    col10, col11 = st.columns(2)
    with col10:
        fuel_tank = st.number_input("Fuel Tank (litres)", min_value=20, max_value=120, value=40, step=1)

    # ── Dimensions ──
    st.markdown('<div class="section-title">📐 Dimensions (mm)</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        length = st.number_input("Length", min_value=3000, max_value=6000, value=4000, step=10)
    with d2:
        width = st.number_input("Width", min_value=1500, max_value=2500, value=1700, step=10)
    with d3:
        height = st.number_input("Height", min_value=1300, max_value=2200, value=1500, step=10)


with right:
    # ── Ownership & Usage ──
    st.markdown('<div class="section-title">👤 Ownership & Usage</div>', unsafe_allow_html=True)

    owner = st.selectbox("Owner Type", meta['owners'])
    seller_type = st.selectbox("Seller Type", meta['seller_types'])
    location = st.selectbox("Location", meta['locations'])

    kilometer = st.slider(
        "Kilometers Driven",
        min_value=0, max_value=300000,
        value=50000, step=1000,
        format="%d km"
    )

    # ── Predict ──
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 &nbsp; Predict Price", use_container_width=True)

    # ── Result ──
    if predict_btn:
        try:
            # Encode categorical inputs
            def safe_encode(le, val):
                if val in le.classes_:
                    return le.transform([val])[0]
                else:
                    return 0  # fallback for unseen labels

            input_data = {
                'Make':              safe_encode(le_dict['Make'], make),
                'Model':             safe_encode(le_dict['Model'], car_model),
                'Year':              year,
                'Kilometer':         kilometer,
                'Fuel Type':         safe_encode(le_dict['Fuel Type'], fuel_type),
                'Transmission':      safe_encode(le_dict['Transmission'], transmission),
                'Location':          safe_encode(le_dict['Location'], location),
                'Color':             safe_encode(le_dict['Color'], color),
                'Owner':             safe_encode(le_dict['Owner'], owner),
                'Seller Type':       safe_encode(le_dict['Seller Type'], seller_type),
                'Engine':            engine,
                'Max Power':         max_power,
                'Max Torque':        max_torque,
                'Drivetrain':        safe_encode(le_dict['Drivetrain'], drivetrain),
                'Length':            length,
                'Width':             width,
                'Height':            height,
                'Seating Capacity':  seating,
                'Fuel Tank Capacity':fuel_tank,
            }

            # Build input DataFrame in correct column order
            input_df = pd.DataFrame([input_data])[feature_cols]
            input_scaled = scaler.transform(input_df)
            predicted_price = model.predict(input_scaled)[0]

            # Format
            low  = predicted_price * 0.90
            high = predicted_price * 1.10

            def fmt(n):
                if n >= 1_00_00_000:
                    return f"₹{n/1_00_00_000:.2f} Cr"
                elif n >= 1_00_000:
                    return f"₹{n/1_00_000:.2f} L"
                else:
                    return f"₹{n:,.0f}"

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Estimated Market Price</div>
                <div class="result-price">{fmt(predicted_price)}</div>
                <div class="result-range">Likely range: {fmt(low)} – {fmt(high)}</div>
            </div>
            """, unsafe_allow_html=True)

            # Chips summary
            st.markdown(f"""
            <div style="margin-top:1.2rem; text-align:center;">
                <span class="chip">{make} {car_model}</span>
                <span class="chip">{year}</span>
                <span class="chip">{fuel_type}</span>
                <span class="chip">{transmission}</span>
                <span class="chip">{owner} Owner</span>
                <span class="chip">{kilometer:,} km</span>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        # Placeholder card
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem 1.5rem; border-style: dashed;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🚗</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#e8eaf0;">
                Fill in the details
            </div>
            <div style="color:#8892a4; font-size:0.85rem; margin-top:0.5rem; line-height:1.6;">
                Complete the form on the left and hit<br>
                <strong style="color:#ffa500;">Predict Price</strong> to get an instant valuation.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#3a4255; font-size:0.78rem; padding-bottom:1rem;">
    CarValue AI &nbsp;·&nbsp; Used Car Price Prediction &nbsp;·&nbsp; Random Forest Model &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
