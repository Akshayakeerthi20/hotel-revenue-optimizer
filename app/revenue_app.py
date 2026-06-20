import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Hotel Revenue Optimizer", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f4f6f9; }
    .main-title { font-size: 30px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
    .sub-title { font-size: 15px; color: #666; margin-bottom: 8px; }
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }
    .section-label {
        font-size: 13px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
    }
    .result-high {
        background: #f0faf3;
        border-left: 6px solid #43a047;
        padding: 24px;
        border-radius: 10px;
    }
    .result-mid {
        background: #fff8e1;
        border-left: 6px solid #f9a825;
        padding: 24px;
        border-radius: 10px;
    }
    .result-low {
        background: #fff0f0;
        border-left: 6px solid #e53935;
        padding: 24px;
        border-radius: 10px;
    }
    .result-title { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
    .insight-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
        font-size: 14px;
        color: #444;
        line-height: 1.8;
    }
    div[data-testid="stButton"] button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ── Load ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USD_TO_INR = 83.5

@st.cache_resource
def load_model():
    rf = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'revenue_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'revenue_scaler.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'revenue_features.pkl'))
    return rf, scaler, features

@st.cache_data
def load_data():
    return pd.read_parquet(os.path.join(BASE_DIR, '..', 'data', 'processed', 'bookings_clean.parquet'))

rf, scaler, features = load_model()
df = load_data()

# ── Header ──
st.markdown('<div class="main-title">Hotel Revenue Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter your hotel details to get the optimal room price recommendation.</div>', unsafe_allow_html=True)
st.markdown("---")

# ── 5 Inputs ──
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Hotel Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
    market_segment = st.selectbox("Target Market", [
        "Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups"
    ])
    month = st.selectbox("Season / Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])

with col2:
    guests = st.slider("Number of Guests", 1, 6, 2)
    stay_length = st.slider("Length of Stay (nights)", 1, 14, 3)

st.markdown('</div>', unsafe_allow_html=True)

# ── Month to week mapping ──
month_to_week = {
    "January": 3, "February": 7, "March": 11, "April": 15,
    "May": 20, "June": 24, "July": 28, "August": 32,
    "September": 37, "October": 41, "November": 46, "December": 50
}

if st.button("Get Optimal Price"):

    input_dict = {
        'hotel': [hotel],
        'meal': ['BB'],
        'market_segment': [market_segment],
        'distribution_channel': ['Direct'],
        'is_repeated_guest': [0],
        'stays_in_weekend_nights': [min(stay_length, 2)],
        'stays_in_week_nights': [max(stay_length - 2, 1)],
        'adults': [min(guests, 4)],
        'children': [max(guests - 4, 0)],
        'arrival_date_month': [month],
        'arrival_date_week_number': [month_to_week[month]],
        'lead_time': [30],
        'customer_type': ['Transient'],
        'total_of_special_requests': [0],
        'required_car_parking_spaces': [0],
        'booking_changes': [0]
    }

    input_df = pd.DataFrame(input_dict)

    # Encode
    cat_cols = input_df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        le.fit(df[col].astype(str))
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except ValueError:
            input_df[col] = 0

    # Predict and convert to INR
    input_scaled = scaler.transform(input_df[features])
    predicted_adr = rf.predict(input_scaled)[0]
    predicted_adr_inr = predicted_adr * USD_TO_INR
    total_nights = stay_length
    revenue_per_booking = predicted_adr_inr * total_nights
    monthly_revenue = revenue_per_booking * 30

    # Category
    if predicted_adr_inr >= 12000:
        result_class = "result-high"
        price_label = "Premium Pricing Recommended"
        price_color = "#43a047"
        advice = "Your hotel parameters support a premium price point. Guests in this segment show high willingness to pay — focus on quality and direct booking incentives."
    elif predicted_adr_inr >= 6500:
        result_class = "result-mid"
        price_label = "Mid-Range Pricing Recommended"
        price_color = "#f9a825"
        advice = "Your parameters align with mid-range market expectations. Consider upselling meal plans or room upgrades to boost revenue per booking."
    else:
        result_class = "result-low"
        price_label = "Budget Pricing Recommended"
        price_color = "#e53935"
        advice = "This segment is price-sensitive. Focus on high occupancy and consider shifting to corporate or direct channels to improve your revenue."

    # Result card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="{result_class}">
            <div class="result-title" style="color:{price_color};">{price_label}</div>
            <div style="font-size:38px;font-weight:700;color:#1a1a2e;margin:10px 0;">
                ₹{predicted_adr_inr:,.2f}
                <span style="font-size:16px;color:#888;">per night</span>
            </div>
            <div style="font-size:13px;color:#777;">
                {hotel} &nbsp;·&nbsp; {market_segment} &nbsp;·&nbsp;
                {month} &nbsp;·&nbsp; {guests} guests &nbsp;·&nbsp; {stay_length} nights
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Revenue Estimates</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Price", f"₹{predicted_adr_inr:,.2f}/night")
    with col2:
        st.metric("Revenue per Booking", f"₹{revenue_per_booking:,.2f}")
    with col3:
        st.metric("Est. Monthly Revenue", f"₹{monthly_revenue:,.2f}")

    st.markdown(
        f'<div class="insight-box">&#128161; <b>Recommendation:</b> {advice}</div>',
        unsafe_allow_html=True
    )

    # Chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Price vs Market Benchmarks</div>', unsafe_allow_html=True)

    budget_inr = 74.48 * USD_TO_INR
    avg_inr = 108.49 * USD_TO_INR
    luxury_inr = 135.00 * USD_TO_INR

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ['Budget\n(25th %)', 'Market\nAverage', 'Your\nPrice', 'Luxury\n(75th %)']
    values = [budget_inr, avg_inr, predicted_adr_inr, luxury_inr]
    colors = ['#90caf9', '#64b5f6', '#1a1a2e', '#42a5f5']

    ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    ax.axhline(predicted_adr_inr, color='#e53935', linestyle='--',
               linewidth=1.5, label=f'Your Price: ₹{predicted_adr_inr:,.2f}')
    ax.set_ylabel('Price per Night (₹)', fontsize=11)
    ax.set_title('Your Recommended Price vs Market', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#ffffff')
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)