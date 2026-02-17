import streamlit as st
import numpy as np
import joblib
import base64

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Mobile Battery Life Predictor",
    page_icon="🔋",
    layout="wide"
)

# -------------------------------
# Set Background Function
# -------------------------------
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background
set_background("images/background.png")

# -------------------------------
# Load Model & Scaler
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .main-title {
        font-size:35px;
        font-weight:bold;
        text-align:center;
        color:white;
    }
    .prediction-box {
        padding:20px;
        border-radius:15px;
        background-color:rgba(0,0,0,0.6);
        color:white;
        text-align:center;
        font-size:22px;
        font-weight:bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown('<p class="main-title">🔋 Mobile Battery Life Predictor</p>', unsafe_allow_html=True)
st.write("")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📊 Input Features")

screen_time = st.sidebar.slider("📱 Screen Time (hours)", 1.0, 15.0, 6.0)
gaming_hours = st.sidebar.slider("🎮 Gaming Hours", 0.0, 8.0, 2.0)
brightness = st.sidebar.slider("💡 Brightness Level (%)", 20.0, 100.0, 60.0)
battery_capacity = st.sidebar.slider("🔋 Battery Capacity (mAh)", 2500.0, 6000.0, 4000.0)
background_apps = st.sidebar.slider("📂 Background Apps", 0, 40, 10)
app_usage = st.sidebar.slider("📈 Daily App Usage Count", 5, 30, 15)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Battery Life"):

    input_data = np.array([[screen_time,
                            gaming_hours,
                            brightness,
                            battery_capacity,
                            background_apps,
                            app_usage]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction < 5:
        risk = "🔴 High Battery Drain"
        color = "red"
    elif prediction < 8:
        risk = "🟡 Moderate Drain"
        color = "orange"
    else:
        risk = "🟢 Good Battery Performance"
        color = "lime"

    st.markdown(f"""
        <div class="prediction-box">
        🔋 Estimated Battery Life: <br><br>
        <span style="color:{color}; font-size:30px;">
        {round(prediction,2)} Hours
        </span>
        <br><br>
        {risk}
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.caption("Built with ❤️ using Streamlit")
