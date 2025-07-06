import streamlit as st
import joblib as jb
import numpy as np

# Page config
st.set_page_config(page_title="🔋 Energy & Solar Power Predictor", page_icon="☀️", layout="centered")

# Load models
energy_model = jb.load(r"C:\Python Skill4Future Session\Appliance_ML Model\appliance_energy_predictor.pkl")
solar_model = jb.load(r"C:\Python Skill4Future Session\Solar_ML Model\solar_power_output(in).pkl")

# Header
st.markdown("""
    <div style="text-align:center">
        <h1 style='color:#00cc66;'> EcoWatt: Smart Energy & Solar Predictor</h1>
        <p style='font-size:12px;'>Enter the environmental values to estimate appliance energy consumption and solar power output.</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Input section
with st.container():
    st.subheader("🌦️ Input Environmental Parameters")

    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        wind_speed = st.number_input("🍃 Wind Speed (m/s)", min_value=0.0, max_value=20.0, value=5.0)
    with col2:
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        radiation = st.number_input("🔆 Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=500.0)

# Buttons and Predictions
st.markdown("## 🔍 Prediction Results")

colA, colB = st.columns(2)

with colA:
    if st.button("🔌 Predict Appliance Energy"):
        input_energy = np.array([[temperature]])
        prediction = energy_model.predict(input_energy)
        st.success(f"⚡ Appliance Energy Consumption: **{prediction[0]:.2f} kWh**")

with colB:
    if st.button("☀️ Predict Solar Power Output"):
        input_solar = np.array([[temperature, humidity, wind_speed, radiation]])
        solar_prediction = solar_model.predict(input_solar)
        st.success(f"🔆 Solar Power Output: **{solar_prediction[0]:.2f} kW**")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:blue;'>Made with ❤️ by Twinkle Ghangare</div>", unsafe_allow_html=True)
