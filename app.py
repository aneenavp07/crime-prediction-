import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Indian Crime Risk Prediction",
    page_icon="🚔",
    layout="centered"
)

# ------------------ LOAD MODELS ------------------
model = joblib.load("crime_future_model.pkl")
state_enc = joblib.load("state_encoder_A0.pkl")
target_enc = joblib.load("target_encoder_A0.pkl")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(
    "C:\Users\akhil\Downloads\indian_crime_with_categories_ordered.csv"
)

# ------------------ HEADER ------------------
st.markdown(
    """
    <h1 style="text-align:center;">🚔 Indian Crime Risk Prediction</h1>
    <p style="text-align:center;">
    AI-based prediction of future crime risk levels across Indian states
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ------------------ INPUTS (NO WRAPPERS) ------------------
state = st.selectbox(
    "🏙 Select Indian State",
    sorted(df["State"].unique())
)

year = st.number_input(
    "📅 Select Future Year",
    min_value=2024,
    max_value=2035,
    step=1
)

# ------------------ BUTTON ------------------
if st.button("🔮 Predict Crime Risk"):
    hist = (
        df[df["State"] == state]
        .sort_values("Year")
        .tail(3)
    )

    lag1 = hist.iloc[-1]["Crime_Rate_Per_100k"]
    lag2 = hist.iloc[-2]["Crime_Rate_Per_100k"]
    lag3 = hist.iloc[-3]["Crime_Rate_Per_100k"]

    input_data = [[
        state_enc.transform([state])[0],
        year,
        lag1,
        lag2,
        lag3
    ]]

    prediction = model.predict(input_data)
    result = target_enc.inverse_transform(prediction)[0]

    if result == "Low":
        st.success(f"🟢 {state} Crime Risk in {year}: LOW")
    elif result == "Medium":
        st.warning(f"🟠 {state} Crime Risk in {year}: MEDIUM")
    else:
        st.error(f"🔴 {state} Crime Risk in {year}: HIGH")

# ------------------ FOOTER ------------------
st.markdown(
    "<hr><p style='text-align:center;'> Machine Learning Project | Crime Risk Level Prediction</p>",
    unsafe_allow_html=True
)





