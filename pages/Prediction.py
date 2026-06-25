import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Prediction Center", page_icon="🔮")

st.title("🔮 House Price Prediction Center")

# Load Model
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.markdown("### Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    medinc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=3.5,
        step=0.1
    )

    houseage = st.number_input(
        "House Age",
        min_value=1.0,
        value=25.0
    )

    averooms = st.number_input(
        "Average Rooms",
        min_value=1.0,
        value=5.0
    )

    avebedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.5,
        value=1.0
    )

with col2:

    population = st.number_input(
        "Population",
        min_value=1,
        value=1000
    )

    aveoccup = st.number_input(
        "Average Occupancy",
        min_value=1.0,
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

# Create Input DataFrame
input_df = pd.DataFrame({
    "MedInc":[medinc],
    "HouseAge":[houseage],
    "AveRooms":[averooms],
    "AveBedrms":[avebedrms],
    "Population":[population],
    "AveOccup":[aveoccup],
    "Latitude":[latitude],
    "Longitude":[longitude]
})

st.subheader("📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("🚀 Predict House Price"):

    prediction = model.predict(input_df)

    predicted_price = prediction[0] * 100000

    st.success("Prediction Generated Successfully!")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted Price",
        f"${predicted_price:,.0f}"
    )

    c2.metric(
        "Median Income",
        round(medinc,2)
    )

    c3.metric(
        "House Age",
        round(houseage,2)
    )

    # Save Prediction History
    history = input_df.copy()
    history["Predicted Price"] = predicted_price

    if os.path.exists("prediction_history.csv"):
        history.to_csv(
            "prediction_history.csv",
            mode="a",
            header=False,
            index=False
        )
    else:
        history.to_csv(
            "prediction_history.csv",
            index=False
        )

    st.balloons()

# History Section
st.divider()

st.subheader("📈 Recent Predictions")

if os.path.exists("prediction_history.csv"):

    history_df = pd.read_csv(
        "prediction_history.csv"
    )

    st.dataframe(
        history_df.tail(10),
        use_container_width=True
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        "⬇ Download Prediction History",
        csv,
        "prediction_history.csv",
        "text/csv"
    )