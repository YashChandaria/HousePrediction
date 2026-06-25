import streamlit as st

st.set_page_config(
    page_title="California Housing Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 California Housing Price Prediction Dashboard")

st.markdown("""
### Welcome

This project demonstrates:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

with model comparison and analytics.

Use the navigation menu on the left.
""")