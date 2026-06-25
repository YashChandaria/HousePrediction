import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Model Comparison")

df = pd.read_csv("model_metrics.csv")

fig1 = px.bar(
    df,
    x="Model",
    y="R2 Score",
    title="R² Score Comparison"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

fig2 = px.bar(
    df,
    x="Model",
    y="RMSE",
    title="RMSE Comparison"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.dataframe(df)