import streamlit as st
import pandas as pd

st.title("🏠 Dashboard Overview")

metrics = pd.read_csv("model_metrics.csv")

best_model = metrics.sort_values(
    "R2 Score",
    ascending=False
).iloc[0]

c1,c2,c3 = st.columns(3)

c1.metric(
    "Best Model",
    best_model["Model"]
)

c2.metric(
    "Best R²",
    round(best_model["R2 Score"],4)
)

c3.metric(
    "Best RMSE",
    round(best_model["RMSE"],4)
)

st.dataframe(metrics)