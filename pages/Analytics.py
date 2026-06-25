import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import fetch_california_housing

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📈"
)

st.title("📈 Housing Analytics Dashboard")

# Load Dataset
housing = fetch_california_housing(as_frame=True)

df = housing.frame

# Dataset Overview
st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    df.shape[0]
)

c2.metric(
    "Columns",
    df.shape[1]
)

c3.metric(
    "Avg House Value",
    f"${df['MedHouseVal'].mean()*100000:,.0f}"
)

c4.metric(
    "Max House Value",
    f"${df['MedHouseVal'].max()*100000:,.0f}"
)

st.divider()

# Distribution Plot
st.subheader("🏠 House Price Distribution")

fig = px.histogram(
    df,
    x="MedHouseVal",
    nbins=50,
    title="Distribution of House Prices"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Income vs Price
st.divider()

st.subheader("💰 Median Income vs House Value")

fig2 = px.scatter(
    df.sample(3000),
    x="MedInc",
    y="MedHouseVal",
    title="Income vs House Value"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# Correlation Heatmap
st.divider()

st.subheader("🔥 Correlation Heatmap")

corr = df.corr()

fig3 = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns
    )
)

fig3.update_layout(
    height=700
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# Feature Distribution
st.divider()

st.subheader("📊 Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]
)

fig4 = px.histogram(
    df,
    x=feature,
    nbins=40,
    title=f"{feature} Distribution"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# Dataset Preview
st.divider()

st.subheader("📝 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# Statistics
st.divider()

st.subheader("📋 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)