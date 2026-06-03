import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
from PIL import Image

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(
    page_title="Smart House Price Analyzer",
    page_icon="🏠",
    layout="wide"
)

# -------------------
# LOAD DATASET
# -------------------
df = pd.read_csv("dataset.csv")

# -------------------
# MODEL
# -------------------
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
score = r2_score(y, y_pred)

# -------------------
# SIDEBAR
# -------------------
st.sidebar.title("🏠 Smart House Analyzer")
st.sidebar.write("Enter house details")

# -------------------
# TITLE
# -------------------
st.markdown(
    "<h1 style='text-align:center;color:green;'>🏠 Smart House Price Analyzer</h1>",
    unsafe_allow_html=True
)

# -------------------
# IMAGE
# -------------------
try:
    image = Image.open("house.jpg")
    st.image(image, use_container_width=True)
except:
    pass

# -------------------
# INPUTS
# -------------------
col1, col2, col3 = st.columns(3)

with col1:
    sqft = st.number_input(
        "Square Feet",
        min_value=500,
        max_value=10000,
        value=1500
    )

with col2:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

with col3:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

# -------------------
# SESSION HISTORY
# -------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------
# PREDICT
# -------------------
if st.button("Predict Price"):

    prediction = model.predict(
        [[sqft, bedrooms, bathrooms]]
    )

    st.success(
        f"Predicted House Price: ₹{prediction[0]:,.2f}"
    )

    st.metric(
        "Model Accuracy (R² Score)",
        f"{score*100:.2f}%"
    )

    # Save history
    st.session_state.history.append({
        "SquareFeet": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Predicted Price": round(prediction[0], 2),
        "Date": datetime.now()
    })

    # Download Report
    report = pd.DataFrame({
        "SquareFeet":[sqft],
        "Bedrooms":[bedrooms],
        "Bathrooms":[bathrooms],
        "Predicted Price":[prediction[0]],
        "Date":[datetime.now()]
    })

    st.download_button(
        "📥 Download Report",
        report.to_csv(index=False),
        file_name="prediction_report.csv",
        mime="text/csv"
    )

    # -------------------
    # PREDICTION GRAPH
    # -------------------
    st.subheader("📈 Prediction Graph")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df["SquareFeet"],
        df["Price"],
        color="blue",
        label="Actual Prices"
    )

    line = model.predict(X)

    ax.plot(
        df["SquareFeet"],
        line,
        color="red",
        label="Regression Line"
    )

    ax.scatter(
        sqft,
        prediction[0],
        color="green",
        s=200,
        marker="X",
        label="Prediction"
    )

    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

# -------------------
# HISTORY
# -------------------
if st.session_state.history:

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(history_df)

# -------------------
# DATASET
# -------------------
st.subheader("📊 Dataset")

st.dataframe(df)

# -------------------
# DISTRIBUTION
# -------------------
st.subheader("📉 House Price Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df["Price"],
    kde=True,
    ax=ax
)

st.pyplot(fig)

# -------------------
# AREA VS PRICE
# -------------------
st.subheader("🏠 Area vs Price")

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x="SquareFeet",
    y="Price",
    data=df,
    s=100,
    ax=ax
)

st.pyplot(fig)

# -------------------
# BEDROOMS VS PRICE
# -------------------
st.subheader("🛏 Bedrooms vs Price")

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    x="Bedrooms",
    y="Price",
    data=df,
    ax=ax
)

st.pyplot(fig)

# -------------------
# HEATMAP
# -------------------
st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8,5))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="YlGnBu",
    ax=ax
)

st.pyplot(fig)

# -------------------
# FOOTER
# -------------------
st.markdown("---")

st.markdown(
"""
### 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

### Features

✔ House Price Prediction  
✔ Linear Regression Model  
✔ R² Score Evaluation  
✔ Prediction History  
✔ CSV Report Download  
✔ Distribution Analysis  
✔ Area vs Price Analysis  
✔ Bedrooms Impact Analysis  
✔ Correlation Heatmap  
✔ Interactive Dashboard
"""
)
