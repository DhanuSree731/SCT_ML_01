import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
from PIL import Image

# Dataset
data = {
    'SquareFeet': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 5],
    'Bathrooms': [1, 2, 2, 3, 4],
    'Price': [300000, 450000, 500000, 650000, 850000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Accuracy
y_pred = model.predict(X)
score = r2_score(y, y_pred)

# Page Title
st.markdown(
    "<h1 style='text-align: center; color: green;'>🏠 House Price Prediction</h1>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("House Price App")
st.sidebar.write("Enter house details to predict price.")

# Optional Image
try:
    image = Image.open("house.jpg")
    st.image(image, caption="House Price Prediction", use_container_width=True)
except:
    st.warning("Add a file named 'house.jpg' to display image.")

# Input Layout
col1, col2, col3 = st.columns(3)

with col1:
    sqft = st.number_input("Square Feet", min_value=500)

with col2:
    bed = st.number_input("Bedrooms", min_value=1)

with col3:
    bath = st.number_input("Bathrooms", min_value=1)

# Prediction Button
if st.button("Predict Price"):

    # Prediction
    prediction = model.predict([[sqft, bed, bath]])

    # Show Prediction
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

    # Accuracy Metric
    st.metric("Model Accuracy", f"{round(score * 100, 2)}%")

    # Create Report
    report = pd.DataFrame({
        "Square Feet": [sqft],
        "Bedrooms": [bed],
        "Bathrooms": [bath],
        "Predicted Price": [prediction[0]],
        "Date": [datetime.now()]
    })

    # Download Button
    st.download_button(
        label="Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="prediction_report.csv",
        mime="text/csv"
    )

    # Graph
    fig, ax = plt.subplots()

    # Scatter Plot
    ax.scatter(df['SquareFeet'], df['Price'], label="Actual Prices")

    # Regression Line
    line = model.predict(X)
    ax.plot(df['SquareFeet'], line, label="Regression Line")

    # Predicted Point
    ax.scatter(
        sqft,
        prediction[0],
        marker='x',
        s=200,
        label="Predicted Price"
    )

    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Price")
    ax.set_title("House Price Prediction Graph")

    ax.legend()

    # Show Graph
    st.pyplot(fig)