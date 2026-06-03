import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart House Price Analyzer",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("dataset.csv")

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Parking', 'LocationRating']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)
accuracy = r2_score(y, pred)

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Prediction",
        "Analytics"
    ]
)

# ==================================================
# DASHBOARD PAGE
# ==================================================
if page == "Dashboard":

    st.title("🏠 Smart House Price Analyzer")
    st.markdown("### Machine Learning Based House Price Prediction Dashboard")

    try:
        image = Image.open("house.jpg")
        st.image(image, use_container_width=True)
    except:
        pass

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Properties",
            len(df)
        )

    with col2:
        st.metric(
            "Average Price",
            f"₹ {int(df['Price'].mean()):,}"
        )

    with col3:
        st.metric(
            "Maximum Price",
            f"₹ {int(df['Price'].max()):,}"
        )

    with col4:
        st.metric(
            "Model Accuracy",
            f"{accuracy*100:.2f}%"
        )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📋 Project Overview")

        st.info("""
        Smart House Price Analyzer predicts
        property prices using Machine Learning.

        Features Used:
        - Square Feet
        - Bedrooms
        - Bathrooms
        - Parking Slots
        - Location Rating
        """)

    with c2:
        st.subheader("🛠 Technology Stack")

        st.success("""
        Python

        Pandas

        Matplotlib

        Seaborn

        Scikit-Learn

        Streamlit
        """)

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Prediction":

    st.title("🏡 House Price Prediction")

    st.markdown("### Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:

        sqft = st.slider(
            "Square Feet",
            500,
            10000,
            1500
        )

        bedrooms = st.slider(
            "Bedrooms",
            1,
            10,
            3
        )

        bathrooms = st.slider(
            "Bathrooms",
            1,
            10,
            2
        )

    with col2:

        parking = st.slider(
            "Parking Slots",
            0,
            5,
            1
        )

        road = st.selectbox(
            "Road Type",
            [
                "Off Road",
                "Road Side"
            ]
        )

        property_type = st.selectbox(
            "Property Type",
            [
                "Apartment",
                "Gated Community",
                "Villa"
            ]
        )

    road_score = 8 if road == "Road Side" else 5

    property_score = {
        "Apartment": 6,
        "Gated Community": 8,
        "Villa": 10
    }

    location_rating = (
        road_score +
        property_score[property_type]
    ) / 2

    if st.button("Predict House Price"):

        result = model.predict([[
            sqft,
            bedrooms,
            bathrooms,
            parking,
            location_rating
        ]])

        st.success(
            f"🏡 Estimated House Price: ₹ {result[0]:,.2f}"
        )

# ==================================================
# ANALYTICS PAGE
# ==================================================
else:

    st.title("📊 Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dataset",
            "Price Distribution",
            "Area vs Price",
            "Heatmap"
        ]
    )

    with tab1:

        st.subheader("Dataset")

        st.dataframe(df)

    with tab2:

        st.subheader("Price Distribution")

        fig, ax = plt.subplots()

        ax.hist(
            df["Price"],
            bins=8
        )

        st.pyplot(fig)

    with tab3:

        st.subheader("Area vs House Price")

        fig, ax = plt.subplots()

        ax.scatter(
            df["SquareFeet"],
            df["Price"]
        )

        ax.set_xlabel("Square Feet")
        ax.set_ylabel("Price")

        st.pyplot(fig)

    with tab4:

        st.subheader("Feature Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            df.corr(),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

st.markdown("---")
st.caption(
    "Developed for SkillCraft Technology ML Internship"
)
