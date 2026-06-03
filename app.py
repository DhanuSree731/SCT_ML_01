import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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
X = df[[
    'SquareFeet',
    'Bedrooms',
    'Bathrooms',
    'Parking',
    'LocationRating',
    'FurnishingType',
    'PropertyAge'
]]

y = df['Price']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

accuracy = r2_score(y, pred)
mse = mean_squared_error(y, pred)

# --------------------------------------------------
# SIDEBAR
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
# DASHBOARD
# ==================================================
if page == "Dashboard":

    st.title("🏠 Smart House Price Analyzer")
    st.markdown(
        "### Machine Learning Based House Price Prediction Dashboard"
    )

    try:
        image = Image.open("house.jpg")

        c1, c2, c3 = st.columns([1, 2, 1])

        with c2:
            st.image(image, width=450)

    except:
        pass

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Properties",
        len(df)
    )

    col2.metric(
        "Average Price",
        f"₹ {int(df['Price'].mean()):,}"
    )

    col3.metric(
        "Maximum Price",
        f"₹ {int(df['Price'].max()):,}"
    )

    col4.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("📋 Project Overview")

        st.info("""
        Smart House Price Analyzer predicts
        property prices using Linear Regression.

        Features Used:

        • Square Feet
        • Bedrooms
        • Bathrooms
        • Parking
        • Location Rating
        • Furnishing Type
        • Property Age
        """)

    with right:

        st.subheader("🛠 Technology Stack")

        st.success("""
        • Python
        • Pandas
        • Matplotlib
        • Seaborn
        • Scikit-Learn
        • Streamlit
        """)

# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "Prediction":

    st.title("🏡 House Price Prediction")

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

        parking = st.slider(
            "Parking Slots",
            0,
            5,
            1
        )

    with col2:

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

        furnishing = st.selectbox(
            "Furnishing Type",
            [
                "Unfurnished",
                "Semi Furnished",
                "Fully Furnished"
            ]
        )

        property_age = st.slider(
            "Property Age (Years)",
            0,
            30,
            5
        )

    road_score = 8 if road == "Road Side" else 5

    property_score = {
        "Apartment": 6,
        "Gated Community": 8,
        "Villa": 10
    }

    furnishing_score = {
        "Unfurnished": 0,
        "Semi Furnished": 1,
        "Fully Furnished": 2
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
            location_rating,
            furnishing_score[furnishing],
            property_age
        ]])

        st.success(
            f"🏡 Estimated House Price: ₹ {result[0]:,.2f}"
        )

        m1, m2 = st.columns(2)

        m1.metric(
            "R² Score",
            round(accuracy, 4)
        )

        m2.metric(
            "Mean Squared Error",
            round(mse, 2)
        )

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Square Feet",
                "Bedrooms",
                "Bathrooms",
                "Parking",
                "Location Rating",
                "Furnishing Type",
                "Property Age"
            ],
            "Value": [
                sqft,
                bedrooms,
                bathrooms,
                parking,
                round(location_rating, 2),
                furnishing,
                property_age
            ]
        })

        st.dataframe(summary)

        report = pd.DataFrame({
            "SquareFeet": [sqft],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Parking": [parking],
            "LocationRating": [location_rating],
            "FurnishingType": [furnishing],
            "PropertyAge": [property_age],
            "PredictedPrice": [round(result[0], 2)]
        })

        st.download_button(
            "📥 Download Prediction Report",
            report.to_csv(index=False),
            "prediction_report.csv",
            "text/csv"
        )

        st.subheader("📈 Linear Regression Graph")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            df["SquareFeet"],
            df["Price"],
            label="Actual Prices"
        )

        line = model.predict(X)

        ax.plot(
            df["SquareFeet"],
            line,
            linewidth=2,
            label="Regression Line"
        )

        ax.scatter(
            sqft,
            result[0],
            marker="X",
            s=250,
            label="Predicted House"
        )

        ax.set_xlabel("Square Feet")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

# ==================================================
# ANALYTICS PAGE
# ==================================================
else:

    st.title("📊 Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Price Distribution",
        "Area vs Price",
        "Heatmap"
    ])

    with tab1:
        st.dataframe(df)
        st.write(df.describe())

    with tab2:
        fig, ax = plt.subplots()
        ax.hist(df["Price"], bins=8)
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots()
        ax.scatter(df["SquareFeet"], df["Price"])
        ax.set_xlabel("Square Feet")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

st.markdown("---")

st.caption(
    "Developed by Dhanu Sree | SkillCraft Technology ML Internship"
)
