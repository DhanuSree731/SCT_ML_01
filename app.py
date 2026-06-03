import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Smart House Price Analyzer",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
div[data-testid="metric-container"]{
    background-color:#ffffff;
    border:1px solid #dddddd;
    padding:15px;
    border-radius:12px;
}
.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD DATASET
# ---------------------------------
df = pd.read_csv("dataset.csv")

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = df[
    [
        "SquareFeet",
        "Bedrooms",
        "Bathrooms",
        "Parking",
        "RoadType",
        "CommunityType"
    ]
]

y = df["Price"]

# ---------------------------------
# TRAIN MODEL
# ---------------------------------
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

# ---------------------------------
# TITLE
# ---------------------------------
st.title("🏠 Smart House Price Analyzer")
st.write("Machine Learning Based House Price Prediction Dashboard")

# ---------------------------------
# IMAGE
# ---------------------------------
try:
    image = Image.open("house.jpg")
    st.image(image, use_container_width=True)
except:
    pass

# ---------------------------------
# TOP METRICS
# ---------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Properties",
        len(df)
    )

with c2:
    st.metric(
        "Average Price",
        f"₹{int(df['Price'].mean()):,}"
    )

with c3:
    st.metric(
        "Model Accuracy",
        f"{r2*100:.2f}%"
    )

# ---------------------------------
# SIDEBAR
# ---------------------------------
st.sidebar.header("Property Details")

sqft = st.sidebar.slider(
    "Square Feet",
    500,
    5000,
    1500
)

bed = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bath = st.sidebar.slider(
    "Bathrooms",
    1,
    10,
    2
)

parking = st.sidebar.slider(
    "Parking Slots",
    0,
    5,
    1
)

road = st.sidebar.selectbox(
    "Road Type",
    ["Off Road", "Main Road"]
)

community = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "Gated Community"]
)

road_value = 1 if road == "Main Road" else 0
community_value = 1 if community == "Gated Community" else 0

# ---------------------------------
# HISTORY
# ---------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.sidebar.button("Predict House Price"):

    prediction = model.predict([[
        sqft,
        bed,
        bath,
        parking,
        road_value,
        community_value
    ]])

    st.success(
        f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}"
    )

    report = pd.DataFrame({
        "SquareFeet": [sqft],
        "Bedrooms": [bed],
        "Bathrooms": [bath],
        "Parking": [parking],
        "RoadType": [road],
        "CommunityType": [community],
        "PredictedPrice": [prediction[0]],
        "Date": [datetime.now()]
    })

    st.download_button(
        "📥 Download Prediction Report",
        report.to_csv(index=False),
        "prediction_report.csv",
        "text/csv"
    )

    st.session_state.history.append({
        "SquareFeet": sqft,
        "Bedrooms": bed,
        "Bathrooms": bath,
        "Parking": parking,
        "RoadType": road,
        "CommunityType": community,
        "PredictedPrice": round(prediction[0], 2)
    })

# ---------------------------------
# TABS
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Analytics",
    "🔥 Heatmap",
    "📊 Dataset",
    "📋 History"
])

# ---------------------------------
# ANALYTICS TAB
# ---------------------------------
with tab1:

    st.subheader("Area vs House Price")

    fig, ax = plt.subplots()

    ax.scatter(
        df["SquareFeet"],
        df["Price"]
    )

    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    st.subheader("Price Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["Price"],
        bins=5
    )

    st.pyplot(fig)

    st.subheader("Bedrooms Impact")

    fig, ax = plt.subplots()

    ax.bar(
        df["Bedrooms"],
        df["Price"]
    )

    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# ---------------------------------
# HEATMAP TAB
# ---------------------------------
with tab2:

    st.subheader("Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------
# DATASET TAB
# ---------------------------------
with tab3:

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.write(
        df.describe()
    )

# ---------------------------------
# HISTORY TAB
# ---------------------------------
with tab4:

    st.subheader("Prediction History")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

    else:
        st.info("No predictions made yet.")
