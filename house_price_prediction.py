import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read dataset
df = pd.read_csv("dataset.csv")

# Features and target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[2000, 3, 2]])

print("Predicted House Price:", prediction[0])

# Graph
plt.scatter(df['SquareFeet'], df['Price'])
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()