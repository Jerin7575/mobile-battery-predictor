import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.title("📱 Mobile Battery Usage Predictor")
st.write("Predict battery drain based on phone usage")


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("smartphone_battery_drain_dataset.csv")
    return data

data = load_data()


# -----------------------------
# DATA CLEANING
# -----------------------------

data = data.drop_duplicates()
data = data.dropna()

numeric_cols = data.select_dtypes(include=np.number).columns

for col in numeric_cols:

    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data = data[(data[col] >= lower) & (data[col] <= upper)]


# -----------------------------
# FEATURE CREATION
# -----------------------------

data["Internet_Usage_Time"] = data["Screen_On_Time_min"]

if "Number_of_Apps_Running" not in data.columns:
    data["Number_of_Apps_Running"] = 5


X = data[
    [
        "Brightness_Level_%",
        "Number_of_Apps_Running",
        "Internet_Usage_Time"
    ]
]

y = data["Battery_Drop_Per_Hour"]


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# MODEL TRAINING
# -----------------------------

@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


model = train_model(X_train, y_train)

y_pred = model.predict(X_test)


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.subheader("Model Performance")

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.write(f"Model Accuracy (R²): {r2*100:.2f}%")
st.write("Mean Absolute Error:", mae)
st.write("Mean Squared Error:", mse)


# -----------------------------
# DATA PREVIEW
# -----------------------------

if st.checkbox("Show dataset preview"):
    st.dataframe(data.head())


# -----------------------------
# VISUALIZATION
# -----------------------------

st.subheader("Battery Drain Distribution")

st.bar_chart(data["Battery_Drop_Per_Hour"])


# -----------------------------
# USER INPUT
# -----------------------------

st.subheader("Enter Phone Usage Details")

brightness = st.slider(
    "Screen Brightness (%)",
    0,
    100,
    50
)

apps = st.slider(
    "Number of Apps Running",
    1,
    20,
    5
)

internet = st.slider(
    "Internet Usage Time (minutes)",
    0,
    300,
    60
)


# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Battery Usage"):

    input_data = pd.DataFrame(
        [[brightness, apps, internet]],
        columns=[
            "Brightness_Level_%",
            "Number_of_Apps_Running",
            "Internet_Usage_Time"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Battery Drop Per Hour: {prediction:.2f}%")

    if prediction > 0:
        battery_duration = 100 / prediction
        st.info(f"Estimated Battery Duration: {battery_duration:.2f} hours")
    else:
        st.warning("Prediction too small to estimate battery duration.")