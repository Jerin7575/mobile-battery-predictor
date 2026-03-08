import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(
    page_title="Mobile Battery Drain Predictor",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Mobile Battery Drain Predictor")

st.write("Predict smartphone battery drain based on usage patterns.")


# Load dataset
data = pd.read_csv("smartphone_battery_drain_dataset.csv")

data = data.drop_duplicates()
data = data.dropna()


# Features
features = [
    "Screen_On_Time_min",
    "Brightness_Level_%",
    "CPU_Usage_%",
    "RAM_Usage_MB"
]

target = "Battery_Drop_Per_Hour"

X = data[features]
y = data[target]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("R² Score", f"{r2*100:.2f}%")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("MSE", f"{mse:.2f}")


# Input section
st.subheader("Enter Phone Usage")

brightness = st.slider("Brightness (%)",0,100,50)
cpu = st.slider("CPU Usage (%)",0,100,30)
ram = st.slider("RAM Usage (MB)",500,8000,2000)
screen = st.slider("Screen Time (minutes)",0,300,60)


if st.button("Predict Battery Drain"):

    input_data = pd.DataFrame(
        [[screen, brightness, cpu, ram]],
        columns=[
            "Screen_On_Time_min",
            "Brightness_Level_%",
            "CPU_Usage_%",
            "RAM_Usage_MB"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Battery Drop Per Hour: {prediction:.2f}%")

    if prediction > 0:
        duration = 100 / prediction
        st.info(f"Estimated Battery Duration: {duration:.2f} hours")
