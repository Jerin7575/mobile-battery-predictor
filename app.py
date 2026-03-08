import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Mobile Battery Drain Predictor",
    page_icon="🔋",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
}

.stButton>button{
background-color:#4CAF50;
color:white;
border-radius:10px;
height:50px;
width:100%;
font-size:16px;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">🔋 Mobile Battery Drain Predictor</p>', unsafe_allow_html=True)

st.write("Predict smartphone battery drain based on usage patterns.")


# --------------------------------
# LOAD DATA
# --------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("smartphone_battery_drain_dataset.csv")

    return data


data = load_data()

data = data.drop_duplicates()
data = data.dropna()


# --------------------------------
# FEATURE SELECTION
# --------------------------------

features = [
    "Screen_On_Time_min",
    "Brightness_Level_%",
    "CPU_Usage_%",
    "RAM_Usage_MB"
]

target = "Battery_Drop_Per_Hour"


X = data[features]
y = data[target]


# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# --------------------------------
# TRAIN MODEL
# --------------------------------

@st.cache_resource
def train_model():

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


model = train_model()


# --------------------------------
# MODEL EVALUATION
# --------------------------------

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy (R²)", f"{r2*100:.2f}%")
col2.metric("Mean Absolute Error", f"{mae:.2f}")
col3.metric("Mean Squared Error", f"{mse:.2f}")


# --------------------------------
# DATA PREVIEW
# --------------------------------

st.subheader("📂 Dataset")

if st.checkbox("Show Dataset Preview"):

    st.dataframe(data.head())


# --------------------------------
# VISUALIZATION
# --------------------------------

st.subheader("📉 Battery Drain Trend")

st.line_chart(data["Battery_Drop_Per_Hour"])


# --------------------------------
# USER INPUT FORM
# --------------------------------

st.subheader("📲 Enter Phone Usage Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        brightness = st.slider(
            "🔆 Brightness Level (%)",
            0,
            100,
            50
        )

        cpu = st.slider(
            "🧠 CPU Usage (%)",
            0,
            100,
            30
        )

    with col2:

        ram = st.slider(
            "💾 RAM Usage (MB)",
            500,
            8000,
            2000
        )

        screen_time = st.slider(
            "⏱ Screen On Time (minutes)",
            0,
            300,
            60
        )

    submit_button = st.form_submit_button("🔋 Predict Battery Drain")


# --------------------------------
# PREDICTION
# --------------------------------

if submit_button:

    input_data = pd.DataFrame(

        [[screen_time, brightness, cpu, ram]],

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

        battery_duration = 100 / prediction

        st.info(f"Estimated Battery Duration: {battery_duration:.2f} hours")

    else:

        st.warning("Prediction too small to estimate battery duration.")
