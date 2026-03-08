# 🔋 Mobile Battery Drain Predictor

A Machine Learning web application that predicts smartphone battery drain based on user activity such as screen time, brightness, CPU usage, and RAM usage.

This project is built using **Python**, **Streamlit**, and **Scikit-learn**.

---

# 🚀 Features

- Predict battery drain per hour
- Estimate total battery life
- Interactive user interface
- Machine learning model using Random Forest
- Dataset preview and visualization

---

# 🧠 Machine Learning Model

The application uses a **Random Forest Regressor** to estimate how quickly a smartphone battery drains.

### Input Features

- Screen On Time (minutes)
- Brightness Level (%)
- CPU Usage (%)
- RAM Usage (MB)

### Output

- Battery Drop Per Hour (%)
- Estimated Battery Duration (hours)

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

# 📂 Project Structure

mobile-battery-predictor
│
├── app.py
├── smartphone_battery_drain_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore  

---

# ⚙️ Installation and Setup

Follow these steps to run the project locally.

## 1️⃣ Clone the repository


git clone https://github.com/jerin7575/mobile-battery-predictor.git⁠

## 2️⃣ Navigate into the project folder

cd mobile-battery-predictor

## 3️⃣ Install the required libraries

pip install -r requirements.txt

## 4️⃣ Run the Streamlit application

streamlit run app.py

## 5️⃣ Open the web application

After running the command above, open your browser and go to: http://localhost:8501⁠



---

## Single Command

git clone https://github.com/jerin7575/mobile-battery-predictor.git && cd mobile-battery-predictor && pip install -r requirements.txt && streamlit run app.py


---

# 📊 How to Use the App

1. Adjust the sliders for:
   - Brightness Level
   - CPU Usage
   - RAM Usage
   - Screen On Time

2. Click **Predict Battery Drain**

3. The app will display:
   - Estimated battery drop per hour
   - Estimated battery duration

---

# 🌐 Deployment

This project can be deployed using **Streamlit Community Cloud**.

Steps:

1. Upload project to GitHub
2. Visit https://share.streamlit.io
3. Connect your GitHub account
4. Select your repository
5. Deploy `app.py`

---

# 📌 Example Use Case

This application demonstrates how machine learning can be used to analyze smartphone usage patterns and predict battery consumption.

---

# 📜 License

This project is open-source and free to use.


