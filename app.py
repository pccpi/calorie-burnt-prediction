import streamlit as st
import numpy as np
import joblib

# Загрузка модели и масштабатора
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Calories Burnt Predictor", layout="centered")
st.title("🔥 Калькулятор сожжённых калорий")

# Пол
gender = st.radio("Пол", ["Мужской", "Женский"])
gender_code = 0 if gender == "Мужской" else 1

# Возраст
st.subheader("Возраст")
age = st.number_input("Введите возраст", min_value=1, max_value=130, value=25)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>1</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>130</b></div>", unsafe_allow_html=True)
st.progress((age - 1) / (130 - 1))

# Рост
st.subheader("Рост (см)")
height = st.number_input("Введите рост", min_value=60, max_value=250, value=170)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>60</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>250</b></div>", unsafe_allow_html=True)
st.progress((height - 60) / (250 - 60))

# Вес
st.subheader("Вес (кг)")
weight = st.number_input("Введите вес", min_value=30, max_value=200, value=70)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>30</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>200</b></div>", unsafe_allow_html=True)
st.progress((weight - 30) / (200 - 30))

# Длительность тренировки
st.subheader("Длительность тренировки (мин)")
duration = st.number_input("Введите длительность", min_value=1, max_value=300, value=30)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>1</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>300</b></div>", unsafe_allow_html=True)
st.progress((duration - 1) / (300 - 1))

# Пульс
st.subheader("Пульс (уд/мин)")
heart_rate = st.number_input("Введите пульс", min_value=50, max_value=220, value=120)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>50</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>220</b></div>", unsafe_allow_html=True)
st.progress((heart_rate - 50) / (220 - 50))

# Температура
st.subheader("Температура тела (°C)")
body_temp = st.number_input("Введите температуру", min_value=35.0, max_value=40.0, value=37.0, step=0.1)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'><b>35.0</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align: right;'><b>40.0</b></div>", unsafe_allow_html=True)
st.progress((body_temp - 35.0) / (40.0 - 35.0))

# Подготовка и предсказание
input_data = np.array([[gender_code, age, height, weight, duration, heart_rate, body_temp]])
scaled_input = scaler.transform(input_data)

if st.button("Рассчитать"):
    calories = model.predict(scaled_input)[0]
    st.success(f"Вы сожгли примерно: {calories:.2f} ккал")streamlit run app.py
