import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

st.set_page_config(page_title="🔥 Calories Burnt Predictor", layout="centered")

xgb_model = joblib.load("model_xgb.pkl")
linear_model = joblib.load("model_linear.pkl")
scaler = joblib.load("scaler.pkl")

with open("feature_bounds.json", "r") as f:
    bounds = json.load(f)

LABELS = {
    "Русский": {
        "title": "🔥 Калькулятор сожжённых калорий",
        "gender": "Пол",
        "male": "Мужской",
        "female": "Женский",
        "age": "Возраст",
        "height": "Рост (см)",
        "weight": "Вес (кг)",
        "duration": "Длительность тренировки (мин)",
        "heart": "Средний пульс (уд/мин)",
        "temp": "Средняя температура тела (°C)",
        "button": "Рассчитать",
        "burned": "Вы сожгли приблизительно:",
        "model_used": "Использована модель:",
        "range": "Диапазон: от {min:.0f} до {max:.0f}, вероятнее всего — {pred:.0f} {unit}",
        "copy": "📋 Копировать данные и результат",
        "download": "⬇️ Скачать результат (.txt)",
        "unit": "ккал"
    },
    "English": {
        "title": "🔥 Calories Burnt Calculator",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "age": "Age",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "duration": "Workout Duration (min)",
        "heart": "Average Heart Rate (bpm)",
        "temp": "Average Body Temperature (°C)",
        "button": "Calculate",
        "burned": "You have burnt approximately:",
        "model_used": "Model used:",
        "range": "Range: from {min:.0f} to {max:.0f}, most likely — {pred:.0f} {unit}",
        "copy": "📋 Copy input and result",
        "download": "⬇️ Download result (.txt)",
        "unit": "kcal"
    },
    "Română": {
        "title": "🔥 Calculator de Calorii Arse",
        "gender": "Gen",
        "male": "Bărbat",
        "female": "Femeie",
        "age": "Vârstă",
        "height": "Înălțime (cm)",
        "weight": "Greutate (kg)",
        "duration": "Durata antrenamentului (min)",
        "heart": "Puls mediu (bpm)",
        "temp": "Temperatura medie a corpului (°C)",
        "button": "Calculează",
        "burned": "Ați ars aproximativ:",
        "model_used": "Model utilizat:",
        "range": "Interval: de la {min:.0f} la {max:.0f}, cel mai probabil — {pred:.0f} {unit}",
        "copy": "📋 Copiază datele și rezultatul",
        "download": "⬇️ Descarcă rezultatul (.txt)",
        "unit": "kcal"
    }
}
# Предзагрузка языка — по умолчанию русский
default_lang = "Русский"
t = LABELS[default_lang]
st.title(t["title"])

# Выбор языка под заголовком
lang = st.selectbox("🌐 Язык / Language / Limba", ["Русский", "English", "Română"], index=0)
t = LABELS[lang]


gender = st.radio(t["gender"], [t["male"], t["female"]])
gender_code = 0 if gender == t["male"] else 1

def input_with_bar(label, val, minv, maxv, step):
    st.subheader(label)
    value = st.number_input(label, min_value=minv, max_value=maxv, value=val, step=step, key=label)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div style='text-align: left;'>{minv}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align: right;'>{maxv}</div>", unsafe_allow_html=True)
    st.progress((value - minv) / (maxv - minv))
    return value

age = input_with_bar(t["age"], 20, 1, 130, step=1)
height = input_with_bar(t["height"], 188, 60, 250, step=1)
weight = input_with_bar(t["weight"], 72, 30, 200, step=1)
duration = input_with_bar(t["duration"], 52, 1, 300, step=1)
heart_rate = input_with_bar(t["heart"], 110, 50, 220, step=1)
body_temp = input_with_bar(t["temp"], 38.0, 35.0, 40.0, step=0.1)

features = {
    "Gender": gender_code,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Duration": duration,
    "Heart_Rate": heart_rate,
    "Body_Temp": body_temp
}

input_df = pd.DataFrame([features])
scaled_input = scaler.transform(input_df)

deviation_score = 0
for k, v in features.items():
    min_val = bounds[k]['min']
    max_val = bounds[k]['max']
    if v < min_val:
        deviation_score += (min_val - v) / (max_val - min_val)
    elif v > max_val:
        deviation_score += (v - max_val) / (max_val - min_val)

if deviation_score == 0:
    prediction = xgb_model.predict(scaled_input)[0]
    model_used = "XGBoost"
elif deviation_score >= 1.5:
    prediction = linear_model.predict(scaled_input)[0]
    model_used = "Linear Regression"
else:
    alpha = min(deviation_score / 1.5, 1)
    xgb_pred = xgb_model.predict(scaled_input)[0]
    lin_pred = linear_model.predict(scaled_input)[0]
    prediction = (1 - alpha) * xgb_pred + alpha * lin_pred
    model_used = f"Blend: {round(alpha * 100)}% Linear"

if st.button(t["button"]):
    min_val = prediction * 0.93
    max_val = prediction * 1.07
    st.success(f"{t['burned']} {prediction:.2f} {t['unit']}")
    st.caption(t["range"].format(min=min_val, max=max_val, pred=prediction, unit=t["unit"]))

    summary = f"""{t['burned']} {prediction:.2f} {t['unit']}
{t['range'].format(min=min_val, max=max_val, pred=prediction, unit=t["unit"])}

{t['gender']}: {gender}
{t['age']}: {age}
{t['height']}: {height}
{t['weight']}: {weight}
{t['duration']}: {duration}
{t['heart']}: {heart_rate}
{t['temp']}: {body_temp}
"""
    st.text_area(t["copy"], summary, height=250)
    st.download_button(t["download"], summary, file_name="calories_result.txt")

