import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Загрузка данных
df1 = pd.read_csv("exercise.csv")
df2 = pd.read_csv("calories.csv")

# Объединение по User_ID
df = pd.merge(df1, df2, on="User_ID")

# Преобразование пола
df.replace({'male': 0, 'female': 1}, inplace=True)

# Удаляем только User_ID (всё остальное используем!)
df.drop(['User_ID'], axis=1, inplace=True)

# Разделение признаков и цели
X = df.drop('Calories', axis=1)
y = df['Calories']

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение модели
model = XGBRegressor()
model.fit(X_scaled, y)

# Сохранение модели и scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Модель и scaler сохранены.")
