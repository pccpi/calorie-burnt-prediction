import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib
import json

# Загрузка данных
df1 = pd.read_csv("exercise.csv")
df2 = pd.read_csv("calories.csv")
df = pd.merge(df1, df2, on="User_ID")
df.replace({'male': 0, 'female': 1}, inplace=True)
df.drop(['User_ID'], axis=1, inplace=True)

# Подготовка
X = df.drop('Calories', axis=1)
y = df['Calories']

# Сохраняем границы признаков
bounds = {
    col: {
        'min': float(X[col].min()),
        'max': float(X[col].max())
    } for col in X.columns
}
with open("feature_bounds.json", "w") as f:
    json.dump(bounds, f)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение моделей
xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_scaled, y)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y)

# Сохранение
joblib.dump(xgb_model, "model_xgb.pkl")
joblib.dump(linear_model, "model_linear.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Модель, scaler и границы признаков сохранены.")
