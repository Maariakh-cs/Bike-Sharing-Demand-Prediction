import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os, json

# ── 1. Generate synthetic dataset (mirrors Kaggle bike sharing structure) ────
np.random.seed(42)
n = 17379  # Same size as UCI bike sharing dataset

hours      = np.random.randint(0, 24, n)
months     = np.random.randint(1, 13, n)
weekdays   = np.random.randint(0, 7, n)
seasons    = np.random.randint(1, 5, n)
holidays   = np.random.choice([0, 1], n, p=[0.97, 0.03])
workingday = ((weekdays >= 1) & (weekdays <= 5) & (holidays == 0)).astype(int)
weather    = np.random.choice([1, 2, 3, 4], n, p=[0.46, 0.39, 0.14, 0.01])
temp       = np.random.uniform(0.02, 1.0, n)
atemp      = temp + np.random.normal(0, 0.05, n)
humidity   = np.random.uniform(0.0, 1.0, n)
windspeed  = np.random.uniform(0.0, 0.85, n)
years      = np.random.choice([0, 1], n)

# Realistic demand formula 
base = (
    50
    + 200 * temp
    - 80  * humidity
    - 30  * windspeed
    + 80  * (weather == 1).astype(int)
    - 50  * (weather == 3).astype(int)
    - 120 * (weather == 4).astype(int)
    + 60  * workingday
    + 80  * years
)

# Hour-of-day pattern (commute peaks)
hour_effect = np.zeros(n)
for i, h in enumerate(hours):
    if h in [7, 8]:     hour_effect[i] = 180
    elif h in [17, 18]: hour_effect[i] = 200
    elif h in [12, 13]: hour_effect[i] = 80
    elif h in [0, 1, 2, 3, 4]: hour_effect[i] = -40
    else:               hour_effect[i] = 40

# Season effect
season_effect = np.where(seasons == 2, 60,
                np.where(seasons == 3, 80,
                np.where(seasons == 1, -20, 20)))

cnt = np.clip(base + hour_effect + season_effect + np.random.normal(0, 30, n), 1, 977).astype(int)
casual    = np.clip((cnt * np.random.uniform(0.1, 0.4, n)).astype(int), 0, cnt)
registered = cnt - casual

df = pd.DataFrame({
    'season': seasons, 'yr': years, 'mnth': months,
    'hr': hours, 'holiday': holidays, 'weekday': weekdays,
    'workingday': workingday, 'weathersit': weather,
    'temp': np.round(temp, 4), 'atemp': np.round(atemp.clip(0,1), 4),
    'hum': np.round(humidity, 4), 'windspeed': np.round(windspeed, 4),
    'casual': casual, 'registered': registered, 'cnt': cnt
})

print(f"Dataset shape: {df.shape}")
print(f"Target range: {df['cnt'].min()} – {df['cnt'].max()}, mean: {df['cnt'].mean():.1f}")

# ── 2. Feature Engineering 
# Lag features & rolling stats 
df = df.sort_values(['yr', 'mnth', 'weekday', 'hr']).reset_index(drop=True)
df['cnt_lag_1h']    = df['cnt'].shift(1).fillna(df['cnt'].mean())
df['cnt_lag_24h']   = df['cnt'].shift(24).fillna(df['cnt'].mean())
df['cnt_rolling_3h']  = df['cnt'].rolling(3, min_periods=1).mean()
df['cnt_rolling_24h'] = df['cnt'].rolling(24, min_periods=1).mean()

# Peak hour flag
df['is_peak_hour'] = df['hr'].isin([7, 8, 17, 18]).astype(int)
# Weekend flag
df['is_weekend'] = (df['weekday'].isin([0, 6])).astype(int)
# Temp x humidity interaction
df['temp_hum'] = df['temp'] * df['hum']

# ── 3. Train/Test Split 
FEATURES = [
    'season','yr','mnth','hr','holiday','weekday','workingday','weathersit',
    'temp','atemp','hum','windspeed',
    'cnt_lag_1h','cnt_lag_24h','cnt_rolling_3h','cnt_rolling_24h',
    'is_peak_hour','is_weekend','temp_hum'
]

X = df[FEATURES]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 4. Train Models 
models = {
    "Linear Regression":   LinearRegression(),
    "Decision Tree":       DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":       RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
}

results = {}
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

best_model = None
best_r2    = -999

for name, model in models.items():
    Xtr = X_train_s if name == "Linear Regression" else X_train
    Xte = X_test_s  if name == "Linear Regression" else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    preds = np.clip(preds, 0, None)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
    print(f"{name:22s} | MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.4f}")

    if r2 > best_r2:
        best_r2    = r2
        best_model = (name, model, scaler if name == "Linear Regression" else None)

print(f"\nBest model: {best_model[0]}  (R²={best_r2:.4f})")

# ── 5. Feature Importance 
best_name, best_m, best_scaler = best_model
if hasattr(best_m, 'feature_importances_'):
    fi = dict(zip(FEATURES, best_m.feature_importances_.round(4)))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
else:
    fi_sorted = {}

# ── 6. Save pipeline 
os.makedirs("model", exist_ok=True)

joblib.dump(best_m,      "model/best_model.pkl")
joblib.dump(scaler,      "model/scaler.pkl")
joblib.dump(FEATURES,    "model/features.pkl")

meta = {
    "best_model":   best_name,
    "r2_score":     best_r2,
    "features":     FEATURES,
    "model_results": results,
    "feature_importance": fi_sorted,
    "target_stats": {
        "min":  int(df['cnt'].min()),
        "max":  int(df['cnt'].max()),
        "mean": round(float(df['cnt'].mean()), 1)
    }
}
with open("model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel, scaler and metadata saved to model/")
print(json.dumps(results, indent=2))
