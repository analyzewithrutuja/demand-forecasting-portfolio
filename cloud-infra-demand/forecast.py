import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ---------- Load ----------
df = pd.read_csv("elb_request_count.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp").asfreq("5min")
df["value"] = df["value"].interpolate()

print("Rows:", len(df))
print("Range:", df.index.min(), "to", df.index.max())
print(df["value"].describe())

# Resample to hourly total request volume — the actual "demand" signal for capacity planning
hourly = df["value"].resample("1h").sum().dropna()
print("\nHourly points:", len(hourly))

# ---------- Train/test split (last 48 hours held out) ----------
horizon = 48
train, test = hourly.iloc[:-horizon], hourly.iloc[-horizon:]
print(f"Train: {len(train)} pts | Test: {len(test)} pts")

# ---------- Econometric model: SARIMA ----------
# Daily seasonality at hourly resolution -> period 24
sarima = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,24),
                  enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima.fit(disp=False)
sarima_fc = sarima_fit.forecast(steps=horizon)

# ---------- Econometric baseline: Holt-Winters ----------
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=24)
hw_fit = hw.fit()
hw_fc = hw_fit.forecast(horizon)

# ---------- ML model: XGBoost with lag/rolling/calendar features ----------
def make_features(series, lags=(1,2,3,6,12,24,48), roll_windows=(6,24)):
    d = pd.DataFrame({"y": series})
    for lag in lags:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["y"].shift(1).rolling(w).std()
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    return d

full_feat = make_features(hourly)
train_feat = full_feat.iloc[:-horizon].dropna()
X_train, y_train = train_feat.drop(columns="y"), train_feat["y"]

xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Recursive multi-step forecast for XGBoost
history = hourly.iloc[:-horizon].copy()
xgb_preds = []
for step in range(horizon):
    feat_row = make_features(history).iloc[[-1]].drop(columns="y")
    pred = xgb_model.predict(feat_row)[0]
    xgb_preds.append(pred)
    next_ts = history.index[-1] + pd.Timedelta(hours=1)
    history.loc[next_ts] = pred
xgb_fc = pd.Series(xgb_preds, index=test.index)

# ---------- Evaluate ----------
def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None))) * 100
    print(f"{name:12s} MAE={mae:6.3f}  RMSE={rmse:6.3f}  MAPE={mape:6.2f}%")
    return mae, rmse, mape

print("\n--- Forecast accuracy on held-out 48 hours ---")
naive_fc = pd.Series([train.iloc[-1]] * horizon, index=test.index)
metrics(test, naive_fc, "Naive")
metrics(test, hw_fc, "Holt-Winters")
metrics(test, sarima_fc, "SARIMA")
metrics(test, xgb_fc, "XGBoost")

# ---------- Feature importance ----------
importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop XGBoost features:")
print(importances.head(8))

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11,5))
train.iloc[-96:].plot(ax=ax, label="History", color="black")
test.plot(ax=ax, label="Actual", color="black", linewidth=2)
sarima_fc.plot(ax=ax, label="SARIMA", linestyle="--")
hw_fc.plot(ax=ax, label="Holt-Winters", linestyle="--")
xgb_fc.plot(ax=ax, label="XGBoost", linestyle="--")
ax.set_title("ELB Request Volume — 48-Hour Infrastructure Demand Forecast")
ax.set_ylabel("Requests per Hour")
ax.legend()
plt.tight_layout()
plt.savefig("forecast_comparison.png", dpi=130)
print("\nSaved forecast_comparison.png")
