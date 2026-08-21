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

# ---------- Load & clean ----------
df = pd.read_csv("household_power_consumption.txt", sep=";", na_values="?",
                  parse_dates={"datetime": ["Date", "Time"]}, dayfirst=True,
                  low_memory=False)
df = df.dropna(subset=["Global_active_power"])
df = df.set_index("datetime")["Global_active_power"]

print("Rows:", len(df))
print("Range:", df.index.min(), "to", df.index.max())

# ---------- Aggregate to hourly mean demand (kW) ----------
hourly = df.resample("1h").mean().dropna()
print("\nHourly points:", len(hourly))
print(hourly.describe())

# Use the most recent ~90 days for tractable model fitting (full series is ~4 years)
hourly = hourly.iloc[-90*24:]
hourly = hourly.asfreq("h").interpolate()  # slicing can drop the explicit freq (breaks SARIMA/HW forecast indexing) and reintroduce gaps

# ---------- Train/test split (last 48 hours held out) ----------
horizon = 48
train, test = hourly.iloc[:-horizon].asfreq("h"), hourly.iloc[-horizon:].asfreq("h")
print(f"Train: {len(train)} pts | Test: {len(test)} pts")

# ---------- Econometric model: SARIMA (daily seasonality) ----------
sarima = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,24),
                  enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima.fit(disp=False)
sarima_fc = sarima_fit.forecast(steps=horizon)

# ---------- Econometric baseline: Holt-Winters ----------
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=24)
hw_fit = hw.fit()
hw_fc = hw_fit.forecast(horizon)

# ---------- ML model: XGBoost, direct multi-step (lesson learned from Project 2) ----------
def make_features(series, lags=(1,2,3,6,12,24,48,168), roll_windows=(6,24)):
    d = pd.DataFrame({"y": series})
    for lag in lags:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["y"].shift(1).rolling(w).std()
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    return d

full_feat = make_features(hourly).drop(columns="y")
feat_train_all = full_feat.iloc[:-horizon]

xgb_preds = []
importances_by_h = {}
for h in range(1, horizon + 1):
    target = hourly.shift(-h).iloc[:-horizon]
    combined = feat_train_all.join(target.rename("target")).dropna()
    X_h, y_h = combined.drop(columns="target"), combined["target"]
    model_h = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_h.fit(X_h, y_h)
    origin_feat = full_feat.loc[[train.index[-1]]]
    pred = max(model_h.predict(origin_feat)[0], 0)
    xgb_preds.append(pred)
    if h == 1:
        importances_by_h[h] = pd.Series(model_h.feature_importances_, index=X_h.columns)

xgb_fc = pd.Series(xgb_preds, index=test.index)

# ---------- Evaluate ----------
def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    print(f"{name:12s} MAE={mae:6.3f}  RMSE={rmse:6.3f}  WAPE={wape:6.2f}%")

print(f"\n--- Forecast accuracy on held-out {horizon} hours ---")
seasonal_naive_fc = pd.Series(hourly.reindex(test.index - pd.Timedelta(hours=24)).values, index=test.index)
metrics(test, seasonal_naive_fc, "SeasonalNaive")
metrics(test, hw_fc, "Holt-Winters")
metrics(test, sarima_fc, "SARIMA")
metrics(test, xgb_fc, "XGBoost")

print("\nTop XGBoost features (1-hour-ahead model):")
print(importances_by_h[1].sort_values(ascending=False).head(8))

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11,5))
train.iloc[-96:].plot(ax=ax, label="History", color="black")
test.plot(ax=ax, label="Actual", color="black", linewidth=2)
sarima_fc.plot(ax=ax, label="SARIMA", linestyle="--")
hw_fc.plot(ax=ax, label="Holt-Winters", linestyle="--")
xgb_fc.plot(ax=ax, label="XGBoost", linestyle="--")
ax.set_title(f"Household Energy Demand — {horizon}-Hour Forecast (kW)")
ax.set_ylabel("Global Active Power (kW)")
ax.legend()
plt.tight_layout()
plt.savefig("forecast_comparison_energy.png", dpi=130)
print("\nSaved forecast_comparison_energy.png")
