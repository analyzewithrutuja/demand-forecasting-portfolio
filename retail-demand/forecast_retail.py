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
df = pd.read_csv("online_retail_combined.csv", parse_dates=["InvoiceDate"])

# Remove cancellations (Invoice starting with 'C') and non-positive quantity/price (returns, adjustments, fees)
df = df[~df["Invoice"].str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

print("Cleaned rows:", len(df))
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

# ---------- Build daily demand series (total units sold per day, all products/countries) ----------
daily = df.set_index("InvoiceDate")["Quantity"].resample("D").sum()
# Drop leading/trailing near-zero days (partial weeks at data boundaries) and fill gaps
daily = daily.asfreq("D").fillna(0)
daily = daily[(daily.index >= "2010-01-01") & (daily.index <= "2011-11-30")]  # trim partial boundary months

print("\nDaily series length:", len(daily))
print(daily.describe())

# ---------- Train/test split (last 14 days held out -- standard demand-planning horizon) ----------
horizon = 14
train, test = daily.iloc[:-horizon], daily.iloc[-horizon:]
print(f"Train: {len(train)} days | Test: {len(test)} days")

# ---------- Econometric model: SARIMA (weekly seasonality) ----------
sarima = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,7),
                  enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = sarima.fit(disp=False)
sarima_fc = sarima_fit.forecast(steps=horizon)

# ---------- Econometric baseline: Holt-Winters ----------
hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7)
hw_fit = hw.fit()
hw_fc = hw_fit.forecast(horizon)

# ---------- ML model: XGBoost ----------
def make_features(series, lags=(1,2,3,7,14,21,28), roll_windows=(7,14)):
    d = pd.DataFrame({"y": series})
    for lag in lags:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in roll_windows:
        d[f"roll_mean_{w}"] = d["y"].shift(1).rolling(w).mean()
        d[f"roll_std_{w}"] = d["y"].shift(1).rolling(w).std()
    d["dow"] = d.index.dayofweek
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    d["month"] = d.index.month
    d["day"] = d.index.day
    return d

# Direct multi-step forecasting: train one model per horizon step, each predicting y[t+h] from
# features anchored at t (using only real historical data, never a prior model's own prediction).
# This avoids the error-compounding that recursive multi-step forecasting suffers on volatile series.
full_feat = make_features(daily).drop(columns="y")
feat_train_all = full_feat.iloc[:-horizon]

xgb_preds = []
importances_by_h = {}
for h in range(1, horizon + 1):
    target = daily.shift(-h).iloc[:-horizon]
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
    # This series has real zero-demand days (Saturdays -- the business is closed), which makes
    # standard MAPE blow up on division-by-near-zero. Use WAPE (aggregate weighted error), the
    # standard robust metric for intermittent/zero-inflated demand data.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    print(f"{name:12s} MAE={mae:9.1f}  RMSE={rmse:9.1f}  WAPE={wape:6.2f}%")

print(f"\n--- Forecast accuracy on held-out {horizon} days ---")
# Seasonal-naive: forecast = value from the same weekday one week earlier (standard baseline for weekly-seasonal demand)
naive_fc = pd.Series(daily.reindex(test.index - pd.Timedelta(days=7)).values, index=test.index)
metrics(test, naive_fc, "SeasonalNaive")
metrics(test, hw_fc, "Holt-Winters")
metrics(test, sarima_fc, "SARIMA")
metrics(test, xgb_fc, "XGBoost")

print("\nTop XGBoost features (1-day-ahead model):")
print(importances_by_h[1].sort_values(ascending=False).head(8))

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11,5))
train.iloc[-60:].plot(ax=ax, label="History", color="black")
test.plot(ax=ax, label="Actual", color="black", linewidth=2)
sarima_fc.plot(ax=ax, label="SARIMA", linestyle="--")
hw_fc.plot(ax=ax, label="Holt-Winters", linestyle="--")
xgb_fc.plot(ax=ax, label="XGBoost", linestyle="--")
ax.set_title(f"Online Retail — {horizon}-Day Daily Demand Forecast (Units Sold)")
ax.set_ylabel("Units Sold per Day")
ax.legend()
plt.tight_layout()
plt.savefig("forecast_comparison_retail.png", dpi=130)
print("\nSaved forecast_comparison_retail.png")
