# Demand Forecasting Portfolio

Three demand-forecasting projects, each comparing classical econometric time-series models
(SARIMA, Holt-Winters) against a gradient-boosted machine-learning model (XGBoost) on real public
data — infrastructure, retail, and energy demand.

Built to demonstrate the combination of econometric modeling and applied ML that real demand-planning
and capacity-forecasting roles require: not just fitting a model, but diagnosing what breaks
(misleading metrics on zero-inflated data, error compounding in recursive forecasting, silent
frequency bugs) and fixing it correctly.

## Projects

| Project | Domain | Data | Best Model | Error (vs. naive baseline) |
|---|---|---|---|---|
| [`cloud-infra-demand/`](cloud-infra-demand/) | Infrastructure request volume | AWS CloudWatch (NAB benchmark) | Holt-Winters | 36.5% MAPE (vs. 60.1% naive) |
| [`retail-demand/`](retail-demand/) | E-commerce daily units sold | UCI Online Retail II (~1M transactions) | XGBoost (direct multi-step) | 12.3% WAPE (vs. 26.1% naive) |
| [`energy-demand/`](energy-demand/) | Household hourly power (kW) | UCI Household Power Consumption (~4 yrs) | XGBoost (direct multi-step) | 31.9% WAPE (vs. 46.1% naive) |

## Common approach across all three

1. **Econometric models** — SARIMA and Holt-Winters, tuned to each series' seasonality (daily or
   weekly).
2. **ML model** — XGBoost with lag, rolling-statistic, and calendar features. Started recursive,
   switched to **direct multi-step forecasting** (one model per horizon step) after recursive
   forecasting compounded errors badly on volatile series — see the retail-demand README for the
   full diagnosis.
3. **Honest evaluation** — backtested against a proper (seasonal-)naive baseline, using WAPE
   instead of MAPE wherever the series has zero-inflated or near-zero values that would otherwise
   make percentage error meaningless.

## Setup

```bash
pip install -r requirements.txt
```

Each project folder has its own README with data-download instructions (where the raw data is too
large to include directly) and results.
