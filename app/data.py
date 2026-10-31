"""Load the committed project outputs for the dashboard.

Reads only files already in the repo (Step 1 panel, Step 2 and Step 3 results),
so the app runs offline and needs no API keys except for the assistant.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/monthly_panel.csv"
DICTIONARY = ROOT / "data/processed/data_dictionary.csv"
STEP2_FORECASTS = ROOT / "results/forecasts.csv"
STEP2_METRICS = ROOT / "results/metrics.csv"
STEP3_DIR = ROOT / "modeling/results/experiments/step3_ml"

# Step 3 picked one ML model per horizon on 2016-19 validation (recommended_models.json).
# The dashboard shows that pick as a single "ML (selected)" model.
ML_SELECTED = {1: "ridge_regional", 3: "ridge_regional", 6: "gradient_boosting"}

MODELS = ["combo_s", "arx", "var_s", "ml", "arima", "arx_s", "naive"]
MODEL_LABELS = {
    "combo_s": "Combo (recommended)",
    "arx": "ARX (indicators)",
    "var_s": "VAR (NY + neighbors)",
    "ml": "ML (validation-selected)",
    "arima": "ARIMA",
    "arx_s": "ARX shrunk",
    "naive": "Naive (no change)",
}
MODEL_DESCRIPTIONS = {
    "combo_s": "Average of naive, ARIMA and ARX, shrunk toward naive",
    "arx": "Regression on NY claims, NY payrolls and US unemployment",
    "var_s": "NY, NJ, CT and PA unemployment modeled together, shrunk",
    "ml": "Ridge with regional data (1 and 3 months), gradient boosting (6 months)",
    "arima": "ARIMA(0,1,2): the rate's own recent movements only",
    "arx_s": "ARX pulled toward naive",
    "naive": "Latest published rate, carried forward",
}
# Categorical slots in fixed order, so a model keeps its color on every chart.
MODEL_COLORS = {
    "combo_s": "#2a78d6",
    "arx": "#eb6834",
    "var_s": "#1baf7a",
    "ml": "#eda100",
    "arima": "#e87ba4",
    "arx_s": "#008300",
    "naive": "#898781",
}
PERIODS = {
    "post_covid": "Turning point · 2022–Aug 2026",
    "calm_2023on": "Calm · 2023–Aug 2026",
    "covid": "COVID shock · 2020–21",
    "validation": "Validation · 2016–19",
}
PERIOD_WINDOWS = {
    "post_covid": ("2021-07-01", "2026-09-01"),
    "calm_2023on": ("2022-10-01", "2026-09-01"),
    "covid": ("2019-10-01", "2022-01-01"),
    "validation": ("2015-10-01", "2020-01-01"),
}


@st.cache_data
def load_panel():
    return pd.read_csv(PANEL, parse_dates=["date"])


@st.cache_data
def load_dictionary():
    return pd.read_csv(DICTIONARY)


@st.cache_data
def load_forecasts():
    """Step 2 forecasts plus the validation-selected Step 3 ML forecast, one row per forecast."""
    columns = ["date", "model", "horizon", "yhat", "lower", "upper", "origin", "actual", "period"]
    step2 = pd.read_csv(STEP2_FORECASTS, parse_dates=["date", "origin"])[columns]

    step3 = pd.read_csv(STEP3_DIR / "forecasts.csv", parse_dates=["date", "origin"])
    step3 = step3[step3["status"] == "ok"]
    picked = step3["model"] == step3["horizon"].map(ML_SELECTED)
    ml = step3[picked].assign(model="ml")[columns]
    return pd.concat([step2, ml], ignore_index=True)


@st.cache_data
def load_metrics():
    """Error vs. naive on the common sample every model was scored on (Step 3 metrics file)."""
    metrics = pd.read_csv(STEP3_DIR / "metrics.csv")
    metrics = metrics[metrics["sample"] == "common_all_models"]
    picked = metrics["model"] == metrics["horizon"].map(ML_SELECTED)
    ml = metrics[picked].assign(model="ml")
    metrics = pd.concat([metrics[metrics["model"].isin(MODELS)], ml], ignore_index=True)

    # Diebold-Mariano p-values vs. naive come from the Step 2 metrics (Step 2 models only).
    dm = pd.read_csv(STEP2_METRICS)[["view", "horizon", "model", "dm_p_vs_naive"]]
    return metrics.merge(dm, on=["view", "horizon", "model"], how="left")


@st.cache_data
def load_ablation():
    return pd.read_csv(STEP3_DIR / "feature_ablation.csv")


def read_text(relative_path):
    return (ROOT / relative_path).read_text()


@st.cache_data
def load_taxable_sales():
    return pd.read_csv(ROOT / "data/processed/quarterly_taxable_sales.csv", parse_dates=["date"])
