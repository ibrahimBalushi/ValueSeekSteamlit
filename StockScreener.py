# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import dask.dataframe as dd
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from valueSeek.config import *
from valueSeek.api import *
from valueSeek.processor import *

# ============================================================
# APP TITLE
# ============================================================

st.write("""
# ValueSeek
A Data-Driven Approach to Identifying Investment Opportunities in the Stock Market
""")

# ============================================================
# LOAD DATA
# ============================================================

# Get the security master (company data)
securityMaster = get_security_master()

st.write("## Company Profile Selection")

# ============================================================
# SECTOR SELECTION
# ============================================================

# List of all sectors
sector_list = securityMaster['sector'].dropna().unique().tolist()

# Default sectors pre-selected
DEFAULT_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"
]

# Helper function to convert list of sectors to a comma-separated string
def sectors_to_str(sector_list):
    return ",".join(sector_list)

# Initialize session state for sectors input
if "sectors_input" not in st.session_state:
    st.session_state.sectors_input = list(DEFAULT_SECTORS)

# Layout columns for sector selection
cols = st.columns([5, 1])
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Multi-select box for sector filtering
    SECTORS = st.multiselect(
        "Sector to filter",
        options=sorted(set(sector_list) | set(st.session_state.sectors_input)),
        default=st.session_state.sectors_input,
        placeholder="Choose sectors to include in your search",
        accept_new_options=True,
    )

# ============================================================
# TIME HORIZON SELECTION
# ============================================================

horizon_map = {
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}

# initialize session state for time horizon
if "horizon" not in st.session_state:
    st.session_state.horizon = "5 Years"


with top_left_cell:
    # Pills selector for time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="5 Years",
    )

# ============================================================
# METRIC RULE BUILDER CONFIG
# ============================================================

st.write("## Metric Rule Builder")

# Available metrics
METRICS = ["revenueGrowth","quickRatio"]

# Calculation mapping
CALC_MAP = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}

# Comparison operators
COMPARISONS = ["<=", ">=", "<", ">", "=="]

# Initialize session state for rules
if "rules" not in st.session_state:
    st.session_state.rules = []

# ------------------------------------------------------------
# ADD NEW RULE
# ------------------------------------------------------------

if st.button("➕ Add Rule"):
    st.session_state.rules.append({
        "metric": METRICS[0],
        "calc_name": "mean",
        "comparison": ">=",
        "threshold": 0.0
    })

st.divider()

# ============================================================
# DISPLAY RULES UI
# ============================================================

for i, rule in enumerate(st.session_state.rules):

    st.markdown(f"### Rule {i+1}")
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 0.5])

    # Metric selection
    with col1:
        rule["metric"] = st.selectbox(
            "Metric",
            METRICS,
            index=METRICS.index(rule["metric"]),
            key=f"metric_{i}"
        )

    # Calculation selection
    with col2:
        rule["calc_name"] = st.selectbox(
            "Calc",
            list(CALC_MAP.keys()),
            index=list(CALC_MAP.keys()).index(rule["calc_name"]),
            key=f"calc_{i}"
        )

    # Comparison operator
    with col3:
        rule["comparison"] = st.selectbox(
            "Operator",
            COMPARISONS,
            index=COMPARISONS.index(rule["comparison"]),
            key=f"comp_{i}"
        )

    # Threshold slider
    min_val = -1.0
    max_val = 1.0
    step_val = 0.05
    with col4:
        rule["threshold"] = st.slider(
            "Threshold",
            min_value=min_val,
            max_value=max_val,
            value=float(rule["threshold"]),
            step=step_val,
            key=f"threshold_{i}"
        )

    # Remove rule button
    with col5:
        if st.button("❌", key=f"remove_{i}"):
            st.session_state.rules.pop(i)
            st.rerun()

    st.divider()

# ============================================================
# COLLECT USER-SPECIFIED RULES
# ============================================================

def collect_user_rules(rules):
    """
    Convert UI rules into clean executable objects
    and store them in session state for later use.
    """
    user_rules = []
    for r in rules:
        user_rules.append({
            "metric": r["metric"],
            "function": CALC_MAP[r["calc_name"]],
            "comparison": r["comparison"],
            "threshold": r["threshold"]
        })
    return user_rules

# Store all rules in session state
st.session_state.user_rules = collect_user_rules(st.session_state.rules)

# ============================================================
# DATA LOADING & AGGREGATION
# ============================================================

st.write("## Financial Data Analysis")

def get_date_range(horizon_name):
    """Convert horizon selection to START_DATE and END_DATE."""
    end_date = datetime.now()
    horizon_days = {
        "1 Year": 365,
        "3 Years": 365 * 3,
        "5 Years": 365 * 5,
        "10 Years": 365 * 10,
    }
    start_date = end_date - timedelta(days=horizon_days.get(horizon_name, 365 * 5))
    return start_date, end_date

# Resolve tickers from selected sectors
tickers_ls = securityMaster[securityMaster["sector"].isin(SECTORS)].index.tolist()

# Compute date range from selected horizon
START_DATE, END_DATE = get_date_range(horizon)

# Build aggregation mapping based on user rules
METRIC_AGG = {}
for rule in st.session_state.user_rules:
    metric_name = rule["metric"].replace(" ", "_").lower()
    calc_name = [k for k, v in CALC_MAP.items() if v == rule["function"]][0]
    agg_key = f"{metric_name}_{calc_name}"
    METRIC_AGG[agg_key] = (metric_name, rule["function"])

st.write("**Data Selection:**")
st.write(f"- Tickers: {len(tickers_ls)} companies selected")
st.write(f"- Date Range: {START_DATE.date()} to {END_DATE.date()}")
st.write(f"- Aggregations: {list(METRIC_AGG.keys())}")

if tickers_ls and st.session_state.user_rules:
    st.write("Loading financial data...")
    try:
        ddf = dd.read_parquet(
            path=f"{FUNDMTL_DB_DIR}/annual/",
            filters=[
                ("symbol", "in", tickers_ls),
                ("filingDate", ">=", pd.Timestamp(START_DATE)),
                ("filingDate", "<=", pd.Timestamp(END_DATE)),
            ],
            engine="pyarrow",
        )

        st.write("### Debug: Available Columns")
        st.write(list(ddf.columns))

        ddf_agg = ddf.groupby("symbol").agg(**METRIC_AGG)
        df = ddf_agg.compute()

        st.success("Data loaded successfully!")
        st.write("### Aggregated Results")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.info("Select sectors and add rules to load data")


# ---------------------------------------------------
# DEBUG: Show consolidated variable
# ---------------------------------------------------
st.subheader("DEBUG ZONE: Collected Rules Variable")
st.json(st.session_state.sectors_input)
st.json({"horizon": st.session_state.horizon})
st.json(st.session_state.user_rules)