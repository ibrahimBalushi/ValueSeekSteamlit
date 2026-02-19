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
# Allow users to filter by market cap, sector, and time horizon

# Helper function to convert list of sectors to a comma-separated string
def list_to_str(item_list):
    return ",".join(item_list)


top_left_cell = st.container(border=True)

# ============================================================
# MARKETCAP SELECTION
# ============================================================

# Ordered market cap options (UI labels)
MCAP_OPTIONS = ["micro", "small", "med", "large", "mega"]
MCAP_TO_DATA_VALUE = {"med": "mid"}

# default market caps pre-selected
DEFAULT_MCAP = ["med", "large", "mega"]

# Initialize session state for mcap input
if "mcap_input" not in st.session_state:
    st.session_state.mcap_input = list(DEFAULT_MCAP)
else:
    st.session_state.mcap_input = [
        "med" if val == "mid" else val
        for val in st.session_state.mcap_input
    ]

with top_left_cell:
    # Multi-select pills for market cap filtering
    MCAPS = st.pills(
        "Market cap",
        options=MCAP_OPTIONS,
        default=st.session_state.mcap_input,
        selection_mode="multi",
    )
    st.session_state.mcap_input = MCAPS if MCAPS is not None else []

# ============================================================
# SECTOR SELECTION
# ============================================================

# List of all sectors
sector_list = securityMaster['sector'].dropna().unique().tolist()

# Default sectors pre-selected
DEFAULT_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"
]

# Initialize session state for sectors input
if "sectors_input" not in st.session_state:
    st.session_state.sectors_input = list(DEFAULT_SECTORS)

with top_left_cell:
    # Multi-select dropdown for sector filtering
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
    # Select data lookback period (1Y, 3Y, 5Y, 10Y)
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default=st.session_state.horizon,
    )

# ============================================================
# METRIC RULE BUILDER CONFIG
# ============================================================

st.write("## Metric Rule Builder")
# Build custom filtering rules: choose metric, aggregation function, operator, and threshold

# Available metrics for analysis
METRICS = ["revenueGrowth","quickRatio"]

# Calculation mapping
CALC_MAP = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}

# Comparison operators for threshold filtering
COMPARISONS = ["<=", ">=", "<", ">", "=="]

# Initialize session state for rules (persists across reruns)
if "rules" not in st.session_state:
    st.session_state.rules = []
if "rules" not in st.session_state:
    st.session_state.rules = []

if "threshold_min" not in st.session_state:
    st.session_state.threshold_min = 0.0

if "threshold_max" not in st.session_state:
    st.session_state.threshold_max = 1.0

# ------------------------------------------------------------
# ADD NEW RULE
# ------------------------------------------------------------

if st.button("➕ Add Rule"):
    st.session_state.rules.append({
        "metric": METRICS[0],
        "calc_name": "mean",
        "comparison": "<",
        "threshold": 1.0
    })

# Display threshold range input controls
range_col_label, range_col_min, range_col_max, range_col_spacer = st.columns([1.2, 1, 1, 3.8])
with range_col_label:
    st.caption("Threshold range [min, max]")
with range_col_min:
    threshold_min_input = st.number_input(
        "Min",
        value=float(st.session_state.threshold_min),
        step=0.01,
        key="threshold_min_input",
        label_visibility="collapsed",
    )
with range_col_max:
    threshold_max_input = st.number_input(
        "Max",
        value=float(st.session_state.threshold_max),
        step=0.01,
        key="threshold_max_input",
        label_visibility="collapsed",
    )

if threshold_max_input <= threshold_min_input:
    st.warning("Max threshold must be greater than min threshold. Using min + 0.01.")
    st.session_state.threshold_min = float(threshold_min_input)
    st.session_state.threshold_max = float(threshold_min_input + 0.01)
else:
    st.session_state.threshold_min = float(threshold_min_input)
    st.session_state.threshold_max = float(threshold_max_input)

st.divider()

# ============================================================
# DISPLAY RULES UI
# ============================================================
# Render edit controls for each rule (metric, calc, operator, threshold, delete)
for i, rule in enumerate(st.session_state.rules):
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
    min_val = float(st.session_state.threshold_min)
    max_val = float(st.session_state.threshold_max)
    step_val = 0.01
    current_threshold = float(rule["threshold"])
    current_threshold = min(max(current_threshold, min_val), max_val)
    with col4:
        rule["threshold"] = st.slider(
            "Threshold",
            min_value=min_val,
            max_value=max_val,
            value=current_threshold,
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
# HELPER FUNCTION
# ============================================================

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

# ============================================================
# DATA LOADING & FILTERING
# ============================================================
# Filter tickers by selected sectors and market caps
selected_mcaps = [MCAP_TO_DATA_VALUE.get(mcap, mcap) for mcap in st.session_state.mcap_input]
security_filtered = securityMaster[
    securityMaster["sector"].isin(SECTORS)
    & securityMaster["marketSize"].isin(selected_mcaps)
]
tickers_ls = security_filtered.index.tolist()

# Compute date range from selected horizon
START_DATE, END_DATE = get_date_range(horizon)

# Build aggregation mapping based on user rules
METRIC_AGG = {}
RULE_FILTERS = []
for rule in st.session_state.user_rules:
    metric_name = rule["metric"]
    calc_name = [k for k, v in CALC_MAP.items() if v == rule["function"]][0]
    agg_key = f"{metric_name}_{calc_name}"
    if agg_key not in METRIC_AGG:
        METRIC_AGG[agg_key] = (metric_name, rule["function"])
    RULE_FILTERS.append(
        {
            "agg_key": agg_key,
            "comparison": rule["comparison"],
            "threshold": rule["threshold"],
        }
    )

# Display data selection summary
st.write("**Data Selection Info:**")
st.write(f"- Date Range: {START_DATE.date()} to {END_DATE.date()}")
st.write(f"- Tickers: {len(tickers_ls)} companies in scope")

# Load and process financial data if selections are valid
if tickers_ls and st.session_state.user_rules:
    st.write("Loading financial data...")
    try:
        # Read parquet data with filters for symbols and date range
        ddf = dd.read_parquet(
            path=f"{FUNDMTL_DB_DIR}/annual/",
            filters=[
                ("symbol", "in", tickers_ls),
                ("filingDate", ">=", pd.Timestamp(START_DATE)),
                ("filingDate", "<=", pd.Timestamp(END_DATE)),
            ],
            engine="pyarrow",
        )
        required_metrics = {source_col for source_col, _ in METRIC_AGG.values()}
        missing_metrics = sorted(required_metrics - set(ddf.columns))
        if missing_metrics:
            raise KeyError(
                f"Missing metric columns in dataset: {missing_metrics}. "
                f"Available columns: {list(ddf.columns)}"
            )

        # Aggregate metrics by symbol using user-specified functions
        ddf_agg = ddf.groupby("symbol").agg(**METRIC_AGG)
        df = ddf_agg.compute()

        # Build filter conditions from user rules (multi-condition AND logic)
        filter_conditions = []
        for rule_filter in RULE_FILTERS:
            col = rule_filter["agg_key"]
            threshold = rule_filter["threshold"]
            comparison = rule_filter["comparison"]

            if threshold is None:
                continue

            if comparison == "<":
                filter_conditions.append(df[col] < threshold)
            elif comparison == ">":
                filter_conditions.append(df[col] > threshold)
            elif comparison == "==":
                filter_conditions.append(df[col] == threshold)
            elif comparison == "<=":
                filter_conditions.append(df[col] <= threshold)
            elif comparison == ">=":
                filter_conditions.append(df[col] >= threshold)
            else:
                raise ValueError(f"Unsupported comparison operator: {comparison}")

        # Apply all filter conditions (rows must satisfy ALL rules)
        if filter_conditions:
            mask = pd.concat(filter_conditions, axis=1).all(axis=1)
            df = df[mask]

        # Prepare display DataFrame: reset index and merge metadata (sector, industry, exchange)
        display_df = df.reset_index()
        if "symbol" not in display_df.columns:
            display_df = display_df.rename(columns={display_df.columns[0]: "symbol"})

        # Merge company metadata
        meta_df = securityMaster.copy()
        if "symbol" not in meta_df.columns:
            index_col_name = meta_df.index.name if meta_df.index.name else "index"
            meta_df = meta_df.reset_index().rename(columns={index_col_name: "symbol"})

        for col in ["sector", "industry", "exchange"]:
            if col not in meta_df.columns:
                meta_df[col] = pd.NA

        meta_df = meta_df[["symbol", "sector", "industry", "exchange"]].drop_duplicates(subset=["symbol"])
        display_df = display_df.merge(meta_df, on="symbol", how="left")

        # Reorder columns: symbol, metadata, then metrics
        metric_cols = [col for col in METRIC_AGG.keys() if col in display_df.columns]
        ordered_cols = ["symbol", "sector", "industry", "exchange", *metric_cols]
        display_df = display_df[ordered_cols]

        # Display results table
        st.success("Data loaded successfully!")
        st.write("### Filtered Results")
        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.info("Select sectors and add rules to load data")

# ============================================================
# DEBUG SECTION
# ============================================================
st.subheader("DEBUG ZONE: Collected Rules Variable")
st.json(st.session_state.sectors_input)
st.json({"horizon": st.session_state.horizon})
st.json(st.session_state.user_rules)