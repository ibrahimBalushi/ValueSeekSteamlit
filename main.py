from valueSeek.config import *
from valueSeek.api import *
from valueSeek.processor import *

import numpy as np
import dask.dataframe as dd
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st


# writing text
st.write("""
# ValueSeek: 
A Data-Driven Approach to Identifying Investment Opportunities in the Stock Market     
""")

# get security master
securityMaster = get_security_master()

st.write("Company Profile Selection")

# set up box for sectors input
cols = st.columns([5, 1])

# list of sectors
SECTORS = securityMaster['sector'].dropna().unique().tolist()

DEFAULT_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"
    ]

# helper function
def sectors_to_str(s):
    return ",".join(s)

# defualt input for sectors
if "sectors_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", sectors_to_str(DEFAULT_SECTORS)
    ).split(",")

# set up box for sectors input
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Selectbox for stock tickers
    tickers = st.multiselect(
        "Sector to filter",
        options=sorted(set(SECTORS) | set(st.session_state.tickers_input)),
        default=st.session_state.tickers_input,
        placeholder="Choose sectors to include in your search",
        accept_new_options=True,
    )

# Time horizon selector
horizon_map = {
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
    "10 Years": "10y",
}

# set up box for time horizon
with top_left_cell:
    # Buttons for picking time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="5 Years",
    )

# ---------------------------------------------------
# METRIC BUILDER
# ---------------------------------------------------

st.write("Metric Rule Builder")

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
METRICS = [
    "ROE",
    "Revenue Growth",
    "PE Ratio",
    "Free Cash Flow"
]

CALC_MAP = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}

COMPARISONS = [">=", "<=", ">", "<"]

# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
if "rules" not in st.session_state:
    st.session_state.rules = []

# ---------------------------------------------------
# ADD RULE BUTTON
# ---------------------------------------------------
if st.button("➕ Add Rule"):
    st.session_state.rules.append({
        "metric": METRICS[0],
        "calc_name": "mean",
        "comparison": ">=",
        "threshold": 0.0
    })

st.divider()

# ---------------------------------------------------
# RULES UI
# ---------------------------------------------------
for i, rule in enumerate(st.session_state.rules):

    st.markdown(f"### Rule {i+1}")

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 0.5])

    # Metric
    with col1:
        rule["metric"] = st.selectbox(
            "Metric",
            METRICS,
            index=METRICS.index(rule["metric"]),
            key=f"metric_{i}"
        )

    # Calculation
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
    with col4:
        rule["threshold"] = st.slider(
            "Threshold",
            min_value=-100.0,
            max_value=100.0,
            value=float(rule["threshold"]),
            step=0.5,
            key=f"threshold_{i}"
        )

    # Remove button
    with col5:
        if st.button("❌", key=f"remove_{i}"):
            st.session_state.rules.pop(i)
            st.rerun()

    st.divider()

# ---------------------------------------------------
# COLLECT USER-SPECIFIED VARIABLES
# ---------------------------------------------------
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

# Save all rules to a single variable
st.session_state.user_rules = collect_user_rules(st.session_state.rules)

# ---------------------------------------------------
# DEBUG: Show consolidated variable
# ---------------------------------------------------
st.subheader("Collected Rules Variable")
st.json(st.session_state.user_rules)

# ---------------------------------------------------
# DATA LOADING & AGGREGATION
# ---------------------------------------------------
st.divider()
st.write("## Financial Data Analysis")

# Convert horizon to date range
def get_date_range(horizon_name):
    """Convert horizon selection to START_DATE and END_DATE"""
    end_date = datetime.now()
    horizon_map = {
        "1 Year": timedelta(days=365),
        "3 Years": timedelta(days=365*3),
        "5 Years": timedelta(days=365*5),
        "10 Years": timedelta(days=365*10),
    }
    start_date = end_date - horizon_map.get(horizon_name, timedelta(days=365*5))
    return start_date, end_date

# Get selected tickers from sectors
tickers_ls = securityMaster[securityMaster['sector'].isin(tickers)]['symbol'].unique().tolist()

# Get date range
START_DATE, END_DATE = get_date_range(horizon)

# Build aggregation dictionary from rules
METRIC_AGG = {}
for rule in st.session_state.user_rules:
    metric_name = rule["metric"].replace(" ", "_").lower()
    calc_name = [k for k, v in CALC_MAP.items() if v == rule["function"]][0]
    agg_key = f"{metric_name}_{calc_name}"
    # This assumes you have metric columns mapped in your data
    # Adjust the column name based on your actual parquet schema
    METRIC_AGG[agg_key] = (metric_name, rule["function"])

st.write(f"**Data Selection:**")
st.write(f"- Tickers: {len(tickers_ls)} companies selected")
st.write(f"- Date Range: {START_DATE.date()} to {END_DATE.date()}")
st.write(f"- Aggregations: {list(METRIC_AGG.keys())}")

# Load parquet data if we have rules and tickers selected
if tickers_ls and st.session_state.user_rules:
    st.write("Loading financial data...")
    
    try:
        # LOAD UP PARQUET DATA BETWEEN TWO YEARS - could be dates too
        ddf = dd.read_parquet(
            path=f"{FUNDMTL_DB_DIR}/annual/",
            filters=[
                ("symbol", "in", tickers_ls),
                ("filingDate", ">=", pd.Timestamp(START_DATE)), 
                ("filingDate", "<=", pd.Timestamp(END_DATE))
            ],
            engine="pyarrow",
        )

        # APPLY SIMPLE USER-SPECIFIED AGGREGATOR on couple of columns
        ddf_agg = ddf.groupby("symbol").agg(**METRIC_AGG)
        df = ddf_agg.compute()
        
        st.success("Data loaded successfully!")
        st.write("### Aggregated Results")
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
else:
    st.info("Select sectors and add rules to load data")
