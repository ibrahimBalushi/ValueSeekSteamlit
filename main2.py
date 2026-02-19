# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import dask.dataframe as dd
import streamlit as st

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
SECTORS = securityMaster['sector'].dropna().unique().tolist()

# Default sectors pre-selected
DEFAULT_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"
]

# Helper function to convert list of sectors to a comma-separated string
def sectors_to_str(sector_list):
    return ",".join(sector_list)

# Initialize session state for sectors input
if "sectors_input" not in st.session_state:
    st.session_state.sectors_input = st.query_params.get(
        "stocks", sectors_to_str(DEFAULT_SECTORS)
    ).split(",")

# Layout columns for sector selection
cols = st.columns([5, 1])
top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Multi-select box for sector filtering
    tickers = st.multiselect(
        "Sector to filter",
        options=sorted(set(SECTORS) | set(st.session_state.sectors_input)),
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
METRICS = ["ROE", "Revenue Growth", "PE Ratio", "Free Cash Flow"]

# Calculation mapping
CALC_MAP = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}

# Comparison operators
COMPARISONS = [">=", "<=", ">", "<"]

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
    with col4:
        rule["threshold"] = st.slider(
            "Threshold",
            min_value=-100.0,
            max_value=100.0,
            value=float(rule["threshold"]),
            step=0.5,
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
# DEBUG: SHOW COLLECTED RULES
# ============================================================

st.subheader("Collected Rules Variable")
st.json(st.session_state.user_rules)
