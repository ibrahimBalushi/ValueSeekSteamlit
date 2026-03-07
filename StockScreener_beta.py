# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import dask.dataframe as dd
from datetime import datetime, timedelta

from valueSeek.config import *
from valueSeek.api import *
from valueSeek.processor import *

st.set_page_config(layout="wide")

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

CALC_MAP = {
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "median": np.median,
}

COMPARISONS = ["<=", ">=", "<", ">", "=="]

MCAP_OPTIONS = ["micro", "small", "med", "large", "mega"]
MCAP_TO_DATA_VALUE = {"med": "mid"}
DEFAULT_MCAP = ["med", "large", "mega"]

DEFAULT_EXCHANGES = ["NASDAQ", "NYSE", "TSX"]
DEFAULT_SECTORS = ["Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"]

HORIZON_MAP = {
    "1 Year": 365,
    "3 Years": 365 * 3,
    "5 Years": 365 * 5,
    "10 Years": 365 * 10,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def init_session_state(available_sectors=None):
    """Initialize session state variables."""
    # Filter default sectors to only include those available in data
    default_sectors = DEFAULT_SECTORS
    if available_sectors:
        default_sectors = [s for s in DEFAULT_SECTORS if s in available_sectors]
        if not default_sectors and available_sectors:
            default_sectors = [available_sectors[0]]
    
    defaults = {
        "exchange_input": DEFAULT_EXCHANGES,
        "mcap_input": list(DEFAULT_MCAP),
        "sectors_input": default_sectors,
        "horizon": "5 Years",
        "rules": [],
        "threshold_min": 0.0,
        "threshold_max": 1.0,
        "show_results": False,  # Don't show results by default
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_metrics_map():
    """Load and structure metrics from CSV."""
    csv_path = "/home/abrahim/Documents/GitHub/valuescreen/valueSeek/CSV_files/valueSeek Project - FMP enrichment.csv"
    df = pd.read_csv(csv_path)
    df = df[['Metric Type', 'Formula', 'Metric varName']].set_index("Metric Type")
    return df.groupby("Metric Type").apply(lambda x: dict(zip(x["Formula"], x["Metric varName"]))).to_dict()


@st.cache_data
def load_security_master():
    """Load security master data with caching."""
    return get_security_master()


@st.cache_data 
def load_metrics_data():
    """Load and process metrics data with caching."""
    return load_metrics_map()


def get_date_range(horizon_name):
    """Convert horizon selection to date range."""
    end_date = datetime.now()
    days = HORIZON_MAP.get(horizon_name, 365 * 5)
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


def collect_user_rules(rules):
    """Convert UI rules to executable format."""
    return [
        {
            "metric": r["metric"],
            "function": CALC_MAP[r["calc_name"]],
            "comparison": r["comparison"],
            "threshold": r["threshold"],
        }
        for r in rules
    ]


def build_metric_aggregation(user_rules):
    """Build metric aggregation mapping and filter specs from rules."""
    metric_agg = {}
    rule_filters = []
    
    for rule in user_rules:
        metric_name = rule["metric"]
        calc_name = [k for k, v in CALC_MAP.items() if v == rule["function"]][0]
        agg_key = f"{metric_name}_{calc_name}"
        
        if agg_key not in metric_agg:
            metric_agg[agg_key] = (metric_name, rule["function"])
        
        rule_filters.append({
            "agg_key": agg_key,
            "comparison": rule["comparison"],
            "threshold": rule["threshold"],
        })
    
    return metric_agg, rule_filters


def apply_filters(df, rule_filters):
    """Apply all filter conditions to dataframe."""
    conditions = []
    
    for rule in rule_filters:
        col = rule["agg_key"]
        threshold = rule["threshold"]
        comp = rule["comparison"]
        
        if threshold is None:
            continue
        
        if comp == "<":
            conditions.append(df[col] < threshold)
        elif comp == ">":
            conditions.append(df[col] > threshold)
        elif comp == "==":
            conditions.append(df[col] == threshold)
        elif comp == "<=":
            conditions.append(df[col] <= threshold)
        elif comp == ">=":
            conditions.append(df[col] >= threshold)
    
    if conditions:
        mask = pd.concat(conditions, axis=1).all(axis=1)
        return df[mask]
    return df


def prepare_display_df(df, security_master, metric_agg):
    """Prepare final display dataframe with metadata."""
    # Reset index and merge metadata
    display_df = df.reset_index()
    if "symbol" not in display_df.columns:
        display_df = display_df.rename(columns={display_df.columns[0]: "symbol"})
    
    # Prepare metadata
    meta_df = security_master.reset_index()
    if "symbol" not in meta_df.columns:
        meta_df = meta_df.rename(columns={meta_df.index.name or "index": "symbol"})
    
    meta_df = meta_df[["symbol", "sector", "industry", "exchange"]].drop_duplicates(subset=["symbol"])
    
    # Merge
    display_df = display_df.merge(meta_df, on="symbol", how="left")
    
    # Reorder columns
    metric_cols = [col for col in metric_agg.keys() if col in display_df.columns]
    ordered_cols = ["symbol", "sector", "industry", "exchange"] + metric_cols
    
    return display_df[[col for col in ordered_cols if col in display_df.columns]]


# ============================================================
# INITIALIZATION
# ============================================================

st.write("# ValueSeek Fundamentals Company Screener")
st.write("Select company profiles and build custom metric rules to filter for investment criteria.")

# Load data
try:
    security_master = load_security_master()
    metric_cat_map = load_metrics_data()
    available_sectors = sorted(security_master['sector'].dropna().unique().tolist())
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

init_session_state(available_sectors)

# ============================================================
# COMPANY PROFILE SELECTION
# ============================================================

st.write("## Company Profile Selection")

col1, col2 = st.columns(2)

with col1:
    exchange_list = sorted(security_master['exchange'].dropna().unique().tolist())
    exchange_selection = st.multiselect(
        "Exchange",
        options=exchange_list,
        key="exchange_input",
    )

with col2:
    sector_selection = st.multiselect(
        "Sector",
        options=available_sectors,
        key="sectors_input",
    )

col3, col4 = st.columns(2)

with col3:
    mcap_selection = st.pills(
        "Market Cap",
        options=MCAP_OPTIONS,
        selection_mode="multi",
        key="mcap_input",
    )

with col4:
    horizon_selection = st.pills(
        "Time Horizon",
        options=list(HORIZON_MAP.keys()),
        key="horizon",
    )

# ============================================================
# METRIC RULE BUILDER
# ============================================================

st.write("## Metric Rule Builder")

metric_types = list(metric_cat_map.keys())
default_type = metric_types[0]
default_formula = list(metric_cat_map[default_type].keys())[0]
default_metric = metric_cat_map[default_type][default_formula]

if st.button("➕ Add Rule"):
    st.session_state.rules.append({
        "metric_type": default_type,
        "metric_formula": default_formula,
        "metric": default_metric,
        "calc_name": "mean",
        "comparison": "<",
        "threshold": 1.0,
    })

# Threshold range
col_min, col_max = st.columns(2)
with col_min:
    min_val = st.number_input(
        "Min Threshold",
        step=0.01,
        key="threshold_min",
    )
with col_max:
    max_val = st.number_input(
        "Max Threshold",
        step=0.01,
        key="threshold_max",
    )

# Validate and adjust thresholds if needed
if st.session_state.threshold_max <= st.session_state.threshold_min:
    st.warning("Max must be greater than min. Adjusting...")
    st.session_state.threshold_max = st.session_state.threshold_min + 0.01

st.divider()

# Display and manage rules
for i, rule in enumerate(st.session_state.rules):
    cols = st.columns([1.5, 1.5, 1, 1, 2, 0.5])
    
    # Metric Type
    with cols[0]:
        current_type = rule.get("metric_type", default_type)
        rule["metric_type"] = st.selectbox(
            "Type",
            metric_types,
            index=metric_types.index(current_type),
            key=f"type_{i}",
        )
    
    # Metric Formula
    with cols[1]:
        formulas = list(metric_cat_map[rule["metric_type"]].keys())
        current_formula = rule.get("metric_formula", formulas[0])
        if current_formula not in formulas:
            current_formula = formulas[0]
        rule["metric_formula"] = st.selectbox(
            "Metric",
            formulas,
            index=formulas.index(current_formula),
            key=f"formula_{i}",
        )
        rule["metric"] = metric_cat_map[rule["metric_type"]][rule["metric_formula"]]
    
    # Calculation
    with cols[2]:
        rule["calc_name"] = st.selectbox(
            "Calc",
            list(CALC_MAP.keys()),
            index=list(CALC_MAP.keys()).index(rule["calc_name"]),
            key=f"calc_{i}",
        )
    
    # Operator
    with cols[3]:
        rule["comparison"] = st.selectbox(
            "Op",
            COMPARISONS,
            index=COMPARISONS.index(rule["comparison"]),
            key=f"comp_{i}",
        )
    
    # Threshold
    with cols[4]:
        min_val = float(st.session_state.threshold_min)
        max_val = float(st.session_state.threshold_max)
        current = min(max(float(rule["threshold"]), min_val), max_val)
        rule["threshold"] = st.slider(
            "Threshold",
            min_value=min_val,
            max_value=max_val,
            value=current,
            step=0.01,
            key=f"threshold_{i}",
        )
    
    # Delete button
    with cols[5]:
        if st.button("❌", key=f"remove_{i}"):
            st.session_state.rules.pop(i)
            st.rerun()
    
    st.divider()

# Query button
if st.button("🔍 Execute Query", type="primary", use_container_width=True):
    st.session_state.show_results = True
    st.rerun()

# ============================================================
# SCREENER RESULTS
# ============================================================

if st.session_state.get("show_results", False):
    st.write("## Screener Results")

    # Prepare data
    user_rules = collect_user_rules(st.session_state.rules)
    selected_mcaps = [MCAP_TO_DATA_VALUE.get(m, m) for m in st.session_state.mcap_input]
    security_filtered = security_master[
        security_master["sector"].isin(st.session_state.sectors_input)
        & security_master["marketSize"].isin(selected_mcaps)
    ]
    tickers = security_filtered.index.tolist()

    start_date, end_date = get_date_range(st.session_state.horizon)

    st.write(f"**Scope:** {len(tickers)} companies | {start_date.date()} to {end_date.date()}")

    # Load and process data
    if tickers and user_rules:
        try:
            metric_agg, rule_filters = build_metric_aggregation(user_rules)
            
            # Load data
            ddf = dd.read_parquet(
                path=f"{FUNDMTL_DB_DIR}/annual/",
                filters=[
                    ("symbol", "in", tickers),
                    ("filingDate", ">=", pd.Timestamp(start_date)),
                    ("filingDate", "<=", pd.Timestamp(end_date)),
                ],
                engine="pyarrow",
            )
            
            # Check for missing metrics
            required_metrics = {col for col, _ in metric_agg.values()}
            missing = required_metrics - set(ddf.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
                st.stop()
            
            # Aggregate and filter
            ddf_agg = ddf.groupby("symbol").agg(**metric_agg)
            df = ddf_agg.compute()
            df = apply_filters(df, rule_filters)
            
            # Prepare display
            display_df = prepare_display_df(df, security_master, metric_agg)
            
            st.success(f"✅ Found {len(display_df)} companies")
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Select sectors and add rules to load data")
else:
    st.info("👆 Build your rules above, then click **Execute Query** to run the screener")
