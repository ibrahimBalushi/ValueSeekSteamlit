# ============================================================
# IMPORTS
# ============================================================
from datetime import datetime as dt
import streamlit as st

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import InverseVolatility, EqualWeighted, RiskBudgeting

from valueSeek.backtesting import *

# ============================================================
# PAGE TITLE
# ============================================================

st.write("""
# ValueSeek Portfolio Optimizer
""")

top_left_cell = st.container(border=True)

# ============================================================
# STOCK SELECTOR
# ============================================================

# List of all stocks
ticker_list = load_sp500_dataset().columns.tolist()

# default stocks pre-selected
DEFLAULT_STOCKS = ["AAPL", "MSFT", "KO", "IMO.TO", "BMO.TO"]

# Initialize session state for stocks
if "stocks_input" not in st.session_state:
    st.session_state.stocks_input = DEFLAULT_STOCKS

with top_left_cell:
    # Multi-select dropdown for stock selection
    STOCKS = st.multiselect(
        "Select Stocks",  # REQUIRED label
        options=sorted(set(ticker_list) | set(st.session_state.stocks_input)),
        default=st.session_state.stocks_input,
        placeholder="Choose stocks to include in your optimization",
        accept_new_options=True,
    )

# ============================================================
# OPTIMIZATION STRATEGY SELECTOR
# ============================================================

strategies = ["Equal Weighted", "Inverse Volatility", "Risk Budgeting"]

rebalancing_map = {"Monthly": "ME", "Quarterly": "QE", "Annually": "YE"}

# Initialize session state for strategy
if "strategy_input" not in st.session_state:
    st.session_state.strategy_input = "Equal Weighted"

with top_left_cell:
    # Multi-select pills for optimization strategy
    STRATEGY = st.pills(
        "Optimization Strategy",
        options=strategies,
        default=st.session_state.strategy_input,
        selection_mode="multi",
    )

# Initialize session rebalancing frequency
if "rebalancing_frequency" not in st.session_state:
    st.session_state.rebalancing_frequency = "Quarterly"

with top_left_cell:
    # Single-select pills for rebalancing frequency
    REBALANCING_FREQUENCY = st.pills(
        "Rebalancing Frequency",
        options=list(rebalancing_map.keys()),
        default=st.session_state.rebalancing_frequency,
        selection_mode="single",
    )
    REBALANCING_FREQUENCY = rebalancing_map[REBALANCING_FREQUENCY]

# Initialize session START_DATE
if "start_date" not in st.session_state:
    st.session_state.start_date = "2010-01-01"

START_DATE = st.date_input("Backtesting start date", 
                           value=st.session_state.start_date,
                           min_value="1995-01-01",
                           max_value="2030-12-31",
                           )
                           


# Initialize session END_DATE
if "end_date" not in st.session_state:
    st.session_state.end_date = "2025-01-01"

END_DATE = st.date_input("Backtesting end date", 
                         value=st.session_state.end_date,
                         min_value="1995-01-01",
                         max_value="2030-12-31",
                         )

# ============================================================
# RUN AND PLOT BACKTEST
# ============================================================

CONFIG = {
        "stocks": STOCKS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "rebalance_frequency": REBALANCING_FREQUENCY,
        "strategies": {
            "Equal Weighted": {"color": "blue"},
            "Inverse Volatility": {"color": "red"}
        }
    }

# # fun methodology
portfolios, metrics = run_comparison_backtests(CONFIG)

# # display results
fig_returns = create_comparison_plot(portfolios, CONFIG, "cumulative_returns")
st.plotly_chart(fig_returns, use_container_width=True)



# ============================================================
# DEBUG SECTION
# ============================================================
st.subheader("DEBUG ZONE: Collected Configuration Variables")
st.write("Your backtesting configuration is:", CONFIG)