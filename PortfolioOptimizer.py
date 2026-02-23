# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import InverseVolatility, EqualWeighted

# ============================================================
# PAGE TITLE
# ============================================================

st.write("""
# ValueSeek Portfolio Optimizer
""")

# ============================================================
# STOCK SELECTOR
# ============================================================

top_left_cell = st.container(border=True)

DEFLAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Initialize session state for stock selector
if "stocks_input" not in st.session_state:
    st.session_state.stocks_input = list(DEFLAULT_STOCKS)

with top_left_cell:
    # Multi-select dropdown for sector filtering
    SECTORS = st.multiselect(
        "Sector to filter",
        # options=sorted(set(sector_list) | set(st.session_state.sectors_input)),
        default=st.session_state.sectors_input,
        placeholder="Choose sectors to include in your search",
        accept_new_options=True,
    )