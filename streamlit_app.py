import streamlit as st


# Define the pages
main_page = st.Page("MainPage.py", title="Main Page")
page_2 = st.Page("StockScreener.py", title="Stock Screener")
page_3 = st.Page("PortfolioOptimizer.py", title="Portfolio Optimizer")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()
