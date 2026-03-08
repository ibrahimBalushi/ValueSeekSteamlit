import streamlit as st


# Define the pages
frontPage = st.Page("FrontPage.py", title="Front Page")
stockScreener = st.Page("StockScreener_beta.py", title="Stock Screener")
valueAnsys = st.Page("ValuationAnalysis.py", title="Valuation Analysis")
portOptimize = st.Page("PortfolioOptimizer.py", title="Portfolio Optimization Demo")

dummyPage = st.Page("dummyPage.py", title="Sandbox")

# Set up navigation
pg = st.navigation([frontPage, stockScreener, valueAnsys, portOptimize, dummyPage])

# Run the selected page
pg.run()
