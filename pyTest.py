from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.optimization import InverseVolatility, EqualWeighted, RiskBudgeting

from valueSeek.backtesting import *

import datetime as dt

if __name__ == '__main__':
# --------------------------------------------------------------------------------------------
# Portfolio Optimization and Backtesting Experiment 2
# --------------------------------------------------------------------------------------------

    # organize user input fields
    STOCKS_PER_MARKET = {"US": ['AAPL','MSFT','KO'], "CA": ['IMO.TO','BMO.TO']}
    BT_START_DATE = dt.date(2013, 1, 1)
    BT_END_DATE   = dt.date(2014, 1, 1)
    REBALANCE_FREQUENCY = "ME"
    STRATEGY = "InverseVolatility"

    STOCKS = [s for stocks in STOCKS_PER_MARKET.values() for s in stocks]

    CONFIG = {
            "stocks": STOCKS,
            "start_date": BT_START_DATE,
            "end_date": BT_END_DATE,
            "rebalance_frequency": REBALANCE_FREQUENCY,
            "strategies": {
                "Equal Weighted": {"color": "blue"},
                "Inverse Volatility": {"color": "red"}
            }
        }

    # run backtest and get portfolios and metrics
    portfolios, metrics = run_comparison_backtests(CONFIG)

    # compare strategies
    fig_returns = create_comparison_plot(portfolios, CONFIG, "cumulative_returns")
    fig_returns.show()
