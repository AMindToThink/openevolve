# EVOLVE-BLOCK-START
"""Stock trading strategy optimization for OpenEvolve"""
import pandas as pd


def calculate_moving_averages(prices, short_window=10, long_window=50):
    """Calculate short and long term moving averages"""
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma, long_ma


def generate_signals(prices, short_window=10, long_window=50):
    """
    Generate buy/sell signals based on moving average crossover strategy.

    Args:
        prices: Series of stock prices
        short_window: Period for short-term moving average
        long_window: Period for long-term moving average

    Returns:
        Series of trading signals (1=buy, -1=sell, 0=hold)
    """
    short_ma, long_ma = calculate_moving_averages(prices, short_window, long_window)

    # Generate signals
    signals = pd.Series(0, index=prices.index)

    # Buy when short MA crosses above long MA
    signals[short_ma > long_ma] = 1

    # Sell when short MA crosses below long MA
    signals[short_ma < long_ma] = -1

    return signals


def optimize_strategy(prices):
    """
    Main strategy optimization function that will be evolved.

    Args:
        prices: Series of historical stock prices

    Returns:
        Series of trading signals (1=buy, -1=sell, 0=hold)
    """
    # Basic moving average crossover strategy
    signals = generate_signals(prices, short_window=10, long_window=50)

    return signals


# EVOLVE-BLOCK-END


def run_stock_optimization(stock_data):
    """Entry point for stock optimization"""
    return optimize_strategy(stock_data["Close"])


if __name__ == "__main__":
    # This would be called by the evaluator with actual stock data
    print("Stock optimization strategy ready for evolution")
