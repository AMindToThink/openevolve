# EVOLVE-BLOCK-START
"""Stock trading strategy optimization for OpenEvolve

CONSTRAINTS FOR EVOLUTION:
- MUST return pandas.Series with same length as input prices 
- MUST return integer values: -1, 0, 1 (sell, hold, buy)
- Use pandas operations, NOT numpy scalars 
- Use .fillna() to handle NaN values safely
- Use .reindex() and .astype(int) for proper formatting
"""
import pandas as pd


def calculate_moving_averages(prices, short_window=10, long_window=50):
    """Calculate short and long term moving averages"""
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma, long_ma


def generate_signals(prices, short_window=10, long_window=50):
    """
    Generate buy/sell signals based on moving average crossover strategy.
    
    IMPORTANT: Always use pandas Series operations, not numpy scalars.
    Use .fillna() and proper indexing to avoid .iloc errors on scalar values.

    Args:
        prices: Series of stock prices
        short_window: Period for short-term moving average
        long_window: Period for long-term moving average

    Returns:
        Series of trading signals (1=buy, -1=sell, 0=hold)
    """
    short_ma, long_ma = calculate_moving_averages(prices, short_window, long_window)
    
    # Fill NaN values to avoid comparison issues
    short_ma = short_ma.fillna(0)
    long_ma = long_ma.fillna(0)

    # Generate signals using pandas Series operations
    signals = pd.Series(0, index=prices.index, dtype=int)

    # Buy when short MA crosses above long MA
    signals.loc[short_ma > long_ma] = 1

    # Sell when short MA crosses below long MA  
    signals.loc[short_ma < long_ma] = -1

    return signals


def optimize_strategy(prices):
    """
    Main strategy optimization function that will be evolved.
    
    CRITICAL CONSTRAINTS:
    - MUST return pandas Series with EXACT same length as input prices
    - MUST return only integer values: -1 (sell), 0 (hold), 1 (buy)
    - NO floating point values allowed - use pd.Series operations
    - Index must match prices.index exactly

    Args:
        prices: Series of historical stock prices

    Returns:
        Series of trading signals (1=buy, -1=sell, 0=hold) with same length as prices
    """
    # Basic moving average crossover strategy
    signals = generate_signals(prices, short_window=10, long_window=50)
    
    # Ensure signals match input length and contain only valid integers
    signals = signals.reindex(prices.index, fill_value=0)  # Align with prices index
    signals = signals.astype(int)  # Ensure integer values only
    
    return signals


# EVOLVE-BLOCK-END


def run_stock_optimization(stock_data):
    """Entry point for stock optimization"""
    return optimize_strategy(stock_data["Close"])


if __name__ == "__main__":
    # This would be called by the evaluator with actual stock data
    print("Stock optimization strategy ready for evolution")
