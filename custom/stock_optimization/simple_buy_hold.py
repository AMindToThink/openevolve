# EVOLVE-BLOCK-START
"""Simple buy-and-hold strategy for comparison"""
import pandas as pd


def optimize_strategy(prices):
    """
    Simple buy-and-hold strategy - buy on first day, hold forever
    
    Args:
        prices: Series of historical stock prices
        
    Returns:
        Series of trading signals (1=buy, -1=sell, 0=hold)
    """
    # Generate simple buy and hold signals
    signals = pd.Series(0, index=prices.index)
    signals.iloc[0] = 1  # Buy on first day
    # Hold for the rest of the period (signals remain 0 = hold)
    
    return signals


def generate_signals(prices):
    """Generate buy-and-hold signals"""
    signals = pd.Series(0, index=prices.index)
    signals.iloc[0] = 1  # Buy on first day
    return signals


# EVOLVE-BLOCK-END


def run_stock_optimization(stock_data):
    """Entry point for stock optimization"""
    return optimize_strategy(stock_data['Close'])


if __name__ == "__main__":
    print("Simple buy-and-hold strategy ready")