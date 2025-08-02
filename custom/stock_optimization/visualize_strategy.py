#!/usr/bin/env python3
"""
Visualization tool for stock trading strategies
"""

import argparse
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from evaluator import load_stock_data


def load_program(program_path):
    """Load a program module from file path"""
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program


def extract_signals_from_strategy(program, stock_data):
    """
    Extract trading signals from a strategy program
    
    This function calls the program's run_stock_optimization function
    to get signals directly.
    """
    # Try the main entry point first
    if hasattr(program, 'run_stock_optimization'):
        try:
            signals = program.run_stock_optimization(stock_data)
            if isinstance(signals, pd.Series):
                # Validate signal length matches stock data
                if len(signals) != len(stock_data):
                    print(f"Warning: Signal length mismatch - got {len(signals)}, expected {len(stock_data)}")
                    # Try to align signals with stock data index, filling missing with 0
                    signals = signals.reindex(stock_data.index, fill_value=0)
                return signals
        except Exception as e:
            print(f"Warning: run_stock_optimization failed: {e}")
    
    # Try direct signal generation function
    if hasattr(program, 'generate_signals'):
        try:
            signals = program.generate_signals(stock_data['Close'])
            if isinstance(signals, pd.Series):
                # Validate signal length matches stock data
                if len(signals) != len(stock_data):
                    print(f"Warning: Signal length mismatch - got {len(signals)}, expected {len(stock_data)}")
                    # Try to align signals with stock data index, filling missing with 0
                    signals = signals.reindex(stock_data.index, fill_value=0)
                return signals
        except Exception as e:
            print(f"Warning: generate_signals failed: {e}")
    
    # Try optimize_strategy function (should return signals now)
    if hasattr(program, 'optimize_strategy'):
        try:
            signals = program.optimize_strategy(stock_data['Close'])
            if isinstance(signals, pd.Series):
                # Validate signal length matches stock data
                if len(signals) != len(stock_data):
                    print(f"Warning: Signal length mismatch - got {len(signals)}, expected {len(stock_data)}")
                    # Try to align signals with stock data index, filling missing with 0
                    signals = signals.reindex(stock_data.index, fill_value=0)
                return signals
        except Exception as e:
            print(f"Warning: optimize_strategy failed: {e}")
    
    # Fallback: generate basic signals using program's logic if available
    if hasattr(program, 'calculate_moving_averages'):
        try:
            short_ma, long_ma = program.calculate_moving_averages(stock_data['Close'])
            signals = pd.Series(0, index=stock_data.index)
            signals[short_ma > long_ma] = 1
            signals[short_ma < long_ma] = -1
            return signals
        except Exception as e:
            print(f"Warning: fallback signal generation failed: {e}")
    
    # If all else fails, return empty signals
    print("Warning: Could not extract signals from strategy, returning empty signals")
    return pd.Series(0, index=stock_data.index)


def calculate_portfolio_value(prices, signals, initial_capital=10000):
    """
    Calculate portfolio value over time based on trading signals
    
    Args:
        prices: Series of stock prices
        signals: Series of trading signals (1=buy, -1=sell, 0=hold)
        initial_capital: Starting capital amount
        
    Returns:
        Series of portfolio values over time
    """
    portfolio_value = pd.Series(index=prices.index, dtype=float)
    cash = initial_capital
    shares = 0
    
    # Shift signals to avoid look-ahead bias
    signals_shifted = signals.shift(1).fillna(0)
    
    for i, (date, price) in enumerate(prices.items()):
        signal = signals_shifted.iloc[i] if i < len(signals_shifted) else 0
        
        # Execute trades based on signal
        if signal == 1 and cash > 0:  # Buy signal
            shares_to_buy = cash // price
            shares += shares_to_buy
            cash -= shares_to_buy * price
        elif signal == -1 and shares > 0:  # Sell signal
            cash += shares * price
            shares = 0
        
        # Calculate total portfolio value
        portfolio_value.iloc[i] = cash + shares * price
    
    return portfolio_value


def plot_strategies(program_paths, symbol='SPY', period='2y', output_file=None):
    """
    Plot stock price with trading signals and portfolio performance for multiple strategies
    
    Args:
        program_paths: List of paths to strategy program files, or single path string
        symbol: Stock symbol to analyze
        period: Time period for data
        output_file: Optional output file path for saving plot
    """
    # Handle single program path
    if isinstance(program_paths, str):
        program_paths = [program_paths]
    
    # Load stock data once
    print(f"Loading {symbol} data for {period}")
    stock_data = load_stock_data(symbol, period)
    
    # Calculate buy-and-hold benchmark
    initial_capital = 10000
    buy_hold_shares = initial_capital / stock_data['Close'].iloc[0]
    buy_hold_value = buy_hold_shares * stock_data['Close']
    
    # Process each program
    strategies = []
    colors = ['green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    for i, program_path in enumerate(program_paths):
        print(f"\nProcessing strategy {i+1}: {program_path}")
        
        # Load the program
        program = load_program(program_path)
        
        # Extract trading signals
        print("  Extracting trading signals...")
        signals = extract_signals_from_strategy(program, stock_data)
        
        # Calculate portfolio value over time
        print("  Calculating portfolio performance...")
        portfolio_value = calculate_portfolio_value(stock_data['Close'], signals)
        
        # Store strategy data
        strategy_name = program_path.split('/')[-1].replace('.py', '')
        strategies.append({
            'name': strategy_name,
            'path': program_path,
            'signals': signals,
            'portfolio_value': portfolio_value,
            'color': colors[i % len(colors)]
        })
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), height_ratios=[3, 2, 1])
    
    # Plot 1: Stock price with signals from first strategy only (to avoid clutter)
    ax1.plot(stock_data.index, stock_data['Close'], 'b-', linewidth=2, label='Close Price')
    
    if strategies:
        # Show signals from first strategy only
        first_strategy = strategies[0]
        signals = first_strategy['signals']
        
        # Mark buy signals (green up triangles)
        buy_signals = signals == 1
        if buy_signals.any():
            ax1.scatter(stock_data.index[buy_signals], stock_data['Close'][buy_signals], 
                       marker='^', color=first_strategy['color'], s=60, 
                       label=f'{first_strategy["name"]} Buy', alpha=0.7, zorder=5)
        
        # Mark sell signals (red down triangles)
        sell_signals = signals == -1
        if sell_signals.any():
            ax1.scatter(stock_data.index[sell_signals], stock_data['Close'][sell_signals], 
                       marker='v', color=first_strategy['color'], s=60, 
                       label=f'{first_strategy["name"]} Sell', alpha=0.7, zorder=5)
        
        # Add moving averages if available (from first strategy)
        try:
            program = load_program(first_strategy['path'])
            if hasattr(program, 'calculate_moving_averages'):
                short_ma, long_ma = program.calculate_moving_averages(stock_data['Close'])
                ax1.plot(stock_data.index, short_ma, 'orange', linewidth=1, alpha=0.7, label='Short MA')
                ax1.plot(stock_data.index, long_ma, 'purple', linewidth=1, alpha=0.7, label='Long MA')
        except:
            pass  # Skip if moving averages fail
    
    ax1.set_title(f'{symbol} Stock Price with Trading Signals', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Plot 2: Portfolio values for all strategies
    ax2.plot(stock_data.index, buy_hold_value, 'b--', linewidth=2, alpha=0.8, label='Buy & Hold')
    
    for strategy in strategies:
        ax2.plot(stock_data.index, strategy['portfolio_value'], 
                color=strategy['color'], linewidth=2, label=strategy['name'])
        
        # Mark significant portfolio changes for first strategy only (to avoid clutter)
        if strategy == strategies[0]:
            portfolio_returns = strategy['portfolio_value'].pct_change()
            significant_gains = portfolio_returns > 0.05  # 5% daily gain
            significant_losses = portfolio_returns < -0.05  # 5% daily loss
            
            if significant_gains.any():
                ax2.scatter(stock_data.index[significant_gains], 
                           strategy['portfolio_value'][significant_gains], 
                           marker='^', color='darkgreen', s=30, alpha=0.8, zorder=5)
            if significant_losses.any():
                ax2.scatter(stock_data.index[significant_losses], 
                           strategy['portfolio_value'][significant_losses], 
                           marker='v', color='darkred', s=30, alpha=0.8, zorder=5)
    
    ax2.set_title('Portfolio Value Over Time', fontsize=14)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Plot 3: Trading signals over time for all strategies
    if len(strategies) == 1:
        # Single strategy - use fill
        signals = strategies[0]['signals']
        ax3.plot(stock_data.index, signals, 'k-', linewidth=1)
        ax3.fill_between(stock_data.index, 0, signals, where=(signals > 0), 
                         color=strategies[0]['color'], alpha=0.3, label='Buy Periods')
        ax3.fill_between(stock_data.index, 0, signals, where=(signals < 0), 
                         color=strategies[0]['color'], alpha=0.3, label='Sell Periods')
    else:
        # Multiple strategies - use offset lines
        for i, strategy in enumerate(strategies):
            offset = (i - (len(strategies)-1)/2) * 0.1  # Slight vertical offset
            signals_offset = strategy['signals'] + offset
            ax3.plot(stock_data.index, signals_offset, 
                    color=strategy['color'], linewidth=2, 
                    label=strategy['name'], alpha=0.8)
    
    ax3.set_title('Trading Signals Over Time', fontsize=14)
    ax3.set_ylabel('Signal', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Sell', 'Hold', 'Buy'])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Calculate summary statistics
    buy_hold_return = (buy_hold_value.iloc[-1] / buy_hold_value.iloc[0] - 1) * 100
    
    if len(strategies) == 1:
        # Single strategy
        strategy = strategies[0]
        signals = strategy['signals']
        portfolio_value = strategy['portfolio_value']
        
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        hold_count = (signals == 0).sum()
        
        strategy_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        outperformance = strategy_return - buy_hold_return
        
        stats_text = f"Strategy: {strategy_return:.1f}% | Buy&Hold: {buy_hold_return:.1f}% | Outperformance: {outperformance:+.1f}%"
        fig.suptitle(f"Trading Strategy Analysis - {stats_text}", fontsize=12, y=0.02)
    else:
        # Multiple strategies
        best_strategy = max(strategies, 
                           key=lambda s: s['portfolio_value'].iloc[-1] / s['portfolio_value'].iloc[0])
        best_return = (best_strategy['portfolio_value'].iloc[-1] / best_strategy['portfolio_value'].iloc[0] - 1) * 100
        
        stats_text = f"Best: {best_strategy['name']} ({best_return:.1f}%) | Buy&Hold: {buy_hold_return:.1f}% | {len(strategies)} Strategies"
        fig.suptitle(f"Multi-Strategy Comparison - {stats_text}", fontsize=12, y=0.02)
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"STRATEGY COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(stock_data)}")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"")
    
    # Sort strategies by performance
    strategies_sorted = sorted(strategies, 
                              key=lambda s: s['portfolio_value'].iloc[-1] / s['portfolio_value'].iloc[0], 
                              reverse=True)
    
    for i, strategy in enumerate(strategies_sorted):
        signals = strategy['signals']
        portfolio_value = strategy['portfolio_value']
        
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        hold_count = (signals == 0).sum()
        
        strategy_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        outperformance = strategy_return - buy_hold_return
        
        rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        
        print(f"{rank_symbol} {strategy['name'].upper()}:")
        print(f"    Final Value: ${portfolio_value.iloc[-1]:,.2f} (Return: {strategy_return:.2f}%)")
        print(f"    Outperformance: {outperformance:+.2f}%")
        print(f"    Signals: {buy_count} Buy ({buy_count/len(stock_data)*100:.1f}%), {sell_count} Sell ({sell_count/len(stock_data)*100:.1f}%), {hold_count} Hold ({hold_count/len(stock_data)*100:.1f}%)")
        print(f"    Max Value: ${portfolio_value.max():,.2f}, Min Value: ${portfolio_value.min():,.2f}")
        print()
    
    # Calculate basic performance using evaluator's logic
    try:
        from evaluator import backtest_strategy, evaluate_strategy_performance
        
        if strategies:
            # Use the first strategy for performance calculation
            first_strategy = strategies[0]
            program = load_program(first_strategy['path'])
            
            if hasattr(program, 'run_stock_optimization'):
                signals = program.run_stock_optimization(stock_data)
                if isinstance(signals, pd.Series):
                    results = backtest_strategy(stock_data['Close'], signals)
                    scores = evaluate_strategy_performance(results)
                    
                    print(f"\nStrategy Performance Metrics:")
                    print(f"  Total Return: {results.get('total_return', 0)*100:.2f}%")
                    print(f"  Volatility: {results.get('annualized_volatility', 0)*100:.2f}%")  
                    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.3f}")
                    print(f"  Max Drawdown: {results.get('max_drawdown', 0)*100:.2f}%")
                    print(f"\nNormalized Scores:")
                    print(f"  Return Score: {scores.get('return_score', 0):.3f}/1.0")
                    print(f"  Volatility Score: {scores.get('volatility_score', 0):.3f}/1.0")
                    print(f"  Sharpe Score: {scores.get('sharpe_score', 0):.3f}/1.0")
                    print(f"  Drawdown Score: {scores.get('drawdown_score', 0):.3f}/1.0")
                    print(f"  Combined Score: {scores.get('combined_score', 0):.3f}/1.0")
    except Exception as e:
        print(f"Could not calculate performance metrics: {e}")


# Backward compatibility wrapper
def plot_strategy(program_path, symbol='SPY', period='2y', output_file=None):
    """Single strategy plotting (backward compatibility)"""
    return plot_strategies(program_path, symbol, period, output_file)


def main():
    parser = argparse.ArgumentParser(description='Visualize stock trading strategies')
    parser.add_argument('programs', nargs='+', help='Path(s) to strategy program files')
    parser.add_argument('--symbol', '-s', default='SPY', help='Stock symbol (default: SPY)')
    parser.add_argument('--period', '-p', default='2y', help='Time period (default: 2y)')
    parser.add_argument('--output', '-o', help='Output file path for plot')
    
    args = parser.parse_args()
    
    try:
        plot_strategies(args.programs, args.symbol, args.period, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()