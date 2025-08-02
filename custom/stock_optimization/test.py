import pandas as pd                                                                        │
│   import yfinance as yf                                                                      │
│   import warnings                                                                            │
│   from evaluator import backtest_strategy                                                    │
│   from datetime import datetime                                                              │
│                                                                                              │
│   warnings.filterwarnings('ignore')                                                          │
│                                                                                              │
│   # Get SPY data for period 2023-08 to 2025-08                                               │
│   ticker = yf.Ticker('SPY')                                                                  │
│   data = ticker.history(start='2023-08-01', end='2025-08-31')                                │
│                                                                                              │
│   print(f'Data period: {data.index[0].strftime(\"%Y-%m-%d\")} to                             │
│   {data.index[-1].strftime(\"%Y-%m-%d\")}')                                                  │
│   print(f'Number of trading days: {len(data)}')                                              │
│                                                                                              │
│   # Create buy-and-hold strategy: buy on first day, hold rest                                │
│   signals = pd.Series(0, index=data.index, dtype=int)                                        │
│   signals.iloc[0] = 1  # Buy on first day                                                    │
│                                                                                              │
│   print(f'Buy-and-hold signals: Buy on day 1, hold for {len(signals)-1} days')               │
│                                                                                              │
│   # Calculate performance using evaluator's backtest function                                │
│   results = backtest_strategy(data['Close'], signals)                                        │
│                                                                                              │
│   print(f'\nBuy-and-hold performance (2023-08 to 2025-08):')                                 │
│   print(f'Total return: {results[\"total_return\"]:.4f}                                      │
│   ({results[\"total_return\"]*100:.2f}%)')                                                   │
│   print(f'Annualized volatility: {results[\"annualized_volatility\"]:.4f}')                  │
│   print(f'Sharpe ratio: {results[\"sharpe_ratio\"]:.4f}')                                    │
│   print(f'Max drawdown: {results[\"max_drawdown\"]:.4f}')                                    │
│                                                                                              │
│   # Also show simple calculation                                                             │
│   start_price = data['Close'].iloc[0]                                                        │
│   end_price = data['Close'].iloc[-1]                                                         │
│   simple_return = (end_price - start_price) / start_price                                    │
│   print(f'\nSimple calculation verification:')                                               │
│   print(f'Start price: ${start_price:.2f}, End price: ${end_price:.2f}')                     │
│   print(f'Simple return: {simple_return:.4f} ({simple_return*100:.2f}%)')   