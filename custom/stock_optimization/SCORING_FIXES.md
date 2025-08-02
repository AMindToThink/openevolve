# Scoring Normalization Fixes

## Problem
The stock optimization evaluator had poorly calibrated scoring thresholds that caused programs to easily saturate at maximum scores, reducing the effectiveness of the evolution process.

## Issues Fixed

### 1. Return Score Saturation
**Before:** `return_score = np.clip(total_return / 0.5, 0.0, 1.0)`
- 50% total return over 5 years (~10% annualized) = max score of 1.0
- Programs with 66.5% return were getting saturated at 1.0

**After:** `return_score = np.clip(total_return / 1.5, 0.0, 1.0)`
- 150% total return over 5 years (~20% annualized) = max score of 1.0
- Much more realistic threshold for exceptional performance

### 2. Volatility Score Range
**Before:** Optimal range 10-40% annualized volatility
- Too narrow and unrealistic for stock strategies

**After:** Optimal range 5-50% annualized volatility  
- More realistic for diverse trading strategies
- Better differentiation between strategies

### 3. Sharpe Ratio Score Ceiling
**Before:** `sharpe_score = np.clip(sharpe_ratio / 1.5, 0.0, 1.0)`
- Sharpe ratio of 1.5 = max score (too easy to achieve)

**After:** `sharpe_score = np.clip(sharpe_ratio / 2.5, 0.0, 1.0)`
- Sharpe ratio of 2.5 = max score (truly exceptional performance)

### 4. Drawdown Score Range
**Before:** 20% max drawdown = score of 0.0
- Too restrictive for realistic trading strategies

**After:** 50% max drawdown = score of 0.0
- More realistic threshold while still penalizing excessive risk

## Impact Example

For a program with 66.5% return, 7.6% volatility, 1.38 Sharpe ratio, -8.5% drawdown:

| Metric | Old Score | New Score | Improvement |
|--------|-----------|-----------|-------------|
| Return | 1.000 (saturated) | 0.443 | No saturation |
| Volatility | 1.000 | 0.942 | Still high, not maxed |
| Sharpe | 0.922 | 0.553 | More realistic |
| Drawdown | 0.576 | 0.830 | Better reward for low drawdown |
| **Combined** | **0.921** | **0.648** | **Room for improvement** |

## Benefits

1. **Prevents Score Saturation**: Programs can no longer easily max out scores
2. **Better Differentiation**: More granular scoring between good strategies  
3. **Realistic Benchmarks**: Thresholds aligned with actual market performance
4. **Improved Evolution**: Fitness landscape has more room for optimization
5. **Maintained Balance**: Scoring weights and overall structure unchanged

These changes will help the evolution process find genuinely better strategies rather than getting stuck at artificially low performance ceilings.