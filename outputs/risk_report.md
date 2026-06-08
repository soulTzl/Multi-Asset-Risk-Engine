# Multi-Asset Risk Engine and Portfolio Optimizer

Generated: 2026-05-27 23:33 UTC

## Executive summary

This project builds a full risk and allocation workflow for a multi-asset portfolio covering equities, rates, credit, commodities, FX, and crypto. The data set runs from 2021-05-28 to 2026-05-27 with 1,253 daily return observations after alignment to the US equity market calendar.

The strongest risk-adjusted portfolio in this long-only random search is **Max Sharpe**. The lowest volatility portfolio is **Minimum variance**, while the least severe historical maximum drawdown is **Minimum variance**.

## Asset universe

| Ticker | Asset | Class |
|---|---|---|
| SPY | US equities | Equity |
| EFA | Developed ex-US equities | Equity |
| TLT | Long-duration US Treasuries | Rates |
| LQD | Investment-grade credit | Credit |
| GLD | Gold | Commodity |
| USO | Oil | Commodity |
| UUP | US dollar | FX |
| BTC-USD | Bitcoin | Crypto |

## Mathematical design

Daily prices are converted to log returns:

`r_t = ln(P_t / P_(t-1))`

For a portfolio with weights `w`, the annualised return and variance are:

`E[R_p] = w' mu`

`Var(R_p) = w' Sigma w`

The covariance matrix uses simple shrinkage:

`Sigma_shrunk = (1 - alpha) Sigma_sample + alpha diag(Sigma_sample)`

This dampens unstable cross-asset correlations without pretending the off-diagonal terms are irrelevant.

Downside risk is measured with:

`VaR_alpha = quantile(losses, alpha)`

`ES_alpha = average(losses | losses >= VaR_alpha)`

The Monte Carlo engine simulates multivariate normal daily log returns and compounds them over a 10-day horizon. The rolling VaR backtest uses a 250-day historical window and checks whether realised one-day losses breach the 99% VaR estimate.

## Portfolio construction methods

- Equal weight: simple benchmark with no risk model.
- Inverse volatility: allocates more weight to lower-volatility assets.
- Risk parity: iteratively targets equal risk contribution across assets.
- Minimum variance: selected from 60,000 long-only random portfolios.
- Max Sharpe: selected from 60,000 long-only random portfolios using a 3.50% risk-free assumption.

## Portfolio weights

| index | SPY | EFA | TLT | LQD | GLD | USO | UUP | BTC-USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal weight | 12.50% | 12.50% | 12.50% | 12.50% | 12.50% | 12.50% | 12.50% | 12.50% |
| Inverse volatility | 10.92% | 11.30% | 11.72% | 21.50% | 10.29% | 5.15% | 25.72% | 3.40% |
| Risk parity | 6.48% | 8.50% | 9.37% | 13.61% | 9.05% | 4.58% | 45.85% | 2.55% |
| Minimum variance | 0.22% | 4.90% | 3.59% | 29.76% | 7.32% | 1.14% | 52.78% | 0.28% |
| Max Sharpe | 8.78% | 5.52% | 0.05% | 1.75% | 22.21% | 2.09% | 57.57% | 2.03% |

## Performance and risk metrics

| portfolio | cagr | annual_volatility | sharpe | max_drawdown | hist_var_99_1d | hist_es_99_1d | mc_var_99_10d | mc_es_99_10d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal weight | 9.61% | 11.45% | 0.534 | -25.05% | 1.99% | 2.66% | 4.84% | 5.62% |
| Inverse volatility | 6.62% | 6.70% | 0.465 | -15.77% | 1.09% | 1.49% | 2.76% | 3.22% |
| Risk parity | 6.65% | 4.71% | 0.670 | -6.91% | 0.77% | 1.15% | 1.94% | 2.27% |
| Minimum variance | 4.86% | 3.80% | 0.356 | -4.28% | 0.62% | 0.80% | 1.67% | 1.95% |
| Max Sharpe | 9.79% | 4.97% | 1.267 | -5.01% | 0.97% | 1.32% | 2.00% | 2.36% |

## Stress test impact for selected portfolio: Max Sharpe

| scenario | portfolio_return | portfolio_loss |
| --- | --- | --- |
| Risk-off growth shock | -0.04% | 0.04% |
| Inflation and rates shock | 1.38% | -1.38% |
| Commodity supply spike | 2.71% | -2.71% |
| Dollar squeeze | 1.91% | -1.91% |
| Crypto deleveraging | -0.59% | 0.59% |

## Rolling 99% VaR backtest for selected portfolio: Max Sharpe

| Metric | Value |
|---|---|
| Observations | 1,003 |
| Exceptions | 16 |
| Expected exception rate | 1.00% |
| Observed exception rate | 1.60% |
| Kupiec LR statistic | 3.040 |
| Kupiec p-value | 0.081 |

Interpretation: a very low Kupiec p-value suggests the rolling VaR model is mis-calibrated; a moderate/high value suggests the exception count is broadly consistent with the selected confidence level. This test only checks exception frequency, not independence or clustering.

## What this demonstrates

- Probability and statistics: log returns, covariance, tail quantiles, shrinkage, simulation, backtesting.
- Financial mathematics: portfolio variance, risk-adjusted optimisation, VaR/Expected Shortfall, stress scenarios.
- Practical finance: cross-asset allocation, risk reporting, scenario design, and model limitations.
- Engineering: data ingestion, caching, reproducible CSV outputs, markdown reporting, and a static dashboard.

## Limitations and extensions

- The optimiser is long-only and uses random search rather than a formal quadratic-programming solver.
- Monte Carlo assumes multivariate normal log returns; a later version could add t-distributions, GARCH volatility, or block bootstrap.
- The VaR backtest covers exception frequency; a Christoffersen independence test would add more rigour.
- ETF proxies are useful for a portfolio project, but institutional implementation would use futures, swaps, bonds, options, and validated market data.

## Output files

- `dashboard.html`: visual dashboard.
- `portfolio_metrics.csv`: portfolio-level metrics.
- `portfolio_weights.csv`: allocation weights.
- `var_summary.csv`: VaR and Expected Shortfall comparison.
- `stress_tests.csv`: scenario results.
- `rolling_var_backtest.csv`: VaR exceptions.
- `efficient_frontier.csv`: random long-only portfolio simulation.
