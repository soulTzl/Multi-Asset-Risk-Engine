# Project Multi-Asset Risk Engine and Portfolio Optimizer

## Objective

This project builds a practical cross-asset risk engine.

The engine compares several portfolio construction methods across equities, bonds, credit, commodities, FX, and crypto. It then measures downside risk using historical, parametric, and Monte Carlo VaR / Expected Shortfall, performs scenario stress tests, and backtests rolling VaR exceptions.

## Asset universe

| Ticker | Asset | Role in portfolio |
|---|---|---|
| SPY | US equity ETF | Growth and equity market beta |
| EFA | Developed ex-US equity ETF | International equity diversification |
| TLT | Long-duration US Treasury ETF | Duration and defensive rates exposure |
| LQD | Investment-grade corporate bond ETF | Credit and income exposure |
| GLD | Gold ETF | Real asset / crisis hedge proxy |
| USO | Oil ETF | Commodity and inflation shock proxy |
| UUP | US dollar ETF | FX / dollar strength proxy |
| BTC-USD | Bitcoin in USD | High-volatility alternative asset |

## Mathematical layers

1. Returns are modelled as daily log returns:

   `r_t = ln(P_t / P_{t-1})`

2. Annualised portfolio return and variance:

   `E[R_p] = w' mu`

   `Var(R_p) = w' Sigma w`

3. Covariance shrinkage is used to reduce sampling noise:

   `Sigma_shrunk = (1 - alpha) Sigma_sample + alpha diag(Sigma_sample)`

4. Downside risk:

   `VaR_alpha = quantile(losses, alpha)`

   `ES_alpha = E[loss | loss >= VaR_alpha]`

5. Monte Carlo uses multivariate normal simulated daily log returns with the shrunk covariance matrix.

6. Rolling VaR backtesting compares realised losses against one-day 99% VaR and reports a Kupiec likelihood-ratio test.

## How to run

Use the bundled Python runtime if available:

```powershell
& 'C:\Users\soult\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\projects\01_multi_asset_risk_engine\src\run_analysis.py
```

Or use your own Python environment with `numpy` and `pandas` installed:

```powershell
python .\projects\01_multi_asset_risk_engine\src\run_analysis.py
```

## Outputs

The script creates:

- `outputs/risk_report.md` - human-readable project write-up.
- `outputs/dashboard.html` - static visual dashboard.
- `outputs/portfolio_metrics.csv` - performance and risk metrics by portfolio.
- `outputs/portfolio_weights.csv` - allocation weights.
- `outputs/var_summary.csv` - VaR and Expected Shortfall comparison.
- `outputs/stress_tests.csv` - scenario stress impacts.
- `outputs/rolling_var_backtest.csv` - rolling VaR exceptions.
- `outputs/efficient_frontier.csv` - simulated long-only frontier.

## Interview/application positioning

This project demonstrates:

- Practical use of probability, covariance, simulation, and tail-risk metrics.
- Understanding that optimisation is fragile unless paired with risk diagnostics.
- Ability to connect mathematical finance to real cross-asset portfolio questions.
- Clean data pipeline, reproducible outputs, and portfolio-ready communication.

