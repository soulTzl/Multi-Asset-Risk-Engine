from __future__ import annotations

import html
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.035
SHRINKAGE_ALPHA = 0.15
N_FRONTIER_PORTFOLIOS = 60_000
N_MONTE_CARLO = 40_000
RANDOM_SEED = 42


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str


ASSETS: list[Asset] = [
    Asset("SPY", "US equities", "Equity"),
    Asset("EFA", "Developed ex-US equities", "Equity"),
    Asset("TLT", "Long-duration US Treasuries", "Rates"),
    Asset("LQD", "Investment-grade credit", "Credit"),
    Asset("GLD", "Gold", "Commodity"),
    Asset("USO", "Oil", "Commodity"),
    Asset("UUP", "US dollar", "FX"),
    Asset("BTC-USD", "Bitcoin", "Crypto"),
]


SCENARIOS: dict[str, dict[str, float]] = {
    "Risk-off growth shock": {
        "SPY": -0.12,
        "EFA": -0.14,
        "TLT": 0.07,
        "LQD": -0.03,
        "GLD": 0.04,
        "USO": -0.10,
        "UUP": 0.03,
        "BTC-USD": -0.28,
    },
    "Inflation and rates shock": {
        "SPY": -0.08,
        "EFA": -0.07,
        "TLT": -0.12,
        "LQD": -0.08,
        "GLD": 0.06,
        "USO": 0.16,
        "UUP": 0.02,
        "BTC-USD": -0.10,
    },
    "Commodity supply spike": {
        "SPY": -0.04,
        "EFA": -0.05,
        "TLT": -0.03,
        "LQD": -0.02,
        "GLD": 0.08,
        "USO": 0.25,
        "UUP": 0.02,
        "BTC-USD": -0.04,
    },
    "Dollar squeeze": {
        "SPY": -0.06,
        "EFA": -0.10,
        "TLT": 0.02,
        "LQD": -0.03,
        "GLD": -0.05,
        "USO": -0.07,
        "UUP": 0.08,
        "BTC-USD": -0.15,
    },
    "Crypto deleveraging": {
        "SPY": -0.03,
        "EFA": -0.03,
        "TLT": 0.02,
        "LQD": 0.00,
        "GLD": 0.01,
        "USO": -0.02,
        "UUP": 0.01,
        "BTC-USD": -0.45,
    },
}


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def yahoo_chart_url(ticker: str, range_: str = "5y") -> str:
    return (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        f"{quote(ticker)}?range={range_}&interval=1d&events=history&includeAdjustedClose=true"
    )


def fetch_yahoo_adjusted_close(ticker: str, range_: str = "5y") -> pd.DataFrame:
    cache_path = RAW_DIR / f"{ticker.replace('-', '_')}.csv"
    req = Request(
        yahoo_chart_url(ticker, range_),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    try:
        with urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]
        quote_data = indicators["quote"][0]
        adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
        close = quote_data.get("close")
        prices = adjclose if adjclose is not None else close
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).date,
                "adjusted_close": prices,
            }
        )
        df = df.dropna().drop_duplicates("date").sort_values("date")
        df.to_csv(cache_path, index=False)
        time.sleep(0.15)
        return df
    except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        if cache_path.exists():
            print(f"Warning: using cached data for {ticker} after fetch error: {exc}", file=sys.stderr)
            return pd.read_csv(cache_path, parse_dates=["date"])
        raise RuntimeError(f"Could not fetch {ticker} and no cache exists: {exc}") from exc


def load_price_matrix() -> pd.DataFrame:
    frames: list[pd.Series] = []
    for asset in ASSETS:
        df = fetch_yahoo_adjusted_close(asset.ticker)
        series = pd.Series(df["adjusted_close"].to_numpy(), index=pd.to_datetime(df["date"]), name=asset.ticker)
        frames.append(series)

    all_prices = pd.concat(frames, axis=1, sort=True).sort_index()
    market_calendar = all_prices["SPY"].dropna().index
    prices = all_prices.loc[market_calendar].ffill().dropna()
    prices.to_csv(OUTPUT_DIR / "price_history.csv", index_label="date")
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.to_csv(OUTPUT_DIR / "returns_history.csv", index_label="date")
    return returns


def shrink_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    sample = daily_returns.cov()
    diagonal = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    return (1 - SHRINKAGE_ALPHA) * sample + SHRINKAGE_ALPHA * diagonal


def portfolio_return_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = weights.reindex(returns.columns).fillna(0.0)
    return returns @ aligned


def max_drawdown(series: pd.Series) -> float:
    wealth = np.exp(series.cumsum())
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def cagr(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    total_growth = float(np.exp(series.sum()))
    years = len(series) / TRADING_DAYS
    return total_growth ** (1 / years) - 1


def skewness(series: pd.Series) -> float:
    centered = series - series.mean()
    std = series.std(ddof=0)
    if std == 0:
        return 0.0
    return float((centered**3).mean() / std**3)


def excess_kurtosis(series: pd.Series) -> float:
    centered = series - series.mean()
    std = series.std(ddof=0)
    if std == 0:
        return 0.0
    return float((centered**4).mean() / std**4 - 3.0)


def var_es_from_losses(losses: pd.Series | np.ndarray, alpha: float) -> tuple[float, float]:
    arr = np.asarray(losses, dtype=float)
    var = float(np.quantile(arr, alpha))
    tail = arr[arr >= var]
    es = float(tail.mean()) if len(tail) else var
    return var, es


def parametric_var_es(daily_returns: pd.Series, alpha: float) -> tuple[float, float]:
    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std(ddof=1))
    z = NormalDist().inv_cdf(alpha)
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    loss_var = -(mu - sigma * z)
    loss_es = -(mu - sigma * pdf / (1 - alpha))
    return float(loss_var), float(loss_es)


def simulate_portfolio_losses(
    returns: pd.DataFrame,
    weights: pd.Series,
    horizon_days: int,
    n_sims: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = returns.mean().to_numpy()
    cov = shrink_covariance(returns).to_numpy()
    weights_array = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    draws = rng.multivariate_normal(mu, cov, size=(n_sims, horizon_days), method="svd")
    simulated_log_returns = draws @ weights_array
    horizon_returns = simulated_log_returns.sum(axis=1)
    return -horizon_returns


def inverse_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    inv_vol = 1.0 / returns.std(ddof=1)
    return inv_vol / inv_vol.sum()


def risk_parity_weights(cov: pd.DataFrame, tolerance: float = 1e-7, max_iter: int = 20_000) -> pd.Series:
    cov_array = cov.to_numpy()
    n = cov_array.shape[0]
    weights = np.ones(n) / n
    target = np.ones(n) / n

    for _ in range(max_iter):
        marginal = cov_array @ weights
        port_var = float(weights @ marginal)
        risk_contrib = weights * marginal / max(port_var, 1e-16)
        error = risk_contrib - target
        if np.max(np.abs(error)) < tolerance:
            break
        weights *= np.power(target / np.maximum(risk_contrib, 1e-12), 0.04)
        weights = np.maximum(weights, 1e-12)
        weights /= weights.sum()

    return pd.Series(weights, index=cov.index)


def random_frontier(
    mean_returns: pd.Series,
    cov: pd.DataFrame,
    n_portfolios: int = N_FRONTIER_PORTFOLIOS,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    tickers = mean_returns.index
    weights = rng.dirichlet(np.ones(len(tickers)), size=n_portfolios)
    mu_ann = mean_returns.to_numpy() * TRADING_DAYS
    cov_ann = cov.to_numpy() * TRADING_DAYS

    ann_return = weights @ mu_ann
    ann_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_ann, weights))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol

    frontier = pd.DataFrame(
        {
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
        }
    )
    for idx, ticker in enumerate(tickers):
        frontier[f"weight_{ticker}"] = weights[:, idx]

    max_sharpe_idx = int(np.nanargmax(sharpe))
    min_vol_idx = int(np.nanargmin(ann_vol))
    max_sharpe = pd.Series(weights[max_sharpe_idx], index=tickers)
    min_variance = pd.Series(weights[min_vol_idx], index=tickers)

    frontier.to_csv(OUTPUT_DIR / "efficient_frontier.csv", index=False)
    return frontier, max_sharpe, min_variance


def build_portfolios(returns: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    cov = shrink_covariance(returns)
    _, max_sharpe, min_variance = random_frontier(returns.mean(), cov)
    equal_weight = pd.Series(np.ones(len(returns.columns)) / len(returns.columns), index=returns.columns)
    inverse_vol = inverse_volatility_weights(returns)
    risk_parity = risk_parity_weights(cov)

    portfolios = {
        "Equal weight": equal_weight,
        "Inverse volatility": inverse_vol,
        "Risk parity": risk_parity,
        "Minimum variance": min_variance,
        "Max Sharpe": max_sharpe,
    }
    weights = pd.DataFrame(portfolios).T
    weights.to_csv(OUTPUT_DIR / "portfolio_weights.csv", index_label="portfolio")
    return portfolios, cov


def portfolio_metrics(returns: pd.DataFrame, portfolios: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, weights in portfolios.items():
        series = portfolio_return_series(returns, weights)
        ann_return = cagr(series)
        ann_vol = float(series.std(ddof=1) * math.sqrt(TRADING_DAYS))
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else float("nan")
        losses = -series
        var95, es95 = var_es_from_losses(losses, 0.95)
        var99, es99 = var_es_from_losses(losses, 0.99)
        pvar99, pes99 = parametric_var_es(series, 0.99)
        mc_losses = simulate_portfolio_losses(returns, weights, 10, N_MONTE_CARLO, RANDOM_SEED)
        mc_var99, mc_es99 = var_es_from_losses(mc_losses, 0.99)
        rows.append(
            {
                "portfolio": name,
                "cagr": ann_return,
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(series),
                "skewness": skewness(series),
                "excess_kurtosis": excess_kurtosis(series),
                "hist_var_95_1d": var95,
                "hist_es_95_1d": es95,
                "hist_var_99_1d": var99,
                "hist_es_99_1d": es99,
                "param_var_99_1d": pvar99,
                "param_es_99_1d": pes99,
                "mc_var_99_10d": mc_var99,
                "mc_es_99_10d": mc_es99,
            }
        )

    metrics = pd.DataFrame(rows).set_index("portfolio")
    metrics.to_csv(OUTPUT_DIR / "portfolio_metrics.csv", index_label="portfolio")
    var_columns = [
        "hist_var_95_1d",
        "hist_es_95_1d",
        "hist_var_99_1d",
        "hist_es_99_1d",
        "param_var_99_1d",
        "param_es_99_1d",
        "mc_var_99_10d",
        "mc_es_99_10d",
    ]
    metrics[var_columns].to_csv(OUTPUT_DIR / "var_summary.csv", index_label="portfolio")
    return metrics


def stress_tests(portfolios: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for scenario, shocks in SCENARIOS.items():
        shock_vector = pd.Series(shocks)
        for name, weights in portfolios.items():
            impact = float(weights.reindex(shock_vector.index).fillna(0.0) @ shock_vector)
            rows.append(
                {
                    "scenario": scenario,
                    "portfolio": name,
                    "portfolio_return": impact,
                    "portfolio_loss": -impact,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "stress_tests.csv", index=False)
    return result


def rolling_var_backtest(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_name: str,
    window: int = 250,
    alpha: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, float]]:
    series = portfolio_return_series(returns, weights)
    records = []
    for i in range(window, len(series)):
        history = series.iloc[i - window : i]
        realised_loss = -float(series.iloc[i])
        var, _ = var_es_from_losses(-history, alpha)
        records.append(
            {
                "date": series.index[i],
                "portfolio": portfolio_name,
                "realised_loss": realised_loss,
                "var_99": var,
                "exception": int(realised_loss > var),
            }
        )
    backtest = pd.DataFrame(records)
    backtest.to_csv(OUTPUT_DIR / "rolling_var_backtest.csv", index=False)

    n = len(backtest)
    exceptions = int(backtest["exception"].sum()) if n else 0
    expected_rate = 1 - alpha
    observed_rate = exceptions / n if n else 0.0
    if exceptions == 0 or exceptions == n:
        kupiec_lr = 0.0 if abs(observed_rate - expected_rate) < 1e-12 else float("inf")
        kupiec_p_value = 0.0 if math.isinf(kupiec_lr) else 1.0
    else:
        unrestricted = (n - exceptions) * math.log(1 - observed_rate) + exceptions * math.log(observed_rate)
        restricted = (n - exceptions) * math.log(1 - expected_rate) + exceptions * math.log(expected_rate)
        kupiec_lr = -2 * (restricted - unrestricted)
        kupiec_p_value = math.erfc(math.sqrt(max(kupiec_lr, 0.0) / 2))

    summary = {
        "observations": float(n),
        "exceptions": float(exceptions),
        "expected_exception_rate": expected_rate,
        "observed_exception_rate": observed_rate,
        "kupiec_lr": kupiec_lr,
        "kupiec_p_value": kupiec_p_value,
    }
    return backtest, summary


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def num(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def table_to_markdown(df: pd.DataFrame, percentage_columns: set[str] | None = None) -> str:
    percentage_columns = percentage_columns or set()
    render_df = df.copy()
    for col in render_df.columns:
        if col in percentage_columns:
            render_df[col] = render_df[col].map(lambda x: pct(float(x)))
        elif pd.api.types.is_float_dtype(render_df[col]):
            render_df[col] = render_df[col].map(lambda x: num(float(x)))
    render_df = render_df.reset_index()
    headers = list(render_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in render_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def wealth_index(returns: pd.Series) -> pd.Series:
    return np.exp(returns.cumsum())


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = wealth_index(returns)
    return wealth / wealth.cummax() - 1.0


def svg_line_chart(series_map: dict[str, pd.Series], title: str, width: int = 920, height: int = 330) -> str:
    margin = {"top": 30, "right": 25, "bottom": 45, "left": 55}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    colours = ["#126782", "#f28f3b", "#4f772d", "#7b2cbf", "#c1121f", "#3a86ff"]

    all_values = pd.concat(series_map.values()).dropna()
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    if math.isclose(y_min, y_max):
        y_min -= 1
        y_max += 1
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    max_len = max(len(s.dropna()) for s in series_map.values())

    def point(idx: int, value: float) -> tuple[float, float]:
        x = margin["left"] + (idx / max(max_len - 1, 1)) * inner_w
        y = margin["top"] + (1 - (value - y_min) / (y_max - y_min)) * inner_h
        return x, y

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin["left"]}" y="20" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + inner_h}" x2="{margin["left"] + inner_w}" y2="{margin["top"] + inner_h}" class="axis"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + inner_h}" class="axis"/>',
    ]

    for i in range(5):
        y_value = y_min + (y_max - y_min) * i / 4
        y = margin["top"] + (1 - i / 4) * inner_h
        elements.append(f'<line x1="{margin["left"]}" y1="{y:.2f}" x2="{margin["left"] + inner_w}" y2="{y:.2f}" class="grid"/>')
        elements.append(f'<text x="8" y="{y + 4:.2f}" class="axis-label">{y_value:.2f}</text>')

    for colour, (label, series) in zip(colours, series_map.items()):
        clean = series.dropna().reset_index(drop=True)
        coords = [point(i, float(value)) for i, value in enumerate(clean)]
        path = " ".join([("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(coords)])
        elements.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2.4"/>')

    legend_x = margin["left"]
    legend_y = height - 18
    for idx, (colour, label) in enumerate(zip(colours, series_map.keys())):
        x = legend_x + idx * 155
        elements.append(f'<rect x="{x}" y="{legend_y - 10}" width="10" height="10" fill="{colour}"/>')
        elements.append(f'<text x="{x + 15}" y="{legend_y}" class="legend">{html.escape(label)}</text>')

    elements.append("</svg>")
    return "\n".join(elements)


def svg_bar_chart(values: pd.Series, title: str, width: int = 920, height: int = 330) -> str:
    values = values.copy()
    margin = {"top": 35, "right": 20, "bottom": 75, "left": 55}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    max_abs = max(abs(float(values.min())), abs(float(values.max())), 1e-9)
    bar_w = inner_w / len(values) * 0.62
    gap = inner_w / len(values)
    zero_y = margin["top"] + inner_h / 2

    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin["left"]}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin["left"]}" y1="{zero_y:.2f}" x2="{margin["left"] + inner_w}" y2="{zero_y:.2f}" class="axis"/>',
    ]
    for idx, (label, value) in enumerate(values.items()):
        x = margin["left"] + idx * gap + (gap - bar_w) / 2
        scaled = abs(float(value)) / max_abs * (inner_h / 2)
        y = zero_y - scaled if value >= 0 else zero_y
        colour = "#126782" if value >= 0 else "#c1121f"
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{scaled:.2f}" fill="{colour}" rx="2"/>')
        elements.append(f'<text x="{x + bar_w / 2:.2f}" y="{height - 43}" class="x-label" text-anchor="middle">{html.escape(str(label))}</text>')
        elements.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 6 if value >= 0 else y + scaled + 14:.2f}" class="bar-label" text-anchor="middle">{pct(float(value))}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def svg_scatter(frontier: pd.DataFrame, selected: pd.DataFrame, width: int = 920, height: int = 330) -> str:
    margin = {"top": 30, "right": 25, "bottom": 48, "left": 60}
    inner_w = width - margin["left"] - margin["right"]
    inner_h = height - margin["top"] - margin["bottom"]
    x = frontier["annual_volatility"]
    y = frontier["annual_return"]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = (x_max - x_min) * 0.06
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def scale(px: float, py: float) -> tuple[float, float]:
        sx = margin["left"] + (px - x_min) / (x_max - x_min) * inner_w
        sy = margin["top"] + (1 - (py - y_min) / (y_max - y_min)) * inner_h
        return sx, sy

    sample = frontier.sample(min(2500, len(frontier)), random_state=RANDOM_SEED)
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Efficient frontier simulation">',
        f'<text x="{margin["left"]}" y="20" class="chart-title">Long-only random portfolio frontier</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"] + inner_h}" x2="{margin["left"] + inner_w}" y2="{margin["top"] + inner_h}" class="axis"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + inner_h}" class="axis"/>',
        f'<text x="{margin["left"] + inner_w / 2}" y="{height - 8}" class="axis-label" text-anchor="middle">Annual volatility</text>',
        f'<text x="16" y="{margin["top"] + inner_h / 2}" class="axis-label" transform="rotate(-90 16,{margin["top"] + inner_h / 2})" text-anchor="middle">Annual return</text>',
    ]
    for _, row in sample.iterrows():
        sx, sy = scale(float(row["annual_volatility"]), float(row["annual_return"]))
        elements.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="1.6" fill="#90a4ae" opacity="0.42"/>')
    colours = {
        "Equal weight": "#126782",
        "Inverse volatility": "#f28f3b",
        "Risk parity": "#4f772d",
        "Minimum variance": "#7b2cbf",
        "Max Sharpe": "#c1121f",
    }
    for name, row in selected.iterrows():
        sx, sy = scale(float(row["annual_volatility"]), float(row["cagr"]))
        colour = colours.get(name, "#111827")
        elements.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="5.5" fill="{colour}" stroke="#ffffff" stroke-width="1.5"/>')
        elements.append(f'<text x="{sx + 8:.2f}" y="{sy - 6:.2f}" class="legend">{html.escape(name)}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def dataframe_to_html_table(df: pd.DataFrame, pct_cols: set[str] | None = None, max_rows: int | None = None) -> str:
    pct_cols = pct_cols or set()
    if max_rows is not None:
        df = df.head(max_rows)
    render = df.copy()
    for col in render.columns:
        if col in pct_cols:
            render[col] = render[col].map(lambda value: pct(float(value)))
        elif pd.api.types.is_float_dtype(render[col]):
            render[col] = render[col].map(lambda value: f"{float(value):.3f}")
    return render.to_html(classes="data-table", border=0, escape=True)


def generate_dashboard(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    metrics: pd.DataFrame,
    stress: pd.DataFrame,
    frontier: pd.DataFrame,
    backtest_summary: dict[str, float],
) -> None:
    selected_returns = {name: portfolio_return_series(returns, weights) for name, weights in portfolios.items()}
    wealth = {name: wealth_index(series) for name, series in selected_returns.items()}
    drawdowns = {name: drawdown_series(series) for name, series in selected_returns.items()}
    max_sharpe_weights = portfolios["Max Sharpe"].sort_values(ascending=False)
    stress_pivot = stress.pivot(index="scenario", columns="portfolio", values="portfolio_return")
    data_start = prices.index.min().date().isoformat()
    data_end = prices.index.max().date().isoformat()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pct_cols_metrics = {
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "hist_var_95_1d",
        "hist_es_95_1d",
        "hist_var_99_1d",
        "hist_es_99_1d",
        "param_var_99_1d",
        "param_es_99_1d",
        "mc_var_99_10d",
        "mc_es_99_10d",
    }

    style = """
    :root {
      --ink: #17202a;
      --muted: #5d6d7e;
      --panel: #ffffff;
      --line: #d7dee5;
      --bg: #f5f7f9;
      --accent: #126782;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }
    header {
      background: #102a43;
      color: white;
      padding: 28px 42px;
    }
    header h1 {
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }
    header p {
      margin: 0;
      color: #dbeafe;
      max-width: 900px;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 26px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px;
      margin-bottom: 18px;
    }
    h2 {
      font-size: 20px;
      margin: 0 0 14px;
    }
    .kpis {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .kpi {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      background: #fbfcfd;
    }
    .kpi .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .kpi .value {
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
    }
    .grid-two {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }
    .chart-wrap {
      overflow-x: auto;
    }
    svg {
      width: 100%;
      height: auto;
      min-width: 620px;
    }
    .axis {
      stroke: #78909c;
      stroke-width: 1;
    }
    .grid {
      stroke: #e5ebf0;
      stroke-width: 1;
    }
    .chart-title {
      font-size: 15px;
      font-weight: 700;
      fill: var(--ink);
    }
    .axis-label, .legend, .x-label, .bar-label {
      font-size: 11px;
      fill: var(--muted);
    }
    .bar-label {
      fill: var(--ink);
      font-weight: 700;
    }
    .data-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }
    .data-table th, .data-table td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    .data-table th:first-child, .data-table td:first-child {
      text-align: left;
    }
    .table-scroll {
      overflow-x: auto;
    }
    @media (max-width: 700px) {
      header { padding: 22px; }
      main { padding: 16px; }
      section { padding: 16px; }
      .grid-two { grid-template-columns: 1fr; }
    }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Asset Risk Engine Dashboard</title>
  <style>{style}</style>
</head>
<body>
  <header>
    <h1>Multi-Asset Risk Engine and Portfolio Optimizer</h1>
    <p>Cross-asset allocation, tail-risk measurement, stress testing, and VaR backtesting using daily market data from {data_start} to {data_end}. Generated {generated}.</p>
  </header>
  <main>
    <section>
      <div class="kpis">
        <div class="kpi"><div class="label">Assets</div><div class="value">{len(ASSETS)}</div></div>
        <div class="kpi"><div class="label">Trading days</div><div class="value">{len(returns):,}</div></div>
        <div class="kpi"><div class="label">Best Sharpe</div><div class="value">{num(float(metrics["sharpe"].max()))}</div></div>
        <div class="kpi"><div class="label">Lowest drawdown</div><div class="value">{pct(float(metrics["max_drawdown"].max()))}</div></div>
      </div>
      <div class="chart-wrap">{svg_scatter(frontier, metrics)}</div>
    </section>
    <section>
      <h2>Portfolio Growth and Drawdowns</h2>
      <div class="grid-two">
        <div class="chart-wrap">{svg_line_chart(wealth, "Wealth index")}</div>
        <div class="chart-wrap">{svg_line_chart(drawdowns, "Drawdown")}</div>
      </div>
    </section>
    <section>
      <h2>Max-Sharpe Allocation and Scenario Returns</h2>
      <div class="grid-two">
        <div class="chart-wrap">{svg_bar_chart(max_sharpe_weights, "Max-Sharpe weights")}</div>
        <div class="table-scroll">{dataframe_to_html_table(stress_pivot, pct_cols=set(stress_pivot.columns))}</div>
      </div>
    </section>
    <section>
      <h2>Metrics</h2>
      <div class="table-scroll">{dataframe_to_html_table(metrics, pct_cols=pct_cols_metrics)}</div>
    </section>
    <section>
      <h2>Rolling VaR Backtest</h2>
      <div class="kpis">
        <div class="kpi"><div class="label">Observations</div><div class="value">{int(backtest_summary["observations"]):,}</div></div>
        <div class="kpi"><div class="label">Exceptions</div><div class="value">{int(backtest_summary["exceptions"])}</div></div>
        <div class="kpi"><div class="label">Observed exception rate</div><div class="value">{pct(backtest_summary["observed_exception_rate"])}</div></div>
        <div class="kpi"><div class="label">Kupiec p-value</div><div class="value">{num(backtest_summary["kupiec_p_value"])}</div></div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    (OUTPUT_DIR / "dashboard.html").write_text(html_doc, encoding="utf-8")


def generate_report(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    metrics: pd.DataFrame,
    stress: pd.DataFrame,
    backtest_summary: dict[str, float],
) -> None:
    weights = pd.DataFrame(portfolios).T
    weight_pct = weights.map(lambda value: pct(float(value)))
    metric_view = metrics[
        [
            "cagr",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "hist_var_99_1d",
            "hist_es_99_1d",
            "mc_var_99_10d",
            "mc_es_99_10d",
        ]
    ]
    pct_cols = {
        "cagr",
        "annual_volatility",
        "max_drawdown",
        "hist_var_99_1d",
        "hist_es_99_1d",
        "mc_var_99_10d",
        "mc_es_99_10d",
    }
    selected = "Max Sharpe"
    stress_selected = stress[stress["portfolio"] == selected].set_index("scenario")[["portfolio_return", "portfolio_loss"]]
    best_sharpe = metrics["sharpe"].idxmax()
    lowest_vol = metrics["annual_volatility"].idxmin()
    lowest_drawdown = metrics["max_drawdown"].idxmax()
    data_start = prices.index.min().date().isoformat()
    data_end = prices.index.max().date().isoformat()

    report = f"""# Multi-Asset Risk Engine and Portfolio Optimizer

Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Executive summary

This project builds a full risk and allocation workflow for a multi-asset portfolio covering equities, rates, credit, commodities, FX, and crypto. The data set runs from {data_start} to {data_end} with {len(returns):,} daily return observations after alignment to the US equity market calendar.

The strongest risk-adjusted portfolio in this long-only random search is **{best_sharpe}**. The lowest volatility portfolio is **{lowest_vol}**, while the least severe historical maximum drawdown is **{lowest_drawdown}**.

## Asset universe

| Ticker | Asset | Class |
|---|---|---|
{chr(10).join(f"| {asset.ticker} | {asset.name} | {asset.asset_class} |" for asset in ASSETS)}

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
- Max Sharpe: selected from 60,000 long-only random portfolios using a {pct(RISK_FREE_RATE)} risk-free assumption.

## Portfolio weights

{table_to_markdown(weight_pct)}

## Performance and risk metrics

{table_to_markdown(metric_view, percentage_columns=pct_cols)}

## Stress test impact for selected portfolio: {selected}

{table_to_markdown(stress_selected, percentage_columns={"portfolio_return", "portfolio_loss"})}

## Rolling 99% VaR backtest for selected portfolio: {selected}

| Metric | Value |
|---|---|
| Observations | {int(backtest_summary["observations"]):,} |
| Exceptions | {int(backtest_summary["exceptions"])} |
| Expected exception rate | {pct(backtest_summary["expected_exception_rate"])} |
| Observed exception rate | {pct(backtest_summary["observed_exception_rate"])} |
| Kupiec LR statistic | {num(backtest_summary["kupiec_lr"])} |
| Kupiec p-value | {num(backtest_summary["kupiec_p_value"])} |

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
"""
    (OUTPUT_DIR / "risk_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_directories()
    prices = load_price_matrix()
    returns = log_returns(prices)
    portfolios, _ = build_portfolios(returns)
    frontier = pd.read_csv(OUTPUT_DIR / "efficient_frontier.csv")
    metrics = portfolio_metrics(returns, portfolios)
    stress = stress_tests(portfolios)
    _, backtest_summary = rolling_var_backtest(returns, portfolios["Max Sharpe"], "Max Sharpe")
    generate_report(prices, returns, portfolios, metrics, stress, backtest_summary)
    generate_dashboard(prices, returns, portfolios, metrics, stress, frontier, backtest_summary)
    print(f"Project complete. Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
