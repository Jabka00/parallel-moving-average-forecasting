from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import pandas as pd

TIMEFRAMES = {
    "day": None,
    "week": "W-FRI",
    "month": "ME",
}

TIMEFRAME_UA = {"day": "день", "week": "тиждень", "month": "місяць"}


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("^", "").replace("/", "_")
    return cache_dir / f"{safe}_{start}_{end}.csv"


def load_prices(ticker: str, years: int, cache_dir: Path, end: str | None = None) -> pd.Series:
    """Денні ціни закриття (з урахуванням спліт/дивідендів) за останні `years` років."""
    end = end or date.today().isoformat()
    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).date().isoformat()

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, ticker, start, end)
    if path.exists():
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"Не вдалося завантажити дані для {ticker}")
        close = df["Close"]
        series = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        series = series.dropna()
        series.to_frame("Close").to_csv(path)

    series.name = "Close"
    series.index.name = "Date"
    return series.astype("float64")


def resample(series: pd.Series, timeframe: str) -> pd.Series:
    """Ціна закриття останнього торгового дня тижня/місяця."""
    rule = TIMEFRAMES[timeframe]
    if rule is None:
        return series
    return series.resample(rule).last().dropna()
