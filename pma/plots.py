from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import TIMEFRAME_UA
from .moving_average import METHODS, SweepResult


def plot_price_ma(
    series: pd.Series,
    ma: dict[str, np.ndarray],
    window: int,
    timeframe: str,
    ticker: str,
    path: Path,
    last_points: int,
) -> None:
    tail = slice(-last_points, None)
    idx = series.index[tail]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(idx, series.values[tail], label="Ціна закриття", color="black", lw=1)
    for method, color in zip(METHODS, ("tab:blue", "tab:red")):
        forecast = np.roll(ma[method], 1)
        ax.plot(idx, forecast[tail], label=f"Прогноз {method}({window})", color=color, lw=1.2)
    ax.set_title(f"{ticker}: прогноз на наступний крок, вікно N={window} ({TIMEFRAME_UA[timeframe]})")
    ax.set_ylabel("Ціна")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_window_errors(sweeps: dict[str, SweepResult], ticker: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6 * len(sweeps), 5))
    for ax, (timeframe, res) in zip(np.atleast_1d(axes), sweeps.items()):
        for j, (method, color) in enumerate(zip(METHODS, ("tab:blue", "tab:red"))):
            ax.plot(res.windows, res.mape[:, j], label=method, color=color)
            best = int(np.argmin(res.mape[:, j]))
            ax.scatter(res.windows[best], res.mape[best, j], color=color, zorder=3)
        ax.set_title(f"Крок: {TIMEFRAME_UA[timeframe]} ({res.count} прогнозів)")
        ax.set_xlabel(f"Розмір вікна N, {TIMEFRAME_UA[timeframe]}")
        ax.set_ylabel("MAPE, %")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{ticker}: похибка прогнозу на наступний крок залежно від вікна")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_speedup(times: dict[int, float], perf_cores: int | None, title: str, path: Path) -> None:
    workers = np.array(sorted(times))
    t = np.array([times[w] for w in workers])
    speedup = t[0] / t
    efficiency = speedup / workers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(workers, speedup, "o-", label="Фактичне прискорення", color="tab:blue")
    ax1.plot(workers, workers, "--", label="Ідеальне (лінійне)", color="gray")
    for w, s in zip(workers, speedup):
        ax1.annotate(f"{s:.2f}", (w, s), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    if perf_cores and perf_cores < workers.max():
        ax1.axvline(perf_cores + 0.5, color="tab:orange", ls=":", label=f"Межа {perf_cores} продуктивних ядер")
    ax1.set_xlabel("Кількість процесів (ядер)")
    ax1.set_ylabel("Прискорення S(p) = T(1) / T(p)")
    ax1.set_xticks(workers)
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.bar(workers, t, color="tab:green", alpha=0.7, label="Час, с")
    ax2.set_xlabel("Кількість процесів (ядер)")
    ax2.set_ylabel("Час обробки, с")
    ax2.set_xticks(workers)
    ax2b = ax2.twinx()
    ax2b.plot(workers, efficiency * 100, "o-", color="tab:purple", label="Ефективність, %")
    ax2b.set_ylabel("Ефективність E(p) = S(p)/p, %")
    ax2b.set_ylim(0, 110)
    ax2.grid(alpha=0.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
