"""Паралельне рухоме середнє (SMA/WMA), прогноз на наступний крок і бенчмарк.

Приклад:
    python main.py --ticker ^GSPC --years 50 --timeframe day --window 20
"""

import os

# BLAS всередині numpy не повинен сам розпаралелюватись, інакше заміри по ядрах некоректні.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pma.data import TIMEFRAME_UA, TIMEFRAMES, load_prices, resample
from pma.moving_average import METHODS, benchmark, moving_averages, window_sweep
from pma.plots import plot_price_ma, plot_speedup, plot_window_errors

ROOT = Path(__file__).resolve().parent
MAX_WINDOW = {"day": 250, "week": 104, "month": 60}


def perf_core_count() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def check_correctness(prices: np.ndarray, ma: dict[str, np.ndarray], window: int) -> None:
    sma_ref = pd.Series(prices).rolling(window).mean().to_numpy()
    w = np.arange(1, window + 1, dtype=float)
    wma_ref = np.full_like(prices, np.nan)
    wma_ref[window - 1 :] = np.convolve(prices, w[::-1] / w.sum(), mode="valid")
    for name, ref in (("SMA", sma_ref), ("WMA", wma_ref)):
        if not np.allclose(ma[name], ref, equal_nan=True, rtol=1e-10):
            raise AssertionError(f"Паралельний {name} не збігається з еталоном")


def forecast_errors(prices: np.ndarray, ma: np.ndarray, window: int) -> dict[str, float]:
    forecast, actual = ma[window - 1 : -1], prices[window:]
    err = actual - forecast
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / actual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="^GSPC", help="тікер Yahoo Finance (за замовчуванням S&P 500)")
    parser.add_argument("--years", type=int, default=50, help="глибина історії, 10-50 років")
    parser.add_argument("--timeframe", choices=list(TIMEFRAMES), default="day", help="одиниця вікна")
    parser.add_argument("--window", type=int, default=20, help="довжина вікна N")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="макс. кількість процесів у бенчмарку")
    parser.add_argument("--bench-windows", type=int, default=500, help="вікна 1..K у бенчмарку (денні дані)")
    parser.add_argument("--repeats", type=int, default=3, help="повторів на кожну кількість процесів")
    args = parser.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    daily = load_prices(args.ticker, args.years, ROOT / "data")
    print(f"{args.ticker}: {len(daily)} торгових днів, {daily.index[0].date()} — {daily.index[-1].date()}")
    summary: dict = {"ticker": args.ticker, "from": str(daily.index[0].date()), "to": str(daily.index[-1].date())}

    # 1. Вибране вікно: паралельний розрахунок, перевірка, похибка, прогноз
    series = resample(daily, args.timeframe)
    prices = series.to_numpy()
    tf = TIMEFRAME_UA[args.timeframe]
    print(f"\n[1] Крок «{tf}», {len(prices)} точок, вікно N={args.window}")
    ma = moving_averages(prices, args.window, workers=os.cpu_count())
    check_correctness(prices, ma, args.window)
    print("    Паралельний результат збігається з еталонним (pandas/numpy)")
    selected = {}
    for method in METHODS:
        errs = forecast_errors(prices, ma[method], args.window)
        next_value = float(ma[method][-1])
        selected[method] = {**errs, "next_step_forecast": next_value}
        print(
            f"    {method}({args.window}): прогноз на наступний {tf} після {series.index[-1].date()} = {next_value:.2f}"
            f" | MAE={errs['MAE']:.2f}  RMSE={errs['RMSE']:.2f}  MAPE={errs['MAPE_%']:.3f}%"
        )
    summary["selected"] = {"timeframe": args.timeframe, "window": args.window, "last_price": float(prices[-1]), **selected}
    plot_price_ma(
        series, ma, args.window, args.timeframe, args.ticker,
        out_dir / f"forecast_{args.timeframe}_N{args.window}.png",
        last_points={"day": 500, "week": 260, "month": 120}[args.timeframe],
    )

    # 2. Який розмір вікна найкращий
    print("\n[2] Перебір розміру вікна (паралельно)")
    sweeps = {}
    rows = []
    summary["best_windows"] = {}
    for timeframe, max_window in MAX_WINDOW.items():
        tf_prices = resample(daily, timeframe).to_numpy()
        res = window_sweep(tf_prices, tuple(range(1, max_window + 1)), workers=os.cpu_count())
        sweeps[timeframe] = res
        summary["best_windows"][timeframe] = {}
        for j, method in enumerate(METHODS):
            for i, w in enumerate(res.windows):
                rows.append((timeframe, method, int(w), res.mae[i, j], res.rmse[i, j], res.mape[i, j]))
            best = int(np.argmin(res.mape[1:, j])) + 1  # N=1 — це наївний прогноз «як учора»
            summary["best_windows"][timeframe][method] = {
                "window": int(res.windows[best]),
                "MAPE_%": float(res.mape[best, j]),
                "naive_MAPE_%": float(res.mape[0, j]),
            }
        sma_best = summary["best_windows"][timeframe]["SMA"]
        wma_best = summary["best_windows"][timeframe]["WMA"]
        print(
            f"    {TIMEFRAME_UA[timeframe]:>8}: наївний (N=1) MAPE={res.mape[0, 0]:.3f}% | "
            f"найкраще N≥2: SMA N={sma_best['window']} ({sma_best['MAPE_%']:.3f}%), "
            f"WMA N={wma_best['window']} ({wma_best['MAPE_%']:.3f}%)"
        )
    pd.DataFrame(rows, columns=["timeframe", "method", "window", "MAE", "RMSE", "MAPE_%"]).to_csv(
        out_dir / "window_errors.csv", index=False
    )
    plot_window_errors(sweeps, args.ticker, out_dir / "window_errors.png")

    # 3. Бенчмарк 1..N ядер
    bench_windows = tuple(range(1, args.bench_windows + 1))
    print(f"\n[3] Бенчмарк: денні дані ({len(daily)} точок), SMA+WMA для вікон 1..{args.bench_windows}")
    times = benchmark(daily.to_numpy(), bench_windows, list(range(1, args.max_workers + 1)), repeats=args.repeats)
    t1 = times[1]
    summary["benchmark"] = {
        str(p): {"time_s": t, "speedup": t1 / t, "efficiency": t1 / t / p} for p, t in times.items()
    }
    pd.DataFrame(
        [(p, t, t1 / t, t1 / t / p) for p, t in times.items()], columns=["workers", "time_s", "speedup", "efficiency"]
    ).to_csv(out_dir / "benchmark.csv", index=False)
    plot_speedup(
        times, perf_core_count(),
        f"Прискорення SMA+WMA ({len(daily)} днів × {len(bench_windows)} вікон)",
        out_dir / "speedup.png",
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nРезультати збережено в {out_dir}")


if __name__ == "__main__":
    main()
