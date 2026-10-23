"""Паралельний розрахунок SMA / WMA та похибок прогнозу на наступний крок.

Схема розпаралелювання — декомпозиція за даними: множина індексів t, для яких
рахується середнє, ділиться на неперервні блоки. Кожен процес отримує весь ряд
один раз (через initializer пулу) і рахує свої блоки незалежно; для точки t
потрібні лише prices[t-N+1 .. t], тому блоки не залежать один від одного.

Прогноз на наступний крок: y_hat[t+1] = MA_N(t).

SMA_N(t) = (1/N) * sum_{k=0}^{N-1} y[t-k]
WMA_N(t) = sum_{k=0}^{N-1} (N-k) * y[t-k] / (N(N+1)/2)   (лінійні ваги, новіші важать більше)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

METHODS = ("SMA", "WMA")

_PRICES: np.ndarray | None = None


def _init_worker(prices: np.ndarray) -> None:
    global _PRICES
    _PRICES = prices


def _weights(method: str, window: int) -> np.ndarray:
    if method == "SMA":
        w = np.ones(window)
    else:
        w = np.arange(1, window + 1, dtype=np.float64)
    return w / w.sum()


def _ma_range(prices: np.ndarray, lo: int, hi: int, window: int, weights: np.ndarray) -> np.ndarray:
    """MA для t у [lo, hi), lo >= window-1. Пряме підсумовування O((hi-lo)·N)."""
    view = sliding_window_view(prices[lo - window + 1 : hi], window)
    return view @ weights


def split_range(lo: int, hi: int, parts: int) -> list[tuple[int, int]]:
    bounds = np.linspace(lo, hi, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]




def _ma_chunk(args: tuple[int, int, int]) -> tuple[int, np.ndarray]:
    start, stop, window = args
    prices = _PRICES
    out = np.full((len(METHODS), stop - start), np.nan)
    lo = max(start, window - 1)
    if lo < stop:
        for j, method in enumerate(METHODS):
            out[j, lo - start :] = _ma_range(prices, lo, stop, window, _weights(method, window))
    return start, out


def moving_averages(prices: np.ndarray, window: int, workers: int = 1) -> dict[str, np.ndarray]:
    """SMA і WMA для всього ряду (NaN, поки не набралось N точок)."""
    n = len(prices)
    result = np.empty((len(METHODS), n))
    chunks = [(a, b, window) for a, b in split_range(0, n, max(1, workers))]
    if workers <= 1:
        _init_worker(prices)
        parts = map(_ma_chunk, chunks)
        for start, block in parts:
            result[:, start : start + block.shape[1]] = block
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(prices,)) as pool:
            for start, block in pool.map(_ma_chunk, chunks):
                result[:, start : start + block.shape[1]] = block
    return dict(zip(METHODS, result))



def _sweep_chunk(args: tuple[int, int, tuple[int, ...]]) -> np.ndarray:
    """Часткові суми похибок прогнозу для t у [start, stop) по всіх вікнах.

    Повертає масив [вікно, метод, (count, sum|e|, sum e^2, sum |e|/y)].
    """
    start, stop, windows = args
    prices = _PRICES
    target = prices[start + 1 : stop + 1]
    out = np.zeros((len(windows), len(METHODS), 4))
    for i, window in enumerate(windows):
        for j, method in enumerate(METHODS):
            forecast = _ma_range(prices, start, stop, window, _weights(method, window))
            err = np.abs(target - forecast)
            out[i, j] = (err.size, err.sum(), (err * err).sum(), (err / target).sum())
    return out


@dataclass
class SweepResult:
    windows: np.ndarray
    mae: np.ndarray  # [вікно, метод]
    rmse: np.ndarray
    mape: np.ndarray  # у відсотках
    count: int


def _reduce(partials: list[np.ndarray], windows: tuple[int, ...]) -> SweepResult:
    total = np.sum(partials, axis=0)
    cnt = total[..., 0]
    return SweepResult(
        windows=np.asarray(windows),
        mae=total[..., 1] / cnt,
        rmse=np.sqrt(total[..., 2] / cnt),
        mape=100 * total[..., 3] / cnt,
        count=int(cnt[0, 0]),
    )


def sweep_tasks(n: int, windows: tuple[int, ...], parts: int) -> list[tuple[int, int, tuple[int, ...]]]:
    """Усі вікна оцінюються на однаковому відрізку цілей, щоб їх можна було порівнювати."""
    eval_start = max(windows) - 1
    return [(a, b, windows) for a, b in split_range(eval_start, n - 1, parts)]


def window_sweep(prices: np.ndarray, windows: tuple[int, ...], workers: int = 1, chunks_per_worker: int = 4) -> SweepResult:
    tasks = sweep_tasks(len(prices), windows, workers * chunks_per_worker)
    if workers <= 1:
        _init_worker(prices)
        return _reduce([_sweep_chunk(t) for t in tasks], windows)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(prices,)) as pool:
        return _reduce(list(pool.map(_sweep_chunk, tasks)), windows)



def _noop(_: int) -> None:
    return None


def benchmark(
    prices: np.ndarray,
    windows: tuple[int, ...],
    worker_counts: list[int],
    repeats: int = 3,
    chunks_per_worker: int = 4,
) -> dict[int, float]:
    """Найкращий з `repeats` час перебору вікон для кожної кількості процесів.

    Пул створюється і прогрівається до заміру, тому час не включає запуск процесів.
    1 процес — те саме ядро без пулу (чисто послідовне виконання).
    """
    times: dict[int, float] = {}
    for workers in worker_counts:
        tasks = sweep_tasks(len(prices), windows, workers * chunks_per_worker)
        best = float("inf")
        if workers == 1:
            _init_worker(prices)
            for _ in range(repeats):
                t0 = time.perf_counter()
                _reduce([_sweep_chunk(t) for t in tasks], windows)
                best = min(best, time.perf_counter() - t0)
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(prices,)) as pool:
                list(pool.map(_noop, range(workers * 4)))
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    _reduce(list(pool.map(_sweep_chunk, tasks)), windows)
                    best = min(best, time.perf_counter() - t0)
        times[workers] = best
        print(f"  {workers:>2} проц.: {best:.3f} с")
    return times
