"""JJ Simon Fair Value Theory backtester, v2.

Core model:
- NQ 1-minute, New York time
- Fair value = 09:30 open and 14:00 open
- continuation early, mean reversion later
- entry requires structure break + decisive displacement on the same candle
- fixed 1.5R target, no discretionary trade management
- ATR tiers: >20 -> 50pt SL, 7-20 -> 25pt SL, <7 -> 16.5pt SL

Optimized timing filters reproduced from the independent video backtest:
- skip the first 3 minutes of the NY-open continuation window
- AM mean reversions only in the first 30 minutes
- PM mean reversions only in the first hour of the afternoon test window

IMPORTANT:
The public description does not specify an exact swing/pivot lookback. This
implementation uses a configurable 2-bar pivot to make BOS/MSB mechanical.
That is a reconstruction choice, not a claim that JJ specified "2 bars".
"""

from __future__ import annotations

import argparse
from datetime import time

import numpy as np
import pandas as pd

# ----------------------------- strategy parameters -----------------------------

WINDOWS = [
    ("am", time(9, 30), time(11, 0)),
    # The published core window is 14:00-15:00. The video optimization explicitly
    # discusses filtering 15:00-16:00 PM reversions, so optimized mode tests 14:00-16:00.
    ("pm", time(14, 0), time(16, 0)),
]

CONT_MIN = 15
ATR_PERIOD = 14
RR = 1.5

# NQ point distances, matching the risk arithmetic in the published strategy.
ATR_HIGH = 20.0
ATR_LOW = 7.0
SL_HIGH = 50.0
SL_NORMAL = 25.0
SL_LOW = 16.5

# Mechanical reconstruction of structure.
SWING_LEFT = 2
SWING_RIGHT = 2

# Published displacement rule: counter-wick <= 20%.
COUNTER_WICK_MAX = 0.20

# Filters reported as improving the independent backtest.
OPTIMIZED_TIMING = True
SKIP_FIRST_3M_CONTINUATION = True
AM_REVERSION_MAX_MINUTE = 30
PM_REVERSION_MAX_MINUTE = 60

# If False, keep the published core session windows (PM ends at 15:00).
USE_PM_16 = True


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index = (
        df.index.tz_localize("UTC")
        if df.index.tz is None
        else df.index.tz_convert("UTC")
    )
    df.index = df.index.tz_convert("America/New_York")
    return df[["open", "high", "low", "close"]].sort_index()


def atr(df: pd.DataFrame, n: int = ATR_PERIOD) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()),
    )
    return tr.rolling(n, min_periods=1).mean().to_numpy()


def stop_distance(atr_value: float) -> float:
    if atr_value > ATR_HIGH:
        return SL_HIGH
    if atr_value >= ATR_LOW:
        return SL_NORMAL
    return SL_LOW


def displacement(open_: float, high: float, low: float, close: float) -> bool:
    """JJ-style decisive candle: counter wick <= 20%.

    For a bullish candle, compare lower wick to the full open-to-high travel.
    For a bearish candle, compare upper wick to the full open-to-low travel.
    The exact pivot geometry is not explicitly published, so this is kept
    isolated as a reconstruction function for later sensitivity testing.
    """
    if close > open_:
        denom = high - open_
        if denom <= 0:
            return False
        counter = open_ - low
    elif close < open_:
        denom = open_ - low
        if denom <= 0:
            return False
        counter = high - open_
    else:
        return False
    return (counter / denom) <= COUNTER_WICK_MAX


def pivot_levels(high: np.ndarray, low: np.ndarray):
    """Confirmed swing levels using a configurable 2-left/2-right pivot."""
    n = len(high)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    L, R = SWING_LEFT, SWING_RIGHT
    for p in range(L, n - R):
        if high[p] > np.max(high[p - L:p]) and high[p] >= np.max(high[p + 1:p + R + 1]):
            swing_high[p] = high[p]
        if low[p] < np.min(low[p - L:p]) and low[p] <= np.min(low[p + 1:p + R + 1]):
            swing_low[p] = low[p]
    return swing_high, swing_low


def latest_confirmed_level(values: np.ndarray, i: int) -> float:
    j = i - SWING_RIGHT
    while j >= 0:
        if not np.isnan(values[j]):
            return float(values[j])
        j -= 1
    return np.nan


def structure_break(
    i: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
):
    """Return +1 for bullish break, -1 for bearish break, 0 otherwise.

    A break is confirmed by candle CLOSE beyond the latest confirmed swing.
    """
    sh = latest_confirmed_level(swing_high, i)
    sl = latest_confirmed_level(swing_low, i)
    bull = not np.isnan(sh) and close[i] > sh
    bear = not np.isnan(sl) and close[i] < sl
    if bull and not bear:
        return 1
    if bear and not bull:
        return -1
    return 0


def simulate_trade(o, h, low, c, i, j_end, direction, stop_d) -> float:
    entry = c[i]
    stop = entry - direction * stop_d
    tgt = entry + direction * RR * stop_d

    for j in range(i + 1, j_end + 1):
        if direction > 0:
            if low[j] <= stop:
                return -1.0
            if h[j] >= tgt:
                return RR
        else:
            if h[j] >= stop:
                return -1.0
            if low[j] <= tgt:
                return RR

    return direction * (c[j_end] - entry) / stop_d


def session_allowed(label: str, minute_from_open: int, phase: str) -> bool:
    if not OPTIMIZED_TIMING:
        return True

    if phase == "continuation" and label == "am" and SKIP_FIRST_3M_CONTINUATION:
        if minute_from_open < 3:
            return False

    if phase == "reversion":
        if label == "am" and minute_from_open >= AM_REVERSION_MAX_MINUTE:
            return False
        if label == "pm" and minute_from_open >= PM_REVERSION_MAX_MINUTE:
            return False

    return True


def collect_entries(df: pd.DataFrame):
    df = df.copy()
    df["atr"] = atr(df)

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    a = df["atr"].to_numpy()

    swing_high, swing_low = pivot_levels(h, low)

    bar_min = int(
        np.median(
            np.diff(df.index.values).astype("timedelta64[m]").astype(int)
        )
    )
    bar_min = max(bar_min, 1)
    cont_bars = max(1, int(np.ceil(CONT_MIN / bar_min)))

    tod = np.array([x.time() for x in df.index])
    dday = df.index.date

    windows = [
        WINDOWS[0],
        WINDOWS[1] if USE_PM_16 else ("pm", time(14, 0), time(15, 0)),
    ]

    entries = []

    for label, w0, w1 in windows:
        inwin = (tod >= w0) & (tod < w1)

        for day in np.unique(dday):
            idxs = np.where(inwin & (dday == day))[0]
            if len(idxs) < 5:
                continue

            anchor = float(o[idxs[0]])
            j_end = int(idxs[-1])
            session_start = idxs[0]

            for i in idxs:
                bars_from_open = i - session_start
                phase = "continuation" if bars_from_open < cont_bars else "reversion"

                if not session_allowed(
                    label, bars_from_open * bar_min, phase
                ):
                    continue

                # Require displacement and structure break on the SAME candle.
                if not displacement(o[i], h[i], low[i], c[i]):
                    continue

                sb = structure_break(
                    i, c, h, low, swing_high, swing_low
                )
                if sb == 0:
                    continue

                dev = np.sign(c[i] - anchor)
                if dev == 0:
                    continue

                if phase == "continuation":
                    # Break must be away from fair value.
                    direction = float(sb)
                    if np.sign(direction) != np.sign(c[i] - anchor):
                        continue
                else:
                    # Break must point back toward fair value.
                    direction = float(sb)
                    if np.sign(direction) != -np.sign(c[i] - anchor):
                        continue

                entries.append(
                    (
                        day,
                        label,
                        phase,
                        int(i),
                        j_end,
                        direction,
                        stop_distance(float(a[i])),
                    )
                )

    return (
        entries,
        (o, h, low, c),
        {
            "bar_min": bar_min,
            "cont_bars": cont_bars,
            "n_bars": len(df),
            "optimized_timing": OPTIMIZED_TIMING,
            "pm_end": "16:00" if USE_PM_16 else "15:00",
        },
    )


def performance(entries, ohlc):
    o, h, low, c = ohlc
    rows = []

    for day, label, phase, i, j_end, direction, stop_d in entries:
        r = simulate_trade(
            o, h, low, c, i, j_end, direction, stop_d
        )
        rows.append(
            {
                "day": day,
                "session": label,
                "phase": phase,
                "R": r,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["day", "session", "phase", "R"]
        )

    return pd.DataFrame(rows)


def print_stats(trades: pd.DataFrame):
    if trades.empty:
        print("No trades.")
        return

    for label in ["all", "continuation", "reversion"]:
        x = trades if label == "all" else trades[trades.phase == label]
        if x.empty:
            continue

        wins = x.R[x.R > 0].sum()
        losses = -x.R[x.R < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        wr = (x.R > 0).mean()

        print(
            f"[{label:13s}] n={len(x):4d} "
            f"win={wr:.3f} PF={pf:.3f} "
            f"R={x.R.sum():+.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OHLC parquet")
    args = ap.parse_args()

    df = load_ohlc(args.data)
    entries, ohlc, meta = collect_entries(df)

    print(
        f"bar={meta['bar_min']}m "
        f"continuation={meta['cont_bars']} bars "
        f"entries={len(entries)} "
        f"optimized_timing={meta['optimized_timing']} "
        f"pm_end={meta['pm_end']}"
    )

    trades = performance(entries, ohlc)
    print_stats(trades)


if __name__ == "__main__":
    main()
