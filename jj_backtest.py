"""Faithful, frequency-agnostic backtester + GATE for JJ Simon's fair-value strategy -- a DROP-IN for the
1-min NQ history when it arrives (validate now on free OHLC; then `python jj_backtest.py --data <1min.parquet>`).
Objectifies his discretionary rules (an improvement, and gate-able):

  * fair-value anchor = the session-window open (morning 09:30, afternoon 14:00).
  * two windows: 09:30-11:00 and 14:00-15:00 NY (his sessions).
  * two-phase: first CONT_MIN minutes = CONTINUATION (trade WITH a displacement candle), rest = REVERSION
    (trade TOWARD the anchor on a displacement candle beyond MIN_DEV) -- his continuation-then-reversion.
  * displacement candle = body/range >= BODY_MIN (objectifies the <20%-counter-wick rule).
  * exit = ATR stop (STOP_ATR*ATR) + 1.5R target, first-touch within the window, else window close; same-bar
    ambiguity resolved stop-first (conservative).

THE GATE (per phase): trades are aggregated to per-DAY R (kills the overlapping-trades t-inflation -> the
unit of evidence is the day), then (1) bootstrap CI on the per-day mean, (2) purged walk-forward replication
across days, (3) a DIRECTION-SHUFFLE placebo -- re-run the SAME entries/exits with random long/short, so the
edge must come from the direction RULE, not the entry/exit mechanics. PASS = mean-2se>0 & folds replicate &
beats placebo (p<0.05). This is the honest significance; on a tiny free sample it will (correctly) not pass.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import argparse
from datetime import time

import numpy as np
import pandas as pd

from src.evaluation.feature_cv import purged_walk_forward

CONT_MIN = 15  # his continuation window (minutes) before the reversion phase
BODY_MIN = 0.5  # displacement candle: body/range >= this (small counter-wick)
MIN_DEV = 0.0005  # min deviation-from-anchor (a real break) to take a reversion entry
STOP_ATR = 1.0  # stop = STOP_ATR * ATR; target = 1.5 * stop (1.5R)
WINDOWS = [(time(9, 30), time(11, 0)), (time(14, 0), time(15, 0))]


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


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()),
    )
    return tr.rolling(n, min_periods=1).mean().to_numpy()


def simulate_trade(o, h, low, c, i, j_end, direction, stop_d) -> float:
    """Walk bars i+1..j_end; exit at first stop/1.5R-target touch (stop-first on same bar), else window
    close. Returns the R-multiple."""
    entry = c[i]
    stop = entry - direction * stop_d
    tgt = entry + direction * 1.5 * stop_d
    for j in range(i + 1, j_end + 1):
        if direction > 0:
            if low[j] <= stop:
                return -1.0
            if h[j] >= tgt:
                return 1.5
        else:
            if h[j] >= stop:
                return -1.0
            if low[j] <= tgt:
                return 1.5
    return direction * (c[j_end] - entry) / stop_d


def collect_entries(df: pd.DataFrame):
    """Precompute every entry signal once: (day, phase, i, j_end, signal_dir, stop_d). The placebo re-runs
    these SAME entries with shuffled directions, so only the direction RULE is under test."""
    df = df.copy()
    df["a"] = atr(df)
    bar_min = int(
        np.median(np.diff(df.index.values).astype("timedelta64[m]").astype(int)) or 5
    )
    cont_bars = max(1, int(np.ceil(CONT_MIN / bar_min)))
    o, h, low, c, a = (df[x].to_numpy() for x in ("open", "high", "low", "close", "a"))
    tod = np.array([t.time() for t in df.index])
    dday = df.index.date
    disp = (np.abs(c - o) / np.maximum(h - low, 1e-12)) >= BODY_MIN
    entries = []
    for w0, w1 in WINDOWS:
        inwin = (tod >= w0) & (tod < w1)
        for day in np.unique(dday):
            idxs = np.where(inwin & (dday == day))[0]
            if len(idxs) < 3:
                continue
            anchor, j_end, k = o[idxs[0]], idxs[-1], 0
            for i in idxs:
                k += 1
                if a[i] <= 0 or not disp[i]:
                    continue
                dev = np.log(c[i] / anchor)
                if k <= cont_bars:
                    d, ph = float(np.sign(c[i] - o[i])), "continuation"
                else:
                    if abs(dev) < MIN_DEV:
                        continue
                    d, ph = -float(np.sign(dev)), "reversion"
                if d != 0:
                    entries.append(
                        (day, ph, int(i), int(j_end), d, float(STOP_ATR * a[i]))
                    )
    return (
        entries,
        (o, h, low, c),
        {"bar_min": bar_min, "cont_bars": cont_bars, "n_bars": len(df)},
    )


def per_day_R(entries, ohlc, phase, directions=None) -> pd.Series:
    """Aggregate trade R to per-day totals for a phase; `directions` (list over ALL entries) overrides the
    signal direction for the placebo."""
    o, h, low, c = ohlc
    byday: dict = {}
    r_list = []
    for idx, (day, ph, i, j_end, d, stop_d) in enumerate(entries):
        if ph != phase:
            continue
        dd = d if directions is None else directions[idx]
        R = simulate_trade(o, h, low, c, i, j_end, dd, stop_d)
        byday[day] = byday.get(day, 0.0) + R
        r_list.append(R)
    return pd.Series(byday).sort_index(), np.array(r_list)


def gate(entries, ohlc, phase, n_placebo=300, seed=0) -> dict:
    dayR, r = per_day_R(entries, ohlc, phase)
    if len(dayR) < 20:
        return {"phase": phase, "n_days": int(len(dayR)), "insufficient": True}
    v = dayR.to_numpy()
    n = len(v)
    rng = np.random.default_rng(seed)
    boot = np.array([v[rng.integers(0, n, n)].mean() for _ in range(2000)])
    mean, se = float(v.mean()), float(boot.std())
    folds = purged_walk_forward(n, n_folds=5, embargo=0.01)
    fold_pos = float(np.mean([v[te].mean() > 0 for _, te in folds])) if folds else 0.0
    ph_idx = [k for k, e in enumerate(entries) if e[1] == phase]
    plac = np.empty(n_placebo)
    for b in range(n_placebo):
        dirs = [0.0] * len(entries)
        for k in ph_idx:
            dirs[k] = float(rng.choice([-1.0, 1.0]))
        plac[b] = per_day_R(entries, ohlc, phase, dirs)[0].mean()
    p_plac = float((plac >= mean).mean())
    passed = (mean - 2 * se > 0) and (fold_pos >= 0.7) and (p_plac < 0.05)
    return {
        "phase": phase,
        "n_trades": int(len(r)),
        "n_days": n,
        "win_rate": float((r > 0).mean()),
        "avg_R": float(r.mean()),
        "mean_dayR": mean,
        "se_dayR": se,
        "t_dayR": float(mean / se) if se > 0 else float("nan"),
        "fold_pos_frac": fold_pos,
        "placebo_p": p_plac,
        "gate_pass": bool(passed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="OHLC parquet (yfinance or 1-min NQ)")
    a = ap.parse_args()
    entries, ohlc, meta = collect_entries(load_ohlc(a.data))
    print(
        f"bar={meta['bar_min']}min cont_bars={meta['cont_bars']} n_bars={meta['n_bars']} entries={len(entries)}"
    )
    for ph in ("continuation", "reversion"):
        g = gate(entries, ohlc, ph)
        if g.get("insufficient"):
            print(f"[{ph:12s}] n_days={g['n_days']} (too few to gate)")
            continue
        print(
            f"[{ph:12s}] GATE={'PASS' if g['gate_pass'] else 'fail'} | n={g['n_trades']} days={g['n_days']} "
            f"win={g['win_rate']:.2f} avgR={g['avg_R']:+.3f} | mean_dayR={g['mean_dayR']:+.3f} "
            f"t={g['t_dayR']:+.2f} foldpos={g['fold_pos_frac']:.2f} placebo_p={g['placebo_p']:.3f}"
        )


if __name__ == "__main__":
    main()
