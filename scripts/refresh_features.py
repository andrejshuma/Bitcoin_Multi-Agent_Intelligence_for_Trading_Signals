"""Refresh the technical features with the latest market data.

Two steps:
  1. For each timeframe (1h, 4h, 1D) download recent OHLCV from Binance (only the
     gap since the last stored bar) and CONCATENATE it onto the existing raw
     parquet -- history is preserved, never overwritten. Overlapping bars are
     de-duplicated keeping the freshly downloaded version.
  2. Rebuild technical_analysis/data/processed/features_1h.parquet from the now
     up-to-date raw data (this derived file is regenerated, as it must be).

No API key needed (Binance public API). No model retraining needed.

Run:  python scripts/refresh_features.py [--overlap-days 2] [--full-history-days 2700]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "technical_analysis" / "data" / "raw"
PROCESSED = REPO_ROOT / "technical_analysis" / "data" / "processed" / "features_1h.parquet"

# Make the project's download helper + feature builder importable.
sys.path.insert(0, str(REPO_ROOT / "technical_analysis" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "technical_analysis" / "src" / "pipeline"))

# (ccxt timeframe, file-name suffix) -- note Binance uses "1d", file is btc_1D.
TIMEFRAMES = [("1h", "1h"), ("4h", "4h"), ("1d", "1D")]


def refresh_raw(tf_ccxt: str, fname: str, overlap_days: float, full_history_days: float) -> dict:
    from apis import fetch_binance_btc  # noqa: PLC0415

    path = RAW_DIR / f"btc_{fname}.parquet"
    old = None
    if path.exists():
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index)
        old = old.sort_index()
        last = old.index[-1]
        now = pd.Timestamp.now("UTC").tz_localize(None)
        gap_days = max(0.0, (now - last).total_seconds() / 86400.0)
        period = gap_days + overlap_days
    else:
        last = None
        period = full_history_days

    print(f"  [{fname}] fetching last {period:.1f} days ...", flush=True)
    new = fetch_binance_btc(tf_ccxt, period)
    new.index = pd.to_datetime(new.index)
    new = new.sort_index()

    if old is not None and len(new):
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    elif old is not None:
        combined = old
    else:
        combined = new

    combined.to_parquet(path)
    added = len(combined) - (len(old) if old is not None else 0)
    return {
        "tf": fname, "rows_before": len(old) if old is not None else 0,
        "rows_after": len(combined), "rows_added": added,
        "old_last": str(last) if last is not None else "—", "new_last": str(combined.index[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate fresh OHLCV and rebuild features.")
    parser.add_argument("--overlap-days", type=float, default=2.0,
                        help="Re-fetch this many extra days before the last stored bar (de-duplicated).")
    parser.add_argument("--full-history-days", type=float, default=2700.0,
                        help="Window to fetch if a timeframe has no existing file.")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1/2 · Refreshing raw OHLCV (append, no overwrite)")
    print("=" * 70)
    summaries = [refresh_raw(tf, fn, args.overlap_days, args.full_history_days) for tf, fn in TIMEFRAMES]
    for s in summaries:
        print(f"  {s['tf']:>3}: {s['rows_before']:,} -> {s['rows_after']:,} "
              f"(+{s['rows_added']:,})   last {s['old_last']} -> {s['new_last']}")

    print("\n" + "=" * 70)
    print("STEP 2/2 · Rebuilding features_1h.parquet")
    print("=" * 70)
    from features import build_features  # noqa: PLC0415

    df = build_features(
        primary_tf="1h",
        htf_list=[("4h", "htf4h"), ("1D", "htf1d")],
        norm_window=200,
        drop_raw_ohlcv=False,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  features_1h.parquet : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  latest bar          : {df.index[-1]}")
    print(f"  path                : {PROCESSED}")


if __name__ == "__main__":
    main()
