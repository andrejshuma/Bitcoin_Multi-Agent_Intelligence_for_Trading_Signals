"""Historical 3-way eval: TA-only vs Agentic vs LLM-Debate.

All three strategies are scored on the SAME sampled bars of the test period with
the SAME realized returns and the SAME historical inputs:
  - technical : real CNN-LSTM trade/long probabilities
  - sentiment : Fear & Greed index (daily history)   [NewsAPI is live-only, so it
                cannot supply per-bar historical headlines -> F&G is the proxy,
                exactly as the Agentic baseline uses]
  - risk      : ATR-percentile risk regime

Only the LLM-Debate path costs Groq calls (~9 per bar), so it runs on a sample.
Results are saved incrementally and the run resumes where it left off.

Run:  python -m scripts.evaluation.debate_eval --n 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "technical_analysis" / "src" / "scripts", REPO_ROOT / "scripts" / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import env_loader  # noqa: F401
except Exception:
    pass

import backtest as bt  # noqa: E402
from technical import evaluate as eval_tech  # noqa: E402
from agentic_prototype.llm_chat import GroqChat, MockChat  # noqa: E402
from agentic_prototype.llm_debate import run_llm_debate  # noqa: E402

ARRAYS_CACHE = REPO_ROOT / "technical_analysis" / "data" / "processed" / "eval_arrays.npz"
RESULTS = REPO_ROOT / "project-context" / "debate_eval_results.json"
ARTIFACT = REPO_ROOT / "project-context" / "debate_eval_raw.json"
_UNIT = {"buy": 1, "sell": -1, "hold": 0}


# --------------------------------------------------------------------------- #
def build_inputs() -> dict:
    if ARRAYS_CACHE.exists():
        d = np.load(ARRAYS_CACHE, allow_pickle=True)
        valid_ts = pd.to_datetime(d["valid_ts"])
        trade_p, dir_p, mr = d["trade_p"], d["dir_p"], d["mr"]
    else:
        print("Running CNN-LSTM inference (once, cached)...", flush=True)
        arr = eval_tech()["_arrays"]
        valid_ts = pd.to_datetime(arr["valid_ts"])
        trade_p = np.array(arr["trade_probs"]); dir_p = np.array(arr["dir_probs"]); mr = np.array(arr["market_returns"])
        np.savez(ARRAYS_CACHE, valid_ts=[str(t) for t in valid_ts], trade_p=trade_p, dir_p=dir_p, mr=mr)

    ta_score = bt._ta_scores(trade_p, dir_p)
    ta_signal = np.where(trade_p < bt.TRADE_THRESHOLD, 0, np.where(dir_p >= 0.5, 1, -1)).astype(int)

    test_feats, _orig, train_feats = bt._load_test_data()
    aligned = test_feats.reindex(valid_ts)

    fg = bt._fetch_fear_greed(limit=700)
    sent = bt._sentiment_scores_from_fg(fg, valid_ts)
    sent_src = "fear_greed"
    if sent is None:
        sent = bt._sentiment_fallback(aligned, train_feats); sent_src = "rsi_proxy"
    risk_action = bt._risk_scores(aligned, train_feats)

    combined = bt.W_TA * ta_score + bt.W_SENTIMENT * sent + bt.W_RISK * risk_action
    agentic_signal = np.where(combined >= bt.AGENTIC_BUY_THRESH, 1,
                              np.where(combined <= bt.AGENTIC_SELL_THRESH, -1, 0)).astype(int)

    return dict(valid_ts=valid_ts, trade_p=trade_p, dir_p=dir_p, mr=mr, ta_score=ta_score,
                ta_signal=ta_signal, sent=sent, sent_src=sent_src, risk_action=risk_action,
                agentic_signal=agentic_signal)


def _risk_level(action: float) -> str:
    rp = (1.0 - action) / 2.0
    return "low_risk" if rp < 0.34 else "high_risk" if rp > 0.66 else "medium_risk"


def briefs_for_bar(inp: dict, i: int) -> dict:
    tp, lp = float(inp["trade_p"][i]), float(inp["dir_p"][i])
    tech = {"source": "model", "trade_probability": round(tp, 4), "long_probability": round(lp, 4),
            "model_signal": "hold" if tp < bt.TRADE_THRESHOLD else ("buy" if lp >= 0.5 else "sell"),
            "trend": "long-biased" if lp >= 0.5 else "short-biased"}
    s = float(inp["sent"][i]); tone = "bullish" if s > 0.15 else "bearish" if s < -0.15 else "mixed"
    sent = {"source": "fear_greed", "finbert_score": round(s, 3), "twitter_tone": tone,
            "macro": "(historical)", "news_headlines": [f"Market sentiment (Fear&Greed) reads {tone} ({s:+.2f})"]}
    risk = {"source": "atr", "risk_level": _risk_level(float(inp["risk_action"][i])),
            "risk_action": round(float(inp["risk_action"][i]), 3)}
    return {"technical": tech, "sentiment": sent, "risk": risk}


def metrics(sig_int: np.ndarray, mr: np.ndarray) -> dict:
    sig_int = np.asarray(sig_int); mr = np.asarray(mr)
    mask = sig_int != 0
    n = int(mask.sum())
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "total_pnl_pct": 0.0, "avg_trade_pct": 0.0,
                "pertrade_sharpe": 0.0, "buy": 0, "sell": 0, "hold": int((sig_int == 0).sum())}
    pnl = sig_int[mask] * mr[mask] - bt.FEE
    return {"n_trades": n, "win_rate": round(int((pnl > 0).sum()) / n, 4),
            "total_pnl_pct": round(float(pnl.sum()) * 100, 2), "avg_trade_pct": round(float(pnl.mean()) * 100, 4),
            "pertrade_sharpe": round(float(pnl.mean() / (pnl.std() + 1e-9)), 3),
            "buy": int((sig_int == 1).sum()), "sell": int((sig_int == -1).sum()), "hold": int((sig_int == 0).sum())}


def summarize(inp: dict, results: list) -> dict:
    mr = inp["mr"]
    ok = [r for r in results if r.get("debate_signal") in _UNIT]
    idx = np.array([r["i"] for r in ok], dtype=int)
    rep = {
        "n_total_bars": int(len(inp["valid_ts"])),
        "n_sampled": len(results),
        "n_evaluated": len(ok),
        "sentiment_source": inp["sent_src"],
        "test_period": {"start": str(inp["valid_ts"][0]), "end": str(inp["valid_ts"][-1])},
        "full_period": {  # all bars, for context (matches backtest.py)
            "ta_only": metrics(inp["ta_signal"], mr),
            "agentic": metrics(inp["agentic_signal"], mr),
        },
    }
    if len(ok):
        deb = np.array([_UNIT[r["debate_signal"]] for r in ok])
        mr_s = mr[idx]
        ta_s = inp["ta_signal"][idx]; ag_s = inp["agentic_signal"][idx]
        rep["sampled"] = {
            "ta_only": metrics(ta_s, mr_s),
            "agentic": metrics(ag_s, mr_s),
            "debate": metrics(deb, mr_s),
            "agreement_debate_vs_ta": round(float(np.mean(deb == ta_s)), 4),
            "agreement_debate_vs_agentic": round(float(np.mean(deb == ag_s)), 4),
        }
    return rep


def print_report(rep: dict) -> None:
    bar = "=" * 78
    print("\n" + bar)
    print(f"DEBATE EVAL — {rep['n_evaluated']}/{rep['n_sampled']} sampled bars "
          f"(of {rep['n_total_bars']:,} test bars) · sentiment={rep['sentiment_source']}")
    print(bar)
    fp = rep["full_period"]
    print("Full period (all bars, no LLM) — context:")
    print(f"  {'strategy':<10}{'trades':>8}{'win%':>8}{'totP&L%':>11}{'avg%':>9}{'ptSharpe':>10}")
    for k in ("ta_only", "agentic"):
        m = fp[k]
        print(f"  {k:<10}{m['n_trades']:>8}{m['win_rate']*100:>7.2f}{m['total_pnl_pct']:>11.1f}{m['avg_trade_pct']:>9.4f}{m['pertrade_sharpe']:>10.3f}")
    if "sampled" in rep:
        s = rep["sampled"]
        print("\nSampled bars (same bars for all three):")
        print(f"  {'strategy':<10}{'trades':>8}{'win%':>8}{'totP&L%':>11}{'avg%':>9}{'ptSharpe':>10}")
        for k in ("ta_only", "agentic", "debate"):
            m = s[k]
            print(f"  {k:<10}{m['n_trades']:>8}{m['win_rate']*100:>7.2f}{m['total_pnl_pct']:>11.1f}{m['avg_trade_pct']:>9.4f}{m['pertrade_sharpe']:>10.3f}")
        print(f"\n  agreement debate vs TA      : {s['agreement_debate_vs_ta']:.1%}")
        print(f"  agreement debate vs Agentic : {s['agreement_debate_vs_agentic']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of evenly-spaced bars to sample")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--report-only", action="store_true", help="just recompute the report from saved results")
    args = ap.parse_args()

    inp = build_inputs()
    n_bars = len(inp["valid_ts"])
    sample_idx = sorted(set(np.linspace(0, n_bars - 1, args.n).astype(int).tolist()))

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {r["ts"] for r in results}

    if not args.report_only:
        backend = GroqChat(temperature=0.0) if os.getenv("GROQ_API_KEY") else MockChat()
        print(f"backend={backend.name} | sampling {len(sample_idx)} bars | already done {len(done)}", flush=True)
        t0 = time.time()
        for j, i in enumerate(sample_idx):
            ts = str(inp["valid_ts"][i])
            if ts in done:
                continue
            scenario = {"id": ts, **briefs_for_bar(inp, i), "expected_signal": None}
            try:
                res = run_llm_debate(scenario, backend=backend, rounds=args.rounds, narrate=False)
                last = res["transcript"][-1]["positions"]
                rec = {"ts": ts, "i": int(i), "debate_signal": res["final"]["signal"], "mode": res["final"]["mode"],
                       "tech": last["technical"]["signal"], "sent": last["sentiment"]["signal"], "risk": last["risk"]["signal"]}
            except Exception as e:
                rec = {"ts": ts, "i": int(i), "debate_signal": "error", "err": str(e)[:140]}
            results.append(rec); done.add(ts)
            RESULTS.write_text(json.dumps(results))
            if (j + 1) % 10 == 0 or j == len(sample_idx) - 1:
                ok = sum(1 for r in results if r.get("debate_signal") in _UNIT)
                rate = (j + 1) / max(1e-9, time.time() - t0)
                print(f"  [{j+1}/{len(sample_idx)}] ok={ok} last={rec['debate_signal']} ({rate*60:.1f}/min)", flush=True)

    rep = summarize(inp, results)
    ARTIFACT.write_text(json.dumps(rep, indent=2))
    print_report(rep)
    print(f"\nartifact: {ARTIFACT}\nraw results: {RESULTS}")


if __name__ == "__main__":
    main()
