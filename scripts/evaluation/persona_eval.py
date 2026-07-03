"""Persona-prompting comparison over the LLM debate (Yang et al., EACL 2026).

Runs the multi-agent debate under a set of persona variants on the SAME sampled
bars of the test period, then compares them the way the paper does:

  * classification/performance  -- signal distribution + P&L / win-rate / Sharpe
                                   against realized returns (our ground truth)
  * directional bias            -- HOLD/BUY/SELL shift vs the no-persona baseline
                                   (the "over-flagging" analog)
  * inter-persona agreement     -- Krippendorff's alpha over persona signals
                                   (how much the persona actually steers the model)
  * reasoning style             -- word count + Flesch Reading Ease per persona
                                   (the paper's CoT analysis, Table 4)
  * significance                -- 95% bootstrap CI on persona-vs-baseline P&L

Only the debate path costs Groq calls (~9 per bar per run). Results are saved
incrementally and the run resumes where it left off, so it can be run in chunks.

Cost note: a full grid is (#personas x runs x #bars) debates. The Groq free tier
is ~1.5 debates/min, so keep the bar sample modest (the persona axis, not the bar
count, is what needs to be dense for a clean comparison). Validate the whole
pipeline for free first:  python -m scripts.evaluation.persona_eval --self-test

Run (real):  python -m scripts.evaluation.persona_eval --n 60 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "technical_analysis" / "src" / "scripts", REPO_ROOT / "scripts" / "evaluation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import env_loader  # noqa: F401
except Exception:
    pass

from agentic_prototype.llm_chat import GroqChat, MockChat  # noqa: E402
from agentic_prototype.llm_debate import run_llm_debate  # noqa: E402
from agentic_prototype.personas import (  # noqa: E402
    ALL_PERSONAS, BY_NAME, DEMOGRAPHIC_PERSONAS, DISPOSITION_PERSONAS,
)

AGENTS = ("technical", "sentiment", "risk")
RESULTS = REPO_ROOT / "project-context" / "persona_eval_results.json"
SUMMARY = REPO_ROOT / "project-context" / "persona_eval_summary.json"
ASSETS = REPO_ROOT / "Presentation" / "assets"
_UNIT = {"buy": 1, "sell": -1, "hold": 0}


# --------------------------------------------------------------------------- #
# Text/statistics helpers (no external deps, for reproducibility)
# --------------------------------------------------------------------------- #
def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:      # silent trailing 'e'
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease (higher = easier). Approximate syllable count."""
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syll = sum(_syllables(w) for w in words)
    return round(206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syll / len(words)), 2)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+", text))


def krippendorff_alpha_nominal(matrix: list[list]) -> float:
    """Krippendorff's alpha for nominal data.

    ``matrix`` is [coder][unit]; entries are hashable labels or None (missing).
    Coders = personas, units = bars. Returns alpha in (-inf, 1]; 1 = perfect
    agreement, 0 = chance, <0 = systematic disagreement.
    """
    n_units = len(matrix[0]) if matrix else 0
    values: set = set()
    coincidence: Counter = Counter()
    total = 0.0
    for u in range(n_units):
        col = [matrix[c][u] for c in range(len(matrix)) if matrix[c][u] is not None]
        m = len(col)
        if m < 2:
            continue
        values.update(col)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(col[i], col[j])] += 1.0 / (m - 1)
        total += m
    if total == 0 or len(values) < 2:
        return float("nan")
    vals = sorted(values, key=str)
    n_c = {v: sum(coincidence[(v, k)] for k in vals) + coincidence[(v, v)] for v in vals}
    n = sum(n_c.values())
    off_diag = sum(coincidence[(c, k)] for c in vals for k in vals if c != k)
    sum_sq = sum(n_c[v] ** 2 for v in vals)
    denom = n * n - sum_sq
    if denom == 0:
        return float("nan")
    return round(1.0 - off_diag * (n - 1) / denom, 4)


def bootstrap_pnl_diff(pnl_a: np.ndarray, pnl_b: np.ndarray, iters: int = 1000, seed: int = 0) -> dict:
    """95% bootstrap CI on mean(pnl_a) - mean(pnl_b) over the same bars."""
    rng = np.random.default_rng(seed)
    n = len(pnl_a)
    if n == 0:
        return {"diff_mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "significant": False}
    diffs = np.empty(iters)
    for k in range(iters):
        idx = rng.integers(0, n, n)
        diffs[k] = pnl_a[idx].mean() - pnl_b[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": round(float((pnl_a - pnl_b).mean()) * 100, 4),
            "ci_lo": round(float(lo) * 100, 4), "ci_hi": round(float(hi) * 100, 4),
            "significant": bool(lo > 0 or hi < 0)}


# --------------------------------------------------------------------------- #
# Per-persona performance
# --------------------------------------------------------------------------- #
def _perf(sig_int: np.ndarray, mr: np.ndarray, fee: float) -> dict:
    mask = sig_int != 0
    n = int(mask.sum())
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "total_pnl_pct": 0.0, "avg_trade_pct": 0.0,
                "pertrade_sharpe": 0.0, "buy": 0, "sell": 0, "hold": int((sig_int == 0).sum())}
    pnl = sig_int[mask] * mr[mask] - fee
    return {"n_trades": n, "win_rate": round(int((pnl > 0).sum()) / n, 4),
            "total_pnl_pct": round(float(pnl.sum()) * 100, 2), "avg_trade_pct": round(float(pnl.mean()) * 100, 4),
            "pertrade_sharpe": round(float(pnl.mean() / (pnl.std() + 1e-9)), 3),
            "buy": int((sig_int == 1).sum()), "sell": int((sig_int == -1).sum()),
            "hold": int((sig_int == 0).sum())}


def _per_bar_pnl(sig_int: np.ndarray, mr: np.ndarray, fee: float) -> np.ndarray:
    return np.where(sig_int != 0, sig_int * mr - fee, 0.0)


def _majority_by_bar(records: list, persona: str, order: list) -> dict:
    """Majority final signal per bar for one persona, across its runs."""
    per_bar: dict = {}
    for r in records:
        if r["persona"] != persona or r.get("signal") not in _UNIT:
            continue
        per_bar.setdefault(r["i"], []).append(r["signal"])
    out = {}
    for i, sigs in per_bar.items():
        out[i] = Counter(sigs).most_common(1)[0][0]
    return out


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(records: list, mr: np.ndarray, personas: list, fee: float,
              n_total_bars: int, period: dict, sentiment_source: str) -> dict:
    order = sorted({r["i"] for r in records if r.get("signal") in _UNIT})
    idx_pos = {i: k for k, i in enumerate(order)}
    mr_s = mr[np.array(order, dtype=int)] if order else np.array([])

    names = [p.name for p in personas]
    maj = {name: _majority_by_bar(records, name, order) for name in names}

    # per-persona signal vectors aligned on `order` (None where the persona has no vote)
    def _vec(name):
        return [maj[name].get(i) for i in order]

    base_vec = _vec("baseline") if "baseline" in names else None
    base_int = None
    if base_vec is not None:
        base_int = np.array([_UNIT.get(s, 0) for s in base_vec])
        base_pnl = _per_bar_pnl(base_int, mr_s, fee)

    personas_out = {}
    for name in names:
        vec = _vec(name)
        sig_int = np.array([_UNIT.get(s, 0) for s in vec])
        rec_p = [r for r in records if r["persona"] == name and r.get("signal") in _UNIT]
        wc = float(np.mean([r["wc"] for r in rec_p])) if rec_p else 0.0
        fre = float(np.mean([r["fre"] for r in rec_p])) if rec_p else 0.0
        entry = {
            "group": BY_NAME[name].group,
            "axis": BY_NAME[name].axis,
            "n_bars": int(sum(1 for s in vec if s is not None)),
            "signal_dist": {"buy": int((sig_int == 1).sum()), "sell": int((sig_int == -1).sum()),
                            "hold": int((sig_int == 0).sum())},
            "hold_rate": round(float((sig_int == 0).mean()), 4) if len(sig_int) else 0.0,
            "performance": _perf(sig_int, mr_s, fee),
            "reasoning": {"word_count_mean": round(wc, 1), "flesch_ease_mean": round(fre, 1)},
        }
        if base_int is not None and name != "baseline":
            agree = float(np.mean([a == b for a, b in zip(vec, base_vec) if a and b])) if order else 0.0
            entry["vs_baseline"] = {
                "agreement": round(agree, 4),
                **bootstrap_pnl_diff(_per_bar_pnl(sig_int, mr_s, fee), base_pnl),
            }
        personas_out[name] = entry

    # inter-persona agreement (Krippendorff's alpha)
    def _alpha(group_names):
        mat = [[maj[n].get(i) for i in order] for n in group_names if n in maj]
        return krippendorff_alpha_nominal(mat) if len(mat) >= 2 and order else float("nan")

    disp = [p.name for p in DISPOSITION_PERSONAS] + ["baseline"]
    demo = [p.name for p in DEMOGRAPHIC_PERSONAS] + ["baseline"]
    agreement = {
        "alpha_all": _alpha(names),
        "alpha_disposition": _alpha([n for n in disp if n in names]),
        "alpha_demographic": _alpha([n for n in demo if n in names]),
    }
    return {
        "n_total_bars": int(n_total_bars),
        "n_evaluated_bars": len(order),
        "sentiment_source": sentiment_source,
        "test_period": period,
        "personas": personas_out,
        "inter_persona_agreement": agreement,
    }


def print_report(rep: dict) -> None:
    bar = "=" * 92
    print("\n" + bar)
    print(f"PERSONA-PROMPTING EVAL — {rep['n_evaluated_bars']} bars "
          f"(of {rep['n_total_bars']:,}) · sentiment={rep['sentiment_source']}")
    print(bar)
    hdr = f"  {'persona':<22}{'grp':<12}{'buy':>5}{'sell':>5}{'hold':>5}{'win%':>7}{'P&L%':>9}{'Sharpe':>8}{'words':>7}{'FRE':>7}{'vsBase':>9}"
    print(hdr)
    for name, e in rep["personas"].items():
        d, perf = e["signal_dist"], e["performance"]
        vb = e.get("vs_baseline", {})
        star = "*" if vb.get("significant") else ""
        vs = f"{vb.get('diff_mean', 0):+.3f}{star}" if vb else "  base"
        print(f"  {name:<22}{e['group'][:11]:<12}{d['buy']:>5}{d['sell']:>5}{d['hold']:>5}"
              f"{perf['win_rate']*100:>6.1f}{perf['total_pnl_pct']:>9.1f}{perf['pertrade_sharpe']:>8.2f}"
              f"{e['reasoning']['word_count_mean']:>7.0f}{e['reasoning']['flesch_ease_mean']:>7.0f}{vs:>9}")
    a = rep["inter_persona_agreement"]
    print(f"\n  Inter-persona Krippendorff's alpha  ->  all={a['alpha_all']}  "
          f"disposition={a['alpha_disposition']}  demographic={a['alpha_demographic']}")
    print("  (alpha near 1 = personas barely change the signal; lower = more steering)")
    print("  * = persona-vs-baseline P&L difference is significant (95% bootstrap CI excludes 0)")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _reasoning_stats(res: dict) -> tuple[int, float]:
    last = res["transcript"][-1]["positions"]
    text = " ".join(last[a]["reasoning"] for a in AGENTS)
    return word_count(text), flesch_reading_ease(text)


def _run_grid(scenarios: list, backend, personas: list, runs: int, rounds: int,
              records: list, done: set, save_path: Path) -> None:
    t0 = time.time()
    total = len(scenarios) * len(personas) * runs
    k = 0
    for persona in personas:
        for run in range(1, runs + 1):
            for sc in scenarios:
                k += 1
                key = f"{persona.name}|{run}|{sc['id']}"
                if key in done:
                    continue
                try:
                    res = run_llm_debate(sc, backend=backend, rounds=rounds, narrate=False, persona=persona)
                    last = res["transcript"][-1]["positions"]
                    wc, fre = _reasoning_stats(res)
                    rec = {"persona": persona.name, "run": run, "ts": str(sc["id"]), "i": int(sc["i"]),
                           "signal": res["final"]["signal"], "mode": res["final"]["mode"],
                           "tech": last["technical"]["signal"], "sent": last["sentiment"]["signal"],
                           "risk": last["risk"]["signal"], "wc": wc, "fre": fre}
                except Exception as e:  # noqa: BLE001
                    rec = {"persona": persona.name, "run": run, "ts": str(sc["id"]), "i": int(sc["i"]),
                           "signal": "error", "err": str(e)[:140]}
                records.append(rec)
                done.add(key)
                save_path.write_text(json.dumps(records))
                if k % 10 == 0 or k == total:
                    ok = sum(1 for r in records if r.get("signal") in _UNIT)
                    rate = k / max(1e-9, time.time() - t0)
                    print(f"  [{k}/{total}] {persona.name}/run{run} -> {rec['signal']} "
                          f"(ok={ok}, {rate*60:.1f}/min)", flush=True)


def _select_personas(args) -> list:
    if args.personas:
        return [BY_NAME[n] for n in args.personas.split(",")]
    if args.groups:
        groups = set(args.groups.split(","))
        sel = [p for p in ALL_PERSONAS if p.group in groups or p.name == "baseline"]
        return sel
    return ALL_PERSONAS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="number of evenly-spaced bars to sample")
    ap.add_argument("--runs", type=int, default=3, help="independent runs per persona (paper uses 3)")
    ap.add_argument("--rounds", type=int, default=2, help="debate rounds")
    ap.add_argument("--fee", type=float, default=0.002, help="round-trip fee fraction")
    ap.add_argument("--personas", default="", help="comma-separated persona names (default: all)")
    ap.add_argument("--groups", default="", help="comma-separated groups: disposition,demographic")
    ap.add_argument("--report-only", action="store_true", help="recompute summary from saved records")
    ap.add_argument("--charts", action="store_true", help="also render comparison charts to Presentation/assets")
    ap.add_argument("--self-test", action="store_true",
                    help="offline end-to-end check on the demo scenarios with MockChat (no API, no data)")
    args = ap.parse_args()

    personas = _select_personas(args)

    if args.self_test:
        from agentic_prototype.llm_debate import _load_scenarios
        demo = _load_scenarios(REPO_ROOT / "agentic_prototype" / "llm_scenarios.jsonl")
        # synthetic returns so the P&L path is exercised (values are meaningless)
        scenarios = [{"id": s["id"], "i": k, **s} for k, s in enumerate(demo)]
        mr = np.array([_UNIT.get(s.get("expected_signal"), 0) * 0.01 for s in demo])
        records: list = []
        selftest_path = REPO_ROOT / "project-context" / "persona_selftest_results.json"
        _run_grid(scenarios, MockChat(), personas, args.runs, args.rounds, records, set(), selftest_path)
        rep = summarize(records, mr, personas, args.fee, len(scenarios),
                        {"start": "demo", "end": "demo"}, "synthetic")
        print_report(rep)
        print("\n[self-test] harness OK — plumbing, stats and summary all run offline.")
        return

    # Real historical evaluation -------------------------------------------- #
    from scripts.evaluation.debate_eval import build_inputs, briefs_for_bar
    inp = build_inputs()
    n_bars = len(inp["valid_ts"])
    sample_idx = sorted(set(np.linspace(0, n_bars - 1, args.n).astype(int).tolist()))
    scenarios = [{"id": str(inp["valid_ts"][i]), "i": int(i), **briefs_for_bar(inp, i),
                  "expected_signal": None} for i in sample_idx]
    period = {"start": str(inp["valid_ts"][0]), "end": str(inp["valid_ts"][-1])}

    records = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {f"{r['persona']}|{r['run']}|{r['ts']}" for r in records}

    if not args.report_only:
        backend = GroqChat(temperature=0.0) if os.getenv("GROQ_API_KEY") else MockChat()
        print(f"backend={backend.name} | {len(personas)} personas x {args.runs} runs x "
              f"{len(scenarios)} bars | already done {len(done)}", flush=True)
        _run_grid(scenarios, backend, personas, args.runs, args.rounds, records, done, RESULTS)

    rep = summarize(records, inp["mr"], personas, args.fee, n_bars, period, inp["sent_src"])
    SUMMARY.write_text(json.dumps(rep, indent=2))
    print_report(rep)
    print(f"\nsummary: {SUMMARY}\nraw records: {RESULTS}")
    if args.charts:
        from scripts.evaluation.persona_charts import render_all
        render_all(rep, ASSETS)
        print(f"charts: {ASSETS}/persona_*.png")


if __name__ == "__main__":
    main()
