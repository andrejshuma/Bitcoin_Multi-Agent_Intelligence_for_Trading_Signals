"""Full multi-agent LLM-debate pipeline, narrated step by step.

Runs the exact flow of CoordinatorAgent.run(mode="debate"), but prints every
stage so you can see what each agent did and how the debate evolved:

  1. Technical agent  (real CNN-LSTM inference)
  2. Risk agent       (real LightGBM, isolated subprocess)
  3. Sentiment        (live NewsAPI headlines -> real FinBERT)
  4. Briefs handed to the three debaters
  5. Debate           (initial positions + each round, with what changed)
  6. Coordinator judge (final signal + reasoning)

Run:  python scripts/run_debate_pipeline.py [--rounds 2]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import env_loader  # load .env (GROQ_API_KEY / NEWSAPI_KEY)  # noqa: F401
except Exception:
    pass

AGENTS = ("technical", "sentiment", "risk")
W = 76


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def hr(ch: str = "-") -> None:
    print(ch * W)


def section(step: str, title: str) -> None:
    print("\n" + "=" * W)
    print(f" {step} · {title}")
    print("=" * W)


def kv(label: str, value, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<18}: {value}")


def wrap(text: str, indent: int = 4) -> str:
    import textwrap

    return textwrap.fill(str(text), width=W - indent,
                         initial_indent=" " * indent, subsequent_indent=" " * indent)


def _sig(p: dict) -> str:
    return f"{p['signal'].upper():<4} ({p['confidence']:.2f})"


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Narrated multi-agent LLM debate pipeline.")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    from agentic_prototype.live_briefs import (
        build_sentiment_brief,
        risk_brief_from_signal,
        technical_brief_from_signal,
    )
    from agentic_prototype.llm_chat import get_backend
    from agentic_prototype.llm_debate import run_llm_debate
    from coordinator_agent.coordinator_agent import CoordinatorAgent

    print("=" * W)
    print(" BITCOIN MULTI-AGENT LLM DEBATE — FULL PIPELINE TRACE".center(W))
    print("=" * W)
    backend = get_backend(verbose=False)
    kv("LLM backend", backend.name + ("" if backend.name != "mock" else "  (set GROQ_API_KEY for real LLM)"))
    kv("News source", "NewsAPI (live)" if os.getenv("NEWSAPI_KEY") else "NewsAPI mock (set NEWSAPI_KEY for live)")
    kv("Debate rounds", args.rounds)

    coord = CoordinatorAgent(repo_root=REPO_ROOT)

    # 1. Technical -------------------------------------------------------------
    section("STEP 1/6", "TECHNICAL AGENT — CNN-LSTM inference on the latest bar")
    print("  running model...", flush=True)
    technical = coord.technical_agent.run().model_dump()
    kv("signal", technical["signal"].upper())
    kv("trade_probability", f"{technical['trade_probability']:.4f}")
    kv("long_probability", f"{technical['long_probability']:.4f}")
    kv("confidence", f"{technical['confidence']:.4f}")
    print(wrap("reasoning: " + technical["reasoning"]))

    # 2. Risk ------------------------------------------------------------------
    section("STEP 2/6", "RISK AGENT — LightGBM (isolated subprocess)")
    print("  running model...", flush=True)
    risk = coord._run_json_script(coord.risk_dir, "run_agent_json.py")
    kv("risk_level", str(risk.get("signal", "?")).upper())
    kv("risk_score", f"{float(risk.get('risk_score', 0)):.4f}")
    kv("confidence", f"{float(risk.get('confidence', 0)):.4f}")
    print(wrap("reasoning: " + str(risk.get("reasoning", "n/a"))))

    # 3. Sentiment -------------------------------------------------------------
    section("STEP 3/6", "SENTIMENT — live headlines -> FinBERT")
    print("  fetching headlines + scoring...", flush=True)
    sentiment_brief = build_sentiment_brief()
    kv("headline_source", sentiment_brief.get("headline_source"))
    kv("finbert_score", sentiment_brief.get("finbert_score"))
    kv("label_dist", sentiment_brief.get("label_distribution", "n/a"))
    kv("tone", sentiment_brief.get("twitter_tone"))
    print("  headlines scored:")
    for h in sentiment_brief.get("news_headlines", [])[:6]:
        print(wrap("- " + h, indent=6))

    # 4. Briefs ----------------------------------------------------------------
    section("STEP 4/6", "BRIEFS HANDED TO THE THREE DEBATERS")
    briefs = {
        "technical": technical_brief_from_signal(technical),
        "sentiment": sentiment_brief,
        "risk": risk_brief_from_signal(risk),
    }
    for name in AGENTS:
        print(f"  [{name}]  (source: {briefs[name].get('source')})")
        for k, v in briefs[name].items():
            if k in ("source", "model_reasoning", "news_headlines", "label_distribution"):
                continue
            print(f"        {k}: {v}")

    # 5. Debate ----------------------------------------------------------------
    section("STEP 5/6", f"DEBATE — {args.rounds} rounds")
    scenario = {"id": "pipeline", **briefs, "expected_signal": None}
    print("  agents debating (this calls the LLM)...", flush=True)
    result = run_llm_debate(scenario, backend=backend, rounds=args.rounds)

    transcript = result["transcript"]
    prev = None
    for step in transcript:
        label = step["phase"].replace("_", " ").title()
        print(f"\n  -- {label} --")
        for name in AGENTS:
            p = step["positions"][name]
            tag = ""
            if prev is not None:
                pp = prev["positions"][name]
                tag = "  [CHANGED]" if pp["signal"] != p["signal"] else "  [held]"
            print(f"    {name:<10} {_sig(p)}{tag}")
            print(wrap(p["reasoning"], indent=8))
        prev = step

    # 6. Coordinator -----------------------------------------------------------
    section("STEP 6/6", "COORDINATOR JUDGE — final decision")
    final = result["final"]
    last = transcript[-1]["positions"]
    kv("final positions", "  ".join(f"{a}={last[a]['signal'].upper()}" for a in AGENTS))
    kv("resolution mode", final["mode"])
    kv("risk veto", final["risk_veto"])
    print(wrap("rationale: " + final["rationale"]))
    print()
    print(wrap("narrative: " + final["reasoning"]))

    # Final summary ------------------------------------------------------------
    conf = float(final["confidence"])
    score = max(-1.0, min(1.0, coord._signal_to_unit(final["signal"]) * conf))
    print("\n" + "=" * W)
    print(" FINAL DECISION".center(W))
    print("=" * W)
    kv("SIGNAL", final["signal"].upper())
    kv("confidence", f"{conf:.3f}")
    kv("score", f"{score:+.3f}")
    kv("risk_level", str(risk.get("signal", "medium_risk")).lower())
    print("=" * W)


if __name__ == "__main__":
    main()
