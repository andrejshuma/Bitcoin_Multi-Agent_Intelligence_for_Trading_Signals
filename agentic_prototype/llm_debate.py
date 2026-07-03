"""Multi-agent *text* debate orchestrator (Du et al. 2023).

Phase 1  Each of the 3 LLM agents independently answers from its own brief.
Phase 2  Fixed 2 rounds: every agent reads the other two agents' latest
         positions and re-emits its own (synchronized -- a round reads the
         previous round's positions).
Phase 3  The coordinator judges: relays consensus, or on a split decides by
         domain priority + risk veto, and writes the final narrative.

Run the demo:  python -m agentic_prototype.llm_debate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .llm_agents import CoordinatorJudge, FinalDecision, LLMAgent, Position
from .llm_chat import get_backend

AGENTS = ("technical", "sentiment", "risk")
DEFAULT_ROUNDS = 2  # fixed, no early stop (per design decision)


def run_llm_debate(scenario: Dict, backend=None, rounds: int = DEFAULT_ROUNDS, narrate: bool = True) -> Dict:
    """Run the full debate for one scenario and return a structured transcript."""
    backend = backend or get_backend(verbose=False)
    agents = {name: LLMAgent(agent=name, backend=backend) for name in AGENTS}

    # Phase 1 -- independent initial positions (sequential; order has no effect).
    current: Dict[str, Position] = {name: agents[name].initial(scenario[name]) for name in AGENTS}
    transcript = [{"phase": "initial", "positions": {n: asdict(p) for n, p in current.items()}}]

    # Phase 2 -- fixed debate rounds.
    for r in range(1, rounds + 1):
        snapshot = current  # each agent revises against the PREVIOUS round
        nxt: Dict[str, Position] = {}
        for name in AGENTS:
            others = [snapshot[o] for o in AGENTS if o != name]
            nxt[name] = agents[name].revise(snapshot[name], others, round_no=r)
        current = nxt
        transcript.append({"phase": f"round_{r}", "positions": {n: asdict(p) for n, p in current.items()}})

    # Phase 3 -- coordinator judge.
    coordinator = CoordinatorJudge(backend=backend)
    decision: FinalDecision = coordinator.decide([current[n] for n in AGENTS], narrate=narrate)

    return {
        "id": scenario.get("id"),
        "expected_signal": scenario.get("expected_signal"),
        "rounds": rounds,
        "final": asdict(decision),
        "transcript": transcript,
    }


# --------------------------------------------------------------------------- #
# Demo runner
# --------------------------------------------------------------------------- #
def _load_scenarios(path: Path) -> List[Dict]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [json.loads(l) for l in lines if l and not l.startswith("//")]


def _print_debate(result: Dict) -> None:
    print("=" * 74)
    print(f"SCENARIO {result['id']}   (expected: {result.get('expected_signal', 'n/a')})")
    print("=" * 74)
    for step in result["transcript"]:
        label = step["phase"].replace("_", " ").title()
        print(f"\n[{label}]")
        for name in AGENTS:
            p = step["positions"][name]
            print(f"  {name:<10} -> {p['signal'].upper():<4} ({p['confidence']:.2f})  {p['reasoning']}")
    f = result["final"]
    print(f"\n[Coordinator]  mode={f['mode']}  risk_veto={f['risk_veto']}")
    print(f"  FINAL -> {f['signal'].upper()} (confidence {f['confidence']:.2f})")
    print(f"  {f['reasoning']}")
    if result.get("expected_signal"):
        ok = "OK" if f["signal"] == result["expected_signal"] else "MISS"
        print(f"  [{ok}] expected {result['expected_signal'].upper()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent LLM debate demo (technical/sentiment/risk).")
    parser.add_argument("--scenarios", default=str(Path(__file__).resolve().parent / "llm_scenarios.jsonl"))
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--json", action="store_true", help="Dump raw JSON transcripts instead of pretty print.")
    parser.add_argument("--live", action="store_true",
                        help="Build ONE scenario from the real models (CNN-LSTM, LightGBM risk, FinBERT) "
                             "instead of the demo scenarios file.")
    args = parser.parse_args()

    backend = get_backend(verbose=True)
    if args.live:
        from .live_briefs import build_live_scenario

        print("[live] building briefs from real models...")
        scenario = build_live_scenario()
        for agent in AGENTS:
            print(f"  {agent:<10} source: {scenario[agent].get('source')}")
        scenarios = [scenario]
    else:
        scenarios = _load_scenarios(Path(args.scenarios))
    results = [run_llm_debate(s, backend=backend, rounds=args.rounds) for s in scenarios]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    correct = 0
    labeled = 0
    for res in results:
        _print_debate(res)
        if res.get("expected_signal"):
            labeled += 1
            correct += res["final"]["signal"] == res["expected_signal"]
    if labeled:
        print("\n" + "=" * 74)
        print(f"Demo accuracy on labeled scenarios: {correct}/{labeled} = {correct / labeled:.0%}")


if __name__ == "__main__":
    main()
