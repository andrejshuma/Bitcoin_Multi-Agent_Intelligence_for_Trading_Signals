from __future__ import annotations

import unittest
from pathlib import Path

from agentic_prototype.llm_agents import CoordinatorJudge, Position
from agentic_prototype.llm_chat import MockChat, get_backend
from agentic_prototype.llm_debate import AGENTS, _load_scenarios, run_llm_debate


class LLMDebateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = MockChat()
        self.scenarios = _load_scenarios(Path(__file__).resolve().parent / "llm_scenarios.jsonl")

    def test_offline_backend_is_mock(self) -> None:
        # With no key in the environment, get_backend must fall back to the mock.
        import os

        saved = os.environ.pop("GROQ_API_KEY", None)
        try:
            self.assertEqual(get_backend(verbose=False).name, "mock")
        finally:
            if saved is not None:
                os.environ["GROQ_API_KEY"] = saved

    def test_transcript_shape(self) -> None:
        res = run_llm_debate(self.scenarios[0], backend=self.backend, rounds=2)
        phases = [s["phase"] for s in res["transcript"]]
        self.assertEqual(phases, ["initial", "round_1", "round_2"])
        for step in res["transcript"]:
            self.assertEqual(set(step["positions"]), set(AGENTS))
        self.assertIn(res["final"]["signal"], {"buy", "sell", "hold"})

    def test_consensus_scenario(self) -> None:
        res = run_llm_debate(self.scenarios[0], backend=self.backend)  # bull_consensus
        self.assertEqual(res["final"]["signal"], "buy")
        self.assertEqual(res["final"]["mode"], "consensus")

    def test_weak_agent_converges(self) -> None:
        # weak_tech_caves: technical starts SELL, should flip to BUY by round 1.
        res = run_llm_debate(self.scenarios[2], backend=self.backend)
        self.assertEqual(res["transcript"][0]["positions"]["technical"]["signal"], "sell")
        self.assertEqual(res["transcript"][1]["positions"]["technical"]["signal"], "buy")
        self.assertEqual(res["final"]["signal"], "buy")

    def test_risk_veto(self) -> None:
        # Strong technical BUY, sentiment HOLD, risk SELL -> deadlock, veto -> HOLD.
        positions = [
            Position("technical", "buy", 0.9, "strong momentum"),
            Position("sentiment", "hold", 0.4, "mixed"),
            Position("risk", "sell", 0.8, "high risk"),
        ]
        decision = CoordinatorJudge(backend=self.backend).decide(positions)
        self.assertTrue(decision.risk_veto)
        self.assertEqual(decision.signal, "hold")
        self.assertEqual(decision.mode, "tie-break")

    def test_demo_accuracy_is_deterministic(self) -> None:
        correct = 0
        for sc in self.scenarios:
            res = run_llm_debate(sc, backend=self.backend)
            correct += res["final"]["signal"] == sc["expected_signal"]
        self.assertEqual(correct, len(self.scenarios))  # mock is designed to hit all


if __name__ == "__main__":
    unittest.main()
