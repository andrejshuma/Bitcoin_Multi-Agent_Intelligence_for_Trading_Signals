"""Tests for the persona-prompting layer and its evaluation statistics."""

from __future__ import annotations

import unittest

from agentic_prototype.llm_agents import DOMAIN_INSTRUCTIONS, LLMAgent
from agentic_prototype.llm_chat import MockChat
from agentic_prototype.llm_debate import run_llm_debate
from agentic_prototype.personas import (
    ALL_PERSONAS, BASELINE, BY_NAME, mock_adjust, persona_injection, resolve,
)


class PersonaLayerTests(unittest.TestCase):
    def test_resolve_accepts_name_object_and_none(self) -> None:
        self.assertIs(resolve(None), BASELINE)
        self.assertIs(resolve("aggressive"), BY_NAME["aggressive"])
        self.assertIs(resolve(BY_NAME["fearful"]), BY_NAME["fearful"])
        with self.assertRaises(ValueError):
            resolve("nope")

    def test_baseline_injection_is_empty(self) -> None:
        self.assertEqual(persona_injection("baseline"), "")
        self.assertEqual(persona_injection(None), "")

    def test_persona_injection_present_and_in_lane(self) -> None:
        inj = persona_injection("conservative")
        self.assertIn("conservative", inj)
        self.assertIn("do not break character", inj.lower())
        self.assertIn("within your analytical domain", inj.lower())

    def test_agent_system_prompt_composition(self) -> None:
        base = LLMAgent(agent="technical", backend=MockChat())._system()
        cons = LLMAgent(agent="technical", backend=MockChat(), persona="conservative")._system()
        self.assertIn(DOMAIN_INSTRUCTIONS["technical"], base)
        self.assertIn(DOMAIN_INSTRUCTIONS["technical"], cons)   # persona keeps the domain lane
        self.assertNotIn("conservative", base)
        self.assertIn("conservative", cons)

    def test_debate_reports_persona_and_runs_for_all(self) -> None:
        scenario = {"id": "t", "technical": {"trade_probability": 0.8, "long_probability": 0.7},
                    "sentiment": {"finbert_score": 0.4}, "risk": {"risk_level": "low_risk"}}
        for p in ALL_PERSONAS:
            res = run_llm_debate(scenario, backend=MockChat(), rounds=2, narrate=False, persona=p)
            self.assertEqual(res["persona"], p.name)
            self.assertIn(res["final"]["signal"], ("buy", "sell", "hold"))


class MockAdjustTests(unittest.TestCase):
    def test_contrarian_flips_direction(self) -> None:
        self.assertEqual(mock_adjust("contrarian", "buy", 0.7, {})[0], "sell")
        self.assertEqual(mock_adjust("contrarian", "sell", 0.7, {})[0], "buy")
        self.assertEqual(mock_adjust("contrarian", "hold", 0.7, {})[0], "hold")

    def test_risk_seeking_activates_hold(self) -> None:
        sig, _ = mock_adjust("aggressive", "hold", 0.5, {"long_probability": 0.7})
        self.assertIn(sig, ("buy", "sell"))
        self.assertEqual(mock_adjust("greedy", "hold", 0.5, {"long_probability": 0.2})[0], "buy")

    def test_risk_averse_fades_weak_directional(self) -> None:
        self.assertEqual(mock_adjust("conservative", "buy", 0.5, {})[0], "hold")
        self.assertEqual(mock_adjust("fearful", "buy", 0.65, {})[0], "hold")

    def test_baseline_is_identity_signal(self) -> None:
        self.assertEqual(mock_adjust("baseline", "buy", 0.6, {})[0], "buy")


class StatisticsTests(unittest.TestCase):
    def test_krippendorff_alpha_bounds(self) -> None:
        from scripts.evaluation.persona_eval import krippendorff_alpha_nominal

        agree = [["buy", "sell", "hold"], ["buy", "sell", "hold"], ["buy", "sell", "hold"]]
        self.assertAlmostEqual(krippendorff_alpha_nominal(agree), 1.0, places=6)
        # systematic disagreement -> alpha below chance (< 0)
        disagree = [["buy", "buy", "buy"], ["sell", "sell", "sell"], ["hold", "hold", "hold"]]
        self.assertLess(krippendorff_alpha_nominal(disagree), 0.0)

    def test_flesch_and_word_count(self) -> None:
        from scripts.evaluation.persona_eval import flesch_reading_ease, word_count

        self.assertEqual(word_count("The cat sat on the mat."), 6)
        self.assertGreater(flesch_reading_ease("The cat sat on the mat."), 80)   # very easy text

    def test_bootstrap_flags_clear_difference(self) -> None:
        import numpy as np

        from scripts.evaluation.persona_eval import bootstrap_pnl_diff

        a = np.full(200, 0.01)
        b = np.zeros(200)
        out = bootstrap_pnl_diff(a, b, iters=500)
        self.assertTrue(out["significant"])
        self.assertGreater(out["diff_mean"], 0)


if __name__ == "__main__":
    unittest.main()
