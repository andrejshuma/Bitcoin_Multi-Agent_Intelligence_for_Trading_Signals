"""The three debating LLM agents and the coordinator judge.

Each agent is one instance of the *same* LLM given a different domain persona
(technical / sentiment / risk). It reasons over a structured brief of its
domain's model outputs and emits ``{signal, confidence, reasoning}``. During
debate it is shown the other agents' positions and may revise its own.

The coordinator is a JUDGE, not a re-calculator: when the agents agree it
relays the consensus; on a split it decides by domain priority (technical is
primary) with a risk veto, then asks the LLM to write the final narrative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm_chat import extract_json

SIGNALS = ("buy", "sell", "hold")


@dataclass
class Position:
    agent: str
    signal: str
    confidence: float
    reasoning: str

    def to_dict(self) -> dict:
        return {"agent": self.agent, "signal": self.signal,
                "confidence": self.confidence, "reasoning": self.reasoning}


@dataclass
class FinalDecision:
    signal: str
    confidence: float
    reasoning: str
    mode: str            # "consensus" | "majority" | "tie-break"
    rationale: str       # short machine reason for the decision
    risk_veto: bool = False


# --------------------------------------------------------------------------- #
# Domain personas
# --------------------------------------------------------------------------- #
_OUTPUT_RULE = (
    'Respond with ONLY a compact JSON object and nothing else: '
    '{"signal": "buy|sell|hold", "confidence": 0.0-1.0, "reasoning": "2-4 sentences"}.'
)

PERSONAS: Dict[str, str] = {
    "technical": (
        "You are a senior Bitcoin technical-analysis trader. You judge ONLY from price action, "
        "momentum, trend and volatility evidence (model trade/long probabilities, RSI, MACD, ATR, "
        "EMA structure). Ignore news and macro. Decide the trade for roughly the next 36 hours. "
        + _OUTPUT_RULE
    ),
    "sentiment": (
        "You are a Bitcoin market sentiment & macro analyst. You judge ONLY from news sentiment, "
        "social tone and macro indicators (FinBERT score, headlines, rates/inflation). Ignore chart "
        "internals. Decide whether the narrative favors buying, selling or holding. " + _OUTPUT_RULE
    ),
    "risk": (
        "You are a Bitcoin risk manager. You judge ONLY from volatility and drawdown risk (risk "
        "regime, ATR, forward-drawdown estimate). Your 'buy' means risk-on (safe to take exposure), "
        "'sell' means risk-off (reduce/avoid exposure), 'hold' means neutral risk. " + _OUTPUT_RULE
    ),
}


def _brief_text(brief: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in brief.items())


@dataclass
class LLMAgent:
    agent: str                 # "technical" | "sentiment" | "risk"
    backend: object            # Chat
    temperature: float = 0.4

    def initial(self, brief: dict) -> Position:
        user = (
            f"Here is your {self.agent} evidence for BTC right now:\n{_brief_text(brief)}\n\n"
            f"Give your independent signal. {_OUTPUT_RULE}"
        )
        raw = self.backend.complete(
            PERSONAS[self.agent], user,
            context={"kind": "agent_initial", "agent": self.agent, "brief": brief},
        )
        return self._parse(raw)

    def revise(self, own: Position, others: List[Position], round_no: int) -> Position:
        others_txt = "\n".join(
            f"- {o.agent} analyst says {o.signal.upper()} (confidence {o.confidence:.2f}): {o.reasoning}"
            for o in others
        )
        user = (
            f"This is debate round {round_no}. Your current position is "
            f"{own.signal.upper()} (confidence {own.confidence:.2f}): {own.reasoning}\n\n"
            f"The other analysts say:\n{others_txt}\n\n"
            f"Reconsider in light of their views, but stay in your {self.agent} lane: weigh your own "
            f"evidence first and only move if their argument genuinely outweighs yours. "
            f"{_OUTPUT_RULE}"
        )
        raw = self.backend.complete(
            PERSONAS[self.agent], user,
            context={"kind": "agent_revise", "agent": self.agent,
                     "own": own.to_dict(), "others": [o.to_dict() for o in others]},
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> Position:
        data = extract_json(raw) if isinstance(raw, str) else (raw or {})
        signal = str(data.get("signal", "hold")).lower().strip()
        if signal not in SIGNALS:
            signal = "hold"
        try:
            conf = float(data.get("confidence", 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        conf = max(0.0, min(1.0, conf))
        reasoning = str(data.get("reasoning", "")).strip() or "(no reasoning returned)"
        return Position(agent=self.agent, signal=signal, confidence=conf, reasoning=reasoning)


# --------------------------------------------------------------------------- #
# Coordinator judge
# --------------------------------------------------------------------------- #
@dataclass
class CoordinatorJudge:
    backend: object
    # Domain priority for tie-breaks; technical is primary (matches the 65/25/10
    # philosophy). Risk holds a veto over BUY.
    primary: str = "technical"

    def decide(self, positions: List[Position], narrate: bool = True) -> FinalDecision:
        by_agent = {p.agent: p for p in positions}
        votes = [p.signal for p in positions]
        counts = {s: votes.count(s) for s in SIGNALS}

        if counts[votes[0]] == len(votes):  # unanimous
            signal, mode, rationale = votes[0], "consensus", "all three agents agreed"
        else:
            top = max(counts, key=counts.get)
            if list(counts.values()).count(counts[top]) == 1 and counts[top] >= 2:
                signal, mode = top, "majority"
                rationale = f"{counts[top]}/3 agents favored {top.upper()}"
            else:  # 3-way split / tie -> domain priority
                signal = by_agent[self.primary].signal
                mode = "tie-break"
                rationale = f"deadlock; deferred to the {self.primary} agent ({signal.upper()})"

        # Risk veto: the risk manager can downgrade a BUY to HOLD.
        risk = by_agent.get("risk")
        risk_veto = bool(signal == "buy" and risk is not None and risk.signal == "sell")
        if risk_veto:
            signal = "hold"
            rationale += "; risk manager vetoed BUY (risk-off) -> HOLD"

        agree = sum(1 for p in positions if p.signal == signal)
        confidence = round(
            min(0.97, (sum(p.confidence for p in positions) / len(positions)) * (0.6 + 0.2 * agree)),
            3,
        )
        decision = {"signal": signal, "confidence": confidence, "rationale": rationale}

        if narrate:
            narrative = self.backend.complete(
                "You are the lead coordinator. Write 2-4 sentences explaining the final BTC trading "
                "decision to a trader, integrating the analysts' debate. Be decisive and concise.",
                self._narrative_prompt(decision, positions),
                context={"kind": "coordinator", "decision": decision,
                         "positions": [p.to_dict() for p in positions]},
            ).strip()
        else:
            narrative = rationale
        return FinalDecision(
            signal=signal, confidence=confidence, reasoning=narrative,
            mode=mode, rationale=rationale, risk_veto=risk_veto,
        )

    @staticmethod
    def _narrative_prompt(decision: dict, positions: List[Position]) -> str:
        pos = "\n".join(
            f"- {p.agent}: {p.signal.upper()} ({p.confidence:.2f}) - {p.reasoning}" for p in positions
        )
        return (
            f"Final decision: {decision['signal'].upper()} (confidence {decision['confidence']:.2f}).\n"
            f"Why: {decision['rationale']}.\n"
            f"Final analyst positions after the debate:\n{pos}\n\n"
            f"Explain the {decision['signal'].upper()} decision."
        )
