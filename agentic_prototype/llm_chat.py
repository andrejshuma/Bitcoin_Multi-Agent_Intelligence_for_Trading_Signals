"""LLM chat backend for the multi-agent *text* debate.

One small interface, ``Chat.complete(system, user, context) -> str``, with two
implementations:

* :class:`GroqChat`  -- real LLM, Groq ``llama-3.1-8b-instant`` (same model the
  rest of the repo already uses). Requires ``GROQ_API_KEY`` and the ``groq``
  package.
* :class:`MockChat`  -- a deterministic stand-in that derives a plausible
  answer from the structured ``context`` so the whole debate pipeline runs and
  is testable offline (no key, no network). It is NOT a model -- it is a
  scaffold so you can see the plumbing work before plugging in real keys.

``get_backend()`` returns :class:`GroqChat` when a key + package are available,
otherwise :class:`MockChat` (printing a one-line notice)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

try:  # load repo-root .env if present (GROQ_API_KEY)
    import env_loader  # noqa: F401
except Exception:
    pass

_SIGNALS = ("buy", "sell", "hold")


def extract_json(text: str) -> dict:
    """Tolerantly pull the first JSON object out of an LLM response."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


@dataclass
class GroqChat:
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.4
    max_tokens: int = 400
    max_retries: int = 6
    name: str = "groq:llama-3.1-8b-instant"

    def complete(self, system: str, user: str, context: Optional[dict] = None) -> str:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    max_tokens=self.max_tokens, messages=messages,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:  # rate limits / transient network -> backoff & retry
                last_err = e
                time.sleep(min(45.0, 3.0 * (attempt + 1) ** 2))
        raise last_err


@dataclass
class MockChat:
    """Deterministic offline stand-in. Uses ``context`` (not the prompt text) to
    produce a believable JSON answer, so the debate transcript is meaningful
    without any API access."""

    name: str = "mock"

    def complete(self, system: str, user: str, context: Optional[dict] = None) -> str:
        context = context or {}
        kind = context.get("kind")
        if kind == "agent_initial":
            return json.dumps(self._initial(context["agent"], context["brief"]))
        if kind == "agent_revise":
            return json.dumps(self._revise(context["agent"], context["own"], context["others"]))
        if kind == "coordinator":
            return self._coordinator_text(context["decision"], context["positions"])
        return json.dumps({"signal": "hold", "confidence": 0.4, "reasoning": "No context."})

    # -- per-domain initial heuristics ------------------------------------- #
    @staticmethod
    def _initial(agent: str, brief: dict) -> dict:
        if agent == "technical":
            tp = float(brief.get("trade_probability", 0.5))
            lp = float(brief.get("long_probability", 0.5))
            if tp < 0.45:
                sig, conf = "hold", round(1.0 - tp, 2)
            else:
                sig = "buy" if lp >= 0.55 else "sell" if lp <= 0.45 else "hold"
                conf = round(tp * max(lp, 1 - lp), 2)
            reasoning = (
                f"Trade-probability {tp:.2f} and long-probability {lp:.2f}; "
                f"{brief.get('trend', 'no clear trend')}. RSI-14 {brief.get('rsi_14', '?')}, "
                f"MACD hist {brief.get('macd_hist', '?')}, price vs EMA200 {brief.get('price_vs_ema200', '?')}."
            )
        elif agent == "sentiment":
            s = float(brief.get("finbert_score", 0.0))
            sig = "buy" if s > 0.2 else "sell" if s < -0.2 else "hold"
            conf = round(min(0.9, 0.4 + abs(s)), 2)
            heads = brief.get("news_headlines", [])
            reasoning = (
                f"FinBERT net sentiment {s:+.2f}; social tone {brief.get('twitter_tone', 'neutral')}, "
                f"macro: {brief.get('macro', 'n/a')}. Top headline: "
                f"\"{heads[0] if heads else 'n/a'}\"."
            )
        elif agent == "risk":
            rl = str(brief.get("risk_level", "medium_risk"))
            sig = {"low_risk": "buy", "medium_risk": "hold", "high_risk": "sell"}.get(rl, "hold")
            conf = {"low_risk": 0.6, "medium_risk": 0.5, "high_risk": 0.78}.get(rl, 0.5)
            reasoning = (
                f"Risk regime {rl}; ATR {brief.get('atr_pct', '?')}% of price, "
                f"forward-drawdown estimate {brief.get('forward_dd_estimate', '?')}. "
                f"{'Risk-off, protect capital.' if sig == 'sell' else 'Risk-on.' if sig == 'buy' else 'Neutral risk.'}"
            )
        else:
            sig, conf, reasoning = "hold", 0.4, "Unknown agent."
        return {"signal": sig, "confidence": conf, "reasoning": reasoning}

    # -- revision: unsure agents move toward an agreeing majority ----------- #
    @staticmethod
    def _revise(agent: str, own: dict, others: list) -> dict:
        votes = [o["signal"] for o in others]
        names = ", ".join(o["agent"] for o in others)
        own_conf = float(own.get("confidence", 0.5))
        if own_conf < 0.6 and len(set(votes)) == 1 and votes[0] != own["signal"]:
            new = votes[0]
            return {
                "signal": new,
                "confidence": round(min(0.9, own_conf + 0.1), 2),
                "reasoning": (
                    f"Both the {names} agree on {new.upper()}, and my {agent} read "
                    f"({own['signal'].upper()}) was only weakly held; I update toward {new.upper()}."
                ),
            }
        return {
            "signal": own["signal"],
            "confidence": own_conf,
            "reasoning": (
                f"I hold {own['signal'].upper()}: my {agent} evidence still outweighs the "
                f"{names} views."
            ),
        }

    @staticmethod
    def _coordinator_text(decision: dict, positions: list) -> str:
        lines = "; ".join(f"{p['agent']}={p['signal'].upper()}" for p in positions)
        return (
            f"Coordinator verdict: {decision['signal'].upper()} (confidence "
            f"{decision['confidence']:.2f}). Resolution: {decision['rationale']} "
            f"Final agent positions after debate: {lines}."
        )


def get_backend(verbose: bool = True):
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            import groq  # noqa: F401

            if verbose:
                print("[llm] backend: Groq llama-3.1-8b-instant")
            return GroqChat()
        except ImportError:
            if verbose:
                print("[llm] GROQ_API_KEY set but 'groq' package missing -> install it. Using MockChat.")
    elif verbose:
        print("[llm] GROQ_API_KEY not set -> using deterministic MockChat (offline demo).")
    return MockChat()
