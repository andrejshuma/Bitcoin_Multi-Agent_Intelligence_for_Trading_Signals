import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

try:
    from .models import FinalCoordinatorSignal
    from .technical_agent import TechnicalAgent
except ImportError:
    from models import FinalCoordinatorSignal
    from technical_agent import TechnicalAgent


class CoordinatorAgent:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.sentiment_dir = repo_root / "sentiment_analysis"
        self.risk_dir = repo_root / "agent_risk"
        self.technical_agent = TechnicalAgent(repo_root=repo_root)

    @staticmethod
    def _signal_to_score(signal: str) -> int:
        mapping = {"buy": 1, "hold": 0, "sell": -1}
        return mapping.get(signal.lower(), 0)

    def _run_json_script(self, cwd: Path, script_name: str) -> Dict:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [sys.executable, script_name],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"{script_name} failed with code {completed.returncode}. stderr={completed.stderr.strip()}"
            )

        stdout_text = completed.stdout or ""
        lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)

        raise RuntimeError(f"No JSON payload found in output of {script_name}.")

    @staticmethod
    def _signal_to_unit(signal: str) -> float:
        return float({"buy": 1.0, "hold": 0.0, "sell": -1.0}.get(str(signal).lower(), 0.0))

    def _build_agentic_case(self, technical: Dict, sentiment: Dict, risk: Dict) -> Dict:
        trend_strength = 2.0 * (float(technical.get("long_probability", 0.5)) - 0.5)
        narrative = " ".join(sentiment.get("key_factors", []))
        return {
            "id": f"live-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "technical_momentum": self._signal_to_unit(technical.get("signal", "hold")) * float(technical.get("confidence", 0.0)),
            "trend_strength": trend_strength,
            "volatility": float(risk.get("volatility_score", risk.get("risk_score", 0.5))),
            "news_sentiment": float(sentiment.get("news_sentiment", sentiment.get("sentiment_score", 0.0))),
            "social_sentiment": float(sentiment.get("twitter_sentiment", 0.0)),
            "macro_sentiment": float(sentiment.get("macro_score", 0.0)),
            "onchain_risk": float(risk.get("onchain_risk", risk.get("risk_score", 0.5))),
            "geopolitical_risk": float(risk.get("geopolitical_risk", risk.get("risk_score", 0.5))),
            "narrative": narrative,
        }

    def _run_agentic(self, technical: Dict, sentiment: Dict, risk: Dict) -> FinalCoordinatorSignal:
        try:
            from agentic_prototype.workflow import run_case
        except ImportError:
            if str(self.repo_root) not in sys.path:
                sys.path.insert(0, str(self.repo_root))
            from agentic_prototype.workflow import run_case

        case = self._build_agentic_case(technical, sentiment, risk)
        prediction = run_case(case)
        risk_level = str(risk.get("signal", "medium_risk")).lower()
        key_factors = [
            f"Technical: {technical['signal'].upper()} ({technical['confidence']:.2f})",
            f"Sentiment: {sentiment.get('signal', 'hold').upper()} ({float(sentiment.get('confidence', 0.0)):.2f})",
            f"Risk: {risk_level.upper()} ({float(risk.get('risk_score', 0.5)):.2f})",
            "Fusion: Agentic skill-based LangGraph workflow",
        ]
        reasoning = (
            f"Agentic coordinator output score {prediction['predicted_score']:+.2f} with signal "
            f"{prediction['predicted_signal'].upper()} and confidence {prediction['predicted_confidence']:.2f}. "
            f"Workflow reasoning: {prediction['reasoning']} Technical reason: {technical['reasoning']} "
            f"Sentiment reason: {sentiment.get('reasoning', 'N/A')} Risk reason: {risk.get('reasoning', 'N/A')}"
        )
        return FinalCoordinatorSignal(
            signal=prediction["predicted_signal"],
            confidence=float(prediction["predicted_confidence"]),
            score=float(prediction["predicted_score"]),
            risk_level=risk_level,
            key_factors=key_factors,
            reasoning=reasoning,
            data_sources=[
                "technical_analysis models",
                "sentiment_analysis",
                "agent_risk",
                "agentic_prototype LangGraph",
            ],
        )

    def _run_debate(self, technical: Dict, risk: Dict) -> FinalCoordinatorSignal:
        """mode='debate': 3 LLM agents (technical/sentiment/risk) debate over the
        real model outputs for 2 rounds, then the coordinator judge gives the final
        signal+reasoning. Sentiment is grounded in live NewsAPI headlines -> FinBERT."""
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from agentic_prototype.live_briefs import (
            build_sentiment_brief,
            risk_brief_from_signal,
            technical_brief_from_signal,
        )
        from agentic_prototype.llm_chat import get_backend
        from agentic_prototype.llm_debate import AGENTS, run_llm_debate

        scenario = {
            "id": f"live-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "technical": technical_brief_from_signal(technical),
            "sentiment": build_sentiment_brief(),  # live NewsAPI headlines -> FinBERT
            "risk": risk_brief_from_signal(risk),
            "expected_signal": None,
        }
        result = run_llm_debate(scenario, backend=get_backend(verbose=False), rounds=2)
        final = result["final"]
        last = result["transcript"][-1]["positions"]

        signal = final["signal"]
        confidence = float(final["confidence"])
        score = max(-1.0, min(1.0, self._signal_to_unit(signal) * confidence))
        risk_level = str(risk.get("signal", "medium_risk")).lower()

        key_factors = [
            f"{a.capitalize()}: {last[a]['signal'].upper()} ({last[a]['confidence']:.2f})" for a in AGENTS
        ]
        key_factors.append(
            f"Debate: {final['mode']}" + (" + risk veto" if final.get("risk_veto") else "")
        )
        key_factors.append(f"Sentiment headlines: {scenario['sentiment'].get('headline_source', 'n/a')}")

        return FinalCoordinatorSignal(
            signal=signal,
            confidence=confidence,
            score=score,
            risk_level=risk_level,
            key_factors=key_factors,
            reasoning=final["reasoning"],
            data_sources=[
                "LLM debate (Groq llama-3.1-8b-instant)",
                "CNN-LSTM technical",
                "FinBERT + NewsAPI sentiment",
                "LightGBM risk",
            ],
        )

    def _combine_signals(self, technical: Dict, sentiment: Dict, risk: Dict) -> Tuple[str, float, float, str]:
        tech_score = self._signal_to_score(technical["signal"]) * float(technical.get("confidence", 0.0))
        sentiment_score = self._signal_to_score(sentiment["signal"]) * float(sentiment.get("confidence", 0.0))

        combined_score = 0.60 * tech_score + 0.40 * sentiment_score

        risk_signal = str(risk.get("signal", "medium_risk")).lower()
        risk_multiplier = {
            "low_risk": 1.10,
            "medium_risk": 1.00,
            "high_risk": 0.65,
        }.get(risk_signal, 1.0)

        adjusted_score = max(-1.0, min(1.0, combined_score * risk_multiplier))

        if risk_signal == "high_risk" and adjusted_score > 0.55:
            final_signal = "hold"
        elif adjusted_score >= 0.25:
            final_signal = "buy"
        elif adjusted_score <= -0.25:
            final_signal = "sell"
        else:
            final_signal = "hold"

        confidence = min(1.0, abs(adjusted_score) + 0.15 * float(risk.get("confidence", 0.0)))
        return final_signal, confidence, adjusted_score, risk_signal

    def run(self, mode: str = "agentic") -> FinalCoordinatorSignal:
        mode_l = mode.lower()
        technical = self.technical_agent.run().model_dump()
        risk = self._run_json_script(self.risk_dir, "run_agent_json.py")

        # Debate mode sources its own sentiment from live headlines (NewsAPI ->
        # FinBERT) inside the debate, so the sentiment subprocess is skipped here.
        if mode_l == "debate":
            return self._run_debate(technical, risk)

        sentiment = self._run_json_script(self.sentiment_dir, "run_agent_json.py")

        if mode_l == "agentic":
            return self._run_agentic(technical, sentiment, risk)
        if mode_l != "legacy":
            raise ValueError("Unsupported coordinator mode. Use 'agentic', 'legacy', or 'debate'.")

        signal, confidence, score, risk_level = self._combine_signals(technical, sentiment, risk)

        key_factors = [
            f"Technical: {technical['signal'].upper()} ({technical['confidence']:.2f})",
            f"Sentiment: {sentiment.get('signal', 'hold').upper()} ({float(sentiment.get('confidence', 0.0)):.2f})",
            f"Risk: {risk_level.upper()} ({float(risk.get('risk_score', 0.5)):.2f})",
        ]

        reasoning = (
            f"Coordinator combined technical and sentiment scores into {score:+.2f}, then adjusted using risk level "
            f"{risk_level}. Final signal is {signal.upper()} with confidence {confidence:.2f}. "
            f"Technical reason: {technical['reasoning']} Sentiment reason: {sentiment.get('reasoning', 'N/A')} "
            f"Risk reason: {risk.get('reasoning', 'N/A')}"
        )

        return FinalCoordinatorSignal(
            signal=signal,
            confidence=confidence,
            score=score,
            risk_level=risk_level,
            key_factors=key_factors,
            reasoning=reasoning,
            data_sources=[
                "technical_analysis models",
                "technical_analysis features_1h.parquet",
                "sentiment_analysis",
                "agent_risk",
            ],
        )


