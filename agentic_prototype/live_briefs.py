"""Build the three agents' debate briefs from the REAL models.

Each brief is the structured evidence one LLM agent reasons over:

* technical -> real CNN-LSTM (coordinator_agent.TechnicalAgent) trade/long probs
* risk      -> real LightGBM model (agent_risk/models/risk_lgbm.pkl) on the latest
               feature bar
* sentiment -> real FinBERT (ProsusAI/finbert) over headlines (demo headlines for
               now, since there is no live news corpus -- swap in real headlines
               by passing ``headlines=[...]``)

Every builder degrades gracefully: if a model/file is missing it returns a clearly
marked demo brief, so the pipeline always runs. Each brief carries a ``source``
field ("model" or "demo") so you can see what was real."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATS = REPO_ROOT / "technical_analysis" / "data" / "processed" / "features_1h.parquet"
_RISK_MODEL = REPO_ROOT / "agent_risk" / "models" / "risk_lgbm.pkl"
_RISK_META = REPO_ROOT / "agent_risk" / "models" / "risk_meta.json"

DEMO_HEADLINES = [
    "Spot Bitcoin ETFs see steady weekly inflows",
    "Fed minutes signal a patient stance on rates",
    "Analysts debate whether BTC can hold key support",
]


# --------------------------------------------------------------------------- #
# Briefs from already-computed agent outputs (used by the coordinator so the
# models are not re-run).
# --------------------------------------------------------------------------- #
def technical_brief_from_signal(sig: Dict) -> Dict:
    lp = float(sig.get("long_probability", 0.5))
    return {
        "source": "model",
        "trade_probability": round(float(sig.get("trade_probability", 0.5)), 4),
        "long_probability": round(lp, 4),
        "model_signal": sig.get("signal", "hold"),
        "model_confidence": round(float(sig.get("confidence", 0.0)), 4),
        "trend": "long-biased" if lp >= 0.5 else "short-biased",
        "model_reasoning": sig.get("reasoning", ""),
    }


def risk_brief_from_signal(r: Dict) -> Dict:
    return {
        "source": "model",
        "risk_level": str(r.get("signal", "medium_risk")).lower(),
        "risk_score": round(float(r.get("risk_score", 0.5)), 4),
        "model_confidence": round(float(r.get("confidence", 0.0)), 4),
        "model_reasoning": r.get("reasoning", ""),
    }


# --------------------------------------------------------------------------- #
# Live headlines via the existing NewsAPI client (real news -> FinBERT)
# --------------------------------------------------------------------------- #
def fetch_live_headlines(days_back: int = 2, max_n: int = 12, repo_root: Path = REPO_ROOT) -> Tuple[List[str], str]:
    """Pull recent BTC headlines using sentiment_analysis/api_clients.NewsAPIClient.
    Returns (headlines, source) where source is 'newsapi-live', 'newsapi-mock', or
    'unavailable:<reason>'."""
    try:
        sdir = str(repo_root / "sentiment_analysis")
        if sdir not in sys.path:
            sys.path.insert(0, sdir)
        from api_clients import NewsAPIClient

        client = NewsAPIClient()
        items = client.fetch_crypto_news(days_back=days_back)
        heads = [i.title for i in items if getattr(i, "title", None)][:max_n]
        return heads, ("newsapi-live" if client.api_key else "newsapi-mock")
    except Exception as exc:
        return [], f"unavailable:{type(exc).__name__}"


# --------------------------------------------------------------------------- #
# Technical -- real CNN-LSTM
# --------------------------------------------------------------------------- #
def build_technical_brief(repo_root: Path = REPO_ROOT) -> Dict:
    try:
        import sys

        cdir = str(repo_root / "coordinator_agent")
        if cdir not in sys.path:
            sys.path.insert(0, cdir)
        from technical_agent import TechnicalAgent

        sig = TechnicalAgent(repo_root=repo_root).run()
        return {
            "source": "model",
            "trade_probability": round(sig.trade_probability, 4),
            "long_probability": round(sig.long_probability, 4),
            "model_signal": sig.signal,
            "model_confidence": round(sig.confidence, 4),
            "sequence_length": sig.sequence_length,
            "trend": "long-biased" if sig.long_probability >= 0.5 else "short-biased",
            "model_reasoning": sig.reasoning,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"source": f"demo (technical model unavailable: {type(exc).__name__})",
                "trade_probability": 0.40, "long_probability": 0.50, "trend": "rangebound"}


# --------------------------------------------------------------------------- #
# Risk -- real LightGBM on the latest feature bar (in a torch-free subprocess)
# --------------------------------------------------------------------------- #
def build_risk_brief(repo_root: Path = REPO_ROOT) -> Dict:
    # LightGBM must NOT share a process with PyTorch (segfaults on macOS), so the
    # prediction runs in an isolated subprocess that imports only pandas+lightgbm.
    import subprocess
    import sys

    try:
        out = subprocess.run(
            [sys.executable, "-m", "agentic_prototype._risk_subproc"],
            cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=120,
        )
        for line in reversed([l.strip() for l in out.stdout.splitlines() if l.strip()]):
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise RuntimeError("no JSON from risk subprocess")
    except Exception as exc:  # pragma: no cover - defensive
        return {"source": f"demo (risk model unavailable: {type(exc).__name__})",
                "risk_level": "medium_risk", "class_probabilities": {"low": 0.4, "medium": 0.4, "high": 0.2}}


# --------------------------------------------------------------------------- #
# Sentiment -- real FinBERT over headlines (demo headlines by default)
# --------------------------------------------------------------------------- #
def build_sentiment_brief(headlines: Optional[List[str]] = None, macro: str = "macro context not fetched") -> Dict:
    # Default: pull REAL headlines from NewsAPI, fall back to demo headlines.
    if headlines is None:
        headlines, headline_source = fetch_live_headlines()
        if not headlines:
            headlines, headline_source = DEMO_HEADLINES, "demo"
    else:
        headline_source = "provided"

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = "ProsusAI/finbert"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}

        with torch.no_grad():
            enc = tok(headlines, return_tensors="pt", padding=True, truncation=True)
            probs = torch.softmax(model(**enc).logits, dim=-1).mean(dim=0)
        dist = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        score = round(dist.get("positive", 0.0) - dist.get("negative", 0.0), 4)
        tone = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "mixed"
        return {
            "source": "model",
            "headline_source": headline_source,
            "finbert_score": score,
            "label_distribution": dist,
            "twitter_tone": tone,
            "macro": macro,
            "news_headlines": headlines,
        }
    except Exception as exc:
        # Lightweight keyword fallback so the pipeline still runs.
        pos = ("inflow", "adoption", "rally", "surge", "etf", "bullish")
        neg = ("probe", "ban", "lawsuit", "selloff", "hack", "bearish", "liquidation")
        text = " ".join(headlines).lower()
        score = round(0.2 * (sum(w in text for w in pos) - sum(w in text for w in neg)), 4)
        return {"source": f"demo (FinBERT unavailable: {type(exc).__name__})",
                "headline_source": headline_source, "finbert_score": score,
                "twitter_tone": "mixed", "macro": macro, "news_headlines": headlines}


def build_live_scenario(repo_root: Path = REPO_ROOT, headlines: Optional[List[str]] = None) -> Dict:
    """Assemble a single live scenario (latest bar) ready for run_llm_debate."""
    return {
        "id": f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "technical": build_technical_brief(repo_root),
        "sentiment": build_sentiment_brief(headlines),
        "risk": build_risk_brief(repo_root),
        "expected_signal": None,
    }
