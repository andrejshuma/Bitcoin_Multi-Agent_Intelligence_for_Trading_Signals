"""Predict the risk regime with the real LightGBM model and print it as JSON.

Run as a SUBPROCESS (``python -m agentic_prototype._risk_subproc``) so that
LightGBM never shares a process with PyTorch -- co-loading the two segfaults on
macOS (duplicate OpenMP runtimes). This process imports only pandas + lightgbm.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATS = REPO_ROOT / "technical_analysis" / "data" / "processed" / "features_1h.parquet"
_MODEL = REPO_ROOT / "agent_risk" / "models" / "risk_lgbm.pkl"
_META = REPO_ROOT / "agent_risk" / "models" / "risk_meta.json"


def predict() -> dict:
    import pandas as pd  # noqa: PLC0415

    with open(_MODEL, "rb") as f:
        model = pickle.load(f)
    meta = json.loads(_META.read_text()) if _META.exists() else {}
    features = meta.get("features", [])
    label_map = meta.get("label_map", {"0": "low_risk", "1": "medium_risk", "2": "high_risk"})

    feats = pd.read_parquet(_FEATS)
    feats.index = pd.to_datetime(feats.index)
    feats = feats.sort_index()
    available = [c for c in features if c in feats.columns]
    row = feats[available].iloc[[-1]].values.astype("float32")

    pred = int(model.predict(row)[0])
    proba = [round(float(p), 4) for p in model.predict_proba(row)[0]]
    return {
        "source": "model",
        "risk_level": label_map.get(str(pred), "medium_risk"),
        "model_confidence": proba[pred],
        "class_probabilities": {"low": proba[0], "medium": proba[1], "high": proba[2]},
        "drawdown_bands": (
            f"low<{meta.get('low_dd_thresh', 0.01) * 100:.0f}% / "
            f"high>{meta.get('high_dd_thresh', 0.03) * 100:.0f}% over "
            f"{meta.get('drawdown_horizon', 48)}h"
        ),
        "as_of": str(feats.index[-1]),
    }


if __name__ == "__main__":
    print(json.dumps(predict()))
