"""Persona-prompting layer for the debate agents (Yang et al., EACL 2026,
*"Persona Prompting as a Lens on LLM Social Reasoning"*).

The paper injects a socio-demographic persona into a classifier with a
"step into the shoes of ... / think out loud as this persona / do not break
character" scaffolding, then measures how the persona steers the output versus a
no-persona baseline. We port that idea to the trading debate: each of the three
debaters (technical / sentiment / risk) keeps its analytical *domain* but is
additionally conditioned on a **persona variant**, so we can compare how
different personas move the same real-model briefs.

Two persona families are provided:

* ``disposition`` -- trading character (aggressive, conservative, contrarian,
  disciplined, fearful, greedy). Domain-meaningful and measurable against
  realized returns.
* ``demographic`` -- a smaller socio-demographic probe (age, experience) that
  directly replicates the paper's "do LLMs resist persona steering / carry
  bias" question in a financial setting.

``baseline`` is the neutral domain persona (no injection) -- the control group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class Persona:
    name: str
    group: str            # "baseline" | "disposition" | "demographic"
    axis: str             # sub-axis for grouping (risk_appetite / style / emotion / age / experience)
    description: str       # the "who" -- goes into "step into the shoes of a real person who ..."


BASELINE = Persona("baseline", "baseline", "baseline", "")

DISPOSITION_PERSONAS: List[Persona] = [
    Persona("aggressive", "disposition", "risk_appetite",
            "is an aggressive, momentum-chasing trader, comfortable taking on risk to catch big "
            "moves and who hates missing a rally"),
    Persona("conservative", "disposition", "risk_appetite",
            "is a conservative, capital-preservation-first trader who prizes protecting the downside "
            "over catching every upside"),
    Persona("contrarian", "disposition", "style",
            "is a contrarian trader who instinctively fades the crowd and is deeply skeptical of "
            "consensus and hype"),
    Persona("disciplined", "disposition", "style",
            "is a disciplined, systematic trader who follows the evidence and rules mechanically, "
            "ignoring emotion and narrative"),
    Persona("fearful", "disposition", "emotion",
            "is a fearful, loss-averse trader, quick to de-risk at the first sign of trouble and "
            "haunted by past drawdowns"),
    Persona("greedy", "disposition", "emotion",
            "is a greedy, FOMO-driven trader, easily excited by upside and terrified of being left "
            "behind"),
]

DEMOGRAPHIC_PERSONAS: List[Persona] = [
    Persona("age_25", "demographic", "age",
            "is a 25-year-old retail trader"),
    Persona("age_65", "demographic", "age",
            "is a 65-year-old trader nearing retirement"),
    Persona("novice_retail", "demographic", "experience",
            "is a novice retail trader with little market experience"),
    Persona("veteran_institutional", "demographic", "experience",
            "is a veteran institutional trader with decades of experience through many market cycles"),
]

ALL_PERSONAS: List[Persona] = [BASELINE] + DISPOSITION_PERSONAS + DEMOGRAPHIC_PERSONAS
BY_NAME: Dict[str, Persona] = {p.name: p for p in ALL_PERSONAS}


def resolve(p: Union[str, Persona, None]) -> Persona:
    """Accept a persona, its name, or None and return a :class:`Persona`."""
    if p is None:
        return BASELINE
    if isinstance(p, Persona):
        return p
    try:
        return BY_NAME[p]
    except KeyError:
        raise ValueError(f"unknown persona '{p}'; known: {list(BY_NAME)}")


def persona_injection(persona: Union[str, Persona, None]) -> str:
    """The persona-conditioning text block (empty for baseline), following the
    paper's step-into-the-shoes / think-out-loud / stay-in-character framing."""
    persona = resolve(persona)
    if persona.group == "baseline" or not persona.description:
        return ""
    return (
        f" Additionally, step into the shoes of a real person who {persona.description}. "
        "Imagine you have lived your whole trading career shaped by this: let it guide what you "
        "notice, how sensitive you are to risk and to reward, and how you weigh the evidence -- but "
        "stay strictly within your analytical domain and never invent data. Reason in character, and "
        "do not break character."
    )


# --------------------------------------------------------------------------- #
# Offline persona effect for MockChat (deterministic, no API).
# --------------------------------------------------------------------------- #
# So the pipeline demonstrates persona steering without a live LLM: each persona
# nudges the evidence-derived (signal, confidence) in a plausible direction. This
# is a SCAFFOLD for offline testing -- the real analysis uses the Groq backend.
_DIRECTIONAL = ("buy", "sell")


def mock_adjust(persona: Union[str, Persona, None], signal: str, conf: float, brief: dict) -> tuple[str, float]:
    """Nudge a MockChat (signal, confidence) to reflect a persona's character."""
    persona = resolve(persona)
    name = persona.name

    def _long_lean() -> str:
        # Best guess of "which way" from whatever the brief exposes.
        lp = brief.get("long_probability")
        if lp is not None:
            return "buy" if float(lp) >= 0.5 else "sell"
        s = brief.get("finbert_score")
        if s is not None:
            return "buy" if float(s) >= 0 else "sell"
        return "buy"

    if name in ("aggressive", "greedy", "age_25", "novice_retail"):
        # Risk-seeking: turn weak HOLDs into a directional call, add conviction.
        if signal == "hold":
            signal = _long_lean() if name != "greedy" else "buy"
        conf = min(0.95, conf + 0.10)
    elif name in ("conservative", "fearful", "age_65", "veteran_institutional"):
        # Risk-averse: fade low-conviction directional calls to HOLD, trim conviction.
        if signal in _DIRECTIONAL and conf < 0.62:
            signal = "hold"
        if name == "fearful" and signal == "buy" and conf < 0.72:
            signal = "hold"
        conf = max(0.05, conf - 0.08)
    elif name == "contrarian":
        # Fade the evidence-implied direction.
        if signal == "buy":
            signal = "sell"
        elif signal == "sell":
            signal = "buy"
    elif name == "disciplined":
        # Follows the evidence; marginally firmer than baseline.
        conf = min(0.95, conf + 0.03)
    # baseline: unchanged.
    return signal, round(max(0.0, min(1.0, conf)), 2)
