"""Render the persona-prompting comparison charts from a summary dict.

Produces (into ``Presentation/assets``):
  persona_signal_distribution.png  -- stacked BUY/SELL/HOLD share per persona
  persona_pnl.png                  -- per-bar P&L vs baseline with 95% bootstrap CI
  persona_reasoning_style.png      -- CoT word count vs Flesch Reading Ease
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_GROUP_COLOR = {"baseline": "#444444", "disposition": "#0b6cff", "demographic": "#e8710a"}


def _order(personas: dict) -> list:
    grp = {"baseline": 0, "disposition": 1, "demographic": 2}
    return sorted(personas, key=lambda n: (grp.get(personas[n]["group"], 9), n))


def render_signal_distribution(rep: dict, out: Path) -> None:
    personas = rep["personas"]
    names = _order(personas)
    buy = [personas[n]["signal_dist"]["buy"] for n in names]
    sell = [personas[n]["signal_dist"]["sell"] for n in names]
    hold = [personas[n]["signal_dist"]["hold"] for n in names]
    tot = [max(1, b + s + h) for b, s, h in zip(buy, sell, hold)]
    buy = [100 * b / t for b, t in zip(buy, tot)]
    sell = [100 * s / t for s, t in zip(sell, tot)]
    hold = [100 * h / t for h, t in zip(hold, tot)]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(names))
    ax.bar(x, buy, label="BUY", color="#2ca02c")
    ax.bar(x, hold, bottom=buy, label="HOLD", color="#b0b0b0")
    ax.bar(x, sell, bottom=[b + h for b, h in zip(buy, hold)], label="SELL", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("share of bars (%)")
    ax.set_title("Signal distribution per persona")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "persona_signal_distribution.png", dpi=140)
    plt.close(fig)


def render_pnl(rep: dict, out: Path) -> None:
    personas = rep["personas"]
    names = _order(personas)
    vals, los, his, colors = [], [], [], []
    for n in names:
        vb = personas[n].get("vs_baseline")
        if n == "baseline" or not vb:
            vals.append(0.0); los.append(0.0); his.append(0.0)
        else:
            vals.append(vb["diff_mean"]); los.append(vb["diff_mean"] - vb["ci_lo"])
            his.append(vb["ci_hi"] - vb["diff_mean"])
        colors.append(_GROUP_COLOR.get(personas[n]["group"], "#888"))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(names))
    ax.bar(x, vals, color=colors, yerr=[los, his], capsize=4)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("mean per-bar P&L vs baseline (%)")
    ax.set_title("Persona P&L vs no-persona baseline (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out / "persona_pnl.png", dpi=140)
    plt.close(fig)


def render_reasoning_style(rep: dict, out: Path) -> None:
    personas = rep["personas"]
    names = _order(personas)
    fig, ax = plt.subplots(figsize=(9, 6))
    for n in names:
        e = personas[n]
        ax.scatter(e["reasoning"]["word_count_mean"], e["reasoning"]["flesch_ease_mean"],
                   s=90, color=_GROUP_COLOR.get(e["group"], "#888"),
                   edgecolor="white", zorder=3)
        ax.annotate(n, (e["reasoning"]["word_count_mean"], e["reasoning"]["flesch_ease_mean"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean CoT word count")
    ax.set_ylabel("mean Flesch Reading Ease (higher = easier)")
    ax.set_title("Reasoning style per persona")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g)
               for g, c in _GROUP_COLOR.items()]
    ax.legend(handles=handles, title="group")
    fig.tight_layout()
    fig.savefig(out / "persona_reasoning_style.png", dpi=140)
    plt.close(fig)


def render_all(rep: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    render_signal_distribution(rep, out)
    render_pnl(rep, out)
    render_reasoning_style(rep, out)


if __name__ == "__main__":
    import json
    import sys

    summary = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "project-context" / "persona_eval_summary.json")
    assets = Path(__file__).resolve().parents[2] / "Presentation" / "assets"
    render_all(json.loads(summary.read_text()), assets)
    print(f"charts written to {assets}")
