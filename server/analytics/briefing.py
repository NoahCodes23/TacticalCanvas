"""Reduce a full tactical-analysis snapshot to a compact coach briefing.

This is the seam between the analytics (server/analytics/experimental.py) and the
language model that speaks to the coach. Its whole job is *selection and framing*:
it turns the large, exhaustive ``analyze()`` dict into a handful of ranked,
salient facts phrased for a defending coach.

Design rule that makes the LLM trustworthy: **every number the model is allowed
to say already appears here, pre-computed, as a string.** The reducer owns the
arithmetic; the LLM only narrates. Nothing downstream should recompute a stat.

The output is deliberately small and stable:

* ``situation`` — one line: who has the ball, where, under how much pressure.
* ``threat``    — the single most dangerous thing the attack can do right now.
* ``weaknesses``— the coaching team's shape problems, worst first.
* ``facts``     — short bullet strings; the only numbers the LLM may quote.
* ``narrative`` — a deterministic prose fallback, so the demo speaks even with
                  no API key and no network.

``build_briefing`` is pure and dependency-free; it never calls a model.
"""

from __future__ import annotations

from typing import Any


def _channel_pressure(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Channels where the attack outnumbers the defence, worst (for us) first.

    ``channelOverloads`` is phrased from the possessing (attacking) team's view:
    ``overload = attackers - defenders``. A positive overload is a channel where
    the *defending* coach is short a body, so we surface those first.
    """
    overloads = context.get("channelOverloads", []) or []
    exposed = [c for c in overloads if c.get("overload", 0) > 0]
    exposed.sort(key=lambda c: c["overload"], reverse=True)
    return exposed


def _weaknesses(
    context: dict[str, Any],
    team_metrics: dict[str, Any],
    pitch_length: float,
) -> list[str]:
    """Ordered shape problems for the coaching team, most actionable first."""
    problems: list[str] = []

    for channel in _channel_pressure(context):
        problems.append(
            f"{channel['overload']}-man overload in the {channel['channel']} channel "
            f"({channel['attackers']}v{channel['defenders']})"
        )

    line = team_metrics.get("lineHeightM")
    # A high defensive line up the pitch leaves space in behind. The threshold is
    # deliberately soft — it flags a talking point, it is not a verdict.
    if line is not None and line >= pitch_length * 0.42:
        problems.append(f"high defensive line at {line:.0f} m — space in behind")

    shape = team_metrics.get("shapeScore")
    if shape is not None and shape < 60:
        width = team_metrics.get("teamWidthM")
        depth = team_metrics.get("teamDepthM")
        problems.append(
            f"stretched shape (score {shape:.0f}/100"
            + (f", {width:.0f} m wide × {depth:.0f} m deep" if width and depth else "")
            + ")"
        )

    space = team_metrics.get("avgOpponentSpaceM")
    if space is not None and space >= 6.0:
        problems.append(f"soft marking — {space:.0f} m of space per attacker on average")

    return problems


def build_briefing(
    analysis: dict[str, Any] | None,
    coaching_team: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> dict[str, Any]:
    """Collapse ``analyze()`` output into a compact briefing for ``coaching_team``.

    ``coaching_team`` is the side the coach is trying to fix — normally the team
    *without* the ball, since the demo is about a goal conceded. The reducer is
    still valid when that team happens to have possession; the threat section
    just describes the best option available to whoever is attacking.
    """
    empty = {
        "coachingTeam": coaching_team,
        "situation": "No tracking data for this frame.",
        "threat": None,
        "weaknesses": [],
        "facts": [],
        "narrative": "There's nothing on the board to analyse yet.",
    }
    if not analysis:
        return empty

    context = analysis.get("context") or {}
    passes = analysis.get("passes") or []
    teams = analysis.get("teams") or {}
    if not context:
        return empty

    attacking_team = context.get("possession", "home")
    team_metrics = teams.get(coaching_team, {}) or {}

    phase = context.get("phase", "open play")
    channel = context.get("channel", "central")
    carrier = context.get("ballCarrierNumber")
    pressure = context.get("pressureIndex", 0)
    tempo = context.get("tempoMps", 0.0)

    on_ball = "your side" if attacking_team == coaching_team else "the opposition"
    carrier_txt = f"#{carrier}" if carrier is not None else "the ball-carrier"
    situation = (
        f"{on_ball.capitalize()} in {phase} through the {channel} channel, "
        f"{carrier_txt} on the ball under {'heavy' if pressure >= 66 else 'moderate' if pressure >= 33 else 'light'} "
        f"pressure (index {pressure:.0f}/100), tempo {tempo:.1f} m/s."
    )

    # The threat is the attack's single best option this frame — passes are
    # already sorted by expected value, so the top one is the thing to stop.
    threat: dict[str, Any] | None = None
    if passes:
        best = passes[0]
        threat = {
            "receiverNumber": best.get("receiverNumber"),
            "completionProbability": best.get("completionProbability"),
            "risk": best.get("risk"),
            "score": best.get("score"),
            "to": best.get("to"),
            "why": best.get("explanation", []),
        }

    weaknesses = _weaknesses(context, team_metrics, pitch_length)

    # ``facts`` is the whitelist of numbers the model may quote. Keep it short.
    facts: list[str] = []
    if threat and threat["completionProbability"] is not None:
        pct = round(threat["completionProbability"] * 100)
        reason = threat["why"][0] if threat["why"] else "an open lane"
        facts.append(
            f"Best attacking option: pass to #{threat['receiverNumber']} — "
            f"{pct}% completion, {threat['risk']} risk ({reason})."
        )
    pc = context.get("pitchControlAtBall")
    if pc is not None:
        facts.append(f"Attack controls {round(pc * 100)}% of the space at the ball.")
    if team_metrics.get("lineHeightM") is not None:
        facts.append(f"Your defensive line sits at {team_metrics['lineHeightM']:.0f} m.")
    if team_metrics.get("shapeScore") is not None:
        facts.append(f"Your shape score is {team_metrics['shapeScore']:.0f}/100.")
    for w in weaknesses[:2]:
        facts.append(w[0].upper() + w[1:] + ".")

    # Deterministic prose so the coach still hears something without an LLM.
    lead = weaknesses[0] if weaknesses else "shape is holding, but stay compact"
    threat_line = ""
    if threat:
        threat_line = (
            f" The danger is a ball to #{threat['receiverNumber']} "
            f"({round((threat['completionProbability'] or 0) * 100)}% on)."
        )
    narrative = f"{situation} Main issue: {lead}.{threat_line}"

    return {
        "coachingTeam": coaching_team,
        "attackingTeam": attacking_team,
        "situation": situation,
        "threat": threat,
        "weaknesses": weaknesses,
        "facts": facts,
        "narrative": narrative,
    }
