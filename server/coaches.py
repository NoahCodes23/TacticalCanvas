"""Coach personas — selectable playstyles that reshape the LLM coach's advice.

Each persona is a *philosophy*, not a different model or a different set of
numbers. Picking one changes which actions the coach emphasises, how much risk
it accepts, and how urgent it sounds — it never changes the tactical model or
lets the LLM invent stats. The persona's ``style_prompt`` is appended to the
shared SYSTEM_PROMPT (see server/coach.py); the "only quote numbers from the
facts whitelist" rule still holds absolutely on top of it.

The accents are deliberately kept clear of the team colours (home #38bdf8,
away #fb7185) so a persona swatch never reads as a side.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Coach:
    id: str
    name: str
    emoji: str
    tagline: str      # one glance: what this coach is about
    accent: str       # hex, drives the carousel UI
    style_prompt: str # appended to the LLM system prompt

    def to_public(self) -> dict[str, str]:
        """The fields the dashboard needs to render the carousel. The
        style_prompt stays server-side — the browser never needs it."""
        return {
            "id": self.id,
            "name": self.name,
            "emoji": self.emoji,
            "tagline": self.tagline,
            "accent": self.accent,
        }


# Ordered — this is the carousel order. The default (balanced) leads.
COACHES: list[Coach] = [
    Coach(
        id="balanced",
        name="Balanced",
        emoji="⚖️",
        tagline="Pragmatic · pick the highest-percentage action",
        accent="#94a3b8",
        style_prompt=(
            "Coach a pragmatic, balanced game. Weigh attacking and defending on "
            "their merits for this exact moment: press when it is genuinely on, "
            "hold shape when it is not, and always favour the highest-percentage "
            "action over the boldest or the safest one."
        ),
    ),
    Coach(
        id="aggressive",
        name="High Press",
        emoji="⚡",
        tagline="Front-foot · hunt the ball high, attack fast",
        accent="#f97316",
        style_prompt=(
            "Coach an aggressive, front-foot game. Hunt the ball high and press "
            "in numbers the instant possession is lost. Force play forward at "
            "speed and back the ambitious forward pass, the runner in behind, and "
            "the overload. Accept real risk to create chances — a turnover high up "
            "is a price worth paying for the chances this pressure creates."
        ),
    ),
    Coach(
        id="defensive",
        name="Low Block",
        emoji="🛡️",
        tagline="Defence-first · stay compact, protect the goal",
        accent="#34d399",
        style_prompt=(
            "Coach a disciplined, defence-first game. Protect the goal above all "
            "else: stay compact between the lines, deny the space in behind, keep "
            "the back line together, and delay the attacker rather than diving in. "
            "Only commit players forward when the ball is genuinely secure; a "
            "conceded chance costs more than a missed attacking opportunity."
        ),
    ),
    Coach(
        id="possession",
        name="Possession",
        emoji="🎯",
        tagline="Patient · keep the ball, move them, wait for the gap",
        accent="#a78bfa",
        style_prompt=(
            "Coach a patient possession game. Keep the ball, build from the back, "
            "and move the opponent with short, secure passes until a gap opens. "
            "Prize control and tempo over the direct ball: a sideways pass that "
            "retains possession beats a hopeful one that risks losing it. Make "
            "them chase; strike only when the picture is clear."
        ),
    ),
    Coach(
        id="counter",
        name="Counter",
        emoji="🏹",
        tagline="Transition · sit in, then break at pace",
        accent="#facc15",
        style_prompt=(
            "Coach a counter-attacking game. Sit in a solid, organised block and "
            "invite pressure, then break at pace the moment the ball is won — get "
            "it forward into the space behind before the opponent recovers shape. "
            "Prioritise the fast, direct transition and the runner in behind over "
            "slow build-up; the danger is in the seconds right after the turnover."
        ),
    ),
]

DEFAULT_COACH_ID = COACHES[0].id

_BY_ID: dict[str, Coach] = {c.id: c for c in COACHES}


def get_coach(coach_id: str) -> Coach | None:
    return _BY_ID.get(coach_id)


def list_coaches() -> list[dict[str, str]]:
    """Public carousel data, in display order."""
    return [c.to_public() for c in COACHES]
