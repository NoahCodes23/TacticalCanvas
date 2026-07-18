"""Translate tactical indicators into coach language via an OpenAI-compatible LLM.

Provider-agnostic. OpenAI is the default, but any OpenAI-compatible
chat-completions endpoint works by pointing env vars at it — "different models"
is a config swap, not a code change:

    TC_COACH_BASE_URL   default https://api.openai.com
    TC_COACH_MODEL      default gpt-4.1-mini
    TC_COACH_API_KEY    falls back to OPENAI_API_KEY, then OPENROUTER_API_KEY

When ``TC_COACH_BASE_URL`` points at OpenRouter, the request keeps OpenRouter's
zero-data-retention flag and attribution headers, so routing tactical data
through it stays private. For any other provider those extras are omitted.

The model is fed a small, pre-shaped ``briefing`` per frame (see
server/analytics/briefing.py), whose ``facts`` list is the *only* set of numbers
it is allowed to say out loud. That whitelist is what keeps the LLM from
inventing stats — this module never computes tactics, it only phrases them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.openai.com"
DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You are the assistant coach for a football (soccer) team.
Translate the supplied tracking-model output into clear, immediately actionable
sideline advice. Treat every numeric output as an experimental heuristic, not a
fact or a validated prediction. Use the five meaningfully spaced snapshots and
recent match events only to understand the short lead-in to the current moment.
Never invent events, player identities, or measurements.

Each snapshot carries a "briefing" with a "facts" list. The ONLY numbers you may
say out loud are ones that appear verbatim in a "facts" entry; if a figure isn't
in "facts", describe it qualitatively instead. Focus on the newest snapshot; the
earlier ones are context for how the moment developed.

The briefing is written entirely from YOUR team's point of view ("your side" vs
"the opposition"), set by "coachingTeam". Coach only that side. When the
opposition has the ball, make it about defending: shape, pressing triggers,
covering the danger, and winning the ball back. When your side has the ball,
make it about keeping and advancing it. Never give the opponent instructions.

Sound like a real coach speaking during a stoppage: urgent, calm, and specific.
Return only 4 to 6 short sentences and stay under 100 words. Use no headings,
preamble, bullet list, or metric dump. Briefly read the moment, give direct
player instructions, name the best immediate action and why, then finish with
the main pressure or offside warning. Refer to players by team and shirt number
only when the supplied data supports it. Use ordinary football language."""


def compose_system_prompt(
    style_prompt: str | None = None, persona_name: str | None = None
) -> str:
    """The base coaching prompt, optionally specialised by a chosen persona.

    The persona only steers emphasis, risk appetite, and tone — it is appended
    *after* the base prompt so every safety rule (especially the facts-only
    numbers whitelist) is still in force and re-asserted here."""
    if not style_prompt:
        return SYSTEM_PROMPT
    identity = f'You are "{persona_name}", ' if persona_name else "You are "
    return (
        SYSTEM_PROMPT
        + "\n\nCOACHING PHILOSOPHY\n"
        + identity
        + "a coach with a distinct footballing philosophy. "
        + style_prompt.strip()
        + " Let this philosophy decide which actions you emphasise, how much risk "
        "you accept, and how urgent you sound — but it never lets you invent "
        "numbers. The rule that you may only state a figure that appears verbatim "
        'in a "facts" entry still holds absolutely.'
    )


class CoachServiceError(RuntimeError):
    """A safe-to-display failure from the external coaching service."""


def default_base_url() -> str:
    return os.environ.get("TC_COACH_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")


def _is_openrouter(base_url: str) -> bool:
    return "openrouter" in base_url


def resolve_api_key() -> str:
    """First configured coach key, or '' if none is set."""
    for name in ("TC_COACH_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if value := os.environ.get(name, "").strip():
            return value
    return ""


def resolve_model() -> str:
    """Model to use: explicit override, OpenRouter's own var when routing there,
    else the default."""
    if model := os.environ.get("TC_COACH_MODEL", "").strip():
        return model
    if _is_openrouter(default_base_url()) and (model := os.environ.get("OPENROUTER_MODEL", "").strip()):
        return model
    return DEFAULT_MODEL


def build_messages(
    frames: list[dict[str, Any]],
    match_label: str | None,
    recent_events: list[dict[str, Any]] | None = None,
    *,
    style_prompt: str | None = None,
    persona_name: str | None = None,
) -> list[dict[str, str]]:
    payload = {
        "match": match_label or "Unknown match",
        "window": {
            "frameCount": len(frames),
            "order": "oldest_to_newest",
            "sourceTrackingRateHz": 25,
            "snapshotSpacingMs": 400,
        },
        "recentEvents": recent_events or [],
        "frames": frames,
    }
    return [
        {"role": "system", "content": compose_system_prompt(style_prompt, persona_name)},
        {
            "role": "user",
            "content": "Analyze this paused tactical window and give coach advice.\n\n"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        },
    ]


def build_request_body(
    frames: list[dict[str, Any]],
    match_label: str | None,
    model: str,
    recent_events: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    *,
    style_prompt: str | None = None,
    persona_name: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": build_messages(
            frames,
            match_label,
            recent_events,
            style_prompt=style_prompt,
            persona_name=persona_name,
        ),
        "max_tokens": 250,
        "stream": False,
    }
    # OpenRouter-only: force providers that retain neither prompt nor completion.
    if _is_openrouter(base_url if base_url is not None else default_base_url()):
        body["provider"] = {"zdr": True}
    return body


async def request_coach_advice(
    frames: list[dict[str, Any]],
    match_label: str | None,
    *,
    api_key: str,
    model: str | None = None,
    recent_events: list[dict[str, Any]] | None = None,
    style_prompt: str | None = None,
    persona_name: str | None = None,
) -> dict[str, str]:
    base_url = default_base_url()
    chosen_model = (model or resolve_model()).strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if _is_openrouter(base_url):
        headers["HTTP-Referer"] = os.environ.get("OPENROUTER_SITE_URL", "http://localhost:8000")
        headers["X-Title"] = os.environ.get("OPENROUTER_APP_NAME", "TacticalCanvas")
    body = build_request_body(
        frames,
        match_label,
        chosen_model,
        recent_events,
        base_url,
        style_prompt=style_prompt,
        persona_name=persona_name,
    )

    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/v1/chat/completions", headers=headers, json=body
            )
    except httpx.TimeoutException as error:
        raise CoachServiceError("The coach model timed out. Please try again.") from error
    except httpx.HTTPError as error:
        raise CoachServiceError(
            "Could not reach the coach model provider. Check the network and try again."
        ) from error

    if response.is_error:
        message = "The coach model provider rejected the request."
        try:
            detail = response.json().get("error", {}).get("message")
            if isinstance(detail, str) and detail.strip():
                message = detail.strip()
        except (ValueError, AttributeError):
            pass
        if response.status_code in (401, 403):
            message = "Coach model authentication failed. Check your API key in .env."
        raise CoachServiceError(message[:300])

    try:
        data = response.json()
        advice = data["choices"][0]["message"]["content"]
        returned_model = data.get("model") or chosen_model
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise CoachServiceError("The coach model returned an unexpected response.") from error

    if not isinstance(advice, str) or not advice.strip():
        raise CoachServiceError("The coach model returned empty advice.")
    return {"advice": advice.strip(), "model": str(returned_model)}
