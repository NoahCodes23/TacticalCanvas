"""OpenRouter-backed translation of tactical indicators into coach language."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "tencent/hy3:free"

SYSTEM_PROMPT = """You are the assistant coach for a football (soccer) team.
Translate the supplied tracking-model output into clear, immediately actionable
sideline advice. Treat every numeric output as an experimental heuristic, not a
fact or a validated prediction. Use the five meaningfully spaced snapshots and
recent match events only to understand the short lead-in to the current moment.
Never invent events, player identities, or measurements.

Sound like a real coach speaking during a stoppage: urgent, calm, and specific.
Return only 4 to 6 short sentences and stay under 100 words. Use no headings,
preamble, bullet list, or metric dump. Briefly read the moment, give direct
player instructions, name the best immediate action and why, then finish with
the main pressure or offside warning. Refer to players by team and shirt number
only when the supplied data supports it. Use ordinary football language."""


class CoachServiceError(RuntimeError):
    """A safe-to-display failure from the external coaching service."""


def build_messages(
    frames: list[dict[str, Any]],
    match_label: str | None,
    recent_events: list[dict[str, Any]] | None = None,
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
        {"role": "system", "content": SYSTEM_PROMPT},
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
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": build_messages(frames, match_label, recent_events),
        "max_completion_tokens": 250,
        "stream": False,
        # Tactical tracking data must only be routed through providers that
        # retain neither the prompt nor the completion.
        "provider": {"zdr": True},
    }


async def request_coach_advice(
    frames: list[dict[str, Any]],
    match_label: str | None,
    *,
    api_key: str,
    model: str | None = None,
    recent_events: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    chosen_model = (model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "TacticalCanvas"),
    }
    body = build_request_body(frames, match_label, chosen_model, recent_events)

    try:
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=body)
    except httpx.TimeoutException as error:
        raise CoachServiceError("OpenRouter timed out. Please try again.") from error
    except httpx.HTTPError as error:
        raise CoachServiceError("Could not reach OpenRouter. Check the network and try again.") from error

    if response.is_error:
        message = "OpenRouter rejected the request."
        try:
            detail = response.json().get("error", {}).get("message")
            if isinstance(detail, str) and detail.strip():
                message = detail.strip()
        except (ValueError, AttributeError):
            pass
        if response.status_code in (401, 403):
            message = "OpenRouter authentication failed. Check OPENROUTER_API_KEY."
        raise CoachServiceError(message[:300])

    try:
        data = response.json()
        advice = data["choices"][0]["message"]["content"]
        returned_model = data.get("model") or chosen_model
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise CoachServiceError("OpenRouter returned an unexpected response.") from error

    if not isinstance(advice, str) or not advice.strip():
        raise CoachServiceError("OpenRouter returned empty coach advice.")
    return {"advice": advice.strip(), "model": str(returned_model)}
