"""Provider-neutral telemetry normalization for agent runs."""

from __future__ import annotations

import json
import re
from typing import Any


_REPAIR_PATTERNS = (
    (
        "repair_requested",
        re.compile(
            r"(?:waiting_for_repair|ExperimentNeedsRepair|repair_request(?:_\d+)?\.json)",
            re.IGNORECASE,
        ),
    ),
    (
        "preflight_failed",
        re.compile(r"(?:PreflightFailed|preflight (?:smoke[- ]?test )?failed)", re.IGNORECASE),
    ),
    (
        "compute_budget_exceeded",
        re.compile(r"(?:ComputeBudgetExceeded|compute budget exceeded)", re.IGNORECASE),
    ),
)
_ATTEMPT_RE = re.compile(r"\battempt\s+(\d+)\b", re.IGNORECASE)


def normalize_usage(raw_usage: Any, *, raw_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map provider-specific token/cost fields into one stable shape.

    Unknown fields remain available in ``raw_usage``/``raw_result`` at the
    persistence layer. Missing values stay ``None`` instead of being guessed.
    """

    usage = raw_usage if isinstance(raw_usage, dict) else {}
    result = raw_result if isinstance(raw_result, dict) else {}

    provider_input_tokens = _first_int(
        usage,
        "input_tokens",
        "prompt_tokens",
        "input",
        "tokens.input",
    )
    cached_input_tokens = _first_int(
        usage,
        "cached_input_tokens",
        "cache_read_input_tokens",
        "prompt_tokens_details.cached_tokens",
        "cache.read",
        "cache_read",
        "tokens.cache.read",
    )
    cache_creation_input_tokens = _first_int(
        usage,
        "cache_creation_input_tokens",
        "cache_write_input_tokens",
        "cache.creation",
        "cache.write",
        "cache_write",
        "tokens.cache.write",
    )
    anthropic_cache_accounting = any(
        key in usage
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens")
    )
    if anthropic_cache_accounting and provider_input_tokens is not None:
        input_tokens = (
            provider_input_tokens
            + (cached_input_tokens or 0)
            + (cache_creation_input_tokens or 0)
        )
        uncached_input_tokens = provider_input_tokens
    else:
        input_tokens = provider_input_tokens
        uncached_input_tokens = None
        if input_tokens is not None:
            uncached_input_tokens = max(input_tokens - (cached_input_tokens or 0), 0)
    output_tokens = _first_int(
        usage,
        "output_tokens",
        "completion_tokens",
        "output",
        "tokens.output",
    )
    reasoning_tokens = _first_int(
        usage,
        "reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
        "completion_tokens_details.reasoning_tokens",
        "reasoning",
        "tokens.reasoning",
    )
    total_tokens = _first_int(usage, "total_tokens", "total", "tokens.total")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    provider_cost_usd = _first_float(
        result,
        "total_cost_usd",
        "cost_usd",
        "cost",
    )
    if provider_cost_usd is None:
        provider_cost_usd = _first_float(
            usage,
            "total_cost_usd",
            "cost_usd",
            "cost",
        )

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "provider_cost_usd": provider_cost_usd,
    }


def normalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return normalized fields for one tool invocation or completion."""

    raw_input = payload.get("input")
    raw_output = payload.get("output")
    if raw_output is None:
        raw_output = payload.get("result")

    status = _first_value(payload, "status", "state.status")
    exit_code = _first_int(payload, "exit_code", "state.exit_code", "state.exitCode")
    is_error = _first_bool(payload, "is_error", "state.is_error", "state.error")
    success = payload.get("success")
    if not isinstance(success, bool):
        if exit_code is not None:
            success = exit_code == 0
        elif is_error is not None:
            success = not is_error
        elif isinstance(status, str) and status.lower() in {
            "completed",
            "success",
            "succeeded",
            "done",
        }:
            success = True
        elif isinstance(status, str) and status.lower() in {
            "failed",
            "error",
            "cancelled",
            "canceled",
        }:
            success = False
        else:
            success = None

    error = _first_value(payload, "error", "state.error", "message")
    if not isinstance(error, str):
        error = None
    explicit_input_bytes = _first_int(payload, "input_bytes")
    explicit_output_bytes = _first_int(payload, "output_bytes")

    return {
        "provider_call_id": _first_value(
            payload,
            "provider_call_id",
            "tool_use_id",
            "id",
            "call_id",
            "state.id",
        ),
        "name": str(payload.get("name") or payload.get("tool") or "tool"),
        "status": status if isinstance(status, str) else None,
        "success": success,
        "duration_ms": _first_float(
            payload,
            "duration_ms",
            "durationMs",
            "state.duration_ms",
            "state.duration",
        ),
        "input_bytes": explicit_input_bytes
        if explicit_input_bytes is not None
        else _json_size(raw_input),
        "output_bytes": explicit_output_bytes
        if explicit_output_bytes is not None
        else _json_size(raw_output),
        "error": error,
    }


def extract_signals(payload: Any) -> list[dict[str, Any]]:
    """Extract stable retry/repair signals from provider tool/result payloads."""

    try:
        text = json.dumps(payload, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        text = str(payload)

    signals: list[dict[str, Any]] = []
    attempt_match = _ATTEMPT_RE.search(text)
    attempt = int(attempt_match.group(1)) if attempt_match else None

    for signal_type, pattern in _REPAIR_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        start = max(match.start() - 120, 0)
        end = min(match.end() + 380, len(text))
        signals.append({
            "signal_type": signal_type,
            "cause": _extract_cause(text, match.end()),
            "attempt": attempt,
            "raw_excerpt": text[start:end],
        })
    return signals


def _extract_cause(text: str, start: int) -> str | None:
    tail = text[start:start + 500]
    tail = re.sub(r"^[\s:;,\-]+", "", tail)
    tail = tail.split("\\n", 1)[0]
    tail = tail.split('","', 1)[0]
    return tail[:300].strip() or None


def _json_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    try:
        return len(json.dumps(value, ensure_ascii=True, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _first_value(data: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _first_int(data: dict[str, Any], *paths: str) -> int | None:
    value = _first_value(data, *paths)
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_float(data: dict[str, Any], *paths: str) -> float | None:
    value = _first_value(data, *paths)
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_bool(data: dict[str, Any], *paths: str) -> bool | None:
    value = _first_value(data, *paths)
    return value if isinstance(value, bool) else None
