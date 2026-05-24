"""LLM proposal providers with a deterministic local fallback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request
from typing import Any, Protocol


class Proposer(Protocol):
    """Protocol for structured experiment proposal providers."""

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Return raw response text and parsed proposal when available."""


def build_prompt(context: dict[str, Any]) -> str:
    """Create the prompt sent to an API-backed proposer."""

    return (
        "You are proposing one controlled insurance burning-cost modelling experiment.\n"
        "Return JSON only. Do not include Markdown.\n"
        "The deterministic framework will validate, run, evaluate, and compare the proposal.\n"
        "Do not request milestone_holdout access or metric changes.\n\n"
        "Required JSON keys: proposal_id, parent_experiment_id, parent_branch_id, branch_action, "
        "experiment_name, rationale, change_summary, expected_benefit, key_risk, experiment_config.\n\n"
        f"Context:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )


class MockProposer:
    """Deterministic local proposer used when no API provider is configured."""

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        champion = context["official_champion"]
        parent_id = champion["champion_id"]
        parent_branch = champion["branch_id"]
        proposal = {
            "proposal_id": f"mock_alpha_branch_{len(context.get('recent_proposals', [])) + 1}",
            "parent_experiment_id": parent_id,
            "parent_branch_id": parent_branch,
            "branch_action": "new_branch",
            "experiment_name": "llm_mock_regularized_direct_alpha_3",
            "rationale": "Try a slightly stronger ridge penalty while keeping the direct target interpretable.",
            "change_summary": "Direct pure premium strategy with alpha increased from the baseline default to 3.0.",
            "expected_benefit": "May reduce validation volatility from noisy sparse claims.",
            "key_risk": "Higher regularisation may underfit local segment effects.",
            "experiment_config": {
                "experiment_name": "llm_mock_regularized_direct_alpha_3",
                "model_family": "regularized_linear",
                "target_strategy": "direct_pure_premium",
                "parent_experiment_id": parent_id,
                "preprocessing": {
                    "claim_capping_enabled": True,
                    "claim_cap_threshold": 100000,
                },
                "model": {
                    "alpha": 3.0,
                    "feature_exclusions": [],
                },
            },
        }
        text = json.dumps(proposal, indent=2, sort_keys=True)
        return text, proposal


class FileProposer:
    """Read the next proposal from a local JSONL file for manual/mock workflows."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if not self.path.exists():
            raise FileNotFoundError(f"Proposal file does not exist: {self.path}")
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                proposal = json.loads(line)
                return line, proposal
        raise ValueError(f"Proposal file contains no JSON proposals: {self.path}")


class OpenAIProposer:
    """Minimal OpenAI Responses API proposer using stdlib HTTP."""

    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        payload = {
            "model": self.model,
            "input": build_prompt(context),
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = _extract_openai_text(body)
        return text, _parse_json(text)


class AnthropicProposer:
    """Minimal Anthropic Messages API proposer using stdlib HTTP."""

    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": build_prompt(context)}],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = "".join(part.get("text", "") for part in body.get("content", []) if part.get("type") == "text")
        return text, _parse_json(text)


def proposer_from_config(config) -> Proposer:
    """Create the configured proposal provider."""

    provider = config.llm_provider.lower()
    if provider == "mock":
        return MockProposer()
    if provider == "file":
        return FileProposer(config.llm_proposal_file)
    if provider == "file_handoff":
        return FileProposer(config.llm_proposal_file)
    if provider == "openai":
        return OpenAIProposer(config.llm_model, config.llm_temperature)
    if provider in {"anthropic", "claude"}:
        return AnthropicProposer(config.llm_model, config.llm_temperature)
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def _extract_openai_text(body: dict[str, Any]) -> str:
    if "output_text" in body:
        return str(body["output_text"])
    texts: list[str] = []
    for item in body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                texts.append(content.get("text", ""))
    return "".join(texts)


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
