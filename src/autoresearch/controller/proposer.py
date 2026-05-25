"""LLM proposal providers with a deterministic rotating mock fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol
import urllib.error
import urllib.request


class Proposer(Protocol):
    """Protocol for structured experiment proposal providers."""

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Return raw response text and parsed proposal when available."""


def build_prompt(context: dict[str, Any]) -> str:
    """Create the prompt sent to an API-backed proposer."""

    return (
        "You are proposing one controlled insurance burning-cost modelling experiment.\n"
        "Return JSON only matching the proposal schema. Do not include Markdown fences.\n"
        "The deterministic framework will validate, run, evaluate, and compare the proposal.\n"
        "Do not request milestone_holdout access or metric changes.\n\n"
        "Required JSON keys: proposal_id, parent_experiment_id, parent_branch_id, branch_action, "
        "experiment_name, rationale, change_summary, expected_benefit, key_risk, experiment_config.\n\n"
        f"Allowed search space:\n{json.dumps(context.get('allowed_search_space', {}), indent=2)}\n\n"
        f"Recent non-promotion summaries (read before proposing to avoid duplication):\n"
        f"{json.dumps(context.get('recent_proposals', []), indent=2)}\n\n"
        f"Context:\n{json.dumps(context, indent=2, sort_keys=True)}"
    )


# ── Mock proposer (rotating pool) ────────────────────────────────────────────

_MOCK_POOL = [
    {
        "branch_action": "new_branch",
        "model_family": "tweedie_glm",
        "target_strategy": "direct_pure_premium",
        "model_params": {"alpha": 0.3, "power": 1.5},
        "rationale": "Reduce GLM regularisation to capture more local segment effects.",
        "change_summary": "Tweedie GLM (power=1.5) with lower alpha=0.3 vs baseline alpha=1.0.",
        "expected_benefit": "Better fit on lower-risk segments where ridge over-smooths.",
        "key_risk": "May overfit on high-volatility segments.",
    },
    {
        "branch_action": "new_branch",
        "model_family": "tweedie_glm",
        "target_strategy": "direct_pure_premium",
        "model_params": {"alpha": 3.0, "power": 1.7},
        "rationale": "Increase power parameter to upweight high-cost tails.",
        "change_summary": "Tweedie GLM with power=1.7 (more tail-heavy than p=1.5) and alpha=3.0.",
        "expected_benefit": "Better calibration on heavy-tailed claim distributions.",
        "key_risk": "May worsen calibration on low-claim segments.",
    },
    {
        "branch_action": "new_branch",
        "model_family": "frequency_severity_glm",
        "target_strategy": "frequency_severity",
        "model_params": {"freq_alpha": 1.0, "sev_alpha": 0.5},
        "rationale": "Separate frequency and severity modelling for better actuarial interpretability.",
        "change_summary": "Poisson GLM for frequency × Gamma GLM for severity.",
        "expected_benefit": "Captures frequency/severity heterogeneity missed by direct model.",
        "key_risk": "Severity model may be noisy with few large claims.",
    },
    {
        "branch_action": "new_branch",
        "model_family": "tweedie_gbm",
        "target_strategy": "direct_pure_premium",
        "model_params": {"max_iter": 300, "max_depth": 5, "learning_rate": 0.05, "min_samples_leaf": 200},
        "rationale": "Non-linear model to capture interaction effects missed by GLM.",
        "change_summary": "Gradient-boosted Tweedie (Poisson loss) with depth=5, 300 trees.",
        "expected_benefit": "Captures non-linear risk interactions (e.g. age × vehicle type).",
        "key_risk": "May overfit on small segments; GBM interpretability lower than GLM.",
    },
    {
        "branch_action": "new_branch",
        "model_family": "tweedie_gbm",
        "target_strategy": "direct_pure_premium",
        "model_params": {"max_iter": 800, "max_depth": 7, "learning_rate": 0.03, "min_samples_leaf": 500},
        "rationale": "Deeper GBM with more conservative learning rate and leaf size.",
        "change_summary": "Tweedie GBM with 800 trees, depth=7, lr=0.03, min_leaf=500.",
        "expected_benefit": "Higher-capacity model with regularised leaves for stable predictions.",
        "key_risk": "Slower training; may not improve over shallower variant.",
    },
]


class MockProposer:
    """Deterministic rotating mock proposer that cycles through a diverse pool."""

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        n_recent = len(context.get("recent_proposals", []))
        entry = _MOCK_POOL[n_recent % len(_MOCK_POOL)]
        champion = context["official_champion"]
        parent_id = champion["champion_id"]
        parent_branch = champion["branch_id"]
        idx = n_recent % len(_MOCK_POOL)
        name = f"mock_proposal_{idx}_{entry['model_family']}"
        model_cfg: dict[str, Any] = {
            "experiment_name": name,
            "model_family": entry["model_family"],
            "target_strategy": entry["target_strategy"],
            "parent_experiment_id": parent_id,
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": dict(entry["model_params"]),
        }
        proposal = {
            "proposal_id": name,
            "parent_experiment_id": parent_id,
            "parent_branch_id": parent_branch,
            "branch_action": entry["branch_action"],
            "experiment_name": name,
            "rationale": entry["rationale"],
            "change_summary": entry["change_summary"],
            "expected_benefit": entry["expected_benefit"],
            "key_risk": entry["key_risk"],
            "experiment_config": model_cfg,
        }
        text = json.dumps(proposal, indent=2, sort_keys=True)
        return text, proposal


# ── File-based proposer ────────────────────────────────────────────────────────

class FileProposer:
    """Read the next proposal from a local JSONL file."""

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


# ── OpenAI proposer ───────────────────────────────────────────────────────────

class OpenAIProposer:
    """OpenAI Responses API proposer with retries and schema validation."""

    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

        prompt = build_prompt(context)
        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
        }
        text, parsed = self._call_with_retries(api_key, payload)
        return text, parsed

    def _call_with_retries(self, api_key: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                time.sleep(4 ** attempt)
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/responses",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = _extract_openai_text(body)
                parsed = _parse_json(text)
                return text, parsed
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    continue
                raise
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"OpenAI API failed after 3 attempts: {last_exc}")


# ── Anthropic proposer ────────────────────────────────────────────────────────

class AnthropicProposer:
    """Anthropic Messages API proposer with retries and schema validation."""

    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        prompt = build_prompt(context)
        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        text, parsed = self._call_with_retries(api_key, payload)
        return text, parsed

    def _call_with_retries(self, api_key: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                time.sleep(4 ** attempt)
            try:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = "".join(
                    part.get("text", "")
                    for part in body.get("content", [])
                    if part.get("type") == "text"
                )
                parsed = _parse_json(text)
                return text, parsed
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    continue
                raise
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Anthropic API failed after 3 attempts: {last_exc}")


# ── Factory ───────────────────────────────────────────────────────────────────

def proposer_from_config(config) -> Proposer:
    """Create the configured proposal provider."""

    provider = config.llm_provider.lower()
    if provider == "mock":
        return MockProposer()
    if provider in ("file", "file_handoff"):
        return FileProposer(config.llm_proposal_file)
    if provider == "openai":
        return OpenAIProposer(config.llm_model, config.llm_temperature)
    if provider in ("anthropic", "claude"):
        return AnthropicProposer(config.llm_model, config.llm_temperature)
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider!r}")


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
    # Strip markdown code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
