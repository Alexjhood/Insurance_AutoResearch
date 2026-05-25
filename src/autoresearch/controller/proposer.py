"""LLM proposal providers with a deterministic rotating mock fallback."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
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
        "For autonomous proposals, every non-global_mean experiment must be backed by a "
        "run-local model script. In file-handoff mode, write that script beside the JSON "
        "and set experiment_config.model.script_path to its filename. Do not rely on "
        "pre-existing src/autoresearch/models implementations; if you choose GLM, GBM, "
        "or another method, write the modelling logic into the run-local script.\n"
        "Do not request milestone_holdout access or metric changes.\n\n"
        "Exploration philosophy — follow this carefully:\n"
        "- Every run begins from the `global_mean` no-model baseline (a flat exposure-weighted\n"
        "  burning rate). Develop relative to it through many small, well-motivated steps.\n"
        "- Strongly prefer cheap, interpretable changes first: a feature transformation, an\n"
        "  interaction term, a one- or few-feature GLM, a modest hyperparameter shift on the\n"
        "  current champion.\n"
        "- Reach for higher-capacity models (GBM, deep trees, ensembles) only after simpler\n"
        "  paths have been explored. A GBM in the first few cycles is almost always premature.\n"
        "- One change at a time. The change_summary should read as 'X relative to the champion'.\n"
        "- Across cycles, prioritise breadth — try several different cheap ideas before\n"
        "  iterating deeply on any one.\n\n"
        "Fixed constraints:\n"
        "- The claim cap is **100,000** and is applied identically to training and testing.\n"
        "  Always set preprocessing.claim_capping_enabled=true and\n"
        "  preprocessing.claim_cap_threshold=100000. No other cap value is allowed.\n\n"
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
        proposal_count = int(context.get("proposal_count", len(context.get("recent_proposals", []))))
        entry = _MOCK_POOL[proposal_count % len(_MOCK_POOL)]
        champion = context["official_champion"]
        parent_id = champion["champion_id"]
        parent_branch = champion["branch_id"]
        idx = proposal_count % len(_MOCK_POOL)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"mock_{stamp}_{idx}_{entry['model_family']}"
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
            "model_script_source": _mock_model_script_source(entry["model_family"], entry["target_strategy"]),
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


class FileHandoffProposer:
    """Use a handoff proposal file when present, otherwise fall back locally."""

    def __init__(self, path: Path) -> None:
        self.file_proposer = FileProposer(path)
        self.fallback = MockProposer()

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        try:
            return self.file_proposer.propose(context)
        except FileNotFoundError:
            return self.fallback.propose(context)
        except ValueError as exc:
            if "contains no JSON proposals" in str(exc):
                return self.fallback.propose(context)
            raise


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
    if provider == "file":
        return FileProposer(config.llm_proposal_file)
    if provider == "file_handoff":
        return FileHandoffProposer(config.llm_proposal_file)
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


def _mock_model_script_source(model_family: str, target_strategy: str) -> str:
    """Return a small self-contained script for mock autonomous proposals."""

    return '''\
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RECORD_ID = "record_id"
EXPOSURE = "exposure_term_a"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
CLAIM_COST = "claim_cost_capped_active"
RAW_CLAIM_COST = "claim_cost_observed_k"
SPLIT = "split"
NON_FEATURE = {RECORD_ID, CLAIM_COUNT, CLAIM_EVENTS, CLAIM_COST, RAW_CLAIM_COST, SPLIT}


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hp):
    features = _feature_columns(train, feature_inclusions, feature_exclusions)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("glm", TweedieRegressor(power=float(hp.get("power", 1.5)), alpha=float(hp.get("alpha", 1.0)), link="log", max_iter=500)),
    ])
    y = train[CLAIM_COST].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-9)
    model.fit(train[features], y, glm__sample_weight=train[EXPOSURE].astype(float))
    pred_pp = np.clip(model.predict(score[features]), 0.0, None)
    return pred_pp * score[EXPOSURE].astype(float).to_numpy(), {
        "model_family": "scripted_tweedie_glm",
        "target_strategy": "direct_pure_premium",
        "feature_columns": features,
        "alpha": float(hp.get("alpha", 1.0)),
        "power": float(hp.get("power", 1.5)),
    }


def _feature_columns(frame, feature_inclusions, feature_exclusions):
    base = [c for c in frame.columns if c not in NON_FEATURE]
    if feature_inclusions:
        base = [c for c in base if c in set(feature_inclusions)]
    if feature_exclusions:
        base = [c for c in base if c not in set(feature_exclusions)]
    if not base:
        raise ValueError("At least one feature column is required")
    return base
'''
