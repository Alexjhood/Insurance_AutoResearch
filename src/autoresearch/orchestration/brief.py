"""The delegation API: what the orchestrator supplies per spawn.

A brief is the *only* channel through which an orchestrator directs a sub-agent's
science. It is rendered into the child run's handoff as an "Orchestration brief"
block, where the standard workflow forces the agent to read it.

Validation is fail-loud: a malformed brief must stop the spawn before a child
process (and its cost) exists, not after.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.utils.io import read_json


@dataclass(frozen=True)
class SeedChampion:
    """Initialise the child run's champion from a prior experiment.

    Saves a follow-up delegation from spending cycles re-beating ``global_mean``.
    ``from_run`` is ``"<track>/<run-id>"``.
    """

    from_run: str
    experiment_id: str

    def __post_init__(self) -> None:
        if "/" not in self.from_run:
            raise ValueError(
                f"seed_champion.from_run must be '<track>/<run-id>'; got {self.from_run!r}"
            )
        if not self.experiment_id:
            raise ValueError("seed_champion.experiment_id must not be empty")

    @property
    def track(self) -> str:
        return self.from_run.split("/", 1)[0]

    @property
    def run_id(self) -> str:
        return self.from_run.split("/", 1)[1]

    def to_dict(self) -> dict[str, Any]:
        return {"from_run": self.from_run, "experiment_id": self.experiment_id}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SeedChampion":
        return cls(
            from_run=str(raw["from_run"]),
            experiment_id=str(raw["experiment_id"]),
        )


@dataclass(frozen=True)
class Brief:
    """A validated delegation brief.

    ``direction`` and ``cycle_budget`` are required; everything else is optional
    context the orchestrator chooses to route into the child.
    """

    direction: str
    cycle_budget: int
    starting_knowledge: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    success_criteria: str | None = None
    seed_champion: SeedChampion | None = None

    def __post_init__(self) -> None:
        if not self.direction.strip():
            raise ValueError("brief.direction must be a non-empty string")
        if self.cycle_budget <= 0:
            raise ValueError(
                f"brief.cycle_budget must be a positive integer, got {self.cycle_budget}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "direction": self.direction,
            "cycle_budget": self.cycle_budget,
            "starting_knowledge": list(self.starting_knowledge),
            "constraints": list(self.constraints),
            "success_criteria": self.success_criteria,
        }
        if self.seed_champion is not None:
            payload["seed_champion"] = self.seed_champion.to_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Brief":
        return validate_brief(raw)


_KNOWN_FIELDS = frozenset(
    {
        "direction",
        "cycle_budget",
        "starting_knowledge",
        "constraints",
        "success_criteria",
        "seed_champion",
    }
)
_REQUIRED_FIELDS = ("direction", "cycle_budget")


def validate_brief(raw: dict[str, Any]) -> Brief:
    """Validate a raw brief mapping and return the frozen record.

    Unknown fields are rejected rather than ignored: a typo'd ``constraint``
    would otherwise silently drop the constraint the orchestrator relied on.
    """

    if not isinstance(raw, dict):
        raise ValueError(f"brief must be a JSON object, got {type(raw).__name__}")

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"brief is missing required field(s): {', '.join(missing)}")

    # Underscore-prefixed keys are framework metadata (e.g. the `_source_path` the
    # spawner stamps on its archived copy), not brief content. A typo'd field name
    # never starts with an underscore, so this keeps the typo guard below intact.
    unknown = sorted(k for k in raw if not k.startswith("_"))
    unknown = sorted(set(unknown) - _KNOWN_FIELDS)
    if unknown:
        raise ValueError(
            f"brief has unknown field(s): {', '.join(unknown)}. "
            f"Known fields: {', '.join(sorted(_KNOWN_FIELDS))}"
        )

    if not isinstance(raw["direction"], str):
        raise ValueError("brief.direction must be a string")
    if isinstance(raw["cycle_budget"], bool) or not isinstance(raw["cycle_budget"], int):
        raise ValueError("brief.cycle_budget must be an integer")

    seed = raw.get("seed_champion")
    seed_champion = SeedChampion.from_dict(seed) if seed else None

    return Brief(
        direction=raw["direction"],
        cycle_budget=raw["cycle_budget"],
        starting_knowledge=tuple(_string_list(raw, "starting_knowledge")),
        constraints=tuple(_string_list(raw, "constraints")),
        success_criteria=(
            str(raw["success_criteria"]) if raw.get("success_criteria") else None
        ),
        seed_champion=seed_champion,
    )


def load_brief(path: Path) -> Brief:
    """Read and validate a brief JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"No brief at {path}")
    return validate_brief(read_json(path))


def _string_list(raw: dict[str, Any], field: str) -> list[str]:
    value = raw.get(field) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"brief.{field} must be a list of strings")
    return [v for v in value if v.strip()]


def render_brief_block(brief: Brief, *, cycle_budget: int, delegation_id: str) -> list[str]:
    """Render the "Orchestration brief" handoff block for a child run.

    Returned as markdown lines so the handoff renderer can splice them in beside
    the Active dataset block, which they sit at parity with.
    """

    lines = [
        "## Orchestration brief",
        "",
        f"You are delegation `{delegation_id}` under an orchestrator. This brief is "
        "**binding**, on par with the Active dataset block.",
        "",
        f"- **Direction**: {brief.direction.strip()}",
        f"- **Cycle budget**: {cycle_budget} (spend it adaptively within the direction)",
    ]
    if brief.success_criteria:
        lines.append(f"- **Success criteria**: {brief.success_criteria.strip()}")
    if brief.starting_knowledge:
        lines.append("- **Starting knowledge** (from the orchestrator; treat as established):")
        lines.extend(f"  - {item}" for item in brief.starting_knowledge)
    if brief.constraints:
        lines.append("- **Constraints** (binding; a stop-condition here ends your run early):")
        lines.extend(f"  - {item}" for item in brief.constraints)
    if brief.seed_champion is not None:
        lines.append(
            f"- **Seeded champion**: this run starts from experiment "
            f"`{brief.seed_champion.experiment_id}` rather than the flat baseline."
        )
    lines.extend(
        [
            "",
            "When your budget is exhausted (or a constraint's stop-condition fires), "
            "finish with `orchestrate finish-delegation --summary \"...\"` and stop.",
            "",
        ]
    )
    return lines
