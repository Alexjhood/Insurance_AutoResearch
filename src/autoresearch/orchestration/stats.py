"""The empirical backend scorecard (design §4.7 Layer 3).

The curated registry says what *may* be spawned; this module says how each entry
has actually behaved on this repo's workload. It reads collected delegation
reports across every orchestration manifest under a root and aggregates them per
backend, so `list-backends` can show measured performance next to the human
judgement in ``backends.toml`` — and Alex can see when the two disagree.

Two rules keep the numbers honest:

* A delegation with no collected report contributes **nothing**. It is counted as
  skipped, never as a zero.
* Cost is ``None`` when no delegation of that backend reported a provider cost.
  A campaign whose tool does not expose spend is unmeasured, not free.

The scorecard never adds or removes a spawnable backend: :mod:`backends` remains
the only source of those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.orchestration import manifest as manifest_mod
from autoresearch.orchestration.manifest import Orchestration
from autoresearch.orchestration.report import DISTRESS_FLAGS
from autoresearch.utils.io import read_json


_PROMOTE = "promote"


def _orchestrations_root(root: Path | None) -> Path:
    """Resolve the campaign root at call time (tests point this at a fixture)."""

    return root if root is not None else manifest_mod.ORCHESTRATIONS_DIR


def _load_orchestrations(root: Path) -> list[tuple[Orchestration, Path]]:
    if not root.exists():
        return []
    found: list[tuple[Orchestration, Path]] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = directory / "orchestration.json"
        if not manifest.exists():
            continue
        try:
            found.append((Orchestration.from_dict(read_json(manifest)), directory))
        except (ValueError, KeyError):
            continue
    return found


class _Accumulator:
    """Mutable per-backend tallies; converted to a frozen summary at the end."""

    def __init__(self) -> None:
        self.campaigns: set[str] = set()
        self.delegations = 0
        self.cycles = 0
        self.promotions = 0
        self.promoted_lifts: list[float] = []
        self.delegations_in_distress = 0
        self.distress_counts: dict[str, int] = {flag: 0 for flag in DISTRESS_FLAGS}
        self.repair_requests = 0
        self.cost_usd = 0.0
        self.delegations_with_cost = 0

    def summary(self) -> dict[str, Any]:
        cost_known = self.delegations_with_cost > 0
        return {
            "campaigns": len(self.campaigns),
            "delegations": self.delegations,
            "cycles": self.cycles,
            "promotions": self.promotions,
            "promotion_rate": _ratio(self.promotions, self.cycles),
            "distress_rate": _ratio(self.delegations_in_distress, self.delegations),
            "distress_rate_by_flag": {
                flag: _ratio(self.distress_counts[flag], self.delegations)
                for flag in DISTRESS_FLAGS
            },
            "repair_attempts_per_cycle": _ratio(self.repair_requests, self.cycles),
            "mean_promoted_gini_lift": (
                round(sum(self.promoted_lifts) / len(self.promoted_lifts), 6)
                if self.promoted_lifts
                else None
            ),
            "cost_usd": (round(self.cost_usd, 6) if cost_known else None),
            "delegations_with_cost": self.delegations_with_cost,
            "cost_per_cycle_usd": (
                round(self.cost_usd / self.cycles, 6) if cost_known and self.cycles else None
            ),
            "cost_per_promotion_usd": (
                round(self.cost_usd / self.promotions, 6)
                if cost_known and self.promotions
                else None
            ),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    """A rate over zero observations is unknown, not zero."""

    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def collect_backend_stats(root: Path | None = None) -> dict[str, Any]:
    """Aggregate every collected delegation report under *root*, keyed by backend."""

    base = _orchestrations_root(root)
    accumulators: dict[str, _Accumulator] = {}
    campaigns = 0
    skipped_missing_report = 0

    for orch, directory in _load_orchestrations(base):
        campaigns += 1
        for delegation in orch.delegations:
            report_file = directory / "reports" / f"{delegation.delegation_id}.json"
            if not report_file.exists():
                skipped_missing_report += 1
                continue
            try:
                report = read_json(report_file)
            except ValueError:
                skipped_missing_report += 1
                continue
            accumulator = accumulators.setdefault(delegation.backend, _Accumulator())
            _absorb(accumulator, orch.orchestration_id, report)

    return {
        "campaigns": campaigns,
        "backends": {name: acc.summary() for name, acc in sorted(accumulators.items())},
        "skipped_missing_report": skipped_missing_report,
    }


def _absorb(acc: _Accumulator, orchestration_id: str, report: dict[str, Any]) -> None:
    acc.campaigns.add(orchestration_id)
    acc.delegations += 1
    acc.cycles += int((report.get("cycles") or {}).get("used") or 0)

    for row in report.get("experiments") or ():
        if row.get("decision") != _PROMOTE:
            continue
        acc.promotions += 1
        if row.get("lift_vs_champion") is not None:
            acc.promoted_lifts.append(float(row["lift_vs_champion"]))

    active = list((report.get("distress") or {}).get("active") or ())
    if active:
        acc.delegations_in_distress += 1
    for flag in active:
        if flag in acc.distress_counts:
            acc.distress_counts[flag] += 1

    acc.repair_requests += int((report.get("repairs") or {}).get("requests") or 0)

    usage = ((report.get("cost") or {}).get("llm_usage") or {})
    if usage.get("cost_usd") is not None:
        acc.cost_usd += float(usage["cost_usd"])
        acc.delegations_with_cost += 1


def scorecard(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Per-backend stats in the shape ``format_backend_table`` appends."""

    try:
        return collect_backend_stats(root)["backends"]
    except OSError:
        return {}


def format_backend_stats(stats: dict[str, Any]) -> str:
    """Render the cross-campaign scorecard for `orchestrate backend-stats`."""

    lines = [
        f"Backend scorecard across {stats['campaigns']} campaign(s):",
        "",
    ]
    if not stats["backends"]:
        lines.append("No collected delegation reports yet — nothing to score.")
        lines.append("")
    for name, entry in stats["backends"].items():
        lines.append(f"- {name}")
        lines.append(
            f"    delegations={entry['delegations']}  cycles={entry['cycles']}  "
            f"campaigns={entry['campaigns']}"
        )
        lines.append(
            f"    promotions={entry['promotions']}  "
            f"promotion_rate={_fmt(entry['promotion_rate'])}  "
            f"mean_promoted_gini_lift={_fmt(entry['mean_promoted_gini_lift'])}"
        )
        lines.append(
            f"    distress_rate={_fmt(entry['distress_rate'])}  "
            f"repair_attempts_per_cycle={_fmt(entry['repair_attempts_per_cycle'])}"
        )
        flags = {
            flag: rate
            for flag, rate in entry["distress_rate_by_flag"].items()
            if rate
        }
        if flags:
            lines.append(
                "    distress by flag: "
                + ", ".join(f"{flag} {_fmt(rate)}" for flag, rate in flags.items())
            )
        lines.append(
            f"    cost_usd={_fmt_cost(entry['cost_usd'])}  "
            f"per_cycle={_fmt_cost(entry['cost_per_cycle_usd'])}  "
            f"per_promotion={_fmt_cost(entry['cost_per_promotion_usd'])}"
        )
        lines.append("")
    if stats["skipped_missing_report"]:
        lines.append(
            f"{stats['skipped_missing_report']} delegation(s) had no collected report "
            "and were excluded. Run `autoresearch orchestrate collect` first."
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _fmt_cost(value: Any) -> str:
    if value is None:
        return "not reported"
    return f"${float(value):.4f}"
