from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .alignment import range_signature, signature_distance
from .hypotheses import fold_reference
from .models import LoopClosure, Scan2D, new_id

ProbeStatus = Literal["planned", "executed", "withdrawn", "reassigned", "restored"]

# Outcome of a *planned* probe for one hypothesis branch.
# "eliminated"        the branch predicts a concrete observation that, if contradicted, kills it
# "unaffected"        the branch makes no prediction that this probe could falsify
# "indistinguishable"  the surviving branches predict exactly the same observation
# "insufficient"      no prediction can be built from the data collected so far
ProbeVerdict = Literal["eliminated", "unaffected", "indistinguishable", "insufficient"]


@dataclass(frozen=True)
class ProbeTarget:
    """One observation location inside a grouped follow-up exploration plan.

    ``target_index`` is the scan index the probe expects to collect; before the probe
    runs it points into the future and must never be treated as observed evidence.
    """

    target_index: int
    location_label: str
    expected_x: float
    expected_y: float
    status: ProbeStatus = "planned"
    executed_index: int | None = None
    executed_at: float | None = None
    analysis: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_index": self.target_index,
            "location_label": self.location_label,
            "expected_x": self.expected_x,
            "expected_y": self.expected_y,
            "status": self.status,
            "executed_index": self.executed_index,
            "executed_at": self.executed_at,
            "analysis": list(self.analysis),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbeTarget":
        return cls(
            target_index=int(data["target_index"]),
            location_label=data["location_label"],
            expected_x=float(data["expected_x"]),
            expected_y=float(data["expected_y"]),
            status=data.get("status", "planned"),
            executed_index=None if data.get("executed_index") is None else int(data["executed_index"]),
            executed_at=None if data.get("executed_at") is None else float(data["executed_at"]),
            analysis=tuple(data.get("analysis", [])),
        )

    def replaced(self, **changes: Any) -> "ProbeTarget":
        payload = self.to_dict()
        if "analysis" in changes:
            payload["analysis"] = list(changes.pop("analysis"))
        payload.update(changes)
        return ProbeTarget.from_dict(payload)


@dataclass(frozen=True)
class ProbePlan:
    """A grouped set of follow-up probes compared across the current hypothesis branches."""

    plan_id: str
    run_id: str
    name: str
    created_at: float
    targets: tuple[ProbeTarget, ...]
    hypothesis_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "targets": [target.to_dict() for target in self.targets],
            "hypothesis_ids": list(self.hypothesis_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbePlan":
        return cls(
            plan_id=data["plan_id"],
            run_id=data["run_id"],
            name=data["name"],
            created_at=float(data["created_at"]),
            targets=tuple(ProbeTarget.from_dict(item) for item in data["targets"]),
            hypothesis_ids=tuple(data.get("hypothesis_ids", [])),
        )

    def with_target(self, updated: ProbeTarget) -> "ProbePlan":
        targets = tuple(updated if target.target_index == updated.target_index else target for target in self.targets)
        return ProbePlan(self.plan_id, self.run_id, self.name, self.created_at, targets, self.hypothesis_ids)


def analyze_probe_target(
    scans: list[Scan2D],
    hypotheses: tuple[Any, ...],
    closures: tuple[LoopClosure, ...],
    target_index: int,
) -> tuple[dict[str, Any], ...]:
    """Compare, per hypothesis branch, what an observation at ``target_index`` would rule out.

    Only data already present in ``scans`` is used; the planned observation itself is
    never consulted. When two fold branches predict the same anchor-side scan they are
    reported as mutually indistinguishable, and missing data is reported as
    "insufficient" instead of being guessed.
    """

    closure_by_id = {closure.closure_id: closure for closure in closures}
    predictions: dict[str, tuple[int, tuple[float, ...]]] = {}
    analyses: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if not hypothesis.closure_ids:
            analyses.append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_name": hypothesis.name,
                "verdict": "unaffected",
                "predicted_anchor_index": None,
                "detail": "no-closure explanation predicts no matching anchor scan",
            })
            continue
        closure = closure_by_id[next(iter(hypothesis.closure_ids))]
        anchor_index = fold_reference(closure, target_index)
        if anchor_index is None or anchor_index >= len(scans) or anchor_index == target_index:
            analyses.append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_name": hypothesis.name,
                "verdict": "insufficient",
                "predicted_anchor_index": anchor_index,
                "detail": "the folded reference point has not been observed yet",
            })
            continue
        signature = range_signature(list(scans[anchor_index].points))
        predictions[hypothesis.hypothesis_id] = (anchor_index, signature)
        analyses.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_name": hypothesis.name,
            "verdict": "eliminated",
            "predicted_anchor_index": anchor_index,
            "detail": "a scan that contradicts the folded anchor view rules this branch out",
        })
    if len(predictions) > 1:
        entries = list(predictions.items())
        indistinguishable: set[str] = set()
        for pos, (first_id, (first_anchor, first_signature)) in enumerate(entries):
            for second_id, (second_anchor, second_signature) in entries[pos + 1:]:
                gap = signature_distance(first_signature, second_signature)
                if first_anchor == second_anchor or gap <= 0.05:
                    indistinguishable.update((first_id, second_id))
        for analysis in analyses:
            if analysis["hypothesis_id"] in indistinguishable and analysis["verdict"] == "eliminated":
                analysis["verdict"] = "indistinguishable"
                analysis["detail"] = "competing fold branches predict the same anchor view at this location"
    return tuple(analyses)


def make_plan(
    run_id: str,
    name: str,
    targets: tuple[tuple[int, str, float, float], ...],
    scans: list[Scan2D],
    hypotheses: tuple[Any, ...],
    closures: tuple[LoopClosure, ...],
) -> ProbePlan:
    planned = tuple(
        ProbeTarget(
            target_index=target_index,
            location_label=label,
            expected_x=x,
            expected_y=y,
            analysis=analyze_probe_target(scans, hypotheses, closures, target_index),
        )
        for target_index, label, x, y in targets
    )
    return ProbePlan(
        plan_id=new_id("plan"),
        run_id=run_id,
        name=name,
        created_at=time.time(),
        targets=planned,
        hypothesis_ids=tuple(hypothesis.hypothesis_id for hypothesis in hypotheses),
    )
