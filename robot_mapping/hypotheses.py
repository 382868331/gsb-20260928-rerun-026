from __future__ import annotations

from .models import LoopClosure


def closures_conflict(first: LoopClosure, second: LoopClosure) -> bool:
    """Two closures conflict when they explain the same repeat place differently.

    Closures that share both endpoints are the same explanation, not a conflict.
    """

    first_pair = (first.anchor_index, first.repeat_index)
    second_pair = (second.anchor_index, second.repeat_index)
    if first_pair == second_pair:
        return False
    return (
        first.anchor_index == second.anchor_index
        or first.repeat_index == second.repeat_index
        or first.anchor_index == second.repeat_index
        or first.repeat_index == second.anchor_index
    )


def fold_reference(closure: LoopClosure, repeat_index: int) -> int | None:
    """Map a later scan to the anchor-side scan predicted by folding along a closure."""

    mapped = closure.anchor_index + (repeat_index - closure.repeat_index)
    if mapped < 0:
        return None
    return mapped


def elimination_reason(
    evidence: tuple[dict[str, object], ...],
    required_failures: int = 2,
    window: int = 3,
) -> str:
    recent = evidence[-window:]
    failures = [
        item
        for item in recent
        if item.get("status", "valid") == "valid"
        and item.get("verdict", "contradictory" if not bool(item.get("supports_fold")) else "consistent") == "contradictory"
    ]
    if len(failures) >= required_failures:
        positions = ", ".join(str(item["repeat_index"]) for item in failures)
        inliers = ", ".join(f"{float(item['inlier_fraction']):.2f}" for item in failures)
        return (
            f"later observations at repeat indices {positions} contradict the fold "
            f"(ICP inlier fractions {inliers}); corridors look similar but connect differently"
        )
    return ""
