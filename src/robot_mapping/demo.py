from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mapper import MappingSession
from .models import MapVersion
from .sensors import AppendOnlyStore
from .simulation import build_drifted_odometry, similar_corridors_scenario, simulate_scan, square_loop_scenario
from .visualization import render_map_svg


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: object):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_session(directory: Path, session: MappingSession, truth, store: AppendOnlyStore, labels: dict[str, str]) -> None:
    write_json(directory / "scans.json", [record.to_dict() for record in store.records("lidar2d")])
    write_json(
        directory / "poses.json",
        {
            "truth": [pose.to_dict() for pose in truth],
            "raw_odometry": [pose.to_dict() for pose in session.raw_poses],
            "current": [pose.to_dict() for pose in session.current_poses],
        },
    )
    write_json(
        directory / "closures.json",
        {"candidates": [candidate.to_dict() for candidate in session.candidates], "accepted": session.accepted_closure_id},
    )
    write_json(directory / "map_versions.json", [version.to_dict() for version in session.versions])
    for version in session.versions:
        label = labels.get(version.version_id, version.event)
        (directory / f"{version.version_id}_{label}.svg").write_text(
            render_map_svg(version, truth, title=f"{directory.name}: {label}"), encoding="utf-8"
        )


def run(output_root: Path) -> None:
    summaries: list[dict[str, object]] = []

    loop_dir = output_root / "loop"
    loop_scenario = square_loop_scenario()
    loop_truth = loop_scenario.path()
    loop_scans = [simulate_scan(pose, i, loop_scenario) for i, pose in enumerate(loop_truth)]
    loop_odometry = build_drifted_odometry(loop_truth)
    loop_store = AppendOnlyStore()
    for i, (scan, reading) in enumerate(zip(loop_scans, loop_odometry)):
        loop_store.append(f"lidar-{i:03d}", "lidar2d", float(i), scan)
        loop_store.append(f"odom-{i:03d}", "odometry2d", float(i), reading)
    loop_session = MappingSession(loop_scans, loop_odometry)
    candidates = loop_session.find_candidates()
    accepted = next((candidate for candidate in candidates if loop_session.evaluate_candidate(candidate) and candidate.accepted), None)
    if accepted is None:
        raise RuntimeError("synthetic loop scenario did not produce an accepted closure")
    accepted_version_id = loop_session.versions[-1].version_id
    loop_session.withdraw_accepted_closure()
    withdrawn_version_id = loop_session.versions[-1].version_id
    save_session(
        loop_dir,
        loop_session,
        loop_truth,
        loop_store,
        {accepted_version_id: "accepted_correct_closure", withdrawn_version_id: "withdrawn_closure"},
    )
    summaries.append({"scenario": "loop", "accepted": accepted.to_dict()})

    similar_dir = output_root / "similar_region"
    similar_scenario = similar_corridors_scenario()
    similar_truth = similar_scenario.path()
    similar_scans = [simulate_scan(pose, i, similar_scenario) for i, pose in enumerate(similar_truth)]
    similar_odometry = build_drifted_odometry(similar_truth)
    similar_store = AppendOnlyStore()
    for i, (scan, reading) in enumerate(zip(similar_scans, similar_odometry)):
        similar_store.append(f"lidar-{i:03d}", "lidar2d", float(i), scan)
        similar_store.append(f"odom-{i:03d}", "odometry2d", float(i), reading)
    similar_session = MappingSession(similar_scans, similar_odometry)
    false_candidate = similar_session.add_forced_candidate(2, 26)
    similar_session.reject_candidate(false_candidate)
    save_session(similar_dir, similar_session, similar_truth, similar_store, {})

    forced_dir = similar_dir / "forced_wrong_closure"
    forced_store = AppendOnlyStore()
    for i, (scan, reading) in enumerate(zip(similar_scans, similar_odometry)):
        forced_store.append(f"lidar-{i:03d}", "lidar2d", float(i), scan)
        forced_store.append(f"odom-{i:03d}", "odometry2d", float(i), reading)
    forced_session = MappingSession(similar_scans, similar_odometry)
    forced_candidate = forced_session.add_forced_candidate(2, 26)
    forced_session.build_version_with_force(forced_candidate)
    save_session(forced_dir, forced_session, similar_truth, forced_store, {forced_session.versions[-1].version_id: "do_not_trust_forced_wrong_closure"})
    summaries.append({"scenario": "similar_region", "rejected": false_candidate.to_dict(), "forced": forced_candidate.to_dict()})

    write_json(output_root / "summary.json", summaries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic 2D mapping scenarios")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="output directory")
    args = parser.parse_args(argv)
    run(args.output)


if __name__ == "__main__":
    main()
