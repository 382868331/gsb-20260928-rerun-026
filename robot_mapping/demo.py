from __future__ import annotations

import json
from pathlib import Path

from .mapper import MappingPlatform
from .simulator import Scenario, similar_corridor_scenario, simulate_scan, true_loop_scenario


def load_scenario_data(platform: MappingPlatform, run_id: str, scenario: Scenario) -> None:
    platform.create_run(run_id)
    for pose in scenario.odometry:
        index = len(platform.runs[run_id].scans)
        points = simulate_scan(scenario.truth[index], scenario.walls)
        platform.add_observation(run_id, pose, points)


def _load_one(platform: MappingPlatform, run_id: str, scenario: Scenario, index: int) -> None:
    pose = scenario.odometry[index]
    points = simulate_scan(scenario.truth[index], scenario.walls)
    platform.add_observation(run_id, pose, points)


def _load_range(platform: MappingPlatform, run_id: str, scenario: Scenario, start: int, end: int) -> None:
    for index in range(start, end):
        _load_one(platform, run_id, scenario, index)


def run_competing_corridor_demo(platform: MappingPlatform, run_id: str, scenario: Scenario) -> dict[str, object]:
    """Two similar corridors: keep both explanations until later connectivity evidence decides."""

    if run_id not in platform.runs:
        platform.create_run(run_id)
    arrival = scenario.expected_repeat - 10
    _load_range(platform, run_id, scenario, 0, arrival + 1)
    raw_map = platform.build_map(run_id, "01_arrival_raw_drift")
    hypotheses = platform.propose_competing_closures(
        run_id,
        [(0, arrival, "same corridor: fold onto first corridor"),
         (scenario.expected_anchor, arrival, "alternate alignment proposal")][:1],
    )
    fold = hypotheses[0]
    distinct = hypotheses[1]
    fold_preview = platform.build_map(run_id, "02_trial_fold_preview", hypothesis_id=fold.hypothesis_id)
    trial_accepted = platform.hypotheses_for(run_id)
    assert all(h.status == "active" for h in trial_accepted), "trial hypotheses must stay undecided"

    # Group the follow-up probes *before* collecting anything. One target lands on
    # a featureless corridor stretch (cannot promise exclusion), another reaches
    # beyond the observed map (information insufficient); neither is evidence yet.
    probe_plan = platform.create_probe_plan(
        run_id,
        "junction after the second similar corridor",
        [arrival + 2, arrival + 40],
        "compare observing a plain stretch vs. a position beyond the mapped corridor",
    )
    comparison = platform.compare_probe_plan(probe_plan.plan_id)
    prediction_labels = {
        (item["planned_index"], item["hypothesis_id"]): item["prediction"]
        for target in comparison["targets"]
        for item in target["predictions"]
    }
    assert any(label == "indistinguishable" for label in prediction_labels.values())
    assert any(label == "insufficient_information" for label in prediction_labels.values())
    # planned probes must not contribute evidence or change hypothesis status
    assert all(len(item.evidence) == 0 for item in platform.hypotheses_for(run_id))

    eliminating_observation = None
    expected_vs_actual_notes: list[str] = []
    for index in range(arrival + 1, len(scenario.truth)):
        _load_one(platform, run_id, scenario, index)
        linked_plan = probe_plan.plan_id if index in probe_plan.target_indices else None
        result = platform.feed_observation_to_hypotheses(
            run_id,
            index,
            plan_id=linked_plan,
            source_position=(scenario.truth[index].x, scenario.truth[index].y),
            collected_at=float(index),
        )
        expected_vs_actual_notes.extend(
            item["expected_vs_actual"] for item in result["hypothesis_results"] if item["expected_vs_actual"]
        )
        if eliminating_observation is None and any(item["eliminated"] for item in result["hypothesis_results"]):
            eliminating_observation = index
    assert eliminating_observation is not None, "later connectivity evidence must eliminate the fold"
    assert any("unlike the plan" in note for note in expected_vs_actual_notes), "plan vs. reality difference must be reported"

    # The first junction view is discovered to belong to another corridor: withdraw
    # it and check the affected plan/hypothesis is restored while other evidence stays.
    withdrawn = platform.withdraw_observation(
        run_id, arrival + 1, "junction scan later attributed to the other corridor", reassigned_corridor=True
    )
    fold = platform.hypothesis(fold.hypothesis_id)
    assert fold.hypothesis_id in withdrawn["restored_hypotheses"]
    assert any(item.get("status") == "reassigned" for item in fold.evidence)
    assert any(item.get("status") == "valid" and not item["supports_fold"] for item in fold.evidence), (
        "independent evidence from other observations must keep counting"
    )
    restored_map = platform.build_map(run_id, "03_fold_restored_after_withdrawal", hypothesis_id=fold.hypothesis_id)

    # A branch-scoped plan advances probing inside the reopened fold branch and
    # predicts the distinctive notch position. Its execution eliminates the fold
    # again: the plan said the position would exclude on conflict, and it does.
    branch_plan = platform.create_probe_plan(
        run_id,
        "fold branch: distinctive notch anchor",
        [arrival + 11],
        "probe the distinctive anchor predicted by the fold",
        branch_hypothesis_id=fold.hypothesis_id,
    )
    branch_target = platform.compare_probe_plan(branch_plan.plan_id)["targets"][0]["predictions"]
    assert any(item["prediction"] == "would_exclude_on_conflict" for item in branch_target)
    branch_result = platform.feed_observation_to_hypotheses(
        run_id,
        arrival + 11,
        plan_id=branch_plan.plan_id,
        source_position=(scenario.truth[arrival + 11].x, scenario.truth[arrival + 11].y),
        collected_at=float(arrival + 11),
    )
    assert any(item["eliminated"] and "matched the plan" in item["expected_vs_actual"] for item in branch_result["hypothesis_results"])

    fold = platform.hypothesis(fold.hypothesis_id)
    distinct = platform.hypothesis(distinct.hypothesis_id)
    elimination_reason_text = fold.reason
    eliminated_map = platform.build_map(run_id, "04_fold_eliminated_distinct_corridors", hypothesis_id=distinct.hypothesis_id)
    # the never-executed far-away plan item must never have become evidence
    assert all(int(item["repeat_index"]) != arrival + 40 for item in fold.evidence)

    reopened = platform.reopen_hypothesis(fold.hypothesis_id)
    reopened_map = platform.build_map(run_id, "05_reopened_fold_for_comparison", hypothesis_id=reopened.hypothesis_id)
    assert {item["repeat_index"] for item in reopened.evidence if not item["supports_fold"]}, "exclusion evidence must remain reviewable"
    fold = platform.eliminate_hypothesis(fold.hypothesis_id, "re-checked against retained evidence; notch connectivity still contradicts the fold")
    platform.reject_closure(next(iter(fold.closure_ids)), fold.reason)
    confirmed = platform.confirm_hypothesis(distinct.hypothesis_id, "connectivity landmarks show two distinct corridors")
    final_map = platform.build_map(run_id, "06_final_distinct_corridors")
    return {
        "run_id": run_id,
        "decision": "fold hypothesis eliminated after later observations; distinct-corridor map selected",
        "fold_hypothesis": fold.hypothesis_id,
        "selected_hypothesis": confirmed.hypothesis_id,
        "eliminating_observation": eliminating_observation,
        "elimination_reason": elimination_reason_text,
        "raw_map": raw_map["map_version_id"],
        "fold_preview": fold_preview["map_version_id"],
        "eliminated_map": eliminated_map["map_version_id"],
        "reopened_map": reopened_map["map_version_id"],
        "restored_map": restored_map["map_version_id"],
        "final_map": final_map["map_version_id"],
        "probe_plan": probe_plan.plan_id,
        "branch_plan": branch_plan.plan_id,
        "expected_vs_actual": expected_vs_actual_notes,
        "evidence": [item for item in fold.evidence],
        "observations": len(scenario.truth),
    }


def run_demo(project_dir: str | Path) -> Path:
    project = Path(project_dir)
    platform = MappingPlatform(project)
    summaries = []
    scenarios = (("true_loop", true_loop_scenario()), ("similar_corridors", similar_corridor_scenario()))
    for run_id, scenario in scenarios:
        if run_id == "similar_corridors":
            platform.create_run(run_id)
            summaries.append(run_competing_corridor_demo(platform, run_id, scenario))
            continue
        load_scenario_data(platform, run_id, scenario)
        proposal = platform.propose_candidate(run_id, scenario.expected_anchor, scenario.expected_repeat)
        raw_map = platform.build_map(run_id, "01_raw_drift")
        if scenario.expected_loop:
            platform.accept_closure(proposal.closure_id)
            accepted_map = platform.build_map(run_id, "02_correct_closure_accepted")
            platform.withdraw_closure(proposal.closure_id)
            withdrawn_map = platform.build_map(run_id, "03_accepted_closure_withdrawn")
            platform.accept_closure(proposal.closure_id)
            final_map = platform.build_map(run_id, "04_correction_restored")
            decision = "accepted then withdrawn and restored"
        summaries.append({
            "run_id": run_id,
            "expected_loop": scenario.expected_loop,
            "decision": decision,
            "closure_id": proposal.closure_id,
            "closure_was_consistent": proposal.reason == "multiple observations agree",
            "score": proposal.score,
            "reason": proposal.reason,
            "raw_map": raw_map["map_version_id"],
            "final_map": final_map["map_version_id"],
            "observations": len(scenario.truth),
            "evidence": proposal.evidence,
        })
    report = project / "DEMO_REPORT.md"
    lines = [
        "# 机器人二维建图闭环演示",
        "",
        "原始扫描和原始里程计只追加保存；接受、拒绝或撤回仅新增闭环事件和地图版本。",
        "",
        "| 场景 | 多观察结论 | 操作 | 观测数 | 初始地图 | 最终地图 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for item in summaries:
        consistency = "一致，可接受" if item.get("expected_loop", False) else "相似但连通关系不同"
        lines.append(
            f"| `{item['run_id']}` | {consistency} | {item['decision']} | {item['observations']} | `{item['raw_map']}` | `{item['final_map']}` |"
        )
    lines.extend([
        "",
        "## 判定依据",
        "",
        "- 先用范围形状粗筛，但相似走廊不会仅凭单帧形状闭环。",
        "- 对候选点前后各一帧共多组观察做独立 ICP 几何配准。",
        "- 比较各组推算出的相对位姿：误差、内点比例、平移散布和角度散布必须同时稳定。",
        "- 正确环路接受后进行位姿图修正；错误相似区域保留拒绝记录且地图不被强行折叠。",
        "- 接受后的修正可撤回，撤回后重新建图；再次接受会生成新的地图版本。",
        "",
        "## 多假设方案",
        "",
        "- 相似走廊场景在机器人到达第二条走廊时同时保留两个解释：折叠为同一条走廊，或确实是两条不同走廊。",
        "- 试用折叠只生成预览地图版本，不改变公共地图；两个假设都保持未决定状态并继续接收后续观测。",
        "- 后续经过第二条走廊的凹槽时，内点率和范围形状与折叠预测矛盾，系统记录每条排除依据并淘汰折叠假设。",
        "- 可回到排除前重开折叠假设并排比较其预览地图；证据仍保留在假设上，重新核对后可再次淘汰。",
        "- 最终选定双走廊解释，公共地图不折叠；局部修正只改动闭环跨度内的位姿，未受影响区域与原始里程计一致。",
        "",
        "工程数据：`runs/*/scans.jsonl`、`runs/*/poses.jsonl`、`events.jsonl`、`map_versions/*/map.pgm`。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (project / "demo_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
