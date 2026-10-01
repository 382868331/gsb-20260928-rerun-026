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

    # Plan a GROUPED follow-up exploration before the observations exist: compare what
    # each visit location would rule out. The last target lies beyond the planned
    # trajectory, so the fold cannot predict an anchor view for it (insufficient info),
    # and it is never executed.
    probe_plan = platform.create_probe_plan(
        run_id,
        "follow-up probes along the second corridor",
        [
            (arrival + 1, "one step into the second corridor", 0.0, -4.0),
            (arrival + 2, "two steps into the second corridor", 0.0, -4.0),
            (arrival + 37, "uncharted end of the new corridor", 0.0, -4.0),
        ],
    )
    planned_verdicts = {
        target.target_index: {entry["hypothesis_name"]: entry["verdict"] for entry in target.analysis}
        for target in probe_plan.targets
    }
    unexecuted_targets = [target.target_index for target in probe_plan.targets if target.status == "planned"]
    assert "insufficient" in planned_verdicts[arrival + 37].values(), "missing data must stay inconclusive at planning time"

    def _load_and(index: int, probe: bool) -> list:
        _load_one(platform, run_id, scenario, index)
        if probe:
            result = platform.execute_probe(probe_plan.plan_id, index, repeat_index=index)
            return [platform.hypothesis(hypothesis_id) for hypothesis_id in result["actual_outcomes"]]
        return platform.feed_observation_to_hypotheses(run_id, index)

    # Two planned probes contradict the fold, then one independent observation provides
    # a third contradiction; the fold is eliminated by the 2-of-3 evidence rule.
    probe_results = [_load_and(index, True) for index in (arrival + 1, arrival + 2)]
    independent_index = arrival + 3
    _load_and(independent_index, False)
    fold = platform.hypothesis(fold.hypothesis_id)
    assert fold.status == "eliminated", "the grouped probes plus one independent scan must exclude the fold"
    eliminating_observation = independent_index
    distinct = platform.hypothesis(distinct.hypothesis_id)
    elimination_reason_text = fold.reason
    eliminated_map = platform.build_map(run_id, "03_fold_eliminated_distinct_corridors", hypothesis_id=distinct.hypothesis_id)

    # The two probe observations are later withdrawn (misattributed corridor). The single
    # independent contradiction can no longer satisfy the 2-of-3 rule, so the fold branch
    # is restored; the independent evidence stays live.
    first_withdrawal = platform.withdraw_probe(
        probe_plan.plan_id, arrival + 1, "observation misattributed to this corridor during review"
    )
    second_withdrawal = platform.withdraw_probe(
        probe_plan.plan_id, arrival + 2, "observation misattributed to this corridor during review"
    )
    restored_fold = platform.hypothesis(fold.hypothesis_id)
    restored_map = platform.build_map(run_id, "04_fold_restored_after_probe_withdrawal", hypothesis_id=restored_fold.hypothesis_id)
    live_evidence = [item for item in restored_fold.evidence if not item.get("withdrawn")]
    withdrawn_evidence = [item for item in restored_fold.evidence if item.get("withdrawn")]
    assert restored_fold.status == "active", "withdrawing the probe observations must restore the affected branch"
    assert len(withdrawn_evidence) == 2 and live_evidence, "withdrawn evidence is retained and independent evidence stays valid"

    # Exploration continues on the restored branch; later independent observations
    # re-eliminate the fold without touching the withdrawn probes.
    reliminated_at = None
    for index in range(arrival + 4, len(scenario.truth)):
        updated = _load_and(index, False)
        for hypothesis in updated:
            if hypothesis.hypothesis_id == restored_fold.hypothesis_id and hypothesis.status == "eliminated":
                reliminated_at = index
                break
        if reliminated_at is not None:
            break
    assert reliminated_at is not None, "independent later observations must re-eliminate the restored fold"
    fold = platform.hypothesis(fold.hypothesis_id)

    reopened = platform.reopen_hypothesis(fold.hypothesis_id)
    reopened_map = platform.build_map(run_id, "05_reopened_fold_for_comparison", hypothesis_id=reopened.hypothesis_id)
    assert {item["repeat_index"] for item in reopened.evidence if not item["supports_fold"]}, "exclusion evidence must remain reviewable"
    fold = platform.eliminate_hypothesis(fold.hypothesis_id, "re-checked against retained evidence; notch connectivity still contradicts the fold")
    platform.reject_closure(next(iter(fold.closure_ids)), fold.reason)
    confirmed = platform.confirm_hypothesis(distinct.hypothesis_id, "connectivity landmarks show two distinct corridors")
    final_map = platform.build_map(run_id, "06_final_distinct_corridors")
    timeline = platform.decision_timeline(run_id)
    assert any(event.get("type") == "probe_plan_created" for event in timeline)
    assert any(event.get("type") == "probe_withdrawn" for event in timeline)
    return {
        "run_id": run_id,
        "decision": "fold hypothesis eliminated after later observations; distinct-corridor map selected",
        "fold_hypothesis": fold.hypothesis_id,
        "selected_hypothesis": confirmed.hypothesis_id,
        "eliminating_observation": eliminating_observation,
        "elimination_reason": elimination_reason_text,
        "probe_plan": probe_plan.plan_id,
        "planned_verdicts": planned_verdicts,
        "unexecuted_targets": unexecuted_targets,
        "restored_hypotheses": first_withdrawal["restored_hypotheses"] + second_withdrawal["restored_hypotheses"],
        "reliminated_at": reliminated_at,
        "raw_map": raw_map["map_version_id"],
        "fold_preview": fold_preview["map_version_id"],
        "eliminated_map": eliminated_map["map_version_id"],
        "restored_map": restored_map["map_version_id"],
        "reopened_map": reopened_map["map_version_id"],
        "final_map": final_map["map_version_id"],
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
        "- 排除前先成组规划多个后续探查位置：标出各位置能排除哪些解释、哪些无法区分、哪些因参照点尚未观测而信息不足。",
        "- 未执行的计划不产生证据；实测按来源位置与采集时刻更新假设，并对比预期与实际排除结果。",
        "- 探查观测被撤回或判定属于另一条走廊时恢复受影响分支，独立证据继续有效；撤回与恢复记录可随项目重开复盘。",
        "- 最终选定双走廊解释，公共地图不折叠；局部修正只改动闭环跨度内的位姿，未受影响区域与原始里程计一致。",
        "",
        "工程数据：`runs/*/scans.jsonl`、`runs/*/poses.jsonl`、`events.jsonl`、`map_versions/*/map.pgm`。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (project / "demo_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
