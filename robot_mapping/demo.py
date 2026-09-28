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


def run_demo(project_dir: str | Path) -> Path:
    project = Path(project_dir)
    platform = MappingPlatform(project)
    summaries = []
    for run_id, scenario in (("true_loop", true_loop_scenario()), ("similar_corridors", similar_corridor_scenario())):
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
        else:
            platform.reject_closure(proposal.closure_id, "adjacent observations describe different corridor features")
            accepted_map = platform.build_map(run_id, "02_false_closure_rejected")
            withdrawn_map = accepted_map
            final_map = accepted_map
            decision = "rejected"
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
        consistency = "一致，可接受" if item["expected_loop"] else "不一致，拒绝"
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
        "工程数据：`runs/*/scans.jsonl`、`runs/*/poses.jsonl`、`events.jsonl`、`map_versions/*/map.pgm`。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (project / "demo_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
