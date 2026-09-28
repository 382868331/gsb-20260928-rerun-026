# 机器人二维建图平台

这是一个可扩展的机器人建图原型。首个功能把连续二维激光扫描接成栅格地图，并在机器人回到旧位置时用位姿图修正累积里程计偏差。

## 设计原则

- 原始扫描、原始里程计和地图版本都是只追加记录，闭环修正不会改写原始数据。
- 相似走廊不会仅凭一帧扫描 accepted；候选闭环必须通过相邻多观测重叠、变换一致性和轨迹相容性检查。
- 闭环可以接受、拒绝，也可以撤回；撤回后用剩余约束重新优化位姿。
- 传感器层使用统一记录接口，后续可继续接入 IMU、轮速计、视觉特征或三维激光。
- 仅使用 Python 标准库，便于审计和运行。

## 运行

```bash
python -m robot_mapping.demo
```

如果未安装包，可在仓库根目录执行：

```bash
PYTHONPATH=src python -m robot_mapping.demo
```

输出位于 `outputs/`：

- `loop/scans.json`：原始二维扫描。
- `loop/poses.json`：原始里程计与各版位姿。
- `loop/map_versions.json`：接受闭环和撤回闭环后的地图版本。
- `loop/*.svg`：路径与地图对照图。
- `similar_region/*`：相似走廊场景，包含错误闭环被拒绝以及“强制接受会发生什么”的对照。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 核心模块

- `robot_mapping.models`：扫描、位姿、传感器记录和地图版本数据模型。
- `robot_mapping.sensors`：传感器注册表与只追加数据存储。
- `robot_mapping.simulation`：带环路和相似走廊的人工场景、射线扫描与带漂移里程计。
- `robot_mapping.scan_matching`：扫描相似度、相邻多观测一致性与轨迹相容性验证。
- `robot_mapping.ray_matching`：以同一激光束角度核对测距观测的局部变换匹配。
- `robot_mapping.pose_graph`：二维位姿图和 Gauss-Newton 优化。
- `robot_mapping.mapper`：把扫描拼接成占据栅格，管理闭环生命周期和地图版本。
- `robot_mapping.visualization`：生成 SVG 对照图。
