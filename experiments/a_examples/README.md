# A 模块的小样本交付记录

这些文件来自实际执行 `python -m search_experiments.self_test --output experiments/a_examples`。用途是交付前检查，不是正式实验或最终 S* 选择。

- Python 3.11.18、NumPy 1.26.4；36 项单元/回归测试通过。
- 生成全部 48 种配置的固定 D0/D1/T 名单；在其中 1 个 D0 配置运行双座严格完整参考对局，无技术失败。
- 在一个抽样固定局面上，对深度 2、3 完成 C0–C3 精确值/最优动作检查，以及冷启动、完整迭代和单独诊断。
- S1–S4 均通过 E0/E1 精确参考检查；限时小样本仅用于测试选择流程，`S_star=null`。
- 相同命令续跑未重复对局；另用 `external_reuse_check.json` 验证跨运行复用相同条件的已完成对局，两局均直接复用。
- 生成的 S4 配置文件通过官方验证器；4 局严格沙箱检查全部通过，无 strikes 或 forfeits；报告的最大 step 为 0.03 秒、峰值 RSS 为 42.6 MB。
- 原始 `agents/student_agent.py`、`agents/evaluation.py`、`agents/symmetry.py`、`agents/game_state.py`、`helpers.py`、`constants.py` 均保持原始字节不变。

缩减预算、实际 seed、完整轨迹、缺失抽样组、机器与版本 hash 均在相邻 JSON / JSONL 中。正式运行请按 [模块 README](../../search_experiments/README.md) 的正式命令，由 F 使用新的输出目录按步骤执行。
