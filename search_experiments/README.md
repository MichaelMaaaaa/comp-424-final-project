# A：第 0–2 步搜索实验模块

对应 `COMP424_Experiment_Plan_Module_Assignments.pdf` 第 11–12 页的 A 分工：实现完整实验入口、共用工具、配置与小样本自测。正式第 0–8 步由 F 合并三人的模块后统一运行。本目录附带的小样本输出仅验证实现与接线，不能用于确定正式 S* 或报告棋力提升。

基于仓库提交 `a407698955bb2ea9931c1264b2d0b7d0f8db479f`。保留原始 `agents/student_agent.py`，也保留 `evaluation.py` 和 `symmetry.py`。新增搜索引擎调用 F 的评价接口及 B 的坐标变换接口。

## 实现内容

| 文件 / 入口 | 内容 |
|---|---|
| `agents/search_engine.py` | 固定深度、迭代加深、C0–C3、S1–S4、TT 后端选择及诊断计数 |
| `agents/experiment_agent.py` | 整步计时、合法 fallback、统一入口、F 评价工厂与 B book hook |
| `search_experiments/exp00_prepare.py` | 48 配置的 D0/D1/T 名单、双座严格对局、参考轨迹及分层局面抽样 |
| `search_experiments/exp01_components.py` | 精确根值及最优动作集合、组件比较、冷启动与完整迭代计时 |
| `search_experiments/exp02_search.py` | 逐项尝试优化、E0/E1 正确性、固定深度与限时比较、开发集确认及配置输出 |
| `search_experiments/runner.py` | 三人共用的唯一对局循环、完整快照、断点续跑、严格复用条件、配对统计 |
| `search_experiments/measure.py` | 固定局面测量、整步测量、计时 CSV 与单独诊断记录 |
| `search_experiments/build.py` | 按模块依赖自动生成允许导入的单文件，附构建来源和 SHA-256 |
| `search_experiments/configs/` | 准备、组件、搜索及单个线上版本的 JSON 模板 |
| `tests/test_search_experiments.py` | 规则相关搜索回归、缓存/超时/对称、接口及日志测试 |

本地 AB 修正了两处与冻结对手有关的行为：整条舍弃经过障碍的获胜线；在一次搜索返回前复查时限，避免提交已超时的最后一层。其搜索仍为原始框架 Minimax/Alpha-Beta、无排序/无 TT、最多深度 3、正式内部预算 1.5 秒。正式运行记录保存修正后源文件和生成对手的 hash；旧仓库的历史对局不视为本轮参考对局。

## 环境与小样本自测

使用 Python 3.11 和 NumPy 1.26.4。在仓库根目录运行：

```bash
python -m unittest tests.test_search_experiments -v
python -m search_experiments.self_test --output experiments/a_smoke
```

第二条依次执行单元测试、1 个 D0 配置的双座严格完整对局、固定局面的深度 2/3 组件检查、S1–S4 在 E0/E1 下的检查、断点续跑检查及生成文件的官方沙箱验证。小样本对局用 0.012 秒候选预算、0.006 秒对手预算、最多深度 2；限时诊断用 0.03 秒。这些缩减条件只在显式 `--smoke` 下生效。

仓库中的 `experiments/a_examples/` 为本次实际执行的小样本交付记录。`SELF_TEST_SUMMARY.json`、文本日志、原始记录与生成文件均可检查；所有结果均标记 `execution_tier=smoke`。如果再自测，请使用 `experiments/a_smoke` 等新目录，保留示例记录。

## F 正式运行的顺序

先合并三人的模块并固定代码，再在同一机器顺序运行。下面的默认配置使用正式预算，不缩短对局，不读取 T 进行开发选择。

```bash
python -m search_experiments.exp00_prepare --config search_experiments/configs/prepare.json --output experiments/a_run/prepare
python -m search_experiments.exp01_components --config search_experiments/configs/components.json --output experiments/a_run/components
python -m search_experiments.exp02_search --config search_experiments/configs/search.json --output experiments/a_run/search
```

也可使用 `--positions`、`--manifest` 覆盖路径。三个入口都接受 `--config`、`--output`。各阶段 JSON 内记录输入路径、hash、机器、预算与来源；不要只按 `student_agent.py` 文件名判断版本。

准备阶段的固定名单为：每种移除大小各有 D0 的 1 角/2 边/1 内部、D1 的另 1 角/2 边/1 内部、T 的其余 2 角/4 边/2 内部，共 12/12/24 配置。先查找匹配配置的 `setup_seed`，正式运行时再核对实际开局。每个配置两局交换候选座位，对手的实际 RNG 在生成的 random/greedy 文件中显式设置。

`manifest_seed` 固定名单；`sample_seed` 固定抽样；名单另存 `opponent_rng_seed`、`collection_seed`。修改 seed 应使用新输出目录。开局名单 CSV 同时给出 `config_id`、障碍行列、移除大小、split 和实际 seed。

第 0 步先对本地 AB 跑原始 S0/E1/N 的 D0 双座 24 局。逐步保存双方行动前的完整状态；按障碍类 × 移除大小 × 阶段，从不同对局各抽 1 个，目标每组 2 个、共 54 个。缺少的组照实写入 `coverage`，不借用其他组或重复相邻状态补齐。阶段为 0–9、10–29、30–99，另有 90–99 标记。

第 1 步固定 E1、关闭 book。C0 完整搜索每个根动作，保存精确值及全部真正最优动作。C1–C3 的根值应相同，返回动作应属于该集合。每个根/目标深度重新清空 TT；关闭排序时也关闭 preferred 提升。冷启动直接搜索到深度 d；完整迭代测量执行 1…d 并保留该次调用的迭代提示/TT。每种计时交替顺序重复 3 次，详细计数另外运行。默认共同保护时限 60 秒；超时项写为未完成，不当作精确参考，也不进入跨版本共同完成项的耗时比较。

第 2 步依次尝试 S1、S2、S3、S4。每项只在此前保留的父配置上改一个开关，并写出实际 `parent_variant`；未保留的修改不会带到下一项。在 E0/E1 下重新检查固定深度精确值，再比较 E1 的完整迭代耗时和 1.6 秒整步预算下完成深度。近 100 ply 和已搜完剩余深度的状态单列。三次重复均胜出且预算/正确性通过才保留；排名反复则保留父版本。

随后 S_diag 与 S0 补齐 D0∪D1 各 48 局。`reference_games` 指向准备阶段日志，完全相同的已有 D0 记录可直接复用；校验代码、配置、book、生成代理、对手、seed、机器、规则和预算，任一变化都不复用。完整开发名单结束后按得分选择 S*，同分保留原始 S0，并输出 S_alt。S_diag 就是 S0 时，备选从已验证候选按完成深度、相对同一 S0 的耗时比及候选顺序选择。

只有 `search_selection.json` 的 `status=selected` 才有正式 S*。此时读取 `selected_search.json` 和 `alternate_search.json` 交给 F 的第 3 步。`smoke_only` 或 `awaiting_development_confirmation` 不代表正式选优；`S_star` 保持 null。正式第 2 步拒绝 smoke 局面或错误/来源不一致的第 1 步结果。

## 搜索开关

| 名称 | Alpha-Beta | 排序及 preferred | TT | 本项变化 |
|---|---|---|---|---|
| C0 | 关 | 关，原始动作顺序 | 关 | 完整 Negamax 参考 |
| C1 | 开 | 关，原始动作顺序 | 关 | 剪枝 |
| C2 | 开 | 当前启发式排序 | 关 | 排序 |
| C3 / S0 | 开 | 当前启发式排序 | 开 | 原始 tuple 键；TT 数值及动作提示整体作用 |
| S1 | 继承父项 | 先构造并搜索合法 preferred | 继承 | 若首着剪枝，不再构造其余后继 |
| S2 | 继承父项 | 复用排序已算的子终局/适用静态分数 | 继承 | 深层搜索仍计算真实子树值 |
| S3 | 继承父项 | 继承 | 15 字节无损原坐标键 | 仅改编码，无对称 |
| S4 | 继承父项 | 继承 | 加当前障碍稳定子群 | 角/边/内部为 2/1/2 个变换，包含恒等 |

紧凑 TT 键包含全部隐藏层、双方储备、行动玩家、绝对 ply 和规则；障碍由棋盘中的三层障碍标志保留。有限对称只调用 B 的 `transform_cell` / `transform_move` / `inverse_transform_move`；读取 TT 动作时逆变换回查询坐标。TT 数值只在相同剩余深度复用。EXACT 直接返回；LOWER/UPPER 只有足以截断当前窗口时才返回，否则不收紧窗口。

TT 默认 50,000 个数值项，到容量后停止新增、允许更新。`tt_capacity` 已是配置项，后续若诊断出现容量拒绝，可通过自定义版本表比较容量；默认流程不自动添加这一实验。诊断输出后继构造、终局、评价、排序、剪枝、TT 行为、键/查询成本和近似 TT 对象内存。计时行的 `statistics.enabled=false`，避免把详细计数开销混入正式速度比较。

## F：评价函数接口

开发引擎直接接受 callable：

```python
engine = SearchEngine(search_config, evaluator=my_evaluate)
fixed = engine.fixed_depth(state, depth=3, deadline=absolute_deadline)
result = engine.search(state, deadline=absolute_deadline)
```

`my_evaluate(state, player)` 返回该玩家视角的非终局分数，并满足双方视角互为相反数。终局先检查，终局效用由 `evaluation.terminal_utility` 处理。一个 SearchEngine 使用固定评价实例，跨步搜索自动清 TT；不要在同一搜索中途更换参数。

F 在自己的 `agents/evaluation.py` 加入：

```python
def make_evaluator(config):
    # config includes evaluation_id, lambda and q.
    # Return a callable evaluate(state, player) using these fixed parameters.
    ...
```

配置例如 `{"evaluation_id":"lambda5_Q1", "lambda":5, "q":[1,50,100]}`。A 的开发工具和构建器都会自动调用此工厂，叶点评价与动作排序共享同一 callable。当前未加入 F 工厂时，只接受原始 E0/E1，其他参数会报错，防止实验看似改了权重却仍在使用 E1。

F 第 3/4 步可以直接复用：

```python
from search_experiments.runner import run_pairs
from search_experiments.measure import measure, measure_online_step
```

`run_pairs(variant, manifest_rows, output, opponent={"kind":"ab","budget":1.5})` 负责全部严格对局、快照和恢复。`variant` 使用配置模板格式，参数改变时设置新的 `variant_id`、`parent_variant`。开发集传 D0 或 D1 的行；最终测试才显式传 `allow_test=True` 并使用 T。F/B 无需重写对局循环。

## B：Book 接口与离线分析

线上配置使用 `book.provider="agents.book_policy"` 等模块名。B 在该模块提供 `make_book_hook(book_config, variant)`，返回一个接收 `(state, legal_moves_tuple, absolute_deadline)` 的函数。

这个函数返回字典：

```python
{
    "matched": True,
    "version": "book-version",
    "direct_move": move_or_none,
    "preferred_move": move_or_none,
    "metadata": {"query_seconds": 0.0001}
}
```

B 在自己的 hook 中实现 N/D/H、版本检查、纯查询计时及具体 book 行为。A 检查动作合法性并执行直接返回/根 preferred 的共用入口；书不会写入搜索 TT。后续迭代使用自身上一完整迭代动作。默认 N 且无 provider 时完全不查询。计数包括是否真实查询、是否匹配、直接返回或排序提示；book/唯一合法动作返回的深度是 null，搜索但无完整第一层是 0。

Book 的加载发生在对象初始化阶段。构建器自动跟随 `provider` 的依赖并内联；模块间采用显式 imports，顶层自定义名称发生冲突时构建器报错，防止静默覆盖。B 负责将运行时 book 数据放在模块常量中；生成文件不能依赖磁盘 book 文件。

离线分析直接用同一 SearchEngine，不调用 book hook，设置新的绝对 deadline。需要真实同分动作集时使用 `fixed_depth(..., collect_ties=True)` 或 `search(..., collect_ties=True, tie_rng=random.Random(collection_seed))`。该模式对每个根动作使用完整窗口，仅从最后完整深度的精确最优集合选择，超时不提交部分集合。

B 的 miss/命中整步成本可使用 `measure_online_step(state, variant)`，它计时真实 `ExperimentAgent.step()`，自动加载配置中的评价及 book 工厂。初始化成本单独记录为 `object_init_s`；正式初始化/RSS 使用严格对局的 referee 记录。

## 单文件构建与验证

```bash
python -m search_experiments.build --config search_experiments/configs/variant.json --output experiments/a_run/built/S0_E1_N
python validate_submission.py experiments/a_run/built/S0_E1_N/student_agent.py
```

生成器读取模块依赖，去掉内部开发 imports，保留官方允许导入，并生成唯一的 `@register_agent("student_agent")` 入口。不使用运行时 exec/eval，不改原始提交文件，不要求三人手工拼接代码。输出旁边的 `.meta.json` 给出源模块 hash、完整配置及生成文件 hash/大小。最终版本仍须由 F 完成官方验证和严格对局检查。

## 输出与断点续跑

| 输出 | 用途 |
|---|---|
| `manifest.json` / `manifest.csv` | 固定 48 配置和 seed |
| `reference/games.jsonl` | 每个计划对局的结果、完整行动前状态和官方裁决 |
| `positions.json` | 固定局面、来源对局、分层缺失情况 |
| `references/measurements.jsonl` | C0 的精确根值和真正最优动作集合 |
| `measurements.jsonl` | 每次固定深度、完整迭代或整步测量；诊断行显式分开 |
| `timings.csv` | 全部测量行及完成/正确性状态 |
| `per_position_timings.csv` | 每个局面/版本/深度三次计时的中位数及范围，未完成不填耗时汇总 |
| `decision.json` / `search_selection.json` | 每项父配置、单项变化、保留原因及正式搜索选择 |
| `plan.json` / `summary.json` | 计划名单、恢复身份、完整性、W/D/L、技术失败、双座配对得分 |

每局记录 `variant_id`、`parent_variant`、code/config/book hash、实际生成文件、对手与 seed、机器、split、座位及官方初始化/step/RSS。`referee.moves[*].agent_time` 是正式的完整 step 耗时；`search_elapsed_s` 只表示内部搜索，不能替代它。轻量搜索信息通过受官方大小限制的 agent 日志带出，详细节点统计在独立诊断中测量。

重复同一命令时，只跳过身份完全相同且已由官方 referee 判定结束的对局。超时、崩溃、初始化失败等按官方裁决进入 W/D/L；没有有效裁决的运行故障保留 `infrastructure_error` 并在续跑时重试。未完成计划名单不输出候选总得分或排名。候选或对手出现技术失败时，记录结果但不用于选优，先修复对应实现。

输出目录的代码、配置或机器身份变化时，恢复过程会报错；另选输出目录保留旧结果即可。不要覆盖已有正式实验日志。固定局面测量也按完整请求身份恢复，未完成搜索仍是未完成项；若需要增加保护预算，使用新的目录。

配对差以一个配置的两座位平均为单位计算，给出改善/持平/下降配置数及配置层面的 bootstrap 区间。重复座位或对称方位不作为独立 Bernoulli 样本。正式耗时、完成深度、得分分别报告；未运行正式开发对局时，不把小样本速度收益写成棋力提升。
