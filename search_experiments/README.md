# A：第 0–2 步搜索实验模块

第 0–8 步的串行 smoke、N/D/H 代表性预检、断点恢复及正式运行命令见 [RUN_EXPERIMENTS.md](../RUN_EXPERIMENTS.md)。Smoke 入口为 `python -m search_experiments.smoke_pipeline`，预检入口为 `python -m search_experiments.acceptance`。两者分别保存 `pipeline_summary.json` 和 `acceptance_summary.json`，仅使用开发集开局。

对应 `COMP424_Experiment_Plan_Module_Assignments.pdf` 第 11–12 页的 A 分工：实现完整实验入口、共用工具、配置与小样本自测。正式第 0–8 步由 F 合并三人的模块后统一运行。本目录附带的小样本输出仅验证实现与接线，不能用于确定正式 S* 或报告棋力提升。

基于仓库提交 `a407698955bb2ea9931c1264b2d0b7d0f8db479f`。保留原始 `agents/student_agent.py`，也保留 `evaluation.py` 和 `symmetry.py`。统一的搜索接口调用 F 的评价工厂及 B 的坐标变换函数。

## 实现内容

| 文件 / 入口 | 内容 |
|---|---|
| `agents/search_engine.py` | C0–C3、S0–S4 的公共调用、搜索结果与诊断汇总 |
| `agents/experiment_agent.py` | 整步计时、合法 fallback、统一入口、F 评价工厂与 B book hook |
| `search_experiments/exp00_prepare.py` | 48 配置的 D0/D1/T 名单、双座严格对局、参考轨迹及分层局面抽样 |
| `search_experiments/exp01_components.py` | 精确根值及最优动作集合、组件比较、冷启动与完整迭代计时 |
| `search_experiments/exp02_search.py` | 逐项尝试优化、E0/E1 正确性、固定深度与限时比较、开发集确认及配置输出 |
| `search_experiments/exp03_evaluation.py` | 固定 S*，用 D0 筛选评价参数，再在完整开发集确认 E* 和 E_alt |
| `search_experiments/exp04_recheck.py` | 复查搜索与评价组合，冻结 N 版本并生成 B 共用的 D1 参考 |
| `search_experiments/exp07_final.py` | 读取冻结选择，验证实际提交文件，运行最终对局并保存结果 |
| `search_experiments/runner.py` | 三人共用的唯一对局循环、完整快照、断点续跑、严格复用条件、配对统计 |
| `search_experiments/measure.py` | 固定局面测量、整步测量、计时 CSV 与单独诊断记录 |
| `search_experiments/build.py` | 按模块依赖自动生成允许导入的单文件，附构建来源和 SHA-256 |
| `search_experiments/configs/` | 准备、组件、搜索、评价、组合复查及单个线上版本的 JSON 模板 |
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

生成新名单时先保留官方验证器默认的两种开局。`helpers.new_game(0)` 的 `r3c1_M` 固定进 D0，`helpers.new_game(1)` 的 `r1c3_M` 固定进 D1，各使用一个 M 边缘名额。其余格子按 `manifest_seed` 在原分层配额中打乱。JSON 保存预留表、规则版本与实际 setup seed；名单散列覆盖这些元数据和所有行。已有目录里的不同名单会被拒绝；历史示例保留原貌。只生成新名单、不开对局，可在仓库根目录运行：

```bash
python -m search_experiments.exp00_prepare \
  --config <(echo '{"manifest_seed":20261125,"run_reference":false}') \
  --output experiments/a_run/prepare
```

第 0 步先对本地 AB 跑原始 S0/E1/N 的 D0 双座 24 局。逐步保存双方行动前的完整状态；按障碍类 × 移除大小 × 阶段，从不同对局各抽 1 个，目标每组 2 个、共 54 个。缺少的组照实写入 `coverage`，不借用其他组或重复相邻状态补齐。阶段为 0–9、10–29、30–99，另有 90–99 标记。

第 1 步固定 E1、关闭 book。C0 完整搜索每个根动作，保存精确值及全部真正最优动作。C1–C3 的根值应相同，返回动作应属于该集合。每个根/目标深度重新清空 TT；关闭排序时也关闭 preferred 提升。冷启动直接搜索到深度 d；完整迭代测量执行 1…d 并保留该次调用的迭代提示/TT。每种计时交替顺序重复 3 次，详细计数另外运行。默认共同保护时限 60 秒；超时项写为未完成，不当作精确参考，也不进入跨版本共同完成项的耗时比较。

第 2 步依次尝试 S1、S2、S3、S4。每项只在此前保留的父配置上改一个开关，并写出实际 `parent_variant`；未保留的修改不会带到下一项。正式运行读取第 1 步保存的 E1 精确参考，核对请求和测量日志散列，并保留当时未完成的项；E0 单独计算精确参考。在 E0/E1 下核对候选的固定深度结果，再比较 E1 的完整迭代耗时和 1.6 秒整步预算下完成深度。近 100 ply 和已搜完剩余深度的状态单列。三次重复均胜出且预算/正确性通过才保留；排名反复则保留父版本。正式比较要求 S0、实际父版本和候选至少在一个共同局面上都有三次限时观测。完成深度相同时，用已核对的共同固定深度耗时打破平局。缺少所需证据，或 S0、最终保留版本曾违反整步预算或硬时限时，写出 `blocked_performance` 并跳过开发对局。违反时限的候选不能作为备选。

随后 S_diag 与 S0 补齐 D0∪D1 各 48 局。`reference_games` 指向准备阶段日志，完全相同的已有 D0 记录可直接复用；校验代码、配置、book、生成代理、对手、seed、机器、规则和预算，任一变化都不复用。完整开发名单结束后按得分选择 S*，同分保留原始 S0，并输出 S_alt。S_diag 就是 S0 时，备选从已验证候选按完成深度、相对同一 S0 的耗时比及候选顺序选择。

只有 `search_selection.json` 的 `status=selected` 才有正式 S*。此时读取 `selected_search.json` 和 `alternate_search.json` 交给 F 的第 3 步。`smoke_only` 或 `awaiting_development_confirmation` 不代表正式选优；`S_star` 保持 null。正式第 2 步拒绝 smoke 局面或错误/来源不一致的第 1 步结果。

每个请求的深度和负载都须有至少一个共同局面，且每个受比较版本在该局面的三次独立计时都完成并通过精确参考。只有迭代负载排除 C0。`no_mismatches` 记录是否发现已核对的错误，`evidence_sufficient` 记录各组是否都有共同证据。`verification_complete` 另核对全部计划的固定深度计时和诊断。部分超时允许继续；任一组为空或已检查行（含诊断）的根值/动作有错时，正式第 2 步写出 `blocked_correctness` 并跳过开发对局。`positions.json` 保存参考对局的完整性、资格、代码和名单散列。

父子性能比较每次重复使用相同的局面集合，两方都须满足预算。开发对局保留官方裁决和得分。出现技术失败，或候选的任一次整步调用超过其配置预算时，该候选失去入选和备选资格。仅有一个合格版本时，备选指回该版本并记录原因。`selected_search.json` 和 `alternate_search.json` 保留实际参赛候选的 ID 及父版本。

`common.deduplicate_variants(variants)` 返回按首次出现保留的原始对象和别名到原始 ID 的映射。例如 E* 或 V2 可指向同一候选。行为判断采用实际搜索配置、有效 book、固定规则、预算、深度上限和日志开销。N 模式按实际 λ/q 比较评价，λ=0 时忽略 q，并忽略未启用的 book。D/H 模式保留 provider 可见的完整规范化评价配置。运行器独立核验原始配置与其余来源字段，角色映射本身不能用来复用对局。

## 第 3 步评价筛选

```bash
python -m search_experiments.exp03_evaluation --config search_experiments/configs/evaluation.json --output experiments/f_run/evaluation
```

正式入口需要同一份冻结名单及当前代码生成的第 2 步 `search_selection.json`、`selected_search.json` 和 `alternate_search.json`。可在 `reuse_paths` 添加已有的对局日志。第 2 步开发日志也会交给共用对局运行器，复用条件以运行器核对的对局身份为准。

第 3 步固定第 2 步的 S*，关闭 book。先用 D0 双座比较五种 λ 及最优非零 λ 的四种 q，再补齐前两种非零 λ 与 Q0、最优 q 的组合。初始筛选有八个不同候选，组合阶段至多新增一个。E1、E0 与 D0 得分最高的合格新评价在 D0∪D1 双座确认。E* 和 E_alt 按完整确认得分挑选，同分优先 E1、E0，再按候选创建顺序。

输出 `evaluation_selection.json` 记录各阶段结果、候选原 ID、名单与代码散列，以及 D0 和确认日志路径。`status=selected` 且 `formal_selection=true` 时，两个同目录角色文件才是正式交接配置。必要对局未完成时，沿用相同请求和输出目录补跑；改动输入或执行层级时使用新目录。显式添加 `--smoke` 会使用每个开发 split 的前 `max_configs` 行和缩减后的预算与深度，结果标记 `smoke_only`。

## 第 4 步组合复查和 D1 交接

```bash
python -m search_experiments.exp04_recheck --config search_experiments/configs/recheck.json --output experiments/f_run/recheck
```

第 4 步需要当前代码生成的第 2/3 步选择及同目录角色文件，以及第 2 步使用的固定局面与冻结名单。对 S*、S_alt 与 E1、E* 的组合按实际行为去重。每种不同评价分别生成精确参考，使用共同局面做三次交替计时与单独诊断。全部评价都要通过固定深度正确性检查。只有 S_alt 在 E* 下三次都胜出且满足预算，才对两种搜索补齐开发集双座对局。对局同分保留 S*。如果 S_alt 胜出，则只在它上面复查 E* 和 E_alt 一次，按第 3 步的同分规则确定评价。

`recheck_selection.json` 保存原始候选 ID、测量别名、各组证据、选择理由和 D1 结果。共用运行器可从第 2/3 步、复查阶段及 `reuse_paths` 复用身份完全一致的对局。最终候选须使用同一个 N 配置在 D1 的 12 个配置各跑双座。只有计划中的完整配置与 24 条合格裁决记录一致才发布 `online_config.json`。B 使用这个配置和同目录的 `d1_reference/games.jsonl`，并读取紧邻日志的 `plan.json`。必要对局未结束时沿用相同请求续跑。变更输入、代码、机器或执行层级时使用新输出目录。

显式 `--smoke` 按缩减的固定局面、开发名单与预算运行。线上计时默认使用 0.03 秒，对局默认使用 0.012 秒候选预算和深度上限 2。小样本交接只供接线检查，标记 `smoke_only` 和 `formal_selection=false`。

## 第 7 步最终测试和提交文件

正式运行需要 Python 3.11、NumPy 1.26.4，以及由当前代码和同一冻结名单生成的第 2、4、5、6 步正式选择。默认输入路径见 `search_experiments/configs/final.json`。

```bash
python -m search_experiments.exp07_final --config search_experiments/configs/final.json --output experiments/f_run/final
```

V0 是第 2 步的 S0/E1/N，V1 是第 4 步实际采用的搜索配 E1，V2 是第 4 步冻结的线上 N 配置。V3 是第 6 步选定的 D/H 提交版本，或选择 N 时具备完整 D0+D1 对局证据的 D/H 对照。只有第 5 步确认 `K=0` 时省略 V3。此时 `final_plan.json` 的请求和 `final_summary.json` 都将 `v3_omission` 记为 `{"reason": "no_common_book", "K": 0}`，保留 V3 时该字段为 null。行为相同的角色共享一次构建和对局，保留 V2 的原始身份。

每个候选都先通过官方静态及严格沙箱验证。官方四局用 seed 0、1，各交换座位。再在名单中预留的同两种开局对本地 AB 跑四局，核对逐局资源记录。要求无技术失败和违规，整步最多 2 秒，初始化最多 30 秒，实测峰值 RSS 低于 500 MB。提交副本单独做静态验证，并核对字节数和 SHA-256。全部候选通过才开放 T。正式测试每个候选对本地 AB 跑 24 配置双座共 48 局，候选预算 1.6 秒，AB 预算 1.5 秒。提交候选还在同一名单分别对 random 和 greedy 各跑 48 局。所有对局记录对手 RNG seed 和生成文件的 hash。技术裁决与原始记录一起保留。提交版本由开发阶段冻结，最终测试结果只供报告。

小样本使用一份指向同级别 smoke 交接文件的配置和单独的输出目录。

```bash
python -m search_experiments.exp07_final --config /path/to/smoke-final.json --output experiments/f_smoke/final --smoke
```

小样本默认取 D0 和 D1 各第一种配置，`smoke.max_configs` 指定每组 1–11 种配置。候选预算、深度上限和 AB 预算与上游 smoke 设置一致。两道验证使用预留的开发开局，所有对局禁用 T，输出标记 `smoke` 和 `formal_selection=false`。

| 输出 | 内容 |
|---|---|
| `final_plan.json` | 输入和代码散列、机器、原始角色与去重映射、名单、对手及构建元数据 |
| `candidates/<variant>/student_agent.py` | 候选验证文件及相邻 SHA-256 和来源元数据 |
| `validation/<variant>/` | 官方 JSON 和文本报告，严格 AB 计划、逐局记录与通过状态 |
| `matches/<variant>/<opponent>/` | 最终计划、原始裁决和汇总 |
| `submission/student_agent.py` | 提交副本和同目录静态报告 |
| `final_summary.json` | 完成、待续跑或阻塞状态，验证和对局结果，提交路径、字节数及 SHA-256 |

中断后用同一配置和目录续跑。输入、代码、机器、预算或名单改变时使用新目录。发布结果前逐项复核保存的计划、实际候选与对手文件、双方座位及种子、裁判裁决和原始记录，并据此重算汇总。每组最终对局都记录计划、JSONL 及汇总文件的 SHA-256。基础设施故障留作重试，已裁决对局保留。普通终局的胜者及结束原因须与最后一步后的盘面一致。技术结果保留裁判裁决。

相同请求及候选文件复用 `official.json` 中已完成的通过或失败结论。如在官方验证写入四局汇总或明确的加载失败后中断，恢复时从已有汇总确定结论。缺少文本报告时从该 JSON 恢复。中断或基础设施故障发生时若缺少完成判定所需的记录，可重试验证。

## 搜索开关

所有 C0–C3、S0–S4 的开发入口和生成单文件均通过 `SearchEngine` 调用同一套源码：`agents/alpha_beta.py`、`agents/move_ordering.py` 和 `agents/iterative_deepening.py`。S0/E1/N 使用上传版的递归、先完整构造再排序、同深度 TT 与超时提交规则。原上传 `agents/student_agent.py` 的 SHA-256 为 `2e9c9dc5a021f615620f8844fbeb3ebdb7b99ae037d9a7bacf3daf7ef3116fed`，可直接用其模块级函数对照。实验整步统一计入框架合法着法、状态转换、搜索和报告，并为返回/报告预留最多 2 ms。公共适配也有开销，固定深度核对不代表逐次限时耗时相同。

N 模式跳过书工厂和 provider。H 的根提示只进入首次迭代，后续使用上一完整层的最佳动作。在线每步创建新 TT，各深度共用。离线请求完整同分集合时，对每一根动作使用完整子窗口，且只保留最后完整、按时返回的一层。诊断只在整次搜索结束后统计 TT 内存。

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

`my_evaluate(state, player)` 返回该玩家视角的非终局分数，并满足双方视角互为相反数。终局先检查，终局效用由 `evaluation.terminal_utility` 处理。一个 SearchEngine 使用固定评价实例，每次在线搜索重新创建 TT；不要在同一搜索中途更换参数。

使用 S4（`symmetry=True`）时，对任何保持障碍格不动的棋盘变换，评价函数都必须给同一玩家相同的分数。S4 会让变换前后的局面共用 TT 项。P1 提供的评价函数都满足这一条件。若自定义函数只给某个特定坐标加分，导致变换后得分不同，应设 `symmetry=False`。

`agents/evaluation.py` 提供工厂：

```python
def make_evaluator(config):
    # config includes evaluation_id, lambda and q.
    # Return a callable evaluate(state, player) using these fixed parameters.
    ...
```

配置例如 `{"evaluation_id":"lambda5_Q1", "lambda":5, "q":[1,50,100]}`。A 的开发工具和构建器都会自动调用此工厂，叶点评价与动作排序共享同一 callable。评估参数在创建工厂实例时校验。

F 第 3/4 步可以直接复用：

```python
from search_experiments.runner import run_pairs
from search_experiments.measure import measure, measure_online_step
```

`run_pairs(variant, manifest_rows, output, opponent={"kind":"ab","budget":1.5})` 负责全部严格对局、快照和恢复。`variant` 使用配置模板格式，参数改变时设置新的 `variant_id`、`parent_variant`。先用 `validate_manifest(manifest)` 取得带完整名单散列的行，再取 D0 或 D1；最终测试才显式传 `allow_test=True` 并使用 T。F/B 无需重写对局循环。

每个 `config_id` 只能调度一次。运行器在构建代理或复用记录前按 `setup_seed` 核对障碍格和移除尺寸。正式候选预算固定为 1.6 秒，AB 对手预算固定为 1.5 秒。缩短对局预算时使用 `execution_tier="smoke"`。构建 AB 时预算须为有限的非负数，可为 0。

开发计时 `measure()` 拒绝标记为 T 的局面。独立入口 `python -m search_experiments.measure` 默认要求由完整、合格且来源匹配的正式参考对局生成的局面。启用整步计时时，正式预算为 1.6 秒。小样本输入使用 `--smoke`，请求及测量行均标记为 `smoke`。

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

B 在自己的 hook 中实现 N/D/H、版本检查、纯查询计时及具体 book 行为。A 检查动作合法性并执行直接返回/根 preferred 的共用入口；书不会写入搜索 TT。后续迭代使用自身上一完整迭代动作。默认 N 且无 provider 时完全不查询。遥测分别记录查询、匹配和采取的动作。搜索按书的提示选取首个根子节点时记录 `book_action='hint'`，即使该次迭代随后超时。book/唯一合法动作返回的深度是 null，搜索但无完整第一层是 0。

Book 的加载发生在对象初始化阶段。构建器支持内联直接写在模块顶层的 `from agents.module import symbol`（可加 `as alias`）和 `import agents.module as alias`。相对导入、包级导入以及写在条件语句或函数等作用域内的开发模块导入都会在构建时被拒绝。可写 `from agents.evaluation import evaluate_e0 as line_score`。顶层名称的定义或导入指向不同对象时，构建器会拒绝覆盖。B 负责将运行时 book 数据放在模块常量中；生成文件不能依赖磁盘 book 文件。

离线分析直接用同一 SearchEngine，不调用 book hook，设置新的绝对 deadline。需要真实同分动作集时使用 `fixed_depth(..., collect_ties=True)` 或 `search(..., collect_ties=True, tie_rng=random.Random(collection_seed))`。该模式对每个根动作使用完整窗口，仅从最后完整深度的精确最优集合选择，超时不提交部分集合。

B 的 miss/命中整步成本可使用 `measure_online_step(state, variant)`，它计时真实 `ExperimentAgent.step()`，自动加载配置中的评价及 book 工厂。初始化成本单独记录为 `object_init_s`；正式初始化/RSS 使用严格对局的 referee 记录。

测量运行期间请保持 provider 源码固定。修改 provider 后，请在下一次测量前重启 Python 进程或使用不同的模块名。

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

每局记录 `variant_id`、`parent_variant`、code/config/book hash、生成文件及其实际输入源码的散列、对手与 seed、机器、split、座位及官方初始化/step/RSS。裁判在每步记录中将整步耗时 `referee.moves[*].agent_time` 保留六位小数。`candidate_max_step_s` 保存候选实际返回的最大未舍入耗时，超出 `candidate_step_budget` 时标记 `candidate_budget_failure`。没有候选调用返回耗时时，最大值为 null。正式请求和每局都保存原始 48 配置名单的散列。相同名单内可从 D0 扩到 D0∪D1；名单不同或旧记录缺少散列时不能复用。轻量搜索信息通过受官方大小限制的 agent 日志带出，详细节点统计在独立诊断中测量。

重复同一命令时，只跳过身份完全相同且已由官方 referee 判定结束的对局。超时、崩溃、初始化失败等按官方裁决进入 W/D/L；没有有效裁决的运行故障保留 `infrastructure_error` 并在续跑时重试。未完成计划名单不输出候选总得分或排名。候选或对手出现技术失败时，记录结果但不用于选优，先修复对应实现。

输出目录的代码、配置或机器身份变化时，恢复过程会报错；另选输出目录保留旧结果即可。不要覆盖已有正式实验日志。固定局面测量将每个局面/深度的参考完成状态、根值和最优动作纳入请求身份。恢复时核对请求文件内容、请求散列和每行的 `request_hash`。非空日志缺失匹配的请求文件时须使用新目录。机器身份包括主机名、解释器及环境路径、父进程与严格子进程的线程环境。未完成搜索仍是未完成项；若需要增加保护预算，使用新的目录。

固定局面测量的版本执行顺序、对局复用来源的优先顺序也属于各自的请求身份。改变顺序时请使用新目录。

配对差以一个配置的两座位平均为单位计算，给出改善/持平/下降配置数及配置层面的 bootstrap 区间。重复座位或对称方位不作为独立 Bernoulli 样本。正式耗时、完成深度、得分分别报告；未运行正式开发对局时，不把小样本速度收益写成棋力提升。

## 第 8 步报告

在生成第 7 步证据的 Python 环境、主机、源码及原始路径下运行。校验器会在临时目录重建候选和各 seed 的对手，以散列核对留存文件。第 8 步只读输入，不启动参赛进程、不选择候选。正式入口的默认配置指向第 7 步汇总和第 1 步独立计数目录。

```bash
python -m search_experiments.exp08_report --config search_experiments/configs/report.json --output experiments/f_run/report
```

开发阶段使用单独的配置和输出目录，把 `final_summary` 指向同一轮 smoke 的文件，并在命令末尾加 `--smoke`。只有完整、通过校验的 formal 赛程可产生 `formal_results=true`。同一最终计划恢复后，可以在同一报告目录重新生成；每次的 `report_summary.json` 保存当次输入及导出文件散列。证据来自不同主机、绝对路径已改变或源码已改变时，当前校验器无法重建该快照。

- `results.csv` 每个实际候选与对手仅一行。汇总给出计划及已裁决局数、W/D/L、完整名单得分、已观察均分、资格及技术失败。合法的超时、崩溃和罚负仍计入实际得分；基础设施重试单列。缺失比赛描述的组显示 pending。汇总 JSON 的 `games` 保存每局配置、座位、终止原因、技术标志、测得的初始化、RSS 及全局最大候选调用耗时。
- `paired_differences.csv` 给出 V1−V0、V2−V1、V3−V2 对共同 AB 的配置差与总体差。每个配置先取两座均值，然后抽取整配置作 2,000 次 bootstrap，默认种子 20261125。缺座时完整名单分数与区间缺失。别名标明共享比赛，K=0 保留 V3 省略原因。完整但无资格的对局保留实际配对差。
- `game_strata.csv` 分解障碍类别、被移除棋子尺寸、座位、配置、开发/测试分组及裁判终止原因。`turn_strata.csv` 展开阶段、双方储备总量、行动方角色和行动方储备量，另给出阶段 × 总储备 × 角色的联合计数。每组分别列出局数和观察步数。同局内以完整行动前局面（除 ply 外的全部字段）统计第二次及后续出现，排除末步后的局面。实际落子或移动由 (player, src, dest, size) 标识，pass 和失败回合单列。100-ply 和局需要 winner=0、reason=ply_cap 且 plies=100。
- `turns.csv` 的 step_s 取自裁判已走步的 agent_time。罚负调用可能未出现在棋谱中；每局的 candidate_max_step_s 仍保留该次计时。深度、停止原因、被中断深度和原始状态分别记录。唯一动作、书直接返回、搜索前超时的报告深度均为 N/A；已进入搜索但首层未完成的深度为 0。缺失遥测、字段缺失、显式 null 和不适用分别有状态标记。旧日志不能确定是否进入搜索的 fallback 保留原始深度。
- `book_metrics.csv` 分列冻结 D1 无书轨迹上的潜在覆盖率与最终比赛的真实查询、合法命中及直接返回/提示。实战命中率的分母是查询确有记录且命中标志已知的查询；同时列出查询和命中标志缺失数。记录提示不保证最终实际走了提示动作。JSON 用 null，CSV 用 N/A。
- `diagnostics.csv` 来自配置中 `diagnostic_dirs` 明确选择的独立目录。每个条目可为目录字符串，或含 `directory`、`positions` 及可选 `max_positions` 的对象。后者指向该次固定局面文件和实际使用的前缀长度，校验实际列表摘要后可精确计算缺失诊断数。仅有局面摘要时，完整性未知。普通计时行不提供节点、剪枝或 TT 计数。每个独立计数行保存请求及原始日志路径和散列、工作负载、目标深度、达到状态和缺失计数器。中断的计数是部分工作量。
- `plots` 默认开启。存在完整分数或差值时分别生成 `scores.png`、`differences.png`。smoke 图标题明确标注 Smoke。
