# B：第 5–6 步 book 实验模块

对应 `COMP424_Experiment_Plan_Module_Assignments.pdf` 第 6–9 页（5A、5B、5C、6）及第 11–12 页的 B 分工：实现 book 表示与构建、来源与规模比较、覆盖率回放和 N/D/H 使用方式的完整实验入口、配置与小样本自测。正式第 5、6 步由 F 在合并三人模块后按 0→8 顺序统一运行；B 不预先建库或跑大规模对局。`experiments/b_examples/` 只验证实现与接线，所有结果均为 `execution_tier=smoke`，不能用于选定正式 book 或报告效果。

基于仓库提交 `97fc767`（A 的第 0–2 步模块）。不修改原始 `agents/student_agent.py`、A 的搜索/运行工具和 F 的 `evaluation.py`。对局一律使用 A 的 `run_pairs`，单文件一律使用 A 的构建器。

## 实现内容

| 文件 / 入口 | 内容 |
|---|---|
| `agents/book_policy.py` | 线上 book：解码、D4 归一化查询、逆变换动作、合法性检查、hash/版本检查、纯查询计时、N/D/H 钩子 |
| `book_experiments/books.py` | book 文件（JSON，含导出顺序、来源信息和原样嵌入数据）、四种表示编码、生成 provider 模块、嵌套子集、book 变体配置 |
| `book_experiments/collect.py` | 5B.1 D0 采集：自我对弈、greedy、本地 AB；最后完整深度的精确同分动作按 `collection_seed` 均匀选择 |
| `book_experiments/candidates.py` | 5B.1–5B.2 候选池、九个初始归一化状态、共同 Nc、B0/B1 排序及 B2 阶段配额 |
| `book_experiments/analysis.py` | 5B.3 每个 key 只分析一次并跨方案复用；5C.1 统一筛选及拒绝原因 |
| `book_experiments/coverage.py` | 5C.3/5C.4 覆盖率回放；6.2 命中、动作采用、深度、整步耗时、初始化和内存统计 |
| `book_experiments/representation.py` | 5A 三组单项表示比较，并逐一核对所有表示返回相同动作 |
| `book_experiments/exp05_book.py` | 第 5 步入口：5A → 5B → 5C，输出选定的 book 文件 |
| `book_experiments/exp06_book_mode.py` | 第 6 步入口：N/D/H、开发集补齐、全部未命中开销，输出最终配置 |
| `book_experiments/configs/` | `book.json`、`book_mode.json` 正式配置模板（含 smoke 段） |
| `book_experiments/self_test.py` | 单元测试、两步 smoke、续跑检查、两个生成文件的官方验证 |
| `tests/test_book_experiments.py` | 对称正逆变换、编码往返、表示一致性、N/D/H、筛选规则、候选选择、导出顺序、回放、构建 |

## 环境与小样本自测

Python 3.11、NumPy 1.26.4，在仓库根目录运行：

```bash
python -m unittest tests.test_book_experiments -v
python -m book_experiments.self_test --output experiments/b_smoke
```

第二条依次运行：B 的单元测试；A 的 `exp00_prepare --smoke`（得到 manifest 和固定局面）；用模板线上配置 `search_experiments/configs/variant.json` 在 1 个 D1 配置上跑 smoke 无 book 双座对局，代替 F 第 4 步的 D1 轨迹；第 5 步 smoke；同一命令重跑并确认采集对局和分析记录不重复执行；第 6 步 smoke；对选定 book 的 D 变体和最终变体分别构建单文件并通过官方 `validate_submission.py`；确认原始 `student_agent.py` 未改变。smoke 只缩小预算和数量（每步 0.012 秒、深度 2、Nc=12、分析 0.3 秒、规模 3/6），并把筛选最低深度降为 1，仅用于走通所有分支。

`experiments/b_examples/` 是本次实际执行的自测记录（`SELF_TEST_SUMMARY.json`、文本日志、JSON 输出）；逐局生成的代理副本和对手副本可重建，未提交。

## F 正式运行

前置输出：A 第 0 步的 `manifest.json`、`positions.json`；F 第 4 步固定的线上配置（无 book 的 variant JSON，格式同 `search_experiments/configs/variant.json`，也可为 `{"variant": {...}}`）以及 D1 无 book 双座 24 局（`run_pairs` 输出目录，含 `plan.json` 与 `games.jsonl`）。路径写在配置中，也可用命令行覆盖：

```bash
python -m book_experiments.exp05_book --config book_experiments/configs/book.json --output experiments/b_run/book \
    --online-config <F 的线上配置> --manifest experiments/a_run/prepare/manifest.json \
    --references <F 的 D1 无 book games.jsonl>
python -m book_experiments.exp06_book_mode --config book_experiments/configs/book_mode.json --output experiments/b_run/book_mode \
    --selected-book experiments/b_run/book/selected_book.json --online-config <同上> \
    --manifest experiments/a_run/prepare/manifest.json --references <同上> \
    --positions experiments/a_run/prepare/positions.json
python -m search_experiments.build --config experiments/b_run/book_mode/final_variant.json --output experiments/b_run/final
```

搜索与评价参数只从线上配置读取（`search`、`evaluation` 传给 A 的 `SearchEngine` 和 F 的 `make_evaluator`），B 的代码不固定任何搜索或评价参数。D1 轨迹的 `plan.json` 会被核对：搜索、评价、步预算、深度上限和 book 模式必须与线上配置一致。两个入口都可中断后用同一命令续跑；代码、配置或机器变化时拒绝续跑，请换输出目录。

## 第 5 步做什么

1. **5A 基准**：用原始 `student_agent.py` 中的 486 条 book 建立表示基准（结束时对选定 book 再测一次）。三组单项比较都以 D4 + 15 字节键 + zlib 为基准：展开全部方位 / D4（条目数、大小、查询时间）；tuple 键 / 紧凑键（建键与查询时间、解码后内存）；不压缩 / zlib（初始化、大小、解码后内存）。展开时每个实际状态都重新调用同一个归一化函数并逆变换推荐动作，因此自对称状态也得到与 D4 查询相同的动作；任一表示改变返回动作即报错。
2. **5B.1 采集**（只用 D0）：己方用线上配置、无 book、每步 1.6 秒，完整窗口收集最后完整深度的精确同分集合后按每局 `collection_seed` 选择；未完成的深度不提供同分集合。B0 每配置 4 场自我对弈；B1 每配置 2 场自我对弈、1 场 greedy、1 场本地 AB（1.5 秒），greedy 与 AB 的己方座位相反并按名单顺序轮换，12 个配置各 6/6。默认 `b1_reuses_b0_selfplay=true`：B1 的自我对弈就是 B0 每配置的前 2 场，使 B1−B0 只差“24 场自我对弈换成 24 场对手局”，并省去约 1 小时重复采集；设为 `false` 则另采 24 场。
3. **5B.2 候选**：行动前完整非终局状态按 D4 key 去重，频次为整个池中的出现次数；三池均含九个初始状态并计数。Nc 默认 600，不足时统一降到共同可用数。B0/B1 按 ply 升序、频次降序、key 升序取前 Nc；B2 复用 B1 的池，按早/中/后（0–9 / 10–29 / 30–99）均分配额，余数依次给早、中；阶段内频次降序、ply 升序、key 升序；初始状态先占早期名额，早期配额不足 9 时先保留初始状态再按早→中→后轮换补满；某阶段不足时从其他阶段按同一轮换补齐；保存实际阶段分布。
4. **5B.3 分析**：三组候选的并集中每个 key 只分析一次，10 秒预算从解码前开始计时；记录每个完成深度的动作、分值、耗时及终止原因（deadline、solved_remaining、max_depth）。记录带 `analysis_id`（搜索、评价、规则、预算及相关源文件 hash），只复用完全相同的记录。报告各组逻辑预算 Nc×10 秒、去重后实际计算时间和采集成本。
5. **5C.1 筛选**：版本、解码、合法性；立即获胜直接保留；其余须完成至少 3 层、最后 2 层推荐同一动作、不立即自败（揭开对手连线）、对手下一步不能直接获胜。每个 key 记录接受或拒绝原因。
6. **5C.2 共同条目数**：K = min(486, a0, a1, a2)。每组先放通过筛选的初始状态，再按本池频次降序、完成深度降序、key 升序取前 K 条；保存“候选→通过→导出”数量和阶段分布。K=0 时输出 `selected_book=null`，跳过来源、规模及使用方式比较，第 6 步只保留 N。
7. **5C.3 来源**：三本 book 回放 F 的同一份 D1 无 book 轨迹（可查询状态、匹配数、按阶段和障碍类别分布；只有一个合法动作的回合不查询，不计入）；再各以 D 模式在 D1 对本地 AB 跑 24 局。按统一规则（完整且无技术失败 → 得分 → 同分取表中靠前、较简单的 B0、B1、B2）选出来源并保存第二名。
8. **5C.4 规模**：从胜出来源的同一导出顺序取 min(128,K)、min(256,K)、K 的嵌套子集（重复只留一次）；先回放，与上一保留规模覆盖相同的较大规模直接舍弃，其余规模在 D1 以 D 模式跑 24 局（K 复用 5C.3 结果），同分取较小 book。若前两名来源同分，同时导出第二名来源的 k* 条 book 供第 6 步使用。

“发现具体问题后才追加的比较”不自动运行；需要时修改配置中的 `screening.min_depth`、`screening.stable_depths`、`analysis.budget` 或 `candidates.nc`，用新的输出目录重跑。

## 第 6 步做什么

1. 读取第 5 步的 book 文件（核对 hash，必要时重建 provider 模块），D/H 使用完全相同的 book 字节。N 不查询 book；D 命中且合法时直接返回；H 只把合法 book 动作作为第一次迭代的根 preferred，后续迭代使用上一完整迭代的最佳动作。book 不写入 TT。
2. 在 D1 对本地 AB 各跑 24 局；N 复用 F 的 D1 无 book 对局，D 复用第 5 步同一 book 的 D 模式对局（A 的运行器逐项核对 code/config/book hash、种子、机器等，完全相同才复用）。
3. 保留 N 及 D/H 中 D1 得分较高者（同分都保留），在 D0∪D1 补齐 48 局后按统一规则选择提交模式（同分依次优先 N、D、H）。若 H 胜出且第 5 步来源前两名同分，在 H 下比较第二名来源的 k* 条 book；若更好则先补齐其开发集再重新排序。
4. 6.2 指标：命中率 = 匹配次数 / 实际查询次数、至少命中一次的对局比例、匹配/直接返回/排序提示按阶段统计、搜索深度（book 返回或唯一合法动作记 N/A，进入搜索但第一层未完成记 0）、裁判记录的整步耗时、初始化和内存。另取对该 book 全部未命中的固定开发局面，用 A 的 `measure_online_step` 交替比较“开启查询但全部未命中”（D）与“完全关闭查询”（N），单列纯查询耗时和整步完成深度。
5. 输出 `final_config.json`、`final_variant.json`（可直接交给 A 的构建器）；若选 N，另存 `book_comparison_variant.json` 作为第 7 步 V3 的有 book 对照。

## Book 接口

线上配置 `book.provider` 指向生成的模块 `agents.books.<label>`。该模块只含 `BOOK_LAYOUT`、`BOOK_DATA` 和 `make_book_hook(book_config, variant)`，逻辑全部在 `agents/book_policy.py`；A 的构建器会内联两者，生成文件不读磁盘。`agents/books/` 为本机生成目录，不提交；`book_experiments/books.py` 的 `load_book()` 可由 book JSON 原样重建。初始化时核对数据 hash 与配置中的 `hash`/`version`/`entries`，不一致即报错。N 模式返回 `None`，代理完全不查询；D/H 返回 A 约定的字典，`metadata` 含 `query_seconds`、`key_found`、`invalid_entry`。

## 输出

| 输出 | 用途 |
|---|---|
| `book/representation/{current,selected}/` | 5A 表示比较（JSON、CSV） |
| `book/collection/` | 采集计划、每局完整记录（可续跑）、采集成本 |
| `book/candidates.json` | 池大小、Nc、三组候选及阶段分布 |
| `book/analysis/records.jsonl`、`book/screening.json` | 每个 key 的分析记录、筛选结论 |
| `book/books/*.json` | B0/B1/B2 的 K 条 book、规模子集及第二名来源的 k* 条 book |
| `book/games/` | 来源与规模的 D 模式对局（A 的 `run_pairs` 格式） |
| `book/book_summary.json`、`book/selected_book.json` | 第 5 步全部表格与选定 book |
| `book_mode/{d1,dev}/` | N/D/H 对局 |
| `book_mode/mode_summary.json` | 对局、指标、配对差、全部未命中开销 |
| `book_mode/final_config.json`、`final_variant.json`、`book_comparison_variant.json` | 最终配置与 V3 对照 |

## 正式运行时间估计（单机、按本地 AB 对局约 2.5–3 分钟/局估算）

| 部分 | 估计 |
|---|---|
| 5B.1 采集：48 场自我对弈（≤160 秒/局）+ 24 场 greedy/AB | 约 2.5–3 小时 |
| 5B.3 分析：最多 1800 个不同 key × 10 秒（去重后通常更少） | 不超过 5 小时 |
| 5C 对局：3 个来源 + 最多 2 个规模，各 24 局 | 约 4–6 小时 |
| 6 对局：H 的 D1 24 局 + 保留模式在 D0 补齐各 24 局 | 约 3–4.5 小时 |
