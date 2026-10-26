# vla_nav —— 阶段 A：语言 → 导航

把「人话指令」变成 Nav2 目标点的最小闭环。不依赖任何大模型也能跑（规则匹配），
配了 LLM key 后可以处理更口语化的说法。

```
用户说 "去A区"  →  llm_nav.py  →  semantic_map.yaml 查坐标  →  /navigate_to_pose
```

## 文件

| 文件 | 作用 |
|---|---|
| `semantic_map.yaml` | 地名词典：地名 → (x, y, yaw)。**你要按自己的地图改这里** |
| `llm_nav.py` | 指令解析 + 发送 Nav2 目标 |
| `record_place.py` | 在 RViz 里点一下，把坐标写进 `semantic_map.yaml` |
| `eval_nav.py` | 导航批量评测：成功率 / 用时 / 终点误差 / recovery 次数 → CSV |
| `semantic_map.py` | 语义地图读取（有 PyYAML 用 PyYAML，没有则内置极小解析器） |
| `test_parse.py` | 解析层单测，23 条断言，**不需要 ROS 也不需要 pyyaml** |

## 安装（在虚拟机里执行一次）

```bash
mkdir -p ~/vla_nav
curl -fsSL <你拿到的下载地址> | base64 -d | tar xz -C ~/vla_nav
ls ~/vla_nav
```

或者直接把这三个文件拷进 `~/vla_nav/`。

> 脚本假设语义地图在 `~/vla_nav/semantic_map.yaml`，用 `--map` 或环境变量 `SEMANTIC_MAP` 可改。

## 用法

**1. 先看有没有读对地图**

```bash
cd ~/vla_nav && python3 llm_nav.py --list
```

**2. 用 RViz 录地名（推荐）**

```bash
python3 record_place.py            # 终端挂起等待
```
RViz 工具栏 → `Publish Point` → 在地图上点目标位置 → 回终端输入地名（如 `A区`）→ 回车。

**3. 下指令导航**

```bash
python3 llm_nav.py "去A区"
python3 llm_nav.py "带我回起点"
python3 llm_nav.py --raw 1.5 0.0 0.0     # 跳过语义解析，直接发坐标
python3 llm_nav.py --dry-run "A区左边1米"  # 只解析不发目标（离线可跑）
```

### 支持的说法

| 类型 | 例子 | 结果（A区在 (1.5, 0)，yaw=0） |
|---|---|---|
| 纯地名 | `去A区` | (1.5, 0) |
| 相对朝向·左/右 | `A区左边1米` | (1.5, **+1.0**) |
| 相对朝向·前/后 | `A区往后1米` | (**0.5**, 0) |
| 绝对罗盘 | `A区北边3米` | (1.5, +3.0) |
| 对角线 | `A区西北2米` | (0.09, 1.41) |
| 多步 | `先回起点然后去B区` | 依次发两个目标 |
| 只有方向 | `A区后面` | 默认 1 m |
| 只有距离 | `走3米` | 按"往前 3 m"（基准默认起点） |

距离支持 `2米` / `1.5m` / `200cm` / `两米` / `半米` / `十五米`。

**两个语义约定（很重要）**：

1. **左/右/前/后 相对该地名自己的朝向** —— 即"站在 A 区、朝 A 区 `yaw` 的方向看"的左右前后。
   所以给地名配 `yaw` 不只是"到达后朝哪"，也决定了"它左边是哪边"。
2. **东/南/西/北 是 map 帧绝对方向**（`+x = 东`，`+y = 北`），与地名朝向无关。

输出会告诉你走到了哪一步：

```
[MAP] /home/xs/vla_nav/semantic_map.yaml  frame=map  places=5
[INSTRUCTION] 去A区
[PARSE] -> A区   (rule-exact)   target=(1.50, 0.00, 0.00)
GOAL_ACCEPTED  (1.50, 0.00, yaw=0.00)
RESULT SUCCEEDED
```

## 评测：把"感觉"换成数字

`eval_nav.py` 反复发目标并记录指标，是后面所有参数调优的依据。

```bash
# 单点重复 5 次
python3 eval_nav.py --runs 5 --goal 1.5 0.0 0.0

# 按语义地图里的地名评测（可多个 --place）
python3 eval_nav.py --runs 5 --map semantic_map.yaml --place A区 --place 走廊尽头

# 落盘 CSV，方便画对比图
python3 eval_nav.py --runs 3 --goal 1.5 0.0 0.0 --csv ~/eval_1026.csv
```

输出：

```
=== 目标 A区  (x=1.50 y=0.00 yaw=0.00) ===
[1/5] SUCCEEDED  12.4s err=0.08m recov=0
...
  runs=5 success=5 (100%)  mean=12.6s  p90=15.1s  err=0.09m  recoveries=0
```

指标口径：

| 指标 | 含义 | 为什么重要 |
|---|---|---|
| `success` | action 返回 `STATUS_SUCCEEDED` 的比例 | 最硬的指标 |
| `mean/p90` | 墙钟耗时 | 用户感知的"快慢" |
| `err` | 过程中 `distance_remaining` 的最小值 | 反映是否真的贴近目标 |
| `recoveries` | 行为树里 recovery 节点触发次数 | **数值高 = 频繁卡住**，这是"慢"的常见真因 |

> `recoveries` 依赖 `/behavior_tree_log` 话题；Jazzy 的 bt_navigator 默认开启，
> 拿不到时会退化为 0，不影响其它指标。

## 跑测试

```bash
python3 test_parse.py
```

不需要 ROS、不需要 pyyaml（缺 pyyaml 时 `semantic_map.py` 自动用内置解析器）。
输出 `ALL PASS` 即通过；改解析逻辑后先跑它，比在机器人上试错快得多。

## 可选：接 LLM 处理更口语化的指令

规则匹配只能命中"地名/别名"。想处理"帮我把快递放到门口那边"这种，
需要配一个 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1     # 用 DeepSeek 时
export LLM_MODEL=deepseek-chat
python3 llm_nav.py "去那个堆放货物的地方"
```

没配 key 时会自动退回规则匹配（`--no-llm` 可强制只用规则）。

## 设计说明

- **解析三层兜底**：地名精确包含 → aliases 命中 → LLM 语义解析。前两层离线可用，
  所以没网也能演示。
- **坐标系**：点位来自 RViz 的 `Publish Point`，已经在 `map` 帧下；`yaw` 是到达朝向，
  0 表示朝 +x。
- **失败可见**：`GOAL_REJECTED` 通常是目标点在障碍/未知区，或 AMCL 没位姿；
  `NO_ACTION_SERVER` 是 nav2 没起。

## 下一步（阶段 A 的延伸）

1. 给 `semantic_map.yaml` 补更多地点和别名，覆盖你常说的说法
2. 加"相对方位"支持：`python3 llm_nav.py "走到起点左边两米"`
3. 接 VLM：把相机图 + 指令一起喂模型，做「找红色箱子」这类视觉接地
4. 失败回退：目标不可达时切到探索模式（Frontier exploration）
