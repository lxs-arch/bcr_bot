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
| `grounding.py` | A2/A3 视觉接地：指令 → 检测框 → 深度反投影 → `map` 坐标 → 导航目标 |
| `test_parse.py` | A1 指令解析单测（表驱动，含真实 yaml） |
| `test_grounding.py` | 几何层：四元数、投影、深度中位数、bbox、落点、CLI |
| `test_pipeline.py` | 主干链路 `locate_object()`：检测+深度+外参 → 落点（含失败路径） |
| `test_detector.py` | 检测器契约：stub / 可选依赖降级 / GroundingDINO 适配层（假 processor 离线验证） |
| `test_ros_glue.py` | ROS 粘合层：`_image_to_array`（编码/通道/毫米转米）、内参、`GroundingNode.locate`、`main()` 全流程（假 rclpy，无需真 ROS） |
| `_testkit.py` | 让同一份断言既能被 pytest 收、也能在没有 pytest 的机器上直接跑 |
| `run_tests.py` | 统一测试入口：语法检查 + pytest（或 standalone）+ 汇总 |

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

一条命令全跑（**自动选 runner**：装了 pytest 用 pytest，没装就逐文件 standalone）：

```bash
python3 run_tests.py              # 语法检查 + 全部单测
python3 run_tests.py --pytest     # CI 里强制 pytest（没装直接报错）
python3 run_tests.py --standalone # 模拟机器人机：完全不用 pytest
python3 run_tests.py -v           # 逐条打印
```

单独跑某个文件两种方式都行：

```bash
python3 test_parse.py            # 零依赖
python3 -m pytest test_pipeline.py -q
```

| 文件 | 覆盖什么 |
|---|---|
| `test_parse.py` | 地名/别名、相对方位、罗盘、距离写法、多步、失败提示、真实 `semantic_map.yaml` |
| `test_grounding.py` | 四元数/位姿、投影反投影、深度中位数、bbox、落点、地面求交、CLI dry-run |
| `test_pipeline.py` | `locate_object()` 端到端：只取框内像素、坏像素过滤、鸭子类型检测器、min_score/min_area/min_depth/max_depth、落点策略、无 ROS 报错 |
| `test_detector.py` | stub 后端、可选依赖缺失时的 `SystemExit` + pip 提示、`make_detector` 参数过滤、GroundingDINO `detect()` 的 prompt/阈值/target_sizes/空结果（PIL 与 ndarray 两种图像形态） |
| `test_ros_glue.py` | `--stub-box` 注入框走完 main、`--frame` 换目标帧、`_image_to_array`（32FC1/16UC1 毫米转米/rgb8/bgr8/未知编码报错）、`_intrinsics_from_info`、订阅 QoS（sensor data / `--qos reliable`）、`ready()` 门控、`locate()` 端到端（TF + 中位深度 + 落点）、`main()` 的 0/2/3/4/5 退出码与 `--navigate` 接线 |

都不需要真 ROS、不需要 GPU、不需要 pyyaml（缺 pyyaml 时 `semantic_map.py` 自动用内置解析器）。
`numpy` 是可选依赖，但**装上才能跑 4 类 ndarray 用例**（深度图/图像以 ndarray 进场的路径，
以及 `test_ros_glue.py` 整个文件）——没装时这些用例打印 `SKIP` 而不是失败；CI 里必装，
否则真机路径的缺陷会静默漏过（历史上就漏过一次，见 C.6）。输出 `RESULT: PASS` 即通过。

**CI**：`.github/workflows/vla_nav-tests.yml` 在 GitHub Actions 上装 `pytest pyyaml numpy` 后
分别跑 `--standalone` 与 `--pytest`，任何推送到 PR 都会触发。

> 设计取舍：断言只写一份（普通 `assert` + 模块级 `test_*` 函数），
> 由 `_testkit.py` 提供 standalone runner。这样机器人机没 pytest 也能跑，
> 不会出现"脚本里一套断、pytest 里另一套断"的两份真相。

## A2：视觉接地（指令 → 画面里的物体 → map 坐标）

A1 把「地名」映射成坐标，前提是**坐标你已经录过**。A2 解决另一半：
指令里只说了物体的样子（"红箱子"），没说是哪个地名。

```
"红箱子"
  → [检测器] bbox + score           ← 可换后端；默认 stub，真模型用 GroundingDINO
  → [深度] bbox 内有效深度中位数     ← 纯函数（过滤 0/NaN/超距）
  → [反投影] 像素 + 深度 + 内参 → 相机系 3D
  → [外参] 相机系 → map（TF）
  → [落点] 目标点沿视线退 standoff 米 → 导航目标
```

**先离线验证几何链**（不需要 ROS、不需要 GPU）：

```bash
python3 grounding.py --dry-run \
    --box 300 200 380 300 --depth 2.0 \
    --intrinsics 600 600 320 240 \
    --camera-pose 1 2 0.3 0 --robot 0 0
```

输出四行，逐级可对数：

```
[BOX] center=(340.0, 250.0) depth=2.00m
[CAM] point=(0.067, 0.033, 2.000)
[MAP] object=(1.067, 2.033, 2.300)
[GOAL] approach=(0.695, 1.325, yaw=1.088) standoff=0.80m
```

**在机器人上真跑**（需要相机 + 深度 + TF）：

```bash
# 只算坐标，不导航
python3 grounding.py "红箱子"

# 算完直接发 Nav2 目标，停在物体前 0.8 m
python3 grounding.py "红箱子" --navigate --standoff 0.8

# 换真模型（见下节 A3；需 pip install torch transformers pillow）
python3 grounding.py "红箱子" --detector dino --navigate
```

默认话题：`/bcr_bot/kinect_camera/color/image_raw`、`.../depth/image_raw`、
`.../color/camera_info`，用 `--color-topic / --depth-topic / --info-topic` 改。

订阅默认用 **sensor data QoS（best_effort）**：相机/深度图基本都用 best_effort 发布，
用默认的 reliable 订阅与发布方不兼容，症状是「一帧都收不到、15 秒后超时退出码 3」。
个别相机用 reliable 发布时加 `--qos reliable` 退回默认 QoS。

> 为什么默认是 `stub`：检测模型要权重、要 GPU、启动几十秒。把几何链先测干净，
> 模型出问题才能一眼分清是"没检出"还是"算错了"。

**没权重也想在真机上验几何链**：`--stub-box X1 Y1 X2 Y2 [--stub-score 0.9]`
把固定框喂给 `StubDetector`，于是走的是**真实彩色图 + 真实深度图 + 真实 TF**
的完整 `locate_object`，只有"框从哪来"这一环是假的。没建图（`map` 帧不存在）时
再加 `--frame odom`。

```bash
# 仿真里已经有相机话题、但没起 SLAM 时，验"深度 → 相机系 → odom 落点"
python3 grounding.py "红箱子" --detector stub \
    --stub-box 300 200 380 300 --frame odom
```

## A3：接真检测器（GroundingDINO）

A2 把几何链钉死了（`test_grounding.py` / `test_pipeline.py`），A3 只做两件事：
**把检测器换成真模型**，以及**让"没装依赖"变成一句能照抄的命令**，而不是 ImportError 栈。

```
grounding.py
  ├─ locate_object(depth, K, detections, tf4, robot_xy, standoff)   ← 纯函数主干（A2 起就测）
  ├─ StubDetector              ← 零依赖，联调/测试用
  └─ GroundingDinoDetector     ← 可选依赖：torch + transformers + 权重
```

**装依赖（可选，不装也能跑全部单测）**

```bash
pip3 install torch transformers pillow     # 有 NVIDIA 卡就装 CUDA 版 torch
```

**权重来源**：默认从 HuggingFace 拉 `IDEA-Research/grounding-dino-tiny`（~700 MB）。

- 有网：第一次 `--detector dino` 会自动下载并缓存。
- 离线机器：先在有网的机器上 `huggingface-cli download IDEA-Research/grounding-dino-tiny`，
  把缓存拷过去并 `export HF_HOME=/path/to/hf-cache`；或者直接把本地权重目录传给 `--model-id`。

**常用参数**

```bash
python3 grounding.py "红箱子" --detector dino \
    --model-id IDEA-Research/grounding-dino-tiny \
    --box-threshold 0.3 --text-threshold 0.25 \
    --device cpu --navigate --standoff 0.8
```

| 参数 | 作用 | 调参直觉 |
|---|---|---|
| `--box-threshold` | 框置信度下限 | **检不到**就降到 0.2；**乱检**就升到 0.4 |
| `--text-threshold` | 文本-框匹配下限 | 提示词是长句时略降 |
| `--device` | `cpu` / `cuda` | 一律自动；显存小可强制 cpu |
| `--model-id` | 权重 id 或本地目录 | 离线/自定义微调权重走这里 |
| `--stub-box` / `--stub-score` | stub 后端注入一个像素框 | 真机上先验几何链，别等权重 |
| `--frame` | 目标坐标系，默认 `map` | 没建图时用 `odom` |

**没 GPU 怎么验证没写错**：`test_detector.py` 用假 processor/model 把 `detect()` 的
适配逻辑（补句点、阈值透传、`target_sizes=[(H, W)]`、空结果 → `[]`）全测了，
所以「模型装不上」和「代码写错了」是两件事，可以分开排查。

> 已知限制：GroundingDINO 在 CPU 上单帧要几秒，导航闭环里建议
> ①先停下车再检测；②用 `--standoff` 留足安全距离；③只在需要时激活检测（别每帧都跑）。

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
2. ~~把 `--detector dino` 接上~~（A3 已完成检测器与几何链拼接；剩下的是**在有 GPU 的机器上真跑一次**并记录指标）
3. 失败回退：目标不可达时切到探索模式（Frontier exploration）
4. 把「检出 → 算坐标 → 导航」串成一次完整 episode，用 `eval_nav.py` 的口径记录成功率/耗时
