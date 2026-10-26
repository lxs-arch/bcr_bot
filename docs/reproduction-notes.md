# bcr_bot 复现实录：从 MuJoCo 仿真到 Nav2 自主导航

> 本文记录在本仓库（`ros2-jazzy` 分支）上完整跑通 **SLAM 建图 → 存图 → Nav2 定位与自主导航 → 语言指令导航** 的全部命令、参数、踩过的坑与修法，以及后续实验用的指标表模板。
>
> **适用版本**：Ubuntu 24.04 + ROS 2 Jazzy + MuJoCo，`bcr_bot` ros2-jazzy 分支
> **验证环境**：Windows 11 宿主 + VMware Workstation 中的 Ubuntu 24.04 虚拟机

---

## 0. 结论速览：这套栈能跑到什么程度

| 能力 | 状态 | 验证方式 |
|---|---|---|
| MuJoCo 仿真 + 键盘遥控 | ✅ | `/bcr_bot/cmd_vel` 有 1 个订阅者，车动 |
| SLAM 建图（slam_toolbox） | ✅ | 覆盖约 5 m × 5 m，墙线闭合 |
| 地图保存（pgm + yaml） | ✅ | `maps/bcr2.*`、`maps/bcr3.*` |
| Nav2 定位 + 自主导航 | ✅ | `Reached the goal!` / `Goal succeeded`，状态码 4（SUCCEEDED） |
| 语言 → 导航（阶段 A，A0–A2） | ✅ | `vla_nav/llm_nav.py` 解析发目标；几何链见附录 C |
| 视觉接地（阶段 A，A3） | 🟡 | 检测器适配 + 离线测试完成；**真机 GPU 推理待验证**（附录 C） |
| 全程自动建图 | ⚠️ | 反应式漫游会因扫描匹配漂移产生拖影，建议用里程计闭环的固定路线 |

---

## 1. 环境准备

```bash
# 1) 系统与 ROS
lsb_release -ds          # Ubuntu 24.04.x LTS
ls /opt/ros              # jazzy

# 2) 工作空间结构（源码目录 == 包目录，注意这层嵌套）
ls ~/bcr_ws/src
# bcr_bot-ros2-jazzy/       <- 仓库根，同时也是 ROS 包
# bcr_bot-ros2-jazzy.zip

# 3) 编译
cd ~/bcr_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# 4) 验证包可见
ros2 pkg prefix bcr_bot
ros2 pkg executables bcr_bot
```

> ⚠️ **每个新开的终端都要 `source ~/bcr_ws/install/setup.bash`**。否则会出现
> `Package 'bcr_bot' not found: searching: ['/opt/ros/jazzy']`。
> 建议写进 `~/.bashrc`：`source ~/bcr_ws/install/setup.bash`

---

## 2. 最小可用流程（照抄即可跑通）

### 2.1 起仿真

```bash
source /opt/ros/jazzy/setup.bash
source ~/bcr_ws/install/setup.bash
ros2 launch bcr_bot mujoco.launch.py
```

MuJoCo 窗口右上角会显示：

```
Status      Running
Desired Speed  100.0%
Actual Speed   45.7%     <- 实时率，见 §4.6
```

> MuJoCo 里按 **空格** 会暂停/继续（状态变为 `Paused`）。误按后点一下视口再按空格恢复。

### 2.2 雷达话题转发（本仓库必做，否则 Nav2 定位静默失效）

底盘只发 `/bcr_bot/scan`，而 Nav2 的 AMCL / costmap 默认订阅 `/scan`，且 **QoS 不兼容**（见 §4.1）。

```bash
python3 ~/scan_relay.py        # 见附录 A：最佳努力 QoS 转发脚本
ros2 topic hz /scan            # 应约 10 Hz
```

### 2.3 SLAM 建图

```bash
ros2 launch bcr_bot mapping.launch.py
# 打开 RViz：Fixed Frame = map，Map 话题 = /map
```

建图推荐用 **里程计闭环的固定路线脚本**（附录 B），不要用"撞到就转向"的漫游——
后者在转向过快时会产生扫描匹配拖影，地图上会出现斜穿房间的虚线和弧状伪影。

建图合格判据：

- [ ] 外墙闭合，白色可达区四周被连续黑线包住
- [ ] 没有斜穿房间的长虚线/弧状拖影
- [ ] 障碍物背面不再显示为白色空闲
- [ ] 回到起点后回环闭合（同一面墙单线，不是双线）

### 2.4 保存地图

```bash
ros2 run nav2_map_server map_saver_cli -f ~/bcr_ws/maps/bcr3
ls -l ~/bcr_ws/maps/
```

失败时用兜底方案（见 §4.3）：

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: /home/<user>/bcr_ws/maps/bcr3}}"
```

### 2.5 起 Nav2 导航

```bash
# 先杀掉建图那一套（slam_toolbox 必须停，否则 map->odom 有两个发布者）
pkill -9 -f slam_toolbox

# 重启仿真，让机器人回到地图原点
pkill -9 -f mujoco && sleep 3
ros2 launch bcr_bot mujoco.launch.py &

ros2 launch bcr_bot nav2.launch.py map:=$HOME/bcr_ws/maps/bcr3.yaml
```

RViz 面板出现 `Navigation: active` / `Localization: active` 即为成功。

### 2.6 给 AMCL 初始位姿（必做）

刚启动时 AMCL 没有位姿，发目标会被拒。机器人位置正好等于地图原点，所以发 `(0,0,0)`：

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
"{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0}, \
orientation: {z: 0.0, w: 1.0}}, covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, \
0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.068]}}"

ros2 topic echo /amcl_pose --once --field pose.pose.position   # 应有输出
```

### 2.7 发目标点

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
"{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0}, \
orientation: {z: 0.0, w: 1.0}}}}"
```

或用 RViz 工具栏 `Nav2 Goal` 在地图上点。成功标志：

```
[controller_server]: Reached the goal!
[bt_navigator]: Goal succeeded
```

状态码含义（`action_msgs/GoalStatus`）：`4 = SUCCEEDED`、`5 = CANCELED`、`6 = ABORTED`。

---

## 3. 关键参数与调优对照表

| 参数 | 默认/原值 | 本项目建议值 | 依据 |
|---|---|---|---|
| `inflation_radius` | `0.45` | `0.25` | 走廊净宽约 1 m，0.45 会把整条走廊吃成高代价区，导致目标点落进膨胀带被拒 |
| `cost_scaling_factor` | `3.0` | `5.0` | 收紧膨胀后提高梯度，让车贴墙时减速更合理 |
| `controller_server FollowPath.vx_max`（MPPI） | `0.5` | `0.8` | 本项目用 **MPPI**，速度上限参数名是 `vx_max`，**不是** RPP 的 `desired_linear_vel` |
| `velocity_smoother max_velocity` | `[0.26, 0.0, 1.0]` | `[0.8, 0.0, 2.0]` | **真正卡速度的是这里**：控制器想出多快都会被截到 0.26 |
| `velocity_smoother max_accel` | `[2.5, 0.0, 3.2]` | 保持 | 加速能力足够，不是瓶颈 |
| `FollowPath.ax_max` | `3.0` | 保持 | MPPI 加速度上限，充足 |
| `max_velocity(x)` 全局规划 | — | 按实车能力 | — |

> ⚠️ `vx_max` / `desired_linear_vel` 设错名字会直接报
> `parameter 'FollowPath.desired_linear_vel' cannot be set because it was not declared`。
> 用 `ros2 param list /controller_server | grep -iE "vel|max|vx"` 先确认控制器类型。

**判断控制器类型的快捷方法**：

| 参数里出现 | 控制器 |
|---|---|
| `FollowPath.vx_max` / `vx_std` / `PathAlignCritic` | **MPPI** |
| `FollowPath.max_vel_x` / `max_speed_xy` | DWB |
| `FollowPath.desired_linear_vel` / `regulated_linear_scaling_min_radius` | RPP |

---

## 4. 六个坑与修法（按踩坑时间排序）

### 4.1 雷达 QoS 不匹配 → AMCL 收不到数据，`/amcl_pose` 永远无输出

**现象**

```
[WARN] [scan_relay]: New subscription discovered on topic '/scan', requesting
incompatible QoS. No messages will be sent to it. Last incompatible policy: RELIABILITY
```

`ros2 topic echo /amcl_pose` 报 `does not appear to be published yet`。

**根因**：雷达发布者用 `BEST_EFFORT`，AMCL / costmap 默认 `RELIABLE`。
DDS 层直接不投递，节点看起来"在跑但没数据"。

**修法**：任何转发节点都必须显式声明 `BEST_EFFORT`（见附录 A），而不是默认构造订阅。

### 4.2 `/clock` 回跳 + 多个仿真实例 → Nav2 生命周期起不来

**现象**

```
[rviz2] [tf2_buffer]: Detected jump back in time. Clearing TF buffer.
ros2 node list | grep -c twist_stamper   ->  7     # 应该只有 1
```

**根因**：反复启动/杀进程时 `ros2 launch` 的**子进程没被杀干净**，多个 MuJoCo 同时在发
`/clock`；`use_sim_time` 的节点看到时间倒退，清空 TF 缓存，AMCL / Nav2 反复重启。

**修法**：不要只用 `pkill -f ros2`，要用进程表兜底：

```bash
ps -eo pid,args | grep -E "mujoco|rviz2|nav2|controller_server|planner_server|\
bt_navigator|scan_relay|component_container|map_server|amcl" \
| grep -v grep | awk '{print $1}' | xargs -r kill -9
ps -eo args | grep -E "mujoco|rviz2|scan_relay" | grep -v grep | wc -l   # 必须是 0
```

起栈后立刻自查：`ros2 node list | grep -c twist_stamper` **必须为 1**。

### 4.3 `map_saver_cli` 报 `Failed to spin map subscription`

**现象**

```
[ERROR] [map_saver]: Failed to spin map subscription
[ros2run]: Process exited with failure 1
```

**根因**：`/map` 是 `transient_local` 的，`map_saver` 订阅端的 QoS 与发布端不匹配时拿不到最后一帧。

**修法**（按顺序尝试）

```bash
# 1) 显式 transient_local
ros2 run nav2_map_server map_saver_cli -f ~/bcr_ws/maps/bcr3 \
  --ros-args -p map_subscribe_transient_local:=true

# 2) 仍失败则走 slam_toolbox 自己的序列化服务
ros2 service list | grep -i save
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: /home/<user>/bcr_ws/maps/bcr3}}"
```

### 4.4 没设初始位姿 → 目标被拒 / 车乱跑

**现象**：`ACCEPTED False`、`STATUS 6`、或车朝着完全错误的方向走。

**根因**：AMCL 没有先验位姿时，`map→odom` 变换不可用或错误，规划器拿到的是错的机器人位置。

**修法**：见 §2.6。**关键前提是"建图起点 == 导航起点"**——地图坐标系原点就是建图时机器人的起始位姿，
所以导航前务必重启仿真让机器人回到原点。

### 4.5 目标落在膨胀/未知区域 → `STATUS 6 (ABORTED)`

**现象**

```
ACCEPTED True
STATUS 6
```

RViz 面板 `Recoveries` 快速上升（十几次）。

**根因**：`inflation_radius = 0.45` 在 1 m 宽的走廊里几乎吃满可通行区；目标点若落在膨胀带或未知区，
planner 无解 → BT 触发恢复行为 → 最终 abort。**这不是"不会避障"，是目标本身不可达。**

**修法**：

1. `inflation_radius` 降到 `0.25`
2. 目标点选在**白色自由区**（RViz 里 Map 显示为白）
3. 远处目标拆成沿走廊的中间点依次发

### 4.6 车"看着很慢"：三个叠加因素

| 因素 | 证据 | 处理 |
|---|---|---|
| **仿真实时率** | MuJoCo `Desired 100% / Actual 36–58%` | 缩窗口、跑任务时最小化 RViz、给 VM 多分 vCPU 并开 3D 加速 |
| **Nav2 默认限速** | `velocity_smoother max_velocity = [0.26, 0.0, 1.0]` | 提到 `[0.8, ...]`；MPPI 的 `vx_max` 同步提到 0.8 |
| **窄走廊 + 大膨胀** | `inflation_radius 0.45` | 降到 0.25，并调 `cost_scaling_factor` |

**换算式**：墙钟速度 ≈ 指令速度 × 实时率。
`0.8 × 0.45 ≈ 0.36 m/s`，这就是当前环境能做到的上限；`0.26 × 0.45 ≈ 0.12 m/s` 是调参前的水平。

### 4.7（附）新终端没有 `source` 工作空间

```
Package 'bcr_bot' not found: "package 'bcr_bot' not found, searching: ['/opt/ros/jazzy']"
```

`source` 只对执行它的那个 shell 生效，新窗口不会继承。见 §1 建议写进 `~/.bashrc`。

---

## 5. 指标表模板

### 5.1 单次导航记录（每次 run 一行）

| run | 日期 | 地图 | 起点 (x,y,yaw) | 目标 (x,y,yaw) | 结果 | 耗时 s | 路径长 m | 平均速度 m/s | Recoveries | ETA 误差 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 2026-10-26 | bcr3 | 0,0,0 | 1.0,0,0 | SUCCEEDED(4) | 19 | 1.0 | 0.05 | 0 | — | 首次验证 |
| 02 | | | | | | | | | | | |

> 采集方法：`Time taken` / `Distance remaining` / `Recoveries` 直接读 RViz 的 Nav2 面板；
> 平均速度用 `ros2 topic echo /bcr_bot/odom --once --field pose.pose.position` 前后差分 ÷ 时间。

### 5.2 参数消融汇总（每换一组参数填一行）

| 参数组 | inflation_radius | vx_max | smoother max_velocity | 成功率 | 平均速度 | 平均 Recoveries | 实时率 |
|---|---|---|---|---|---|---|---|
| baseline | 0.45 | 0.5 | 0.26 | | | | |
| 放宽限速 | 0.45 | 0.8 | 0.8 | | | | |
| 收紧膨胀 | 0.25 | 0.8 | 0.8 | | | | |

### 5.3 定位精度评估

| 指标 | 方法 | 目标 |
|---|---|---|
| 定位 RMSE | 在固定点位记录 `/amcl_pose` 与真值（MuJoCo 里的位姿）差 | < 0.1 m |
| 重定位时间 | 从给初始位姿到粒子收拢（RViz 粒子云收敛） | < 5 s |
| 建图一致性 | 回到起点后同一面墙的偏移 | < 0.1 m |

---

## 6. 每次实验的 SOP

```
1. 记录环境：实时率(Actual Speed)、节点数(ros2 node list | wc -l)、twist_stamper 数
2. 记录参数快照：ros2 param dump /controller_server > logs/params_YYYYmmdd_HHMM.yaml
3. 记录地图版本：md5sum maps/bcr3.yaml
4. 跑任务，同时 ros2 bag record /tf /tf_static /scan /amcl_pose /plan /bcr_bot/odom
5. 结果写进 §5.1 一行
6. 失败必须写"失败原因假设"，比成功记录更有价值
```

---

## 7. 常见错误速查

| 报错 / 现象 | 最可能原因 | 一行修法 |
|---|---|---|
| `Package 'bcr_bot' not found` | 没 source 工作空间 | `source ~/bcr_ws/install/setup.bash` |
| `/amcl_pose` 无输出 | `/scan` QoS 不匹配 / 没给初始位姿 | 用 BEST_EFFORT 转发 + 发 `/initialpose` |
| `Detected jump back in time` | 多个 MuJoCo 实例 / 时钟回跳 | 进程表清理，确认 `twist_stamper` 只有 1 个 |
| `Failed to spin map subscription` | `/map` 的 transient_local 不匹配 | 加 `-p map_subscribe_transient_local:=true` 或走 `save_map` 服务 |
| 目标 `REJECTED` | 目标在障碍/未知区，或定位未就绪 | 换白色自由区 + 确认 AMCL 有位姿 |
| 目标 `ABORTED(6)`、Recoveries 飙升 | 目标不可达 / 膨胀带太宽 | 降 `inflation_radius`，拆中间点 |
| 车比预期慢 3 倍 | 实时率 < 100% + `max_velocity` 0.26 | 见 §4.6 |
| MuJoCo 不动 | 误按空格进 `Paused` | 点视口后按空格 |
| 地图有斜线/弧状伪影 | 转向过快导致扫描匹配漂移 | 改用里程计闭环的固定路线，降角速度 |

---

## 附录 A：雷达 QoS 转发脚本 `scan_relay.py`

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

SRC = '/bcr_bot/scan'
DST = '/scan'

class Relay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(LaserScan, DST, qos)
        self.sub = self.create_subscription(LaserScan, SRC, self.cb, qos)
        self.get_logger().info('relay %s -> %s (BEST_EFFORT)' % (SRC, DST))

    def cb(self, msg):
        self.pub.publish(msg)

rclpy.init()
rclpy.spin(Relay())
```

> 也可以 `sudo apt install ros-jazzy-topic-tools` 后用 `ros2 run topic_tools relay`，
> 但**必须**额外用 `--qos-reliability best_effort` 之类的方式对齐 QoS，否则同样静默失败。

## 附录 B：确定性建图路线（里程计闭环）

思路：不靠避障，靠**里程计反馈**把每一段走准，减少转向时的扫描匹配漂移。

```python
# 片段：按 odom 闭环走一段直线 + 转固定角度
import math, rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))

# 直行：累计位移 ≥ dist 或前方 < 0.9 m 即停
cmd.linear.x = 0.15
# 转向：累计角度变化 ≥ target_rad 即停
cmd.angular.z = 0.35
```

推荐参数：`linear.x = 0.15 m/s`、`angular.z = 0.35 rad/s`、每段后停 1 s 让扫描匹配收敛。

## 附录 C：阶段 A 里程碑（A0–A3）—— 语言 → 导航 → 视觉接地

代码在 `vla_nav/`，全部**离线可测**（不需要 ROS / GPU / 相机）。

### C.1 里程碑与状态

| 里程碑 | 内容 | 状态 | 验证方式 |
|---|---|---|---|
| A0 | 语义地图 + 录点 + 发目标的基础设施（`semantic_map.py`/`record_place.py`） | ✅ | `llm_nav.py --list`、RViz 点选写入 |
| A1 | 指令解析：相对方位（前后左右）/ 绝对方位（罗盘）/ 多步指令 | ✅ | `test_parse.py` 表驱动用例 |
| A2 | 视觉接地几何链：检测框 → 深度中位数 → 反投影 → TF → `map` 落点 | ✅ | `test_grounding.py` + `test_pipeline.py` |
| A3 | 真检测器适配（GroundingDINO）+ 可选依赖降级 + 统一 pytest 套件 + CI | ✅（代码/离线测试） | `test_detector.py`；**真机 GPU 推理尚未在本仓库验证** |

> 诚实边界：A3 只把「检测器接口 + 适配层 + 依赖缺失时的可执行提示」做完并测了；
> HuggingFace 权重下载、GPU 上的实际延迟、真实 Kinect 深度噪声下的落点精度，
> 都还需要一台带 GPU 的机器跑一次才能填数（见 C.5 指标表）。

### C.2 文件清单

| 文件 | 作用 |
|---|---|
| `semantic_map.yaml` | 地名 → `(x, y, yaw)`，按自己的地图改 |
| `llm_nav.py` | 指令解析（规则 → aliases → LLM 三层兜底）+ 发 Nav2 目标 |
| `record_place.py` | RViz `Publish Point` 点一下自动写入 yaml |
| `eval_nav.py` | 批量导航评测：成功率 / 用时 / 终点误差 / recovery 次数 → CSV |
| `grounding.py` | 视觉接地：`locate_object()` 纯函数主干 + `stub` / `dino` 两种检测器后端 |
| `test_parse.py` / `test_grounding.py` / `test_pipeline.py` / `test_detector.py` | 四层测试 |
| `_testkit.py` / `conftest.py` / `pytest.ini` / `run_tests.py` | 测试脚手架：pytest 与 standalone 双跑 |

### C.3 命令

```bash
# A1：地名 / 相对方位 / 多步
python3 llm_nav.py --list
python3 record_place.py
python3 llm_nav.py "去A区"
python3 llm_nav.py --dry-run "A区左边1米"      # 只解析，不发目标（离线）

# A2：只算几何，不需要 ROS
python3 grounding.py --dry-run --box 300 200 380 300 --depth 2.0 \
    --intrinsics 600 600 320 240 --camera-pose 1 2 0.3 0 --robot 0 0

# A3：真检测器（可选依赖 torch/transformers/pillow + 权重）
python3 grounding.py "红箱子" --detector dino --box-threshold 0.3 --navigate --standoff 0.8
```

### C.4 测试（本仓库的「先写测试再合并」约定）

```bash
cd vla_nav
python3 run_tests.py              # 自动选 runner：有 pytest 用 pytest，没装逐文件跑
python3 run_tests.py --pytest     # CI 用：强制 pytest
python3 run_tests.py --standalone # 机器人机：完全不用 pytest
```

- 覆盖解析 / 语义词典 / 几何投影 / 落点策略 / 检测器契约 / 依赖降级；
- CI：`.github/workflows/vla_nav-tests.yml`（push / PR 触发，`ubuntu-latest` + `pip install pytest pyyaml`）；
- numpy 只在装了的时候多测一条互操作用例；pyyaml 缺失时走内置解析器。

### C.5 阶段 A 指标表模板

每次「一句话 → 到达」的 episode 填一行，用来判断失败到底出在解析、接地还是导航：

| 指令 | 类型 | 检出 | 一次成功 | 重试 | 用时(s) | 终点误差(m) | 失败环节 | 备注 |
|---|---|---|---|---|---|---|---|---|
| `去A区` | 地名 | n/a | ✅ | 0 | 12.4 | 0.08 | — | 基线 |
| `A区左边1米` | 相对方位 | n/a | ✅ | 0 | 13.1 | 0.11 | — | yaw 决定"左" |
| `红箱子` | 视觉接地(stub) | ✅ | — | — | — | — | — | 只验证链路 |
| `红箱子` | 视觉接地(dino) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | **待 GPU 机器填** |

失败环节分类建议：`parse`（没认出）/ `detect`（没检出/误检）/ `depth`（深度缺失）/
`tf`（相机外参查不到）/ `nav`（目标被拒或超时）。

### C.6 阶段 A 踩坑补记

| 现象 | 原因 | 修法 |
|---|---|---|
| `grounding.py` 一启动就 ImportError 栈 | torch/transformers 是可选依赖 | 构造前先 `missing_deps()`，直接给 `pip3 install …` 提示；联调用 `--detector stub` |
| 算出的点"贴着车头" | 深度图里 0 被当成 0 m 参与平均 | `median_depth` 过滤 0 / NaN / inf / 超距，取中位数 |
| 物体在画面里但坐标乱飞 | 只用 yaw 近似相机外参，把光学系的"下"当成了 map 的 +y | TF 必须带完整四元数；`test_pipeline.py::test_optical_rotation_maps_height` 钉住这一点 |
| 目标点撞在物体上 | 直接把物体坐标当导航目标 | 沿「机器人→物体」连线退 `--standoff` 米 |
| CPU 上 dino 每帧好几秒 | GroundingDINO 是 ViT 检测器 | 停车再检测、`--standoff` 留余量、只在需要时触发 |

---

## 参考

- bcr_bot 仓库：<https://github.com/blackcoffeerobotics/bcr_bot>（`ros2-jazzy` 分支）
- Nav2 文档：<https://docs.nav2.org/>
- slam_toolbox：<https://github.com/SteveMacenski/slam_toolbox>
- MuJoCo：<https://mujoco.readthedocs.io/>
