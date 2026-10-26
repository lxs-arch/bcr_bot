#!/usr/bin/env python3
"""grounding.py —— 视觉接地（A2）：自然语言 → 画面里的物体 → map 坐标

链路:

    指令 "红箱子"
      → [检测器] 输出 bbox (x1,y1,x2,y2) + score        ← 需要 GPU/权重，可换后端
      → [深度] 取 bbox 内有效深度中位数                    ← 纯函数
      → [反投影] 像素 + 深度 + 内参 → 相机系 3D 点          ← 纯函数
      → [外参] 相机系 → map（TF 或给定 4x4）               ← 纯函数
      → [落点] 目标点 + 保持距离 dist → 导航目标            ← 纯函数

设计约束（重要）:

* 几何计算**不依赖 numpy / ROS / 第三方库**，因此可以离线单测；
  只有真正读图（sensor_msgs/Image 解码）才用 numpy，并且延迟导入。
* 整条链路被压成一个纯函数 `locate_object(...)`：检测结果 + 深度数组 + 内参 +
  相机→map 外参 → map 坐标与导航落点。ROS 节点只做「解码 + 查 TF」。
* 检测器是一个「后端接口」。默认 `StubDetector` 用于测试与联调；
  真模型（GroundingDINO / OWLv2）在 `GroundingDinoDetector` 里延迟导入，
  **torch / transformers / 权重都是可选依赖**：没装时 `missing_deps()` 能提前报出来，
  构造时给出可执行的 pip 提示而不是 ImportError 堆栈，CI 与无 GPU 机器照常跑单测。

用法:

    # 纯几何自检（不需要 ROS / GPU）
    python3 grounding.py --dry-run --box 300 200 380 300 --depth 2.0 \\
        --intrinsics 600 600 320 240 --camera-pose 1.0 2.0 0.3 0.0

    # 真机：stub 先联调链路，再换真模型 + 直接导航过去
    python3 grounding.py "红箱子" --detector stub
    python3 grounding.py "红箱子" --detector dino --navigate --standoff 0.8
"""

import argparse
import math
import os
import sys

# ---------------------------------------------------------------- 基础几何
# 统一用「3x3 = list[list[float]]」「4x4 = list[list[float]]」表示矩阵，
# 不引入 numpy，保证本文件可在任何 Python 3.8+ 环境导入与单测。


def mat_mul(a, b):
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def mat_vec(a, v):
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def transpose(a):
    return [list(row) for row in zip(*a)]


def identity(n=4):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def quat_to_matrix(qx, qy, qz, qw):
    """四元数 → 3x3 旋转矩阵（ROS 顺序 x,y,z,w）。"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def pose_to_matrix(x, y, z, qx, qy, qz, qw):
    """位姿 → 4x4 齐次矩阵。"""
    m = identity(4)
    r = quat_to_matrix(qx, qy, qz, qw)
    for i in range(3):
        for j in range(3):
            m[i][j] = r[i][j]
    m[0][3], m[1][3], m[2][3] = x, y, z
    return m


def transform_point(tf4, p):
    """用 4x4 把 3D 点从一个坐标系变到另一个。"""
    v = mat_vec([row[:3] for row in tf4], p)
    return [v[0] + tf4[0][3], v[1] + tf4[1][3], v[2] + tf4[2][3]]


# ---------------------------------------------------------------- 投影 / 反投影
def project_point(p, k):
    """相机系点 → 像素 (u, v)。p 必须在相机前方（z>0）。"""
    if p[2] <= 1e-6:
        raise ValueError('point behind camera: z=%.4f' % p[2])
    u = k['fx'] * p[0] / p[2] + k['cx']
    v = k['fy'] * p[1] / p[2] + k['cy']
    return u, v


def backproject(u, v, depth, k):
    """像素 + 深度 → 相机系 3D 点（光学坐标系：x 右, y 下, z 前）。"""
    if depth is None or depth <= 0:
        raise ValueError('invalid depth: %r' % (depth,))
    x = (u - k['cx']) * depth / k['fx']
    y = (v - k['cy']) * depth / k['fy']
    return [x, y, depth]


# ---------------------------------------------------------------- 深度取样
def median_depth(values, min_depth=0.10, max_depth=8.0):
    """bbox 内有效深度中位数。过滤 0 / NaN / inf / 越界。

    真实深度图里 0 表示"没有数据"，不能当 0 米参与平均——
    这是机器人抓"半米处的幽灵"最常见的原因。
    """
    good = []
    for d in values:
        if d is None:
            continue
        try:
            f = float(d)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        if min_depth <= f <= max_depth:
            good.append(f)
    if not good:
        return None
    good.sort()
    n = len(good)
    mid = n // 2
    return good[mid] if n % 2 else 0.5 * (good[mid - 1] + good[mid])


def depth_patch(depth, box, pad=0):
    """取 bbox 内的一维深度序列（行优先），越界自动夹到图像范围内。

    `depth` 既可以是「行列表」(`list[list[float]]`)，也可以是 numpy 二维数组——
    两者都支持 `depth[y1:y2]` 与 `row[x1:x2]` 切片，所以这里不必 import numpy。
    空图像 / 完全越界返回 []，交给 `median_depth` 判空。

    判空必须用 `len(depth) == 0`：写 `if not depth:` 时 numpy 数组会抛
    "truth value of an array is ambiguous"，而真节点喂进来的正好是 ndarray。
    """
    if depth is None or len(depth) == 0:
        return []
    h = len(depth)
    w = len(depth[0]) if h else 0
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, min(x1, x2) - pad), min(w, max(x1, x2) + 1 + pad)
    y1, y2 = max(0, min(y1, y2) - pad), min(h, max(y1, y2) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return []
    out = []
    for row in depth[y1:y2]:
        out.extend(row[x1:x2])
    return out


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def pick_best(detections, min_score=0.3, min_area=64.0):
    """从检测结果里挑最可信的一个：先过滤，再按 score 排序。"""
    cand = [d for d in detections
            if d.score >= min_score and box_area(d.box) >= min_area]
    if not cand:
        return None
    return max(cand, key=lambda d: d.score)


# ---------------------------------------------------------------- 落点
def ground_hit(origin, direction, z_ground=0.0):
    """射线与水平面求交（深度缺失时的兜底）。方向需指向地面。"""
    dz = direction[2]
    if abs(dz) < 1e-6:
        return None
    t = (z_ground - origin[2]) / dz
    if t <= 0:
        return None
    return [origin[0] + t * direction[0],
            origin[1] + t * direction[1],
            z_ground]


def standoff_point(target_xy, robot_xy, dist=0.8):
    """在「机器人 → 目标」连线上、距目标 dist 处停下，避免直接撞进物体。"""
    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        # 机器人已在目标点，退到目标后方，避免除零
        return [target_xy[0] - dist, target_xy[1]]
    if norm <= dist:
        return list(robot_xy)                      # 已经足够近，原地不动
    k = (norm - dist) / norm
    return [robot_xy[0] + dx * k, robot_xy[1] + dy * k]


def yaw_towards(from_xy, to_xy):
    return math.atan2(to_xy[1] - from_xy[1], to_xy[0] - from_xy[0])


# ---------------------------------------------------------------- 整条管线（纯函数）
def locate_object(depth, k, detections, tf4, robot_xy, standoff=0.8,
                  min_score=0.3, min_area=64.0, min_depth=0.10, max_depth=8.0):
    """检测结果 + 深度 + 内参 + 相机→map 外参 → 物体 map 坐标与导航落点。

    这是 A2/A3 的核心链路，**完全纯函数**：不触碰 ROS、不读 topic、不需要 numpy。
    单元测试可以直接喂「手写的深度二维数组 + Detection 列表 + 4x4」，
    真机路径 (`GroundingNode.locate`) 只是把 sensor_msgs 解码成同样形状再调它。

    返回 dict（含 `x/y/yaw` 兼容字段），任一环节没数据则返回 None：

        {'object': (x, y, z),   # 物体在 map 下的 3D 位置
         'goal':   (x, y, yaw), # 机器人该去的落点（离物体 standoff 米）
         'x': ..., 'y': ..., 'yaw': ...,
         'score': ..., 'depth': ..., 'box': (x1, y1, x2, y2)}
    """
    best = pick_best(detections, min_score=min_score, min_area=min_area)
    if best is None:
        return None
    z = median_depth(depth_patch(depth, best.box),
                     min_depth=min_depth, max_depth=max_depth)
    if z is None:
        return None
    u, v = box_center(best.box)
    p_cam = backproject(u, v, z, k)
    p_map = transform_point(tf4, p_cam)
    goal = standoff_point((p_map[0], p_map[1]), tuple(robot_xy), standoff)
    yaw = yaw_towards(tuple(robot_xy), (p_map[0], p_map[1]))
    return {'object': (p_map[0], p_map[1], p_map[2]),
            'goal': (goal[0], goal[1], yaw),
            'x': goal[0], 'y': goal[1], 'yaw': yaw,
            'score': best.score, 'depth': z, 'box': best.box}


# ---------------------------------------------------------------- 检测器后端
class Detection(object):
    __slots__ = ('box', 'score', 'label')

    def __init__(self, box, score, label=''):
        self.box = tuple(float(v) for v in box)
        self.score = float(score)
        self.label = label

    def __repr__(self):
        return 'Detection(box=%s, score=%.3f, label=%r)' % (
            tuple(round(v, 1) for v in self.box), self.score, self.label)


class StubDetector(object):
    """测试/联调用：返回预设结果，不加载任何模型。"""

    def __init__(self, detections=None):
        self.detections = list(detections or [])

    def detect(self, image, prompt):
        return list(self.detections)


def _image_hw(image):
    """取图像的 (高, 宽)。

    * numpy 数组（ROS 节点传进来的形态）：`shape` 是 `(H, W[, C])`；
    * PIL Image：`size` 是 `(W, H)`，要反过来。

    之前直接写 `image.size[::-1]`：PIL 能过，ndarray 上 `image.size` 是整数，
    `[::-1]` 直接 TypeError——假图像测试掩盖了这个坑。
    """
    shape = getattr(image, 'shape', None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    w, h = image.size
    return int(h), int(w)


class GroundingDinoDetector(object):
    """真模型后端（可选依赖）。torch + transformers 没装 / 权重没下时给出可执行提示。

    调用示例::

        det = GroundingDinoDetector(model_id='IDEA-Research/grounding-dino-tiny')
        dets = det.detect(pil_image, '红箱子')

    设备选择: torch.cuda.is_available() 有就用 cuda，否则 cpu（会很慢，仅演示）。
    权重来源: 默认从 HuggingFace 拉；离线机器先 `huggingface-cli download` 再把
    `HF_HOME` 指到本地缓存目录，或直接把本地权重路径当 `model_id` 传进来。
    """

    REQUIREMENTS = ('torch', 'transformers')

    def __init__(self, model_id='IDEA-Research/grounding-dino-tiny',
                 box_threshold=0.3, text_threshold=0.25, device=None):
        missing = self.missing_deps()
        if missing:
            raise SystemExit(
                '缺少可选依赖：%s\n'
                '装装看：pip3 install %s pillow\n'
                '（没有 GPU 也能跑，就是慢；先用 --detector stub 联调几何链）' % (
                    ', '.join(missing), ' '.join(missing)))
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._torch = torch
        try:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:                              # noqa: BLE001  某些装了 CPU 版但驱动坏的情况
            self.device = 'cpu'
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    @classmethod
    def missing_deps(cls):
        """返回缺失的可选依赖名列表；空列表表示可以加载真模型。

        用 `__import__` 而不是 `find_spec`，这样 sys.modules 里被置 None
        （测试里模拟"没装"）也能正确判定。
        """
        missing = []
        for mod in cls.REQUIREMENTS:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        return missing

    def detect(self, image, prompt):
        text = prompt if prompt.strip().endswith('.') else prompt.strip() + '.'
        inputs = self.processor(images=image, text=text,
                                return_tensors='pt').to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[_image_hw(image)])[0]
        out = []
        for box, score, label in zip(res['boxes'].tolist(), res['scores'].tolist(),
                                     res.get('labels', [])):
            out.append(Detection(box, score, str(label)))
        return out


def make_detector(name, **kw):
    """按名字建检测器。未知后端 / 缺依赖都给出可读错误（SystemExit）。"""
    if name == 'stub':
        return StubDetector(kw.get('detections'))
    if name in ('dino', 'groundingdino'):
        allowed = ('model_id', 'box_threshold', 'text_threshold', 'device')
        return GroundingDinoDetector(**{k: v for k, v in kw.items()
                                        if k in allowed and v is not None})
    raise SystemExit('未知检测器后端: %s（可选 stub / dino）' % name)


# ---------------------------------------------------------------- ROS 部分
try:                                              # 延迟导入，保证纯几何可单测
    import rclpy                                     # noqa: F401
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False


if HAVE_ROS:
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from geometry_msgs.msg import PoseStamped
    from tf2_ros import Buffer, TransformListener
    from rclpy.qos import qos_profile_sensor_data

    def _intrinsics_from_info(msg):
        return {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def _image_to_array(msg):
        """sensor_msgs/Image → (H, W) / (H, W, C) ndarray。仅支持常见编码。"""
        if msg.encoding in ('32FC1', '16UC1'):
            dtype = np.float32 if msg.encoding == '32FC1' else np.uint16
            arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
            if msg.encoding == '16UC1':
                arr = arr.astype(np.float32) / 1000.0    # mm → m
            return arr
        if msg.encoding in ('rgb8', 'bgr8'):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr[:, :, ::-1] if msg.encoding == 'bgr8' else arr
        raise ValueError('不支持的编码: %s' % msg.encoding)

    class GroundingNode(Node):
        """订阅彩色/深度/相机内参 + TF，把"指令里的物体"变成 map 坐标。"""

        def __init__(self, color_topic, depth_topic, info_topic,
                     target_frame='map', standoff=0.8, qos=None):
            """qos: None/'sensor' → sensor data QoS（best_effort）；'reliable' → 默认 QoS。

            默认必须是 sensor data：相机/深度图普遍用 best_effort 发布，reliable 订阅
            与它们不兼容，表现就是「一帧都收不到、15s 后超时」。
            """
            super().__init__('grounding')
            self.standoff = standoff
            self.target_frame = target_frame
            self.qos_name = 'reliable' if qos == 'reliable' else 'sensor'
            self.qos = 10 if self.qos_name == 'reliable' else qos_profile_sensor_data
            self.k = None
            self.color = None
            self.depth = None
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            self.create_subscription(CameraInfo, info_topic, self._on_info, self.qos)
            self.create_subscription(Image, color_topic, self._on_color, self.qos)
            self.create_subscription(Image, depth_topic, self._on_depth, self.qos)
            self.get_logger().info('grounding 就绪: color=%s depth=%s info=%s qos=%s' % (
                color_topic, depth_topic, info_topic, self.qos_name))

        def _on_info(self, msg):
            self.k = _intrinsics_from_info(msg)

        def _on_color(self, msg):
            self.color = msg

        def _on_depth(self, msg):
            self.depth = msg

        def ready(self):
            return self.k is not None and self.color is not None and self.depth is not None

        def locate(self, detector, prompt, robot_xy):
            """读一帧 → 调检测器 → 交给纯函数 locate_object；失败返回 None。

            ROS 部分只负责「把消息变成数组 + 查一次 TF」，几何全部在
            `locate_object` 里，因此同样的链路可以离线单测。
            """
            if not self.ready():
                return None
            img = _image_to_array(self.color)
            dep = _image_to_array(self.depth)
            dets = detector.detect(img, prompt)
            if not dets:
                return None
            tf_msg = self.tf.lookup_transform(self.target_frame, self.color.header.frame_id,
                                              rclpy.time.Time())
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            tf4 = pose_to_matrix(t.x, t.y, t.z, q.x, q.y, q.z, q.w)
            return locate_object(dep, self.k, dets, tf4, robot_xy, self.standoff)


# ---------------------------------------------------------------- CLI
def _parse_args(argv):
    ap = argparse.ArgumentParser(description='视觉接地：指令 → map 坐标')
    ap.add_argument('prompt', nargs='*', help='要寻找的目标，例如 "红箱子"')
    ap.add_argument('--detector', default='stub', choices=['stub', 'dino'])
    ap.add_argument('--model-id', default='IDEA-Research/grounding-dino-tiny',
                    help='dino 权重（HF id 或本地路径）；离线机器把 HF_HOME 指到缓存目录')
    ap.add_argument('--box-threshold', type=float, default=0.3, help='dino 框阈值')
    ap.add_argument('--text-threshold', type=float, default=0.25, help='dino 文本阈值')
    ap.add_argument('--device', default=None, help='dino 设备：cuda / cpu（默认自动）')
    ap.add_argument('--color-topic', default='/bcr_bot/kinect_camera/color/image_raw')
    ap.add_argument('--depth-topic', default='/bcr_bot/kinect_camera/depth/image_raw')
    ap.add_argument('--info-topic', default='/bcr_bot/kinect_camera/color/camera_info')
    ap.add_argument('--standoff', type=float, default=0.8, help='距目标保持多少米')
    ap.add_argument('--qos', default='sensor', choices=['sensor', 'reliable'],
                    help='订阅 QoS：sensor=best_effort（相机默认），reliable=少数相机用')
    ap.add_argument('--navigate', action='store_true', help='算完直接发 /navigate_to_pose')
    ap.add_argument('--dry-run', action='store_true', help='只做纯几何计算，不需要 ROS')
    ap.add_argument('--box', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'))
    ap.add_argument('--depth', type=float)
    ap.add_argument('--intrinsics', nargs=4, type=float, metavar=('FX', 'FY', 'CX', 'CY'))
    ap.add_argument('--camera-pose', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'YAW'),
                    help='相机在 map 中的位姿（dry-run 用，简化成只有 yaw）')
    ap.add_argument('--robot', nargs=2, type=float, metavar=('X', 'Y'), default=[0.0, 0.0])
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.dry_run:
        need = [('--box', args.box), ('--depth', args.depth),
                ('--intrinsics', args.intrinsics), ('--camera-pose', args.camera_pose)]
        missing = [n for n, v in need if v is None]
        if missing:
            print('[ERR] dry-run 需要: %s' % ', '.join(missing))
            return 2
        k = dict(zip(('fx', 'fy', 'cx', 'cy'), args.intrinsics))
        u, v = box_center(args.box)
        p_cam = backproject(u, v, args.depth, k)
        cx, cy, cz, cyaw = args.camera_pose
        qz, qw = math.sin(cyaw / 2), math.cos(cyaw / 2)
        tf4 = pose_to_matrix(cx, cy, cz, 0.0, 0.0, qz, qw)
        p_map = transform_point(tf4, p_cam)
        goal = standoff_point((p_map[0], p_map[1]), tuple(args.robot), args.standoff)
        yaw = yaw_towards(tuple(args.robot), (p_map[0], p_map[1]))
        print('[BOX] center=(%.1f, %.1f) depth=%.2fm' % (u, v, args.depth))
        print('[CAM] point=(%.3f, %.3f, %.3f)' % tuple(p_cam))
        print('[MAP] object=(%.3f, %.3f, %.3f)' % tuple(p_map))
        print('[GOAL] approach=(%.3f, %.3f, yaw=%.3f) standoff=%.2fm' % (goal[0], goal[1], yaw, args.standoff))
        return 0

    if not HAVE_ROS:
        print('[ERR] 需要 ROS2 环境（--dry-run 可离线跑）')
        return 2

    prompt = ' '.join(args.prompt).strip()
    if not prompt:
        print('[ERR] 需要给出要寻找的目标，例如: python3 grounding.py "红箱子"')
        return 2

    # 先把检测器建起来：可选依赖缺失时立刻给出可执行提示，别等 ROS 起来才报错。
    det = make_detector(args.detector,
                        model_id=args.model_id,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                        device=args.device)

    rclpy.init()
    node = GroundingNode(args.color_topic, args.depth_topic, args.info_topic,
                         standoff=args.standoff, qos=args.qos)
    try:
        import time
        t0 = time.time()
        while not node.ready() and time.time() - t0 < 15.0:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.ready():
            print('[ERR] 15s 内没收到 相机图像/深度/内参，检查 topic 名')
            return 3
        res = node.locate(det, prompt, args.robot)
        if res is None:
            print('[MISS] 没找到 %r（降低阈值或换说法再试）' % prompt)
            return 4
        obj = res['object']
        print('[FOUND] %r score=%.3f depth=%.2fm object=(%.2f, %.2f, %.2f) '
              '-> goal=(%.2f, %.2f, yaw=%.2f)' % (
                  prompt, res['score'], res['depth'], obj[0], obj[1], obj[2],
                  res['x'], res['y'], res['yaw']))
        if args.navigate:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import llm_nav
            from rclpy.action import ActionClient
            from nav2_msgs.action import NavigateToPose
            ac = ActionClient(node, NavigateToPose, 'navigate_to_pose')
            label = llm_nav.goto(node, ac, NavigateToPose, 'map',
                                 res['x'], res['y'], res['yaw'])
            print('[NAV] %s' % label)
            return 0 if label == 'SUCCEEDED' else 5
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
