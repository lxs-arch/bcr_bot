#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 粘合层测试：_image_to_array / _intrinsics_from_info / GroundingNode / main()。

为什么需要这个文件：`grounding.py` 的几何部分被 test_pipeline.py 扎实覆盖，但
「消息 → 数组 → 节点 → 主流程」这一层只用真机跑过，而真机跑一次要开一整个
Jazzy + MuJoCo。历史上这里藏过一个必崩的 bug —— 节点喂给几何链的是 ndarray，
而 depth_patch 里写了 `if not depth:`（numpy 上抛 ambiguous），假图像测试完全
看不见。

做法：不用 ROS，塞一套**够用就行的假 rclpy / sensor_msgs / tf2_ros / geometry_msgs**
进 sys.modules，再 importlib.reload(grounding) 让 `if HAVE_ROS:` 分支真正执行，
从而可以在 Mac 上验证粘合层。测完把 sys.modules 还原并 reload 回纯几何版本，
保证别的测试文件看到的仍然是 HAVE_ROS=False。

几何约定（本文件所有期望值都基于它）：假帧 64x48、内参 fx=fy=500、主点 (32,24)；
相机光学系装在 map 原点上方 0.5m、绕 x 转 -90°（光学 z 轴 → map +x），也就是
"小车在原点朝 +x，相机水平朝前"。于是光学系里的 (0, 0, 2.5) → map (2.5, 0, 0.5)。

限制（诚实标注）：假模块只实现被调用的接口，验证的是「接线正确」，不是
「ROS2 API 语义正确」。TF 时机/超时、QoS、真实 Image 的内存布局仍须远程机验证。
"""
import contextlib
import importlib
import os
import struct
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as _np                                          # noqa: F401
    HAVE_NUMPY = True
except ImportError:                                              # 机器人机上可能没有 numpy
    HAVE_NUMPY = False

import _testkit                                                    # noqa: E402
import grounding as G                                              # noqa: E402

PY_ROOTS = ('rclpy', 'sensor_msgs', 'geometry_msgs', 'tf2_ros', 'nav2_msgs', 'llm_nav')

W, H = 64, 48
DEPTH_M = 2.5                                  # 米，全局统一的假深度
BOX = (24.0, 16.0, 40.0, 32.0)                 # 16x16=256 px²，够过 min_area=64
BOX_CENTER = (32.0, 24.0)                      # == 主点 → 光学系里 x=y=0


# ------------------------------------------------------------------ 假 ROS 世界
class FakeHeader(object):
    def __init__(self, frame_id='bcr_bot/kinect_camera_optical_frame'):
        self.frame_id = frame_id
        self.stamp = None


class FakeImage(object):
    def __init__(self, data, height, width, encoding, frame_id=None):
        self.data = data
        self.height = height
        self.width = width
        self.encoding = encoding
        self.header = FakeHeader() if frame_id is None else FakeHeader(frame_id)


class FakeCameraInfo(object):
    def __init__(self, fx=500.0, fy=500.0, cx=BOX_CENTER[0], cy=BOX_CENTER[1]):
        self.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


class FakeVector(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class FakeQuat(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class FakeTransformStamped(object):
    def __init__(self, xyz, quat):
        self.transform = types.SimpleNamespace(translation=FakeVector(*xyz),
                                               rotation=FakeQuat(*quat))


S = 0.7071067811865476                          # sin(45°) = cos(45°)
SENSOR_QOS = object()                           # 假 rclpy.qos.qos_profile_sensor_data
# 光学系 → map：绕 y 转 +90°（光学 z 前 → map x 前），再抬高 0.5m 当相机安装高度
# 注意是绕 y 不是绕 x：绕 x 转 -90° 会把光学 z 映射到 map +y（小车"朝左"）。
OPTICAL_TO_MAP = FakeTransformStamped((0.0, 0.0, 0.5), (0.0, S, 0.0, S))


class FakeBuffer(object):
    def __init__(self, answer=None):
        self.answer = answer or OPTICAL_TO_MAP
        self.calls = []

    def lookup_transform(self, target, source, when):
        self.calls.append((target, source))
        return self.answer


class FakeLogger(object):
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)


class FakeWorld(object):
    """一次假 ROS 会话：所有节点自动登记，可主动投递消息或排队交给 spin_once。"""

    def __init__(self):
        self.nodes = []
        self.inited = False
        self.shutdown_count = 0
        self.spin_scripts = []          # [(topic, message)]，spin_once 依次吐出

    # -- 节点登记
    def node_class(self):
        world = self

        class RegisteredNode(FakeNode):
            def __init__(self, name):
                FakeNode.__init__(self, name)
                world.nodes.append(self)

        return RegisteredNode

    # -- 投递（按 topic，因为彩色和深度都是 sensor_msgs/Image，类型区分不了）
    def deliver(self, topic, message, msgtype=None):
        hits = 0
        for node in self.nodes:
            for mtype, stopic, cb, _qos in node.subs:
                if stopic == topic and (msgtype is None or mtype is msgtype):
                    cb(message)
                    hits += 1
        return hits

    def spin_once(self, node, timeout_sec=0.0):
        if self.spin_scripts:
            topic, msg = self.spin_scripts.pop(0)
            self.deliver(topic, msg)


class FakeNode(object):
    """只实现 grounding.py 用到的那几个方法。"""

    def __init__(self, name):
        self.name = name
        self.subs = []                  # [(msgtype, topic, callback)]
        self.logger = FakeLogger()
        self.destroyed = False

    def create_subscription(self, msgtype, topic, callback, qos):
        self.subs.append((msgtype, topic, callback, qos))

    def get_logger(self):
        return self.logger

    def destroy_node(self):
        self.destroyed = True


def _fake_action_client(node, action_type, name):
    return types.SimpleNamespace(node=node, action_type=action_type, name=name)


@contextlib.contextmanager
def ros_env(goto_result='SUCCEEDED'):
    """装好假 ROS → reload(grounding) → 交出 (模块, 世界) → 无条件还原。"""
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in PY_ROOTS}
    for k in list(saved):
        sys.modules.pop(k, None)

    world = FakeWorld()
    rclpy = types.ModuleType('rclpy')

    def init():
        world.inited = True

    def shutdown():
        world.shutdown_count += 1

    rclpy.init = init
    rclpy.shutdown = shutdown
    rclpy.spin_once = world.spin_once
    rclpy.node = types.SimpleNamespace(Node=world.node_class())
    rclpy.time = types.SimpleNamespace(Time=lambda: 'now')
    rclpy.action = types.SimpleNamespace(ActionClient=_fake_action_client)
    rclpy.qos = types.SimpleNamespace(qos_profile_sensor_data=SENSOR_QOS)

    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs.msg = types.SimpleNamespace(Image=FakeImage, CameraInfo=FakeCameraInfo)
    geometry_msgs = types.ModuleType('geometry_msgs')
    geometry_msgs.msg = types.SimpleNamespace(PoseStamped=object)
    tf2_ros = types.ModuleType('tf2_ros')
    tf2_ros.Buffer = FakeBuffer
    tf2_ros.TransformListener = lambda buf, node: None
    nav2_msgs = types.ModuleType('nav2_msgs')
    nav2_msgs.action = types.SimpleNamespace(NavigateToPose=object)
    llm_nav = types.ModuleType('llm_nav')
    llm_nav.calls = []

    def goto(node, ac, action_type, frame, x, y, yaw):
        llm_nav.calls.append((frame, x, y, yaw))
        return goto_result

    llm_nav.goto = goto

    sys.modules.update({'rclpy': rclpy, 'sensor_msgs': sensor_msgs,
                        'geometry_msgs': geometry_msgs, 'tf2_ros': tf2_ros,
                        'nav2_msgs': nav2_msgs, 'llm_nav': llm_nav})
    for name, mod in (('rclpy.node', rclpy.node), ('rclpy.time', rclpy.time),
                      ('rclpy.qos', rclpy.qos),
                      ('rclpy.action', rclpy.action),
                      ('sensor_msgs.msg', sensor_msgs.msg),
                      ('geometry_msgs.msg', geometry_msgs.msg),
                      ('nav2_msgs.action', nav2_msgs.action)):
        sys.modules[name] = mod

    try:
        mod = importlib.reload(G)
        mod._world = world
        mod._llm_nav = llm_nav
        yield mod, world
    finally:
        for k in [m for m in list(sys.modules) if m.split('.')[0] in PY_ROOTS]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        importlib.reload(G)          # 还原成纯几何版（HAVE_ROS=False）


# ----------------------------------------------------------------- 造像素字节
def depth_bytes(value=DEPTH_M, dtype='float32', w=W, h=H):
    """造一段 32FC1 / 16UC1 原始像素（行优先）。"""
    n = w * h
    if dtype == 'uint16':
        return struct.pack('<%dH' % n, *([int(value * 1000)] * n))
    return struct.pack('<%df' % n, *([float(value)] * n))


def color_bytes(w=W, h=H, rgb=(10, 20, 30)):
    return bytes(bytearray(list(rgb) * (w * h)))


def depth_image(value=DEPTH_M, dtype='float32', w=W, h=H):
    return FakeImage(depth_bytes(value, dtype, w, h), h, w,
                     '32FC1' if dtype == 'float32' else '16UC1')


def color_image(encoding='rgb8', rgb=(10, 20, 30), w=W, h=H):
    return FakeImage(color_bytes(w, h, rgb), h, w, encoding)


def standard_detector(mod, score=0.9):
    return mod.StubDetector([mod.Detection(BOX, score, 'box')])


def needs_numpy(fn):
    """没有 numpy 就跳过：ROS 粘合层本身要 numpy（节点里 `import numpy as np`），
    而且这里要验的正是 ndarray 路径，模拟成行列表就失去意义了。"""
    def wrapper(*args, **kwargs):
        if not HAVE_NUMPY:
            print('SKIP %s（没有 numpy，装 numpy 后再测这一层）' % fn.__name__)
            return None
        return fn(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------- 默认 topic（与 CLI 一致）
COLOR_T = '/bcr_bot/kinect_camera/color/image_raw'
DEPTH_T = '/bcr_bot/kinect_camera/depth/image_raw'
INFO_T = '/bcr_bot/kinect_camera/color/camera_info'


def make_node(mod, standoff=0.8):
    return mod.GroundingNode(COLOR_T, DEPTH_T, INFO_T, standoff=standoff)


def feed(world, mod, info=None, color=None, depth=None):
    """按真实启动顺序投递：内参 → 彩色 → 深度（Kinect 深度按毫米发，即 16UC1）。"""
    world.deliver(INFO_T, FakeCameraInfo() if info is None else info)
    if color is not False:
        world.deliver(COLOR_T, color_image() if color is None else color)
    if depth is not False:
        world.deliver(DEPTH_T, depth_image(dtype='uint16') if depth is None else depth)


def queue_for_main(world, mod):
    """给 main() 的 spin_once 排队：内参 → 彩色 → 深度。"""
    world.spin_scripts = [(INFO_T, FakeCameraInfo()),
                          (COLOR_T, color_image()),
                          (DEPTH_T, depth_image(dtype='uint16'))]


# ------------------------------------------------------- _image_to_array / 内参
@needs_numpy
def test_image_to_array_32fc1():
    with ros_env() as (mod, _world):
        arr = mod._image_to_array(depth_image())
        assert arr.shape == (H, W), '32FC1 应解成 (高, 宽)=(%d, %d)，实际 %r' % (H, W, arr.shape)
        assert arr.dtype.name == 'float32', arr.dtype
        assert _testkit.approx(float(arr[2][3]), DEPTH_M, note='行优先索引深度值') is None


@needs_numpy
def test_image_to_array_16uc1_converts_mm_to_m():
    with ros_env() as (mod, _world):
        arr = mod._image_to_array(depth_image(value=2.5, dtype='uint16'))
        assert arr.dtype.name == 'float32', '转米后应是 float32，实际 %r' % (arr.dtype,)
        assert _testkit.approx(arr[0][0], 2.5, note='16UC1 毫米转米') is None


@needs_numpy
def test_image_to_array_color_channel_order():
    with ros_env() as (mod, _world):
        rgb = mod._image_to_array(color_image('rgb8'))
        bgr = mod._image_to_array(color_image('bgr8'))
        assert tuple(rgb[0, 0]) == (10, 20, 30), tuple(rgb[0, 0])
        assert tuple(bgr[0, 0]) == (30, 20, 10), 'bgr8 必须翻转通道，实际 %r' % (tuple(bgr[0, 0]),)


@needs_numpy
def test_image_to_array_rejects_unknown_encoding():
    with ros_env() as (mod, _world):
        try:
            mod._image_to_array(color_image('mono16'))
        except ValueError as exc:
            assert 'mono16' in str(exc), str(exc)
        else:
            raise AssertionError('不认识的编码必须报错，不能静默返回垃圾')


@needs_numpy
def test_intrinsics_from_info():
    with ros_env() as (mod, _world):
        k = mod._intrinsics_from_info(FakeCameraInfo(fx=520.0, fy=521.0, cx=3.5, cy=2.5))
        assert k == {'fx': 520.0, 'fy': 521.0, 'cx': 3.5, 'cy': 2.5}, k


# ------------------------------------------------------------------- 节点接线
@needs_numpy
def test_node_subscribes_with_sensor_data_qos():
    """相机/深度是 best_effort 发布的，默认 reliable 订阅会一帧都收不到
    （症状：15s 超时、退出码 3）。三个订阅都必须用 sensor data QoS。"""
    with ros_env() as (mod, _world):
        node = mod.GroundingNode(COLOR_T, DEPTH_T, INFO_T)
        assert len(node.subs) == 3, node.subs
        for _msgtype, topic, _cb, qos in node.subs:
            assert qos is mod.qos_profile_sensor_data, (
                '%s 订阅的 QoS 应是 qos_profile_sensor_data，实际 %r' % (topic, qos))


@needs_numpy
def test_node_reliable_qos_opt_in():
    """少数相机用 reliable 发布时可显式退回默认 QoS（队列深度 10）。"""
    with ros_env() as (mod, _world):
        node = mod.GroundingNode(COLOR_T, DEPTH_T, INFO_T, qos='reliable')
        assert [sub[3] for sub in node.subs] == [10, 10, 10], node.subs


@needs_numpy
def test_node_ready_gating():
    with ros_env() as (mod, world):
        node = make_node(mod)
        assert node.ready() is False, '刚起来不能 ready'
        world.deliver(INFO_T, FakeCameraInfo())
        assert node.ready() is False, '只有内参还不够'
        world.deliver(COLOR_T, color_image())
        assert node.ready() is False, '有内参 + 彩色、缺深度时还不算 ready'
        world.deliver(DEPTH_T, depth_image(dtype='uint16'))
        assert node.ready() is True, '内参 + 彩色 + 深度齐了才是 ready'


@needs_numpy
def test_node_locate_end_to_end():
    """假相机装在原点上方 0.5m 朝 +x，框在图像正中 → 物体 map (2.5, 0, 0.5)。"""
    with ros_env() as (mod, world):
        node = make_node(mod)
        feed(world, mod)
        res = node.locate(standard_detector(mod), '红箱子', (0.0, 0.0))
        assert res is not None, '粘合层必须能走通 locate（ndarray 深度图不该崩）'
        obj = res['object']
        _testkit.approx(obj[0], 2.5, tol=1e-3, note='map x（光学 z 前 = map +x）')
        _testkit.approx(obj[1], 0.0, tol=1e-3, note='map y（框居中）')
        _testkit.approx(obj[2], 0.5, tol=1e-3, note='map z（相机安装高度）')
        _testkit.approx(res['depth'], DEPTH_M, note='中位深度')
        _testkit.approx(res['x'], 1.7, tol=1e-6, note='goal 在物体前 0.8m')
        assert node.tf.calls == [('map', 'bcr_bot/kinect_camera_optical_frame')], \
            'TF 必须查 target=map ← 彩色图 frame，实际 %r' % (node.tf.calls,)


@needs_numpy
def test_node_locate_returns_none_without_detections():
    with ros_env() as (mod, world):
        node = make_node(mod)
        feed(world, mod)
        assert node.locate(mod.StubDetector([]), '红箱子', (0.0, 0.0)) is None


@needs_numpy
def test_node_locate_returns_none_when_not_ready():
    with ros_env() as (mod, _world):
        node = make_node(mod)
        assert node.locate(standard_detector(mod), '红箱子', (0.0, 0.0)) is None


# --------------------------------------------------------------------- main()
@needs_numpy
def test_main_runs_whole_flow_with_stub():
    with ros_env() as (mod, world):
        mod.make_detector = lambda name, **kw: standard_detector(mod)
        queue_for_main(world, mod)
        assert mod.main(['红箱子', '--robot', '0', '0']) == 0
        assert world.inited, 'rclpy.init 必须被调用'
        assert world.shutdown_count == 1, 'rclpy.shutdown 必须调用一次，实际 %d' % world.shutdown_count
        assert world.nodes[0].destroyed is True, 'finally 里必须 destroy_node'


@needs_numpy
def test_main_returns_4_when_nothing_found():
    with ros_env() as (mod, world):
        mod.make_detector = lambda name, **kw: mod.StubDetector([])
        queue_for_main(world, mod)
        assert mod.main(['红箱子']) == 4


@needs_numpy
def test_main_navigate_wiring():
    """--navigate 应把 locate 出的 goal 交给 llm_nav.goto；goto 自身逻辑不在此验证。"""
    with ros_env(goto_result='SUCCEEDED') as (mod, world):
        mod.make_detector = lambda name, **kw: standard_detector(mod)
        queue_for_main(world, mod)
        assert mod.main(['红箱子', '--navigate', '--robot', '0', '0']) == 0
        (frame, x, y, yaw), = mod._llm_nav.calls
        assert frame == 'map', frame
        _testkit.approx(x, 1.7, tol=1e-6, note='goal 在物体前 0.8m')
        _testkit.approx(y, 0.0, tol=1e-6, note='框居中 → y≈0')
        _testkit.approx(yaw, 0.0, tol=1e-6, note='yaw 朝物体')


@needs_numpy
def test_main_navigate_failure_exit_code():
    with ros_env(goto_result='ABORTED') as (mod, world):
        mod.make_detector = lambda name, **kw: standard_detector(mod)
        queue_for_main(world, mod)
        assert mod.main(['红箱子', '--navigate']) == 5, '导航失败应返回 5'


@needs_numpy
def test_main_missing_prompt():
    with ros_env() as (mod, _world):
        assert mod.main([]) == 2, '没有目标词应返回 2'


if __name__ == '__main__':
    sys.exit(_testkit.run(globals()))
