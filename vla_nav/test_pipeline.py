#!/usr/bin/env python3
"""test_pipeline.py —— 端到端「检测框 → 深度 → map → 落点」纯函数链路的测试

这条链路是 A2/A3 的公共主干：`grounding.locate_object(...)`。
ROS 节点、真机、未来换任何检测器（GroundingDINO / OWLv2 / 云端）都复用同一份几何，
所以它必须被单测钉死——这里用「手写深度表 + 手工构造的 Detection」验证，
不需要相机、不需要 ROS、不需要 torch。

    python3 test_pipeline.py
    python3 -m pytest test_pipeline.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _testkit
import grounding as G

K = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}
W, H = 640, 480
BOX = (300.0, 200.0, 380.0, 300.0)
IDENTITY_AT_ORIGIN = G.pose_to_matrix(0, 0, 0, 0, 0, 0, 1)


def flat_depth(w=W, h=H, value=2.0):
    """整张深度图同一个值。"""
    return [[value] * w for _ in range(h)]


def depth_with(value_in_box, value_out, box=BOX, w=W, h=H):
    """框内按 `value_in_box` 填充（标量，或按顺序循环的序列）、框外另一个值。

    用序列是为了验证"坏像素被过滤"：比如 [0, nan, inf, 2.0, 2.0] 铺满框内。
    """
    seq = value_in_box if isinstance(value_in_box, (list, tuple)) else None
    count = 0
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            inside = box[0] <= x < box[2] and box[1] <= y < box[3]
            if not inside:
                row.append(value_out)
            elif seq is None:
                row.append(value_in_box)
            else:
                row.append(seq[count % len(seq)])
                count += 1
        rows.append(row)
    return rows


class DuckDetection(object):
    """只有 box/score 属性的对象——验证 locate_object 不强制 isinstance。"""

    def __init__(self, box, score):
        self.box = box
        self.score = score


# ---------------------------------------------------------------- depth_patch
def test_depth_patch_extracts_bbox_rows_major():
    dep = [[float(x + 1000 * y) for x in range(W)] for y in range(H)]
    patch = G.depth_patch(dep, (10, 20, 12, 22))
    assert len(patch) == 9, '3x3 框应取 9 个像素，实际 %d' % len(patch)
    expect = [1000 * y + x for y in (20, 21, 22) for x in (10, 11, 12)]
    assert [int(v) for v in patch] == expect, '行优先顺序不对: %r' % (patch,)


def test_depth_patch_clamps_and_handles_empty():
    dep = flat_depth()
    assert G.depth_patch(dep, (-50, -50, -10, -10)) == [], '全在图像外应返回 []'
    assert G.depth_patch(dep, (W + 10, H + 10, W + 20, H + 20)) == [], '全在图像外应返回 []'
    assert len(G.depth_patch(dep, (-5, -5, 5, 5))) == 36, '越界框应夹到左上角 6x6'
    assert G.depth_patch([], BOX) == [], '空图像应返回 []'
    assert G.depth_patch(dep, (10, 10, 10, 10)) != [], '退化框仍取到 1 个像素（+1 语义）'


def test_depth_patch_accepts_numpy():
    try:
        import numpy as np
    except ImportError:
        print('SKIP numpy 互操作（本机没有 numpy）')
        return
    arr = np.full((H, W), 2.5, dtype=np.float32)
    patch = G.depth_patch(arr, BOX)
    _testkit.approx(G.median_depth(patch), 2.5, note='numpy 深度图应能直接喂')


# ---------------------------------------------------------------- 主干 happy path
def test_locate_object_happy_path():
    """手算核对：像素中心 (340,250)、深度 2.0、相机在 map(1,2,0.3) 朝 +x。"""
    det = G.Detection(BOX, 0.9, 'box')
    res = G.locate_object(flat_depth(), K, [det], G.pose_to_matrix(1, 2, 0.3, 0, 0, 0, 1),
                          (0.0, 0.0), standoff=0.8)
    assert res is not None, '正常输入不应返回 None'

    # cam: x=(340-320)*2/600, y=(250-240)*2/600, z=2 → map 同值 + 平移
    obj = (1 + 20 * 2.0 / 600, 2 + 10 * 2.0 / 600, 0.3 + 2.0)
    _testkit.approx_list(res['object'], obj, note='object map 坐标')

    # 落点 = 从机器人(0,0)朝物体方向、距物体 0.8 m
    norm = math.hypot(obj[0], obj[1])
    goal = (obj[0] - 0.8 * obj[0] / norm, obj[1] - 0.8 * obj[1] / norm)
    _testkit.approx_list(res['goal'][:2], goal, note='落点')
    _testkit.approx_list(res['goal'], goal + (res['goal'][2],), note='goal 三元组')
    _testkit.approx(res['x'], goal[0], note='x 兼容字段')
    _testkit.approx(res['y'], goal[1], note='y 兼容字段')
    _testkit.approx(res['yaw'], math.atan2(obj[1], obj[0]), note='yaw 朝向物体')

    # 附带的元信息要能被上层拿去做日志/阈值调参
    _testkit.approx(res['score'], 0.9, note='score')
    _testkit.approx(res['depth'], 2.0, note='depth')
    assert tuple(res['box']) == BOX, 'box 应原样回传，实际 %r' % (res['box'],)


def test_yaw_matches_object_bearing():
    """任意象限：yaw 必须等于「机器人 → 物体」的方位角（不是固定 +x）。"""
    tf = G.pose_to_matrix(0, 0, 0, 0, 0, 0, 1)     # 相机在 map 原点、光学系与 map 同向
    robot = (-1.0, 0.5)                            # 机器人不在相机原点上，方位角才不为 0
    for center in [(340, 240), (320, 120), (320, 360), (180, 300), (500, 180)]:
        box = (center[0] - 20, center[1] - 20, center[0] + 20, center[1] + 20)
        res = G.locate_object(flat_depth(), K, [G.Detection(box, 0.9)], tf, robot, 0.8)
        assert res is not None, '%r 应解析成功' % (center,)
        obj = res['object']
        _testkit.approx(res['yaw'], math.atan2(obj[1] - robot[1], obj[0] - robot[0]),
                        note='yaw 应指向物体 (%r)' % (center,))


def test_optical_rotation_maps_height():
    """相机抬头轴向上装（绕 x 转 -90°，朝 map +y）：像素上下偏移要变成高度差。

    这是最容易写错的一处：像素 v 变小（图像上方）→ 物体更高，不是更远。
    """
    qx, qw = -math.sin(math.pi / 4), math.cos(math.pi / 4)      # Rx(-90°)
    tf = G.pose_to_matrix(0, 0, 0.5, qx, 0, 0, qw)              # 相机离地 0.5 m
    for v, expect_z in [(140, 0.5 + (240 - 140) * 2.0 / 600),   # 上方 → 更高
                        (340, 0.5 - (340 - 240) * 2.0 / 600)]:  # 下方 → 更低
        box = (300, v - 20, 340, v + 20)
        res = G.locate_object(flat_depth(), K, [G.Detection(box, 0.9)], tf, (0, 0), 0.8)
        assert res is not None, 'v=%d 应解析成功' % v
        obj = res['object']
        _testkit.approx(obj[1], 2.0, note='应落在相机正前方 2 m（map +y）')
        _testkit.approx(obj[0], (320 - 320) * 2.0 / 600, note='水平居中时 map x≈0')
        _testkit.approx(obj[2], expect_z, note='v=%d 的高度' % v)


def test_only_bbox_pixels_used():
    """框外像素（比如墙 5 m）不能污染物体深度。"""
    dep = depth_with(2.0, 5.0)
    res = G.locate_object(dep, K, [G.Detection(BOX, 0.9)], IDENTITY_AT_ORIGIN, (0, 0))
    _testkit.approx(res['depth'], 2.0, note='应只取框内 2.0，而不是混进 5.0')


def test_depth_robust_to_zero_nan_inf():
    """真实深度图的 0/NaN/inf 是"没数据"，不能被当成 0 m 或 1e9 m。"""
    bad = [0.0, float('nan'), float('inf'), 0.0, 0.0]
    dep = depth_with(bad, 0.0)                        # 框内全是坏值
    assert G.locate_object(dep, K, [G.Detection(BOX, 0.9)], IDENTITY_AT_ORIGIN,
                           (0, 0)) is None, '坏像素应判"没数据"返回 None'

    good = [0.0, float('nan'), float('inf'), 2.0, 2.0]
    dep = depth_with(good, 0.0)
    res = G.locate_object(dep, K, [G.Detection(BOX, 0.9)], IDENTITY_AT_ORIGIN, (0, 0))
    _testkit.approx(res['depth'], 2.0, note='坏像素应被过滤后取 2.0')


# ---------------------------------------------------------------- 主干失败路径
def test_locate_object_returns_none():
    dep = flat_depth()
    det = G.Detection(BOX, 0.9)
    ids = "locate_object 失败路径"
    assert G.locate_object(dep, K, [], IDENTITY_AT_ORIGIN, (0, 0)) is None, ids + ': 无检测'
    assert G.locate_object(dep, K, [G.Detection(BOX, 0.1)], IDENTITY_AT_ORIGIN,
                           (0, 0)) is None, ids + ': score 太低'
    assert G.locate_object(dep, K, [G.Detection((0, 0, 5, 5), 0.99)], IDENTITY_AT_ORIGIN,
                           (0, 0)) is None, ids + ': 框太小'
    assert G.locate_object([], K, [det], IDENTITY_AT_ORIGIN, (0, 0)) is None, ids + ': 空深度图'
    # 框完全在图像外 → 取不到深度
    assert G.locate_object(dep, K, [G.Detection((700, 600, 800, 700), 0.9)],
                           IDENTITY_AT_ORIGIN, (0, 0)) is None, ids + ': 框越界'


def test_locate_object_threshold_kwargs():
    """阈值要能从上往下透传（真机调参靠它）。"""
    dep = flat_depth(value=2.0)
    det = G.Detection(BOX, 0.5)
    assert G.locate_object(dep, K, [det], IDENTITY_AT_ORIGIN, (0, 0),
                           min_score=0.6) is None, 'min_score 应生效'
    assert G.locate_object(dep, K, [det], IDENTITY_AT_ORIGIN, (0, 0),
                           min_area=1e6) is None, 'min_area 应生效'
    assert G.locate_object(dep, K, [det], IDENTITY_AT_ORIGIN, (0, 0),
                           min_depth=3.0) is None, 'min_depth 应生效'
    assert G.locate_object(dep, K, [det], IDENTITY_AT_ORIGIN, (0, 0),
                           max_depth=1.0) is None, 'max_depth 应生效'


def test_multiple_detections_picks_best():
    dep = depth_with(2.0, 0.0, box=BOX)
    weak = G.Detection(BOX, 0.4)
    strong = G.Detection(BOX, 0.95)
    res = G.locate_object(dep, K, [weak, strong], IDENTITY_AT_ORIGIN, (0, 0))
    _testkit.approx(res['score'], 0.95, note='应选 score 更高的框')


def test_duck_typed_detection():
    """真检测器的返回对象只要带 box/score 就能用，不强制 isinstance(Detection)。"""
    dep = flat_depth()
    res = G.locate_object(dep, K, [DuckDetection(BOX, 0.8)], IDENTITY_AT_ORIGIN, (0, 0))
    assert res is not None, '鸭子类型应可用'
    _testkit.approx(res['score'], 0.8, note='鸭子类型 score')


# ---------------------------------------------------------------- 落点策略
def test_standoff_policy():
    """机器人到物体距离不同时，落点策略要可预期。"""
    dep = flat_depth()
    det = G.Detection(BOX, 0.9)
    tf = G.pose_to_matrix(1, 2, 0.3, 0, 0, 0, 1)

    # 远：停在物体前 0.8 m
    res = G.locate_object(dep, K, [det], tf, (0.0, 0.0), standoff=0.8)
    obj = res['object']
    d = math.hypot(obj[0] - res['goal'][0], obj[1] - res['goal'][1])
    _testkit.approx(d, 0.8, note='落点距物体 0.8 m')

    # 近（比 standoff 还近）：原地不动，别后退
    res = G.locate_object(dep, K, [det], tf, (1.0, 1.9), standoff=0.8)
    _testkit.approx_list(res['goal'][:2], [1.0, 1.9], note='太近就原地')

    # 正好在物体位置：退到物体后方 0.8 m，不能除零
    res = G.locate_object(dep, K, [det], tf, (obj[0], obj[1]), standoff=0.8)
    _testkit.approx_list(res['goal'][:2], [obj[0] - 0.8, obj[1]], note='重合时退后')


def test_need_ros_message_when_not_dry_run():
    """没有 ROS 时，非 dry-run 的实体流程要明确拒绝而不是崩栈。"""
    if G.HAVE_ROS:
        print('SKIP 本机有 ROS（不该有这个测试环境）')
        return
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = G.main(['红箱子'])
    assert code == 2, '无 ROS 应退出 2，实际 %r' % code
    assert 'ROS2' in buf.getvalue(), '应提示需要 ROS2:\n%s' % buf.getvalue()


if __name__ == '__main__':
    sys.exit(_testkit.run(globals()))
