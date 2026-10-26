#!/usr/bin/env python3
"""test_grounding.py —— A2 视觉接地里「纯几何」部分的单元测试（pytest 兼容）

只测不需要 ROS / GPU / numpy / 图像的部分：
四元数与位姿、投影/反投影、深度取样、bbox 工具、落点与地面求交、CLI dry-run。

整条管线级的用例在 `test_pipeline.py`，检测器后端在 `test_detector.py`。

    python3 test_grounding.py        # 没有 pytest 的机器用这个
    python3 -m pytest test_grounding.py -q
"""

import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _testkit
import grounding as G

K = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}


def _raises(fn, exc):
    """断言 fn() 抛 exc（不依赖 pytest.raises，两种运行方式都能用）。"""
    try:
        fn()
    except exc:
        return
    except Exception as other:                                # noqa: BLE001
        raise AssertionError('抛了 %r，期望 %s' % (other, exc.__name__))
    raise AssertionError('没有抛 %s' % exc.__name__)


# ---------------------------------------------------------------- 位姿 / 矩阵
def test_quaternion():
    m = G.quat_to_matrix(0, 0, 0, 1)
    _testkit.approx_list([m[0][0], m[1][1], m[2][2]], [1, 1, 1], note='identity diag')

    half = math.sin(math.pi / 4)
    m = G.quat_to_matrix(0, 0, half, half)                    # 绕 z 转 90°
    _testkit.approx_list(G.mat_vec(m, [1, 0, 0]), [0, 1, 0], note='yaw90 maps +x to +y')

    _, _, qz, qw = G.yaw_to_quat(0.7)
    m2 = G.quat_to_matrix(0, 0, qz, qw)
    _testkit.approx_list(G.mat_vec(m2, [1, 0, 0]),
                         [math.cos(0.7), math.sin(0.7), 0], note='yaw_to_quat')


def test_pose_transform():
    tf = G.pose_to_matrix(1, 2, 3, 0, 0, 0, 1)
    _testkit.approx_list(G.transform_point(tf, [0.5, -0.5, 0.0]), [1.5, 1.5, 3.0],
                         note='identity rot + translate')

    half = math.sin(math.pi / 4)
    tf = G.pose_to_matrix(1, 0, 0, 0, 0, half, half)          # 先转 90° 再平移
    _testkit.approx_list(G.transform_point(tf, [1, 0, 0]), [1, 1, 0],
                         note='rot then translate')

    _testkit.approx_list([G.identity(4)[i][i] for i in range(4)], [1, 1, 1, 1],
                         note='identity diag')


def test_matrix_helpers():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    got = G.mat_mul(a, b)
    _testkit.approx_list([got[0][0], got[0][1], got[1][0], got[1][1]],
                         [19, 22, 43, 50], note='mat_mul')
    t = G.transpose(a)
    _testkit.approx_list([t[0][0], t[0][1], t[1][0], t[1][1]], [1, 3, 2, 4],
                         note='transpose')


# ---------------------------------------------------------------- 投影
def test_projection():
    _testkit.approx_list(G.project_point([0, 0, 2], K), [320, 240],
                         note='optical center hits principal point')
    _testkit.approx_list(G.backproject(320, 240, 2.0, K), [0, 0, 2.0],
                         note='backproject center')

    world = [0.5, -0.3, 1.8]
    u, v = G.project_point(world, K)
    _testkit.approx_list(G.backproject(u, v, world[2], K), world,
                         note='project/backproject round trip')

    _raises(lambda: G.project_point([0, 0, 0], K), ValueError)
    _raises(lambda: G.backproject(320, 240, 0.0, K), ValueError)


# ---------------------------------------------------------------- 深度取样
def test_median_depth():
    _testkit.approx(G.median_depth([0.0, float('nan'), 1.5, 1.7, 99.0]), 1.6,
                    note='filters 0/NaN/too-far')
    _testkit.approx(G.median_depth([1.0, 2.0, 3.0]), 2.0, note='odd count middle')
    _testkit.approx(G.median_depth([1.0, 3.0]), 2.0, note='even count average')
    assert G.median_depth([]) is None, '空输入应返回 None'
    assert G.median_depth([0.0, float('inf'), None, 20.0]) is None, '无有效像素应返回 None'
    _testkit.approx(G.median_depth([0.05, 0.5]), 0.5, note='min_depth respected')
    _testkit.approx(G.median_depth([1.0, 2.0], max_depth=1.5), 1.0,
                    note='max_depth respected')


# ---------------------------------------------------------------- bbox / 选择
def test_boxes():
    _testkit.approx_list(G.box_center((10, 20, 30, 40)), [20, 30], note='box center')
    _testkit.approx(G.box_area((0, 0, 8, 8)), 64.0, note='box area')
    _testkit.approx(G.box_area((5, 5, 4, 4)), 0.0, note='inverted box clamps to zero')


def test_pick_best():
    dets = [G.Detection((0, 0, 5, 5), 0.9),                   # 面积 25 < min_area 64 → 过滤
            G.Detection((0, 0, 20, 20), 0.5),
            G.Detection((0, 0, 30, 30), 0.35)]
    best = G.pick_best(dets)
    assert best is not None, '过滤后应还剩候选'
    _testkit.approx(best.score, 0.5, note='small box filtered, highest remaining wins')

    assert G.pick_best([G.Detection((0, 0, 5, 5), 0.99)]) is None, '全被面积过滤应返回 None'
    assert G.pick_best([]) is None, '空列表应返回 None'
    assert G.pick_best(dets, min_score=0.95) is None, 'min_score 过滤掉全部应返回 None'


# ---------------------------------------------------------------- 落点 / 地面
def test_standoff_and_yaw():
    _testkit.approx_list(G.standoff_point((5, 0), (0, 0), 0.8), [4.2, 0.0],
                         note='normal standoff')
    _testkit.approx_list(G.standoff_point((0, 0), (0, 0), 0.8), [-0.8, 0.0],
                         note='robot on target, no div0')
    _testkit.approx_list(G.standoff_point((0.3, 0), (0, 0), 0.8), [0.0, 0.0],
                         note='closer than standoff, stay put')
    _testkit.approx(G.yaw_towards((0, 0), (0, 3)), math.pi / 2, note='yaw toward +y')


def test_ground_hit():
    hit = G.ground_hit([0, 0, 1.0], [0, 1, -1], 0.0)
    assert hit is not None, '射线应打中地面'
    _testkit.approx_list(hit, [0, 1, 0], note='45° down hits at 1 m')
    assert G.ground_hit([0, 0, 1.0], [0, 1, 0], 0.0) is None, '平行射线应返回 None'
    assert G.ground_hit([0, 0, 1.0], [0, 1, 1], 0.0) is None, '朝上的射线应返回 None'


# ---------------------------------------------------------------- CLI dry-run
def run_cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = G.main(argv)
    return code, buf.getvalue()


def test_cli_dry_run():
    argv = ['--dry-run', '--box', '300', '200', '380', '300', '--depth', '2.0',
            '--intrinsics', '600', '600', '320', '240',
            '--camera-pose', '1', '2', '0.3', '0',
            '--robot', '0', '0', '--standoff', '0.8']
    code, out = run_cli(argv)
    assert code == 0, 'dry-run 应退出 0，实际 %r\n%s' % (code, out)
    assert '[MAP]' in out, '应打印 map 点:\n%s' % out
    assert '[GOAL]' in out, '应打印落点:\n%s' % out

    # 手算: 像素框中心 (340,250) → cam(0.0667, 0.0333, 2.0) → map(1.0667, 2.0333, 2.3)
    obj = [1 + (340 - 320) * 2.0 / 600, 2 + (250 - 240) * 2.0 / 600]
    _testkit.approx_list(obj, [1.0666667, 2.0333333], note='hand-computed object xy')
    goal = G.standoff_point((obj[0], obj[1]), (0, 0), 0.8)
    assert '[GOAL] approach=(%.3f, %.3f' % (goal[0], goal[1]) in out, (
        'GOAL 与手算不一致:\n%s' % out)


def test_cli_missing_args():
    code, out = run_cli(['--dry-run', '--depth', '2.0'])
    assert code == 2, '缺参数应退出 2，实际 %r' % code
    assert '[ERR]' in out, '应解释缺什么参数:\n%s' % out


if __name__ == '__main__':
    sys.exit(_testkit.run(globals()))
