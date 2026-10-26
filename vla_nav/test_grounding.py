#!/usr/bin/env python3
"""test_grounding.py —— A2 视觉接地里「纯几何」部分的单元测试

只测不需要 ROS / GPU / 图像的部分：
四元数与位姿、投影/反投影、深度取样、落点计算、检测器选择、CLI dry-run。

    python3 test_grounding.py            # 全跑
    python3 test_grounding.py -v         # 打印每条断言

没有 pytest 依赖，退出码非 0 表示失败，方便 CI。
"""

import io
import math
import os
import sys
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grounding as G

K = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}
CASE = [0, 0]
FAILED = []


def eq(actual, expect, tol=1e-9, note=''):
    CASE[0] += 1
    ok = abs(actual - expect) <= tol
    if ok:
        if VERBOSE:
            print('  ok  #%d %s' % (CASE[0], note or '%.6f == %.6f' % (actual, expect)))
    else:
        FAILED.append('#%d %s: got %r want %r' % (CASE[0], note, actual, expect))
        print('  FAIL #%d %s: got %r want %r' % (CASE[0], note, actual, expect))


def eq_list(actual, expect, tol=1e-6, note=''):
    assert len(actual) == len(expect), 'length mismatch in %s' % note
    for i, (a, e) in enumerate(zip(actual, expect)):
        eq(a, e, tol, '%s[%d]' % (note, i))


def truthy(value, note=''):
    eq(1.0 if value else 0.0, 1.0, 0, note)


def is_none(value, note=''):
    CASE[0] += 1
    if value is None:
        if VERBOSE:
            print('  ok  #%d %s' % (CASE[0], note or 'is None'))
    else:
        FAILED.append('#%d %s: expected None, got %r' % (CASE[0], note, value))
        print('  FAIL #%d %s: expected None, got %r' % (CASE[0], note, value))


def raises(fn, exc, note=''):
    CASE[0] += 1
    try:
        fn()
    except exc:
        if VERBOSE:
            print('  ok  #%d %s' % (CASE[0], note or 'raised %s' % exc.__name__))
        return
    except Exception as other:                              # noqa: BLE001
        FAILED.append('#%d %s: raised %r, want %s' % (CASE[0], note, other, exc.__name__))
        print('  FAIL #%d %s: raised %r' % (CASE[0], note, other))
        return
    FAILED.append('#%d %s: did not raise %s' % (CASE[0], note, exc.__name__))
    print('  FAIL #%d %s: did not raise %s' % (CASE[0], note, exc.__name__))


# ---------------------------------------------------------------- 位姿 / 矩阵
def test_quaternion():
    m = G.quat_to_matrix(0, 0, 0, 1)
    eq_list([m[0][0], m[1][1], m[2][2]], [1, 1, 1], note='identity diag')

    half = math.sin(math.pi / 4)
    m = G.quat_to_matrix(0, 0, half, half)          # 绕 z 转 90°
    eq_list(G.mat_vec(m, [1, 0, 0]), [0, 1, 0], note='yaw90 maps +x to +y')

    _, _, qz, qw = G.yaw_to_quat(0.7)
    m2 = G.quat_to_matrix(0, 0, qz, qw)
    eq_list(G.mat_vec(m2, [1, 0, 0]), [math.cos(0.7), math.sin(0.7), 0], note='yaw_to_quat')


def test_pose_transform():
    tf = G.pose_to_matrix(1, 2, 3, 0, 0, 0, 1)
    eq_list(G.transform_point(tf, [0.5, -0.5, 0.0]), [1.5, 1.5, 3.0], note='identity rot + translate')

    half = math.sin(math.pi / 4)
    tf = G.pose_to_matrix(1, 0, 0, 0, 0, half, half)   # 先转 90° 再平移
    eq_list(G.transform_point(tf, [1, 0, 0]), [1, 1, 0], note='rot then translate')

    eq_list([1.0 if G.identity(4)[i][i] == 1 else 0.0 for i in range(4)], [1, 1, 1, 1],
            note='identity matrix')


# ---------------------------------------------------------------- 投影
def test_projection():
    eq_list(G.project_point([0, 0, 2], K), [320, 240], note='optical center hits principal point')
    p = G.backproject(320, 240, 2.0, K)
    eq_list(p, [0, 0, 2.0], note='backproject center')

    world = [0.5, -0.3, 1.8]
    u, v = G.project_point(world, K)
    back = G.backproject(u, v, world[2], K)
    eq_list(back, world, tol=1e-9, note='project/backproject round trip')

    raises(lambda: G.project_point([0, 0, 0], K), ValueError, note='zero depth rejected')
    raises(lambda: G.backproject(320, 240, 0.0, K), ValueError, note='zero depth backproject')


# ---------------------------------------------------------------- 深度取样
def test_median_depth():
    eq(G.median_depth([0.0, float('nan'), 1.5, 1.7, 99.0]), 1.6, 1e-9, note='filters 0/NaN/too-far')
    eq(G.median_depth([1.0, 2.0, 3.0]), 2.0, 1e-9, note='odd count middle')
    eq(G.median_depth([1.0, 3.0]), 2.0, 1e-9, note='even count average')
    is_none(G.median_depth([]), note='empty input')
    is_none(G.median_depth([0.0, float('inf'), None, 20.0]), note='nothing valid')
    eq(G.median_depth([0.05, 0.5]), 0.5, 1e-9, note='min_depth respected')
    eq(G.median_depth([1.0, 2.0], max_depth=1.5), 1.0, 1e-9, note='max_depth respected')


# ---------------------------------------------------------------- bbox / 选择
def test_boxes():
    eq_list(G.box_center((10, 20, 30, 40)), [20, 30], note='box center')
    eq(G.box_area((0, 0, 8, 8)), 64.0, note='box area')
    eq(G.box_area((5, 5, 4, 4)), 0.0, note='inverted box clamps to zero')

    dets = [G.Detection((0, 0, 5, 5), 0.9),              # 面积 25 < min_area 64 → 过滤
            G.Detection((0, 0, 20, 20), 0.5),
            G.Detection((0, 0, 30, 30), 0.35)]
    best = G.pick_best(dets)
    truthy(best is not None, 'pick_best returns something')
    eq(best.score, 0.5, note='small box filtered, highest remaining wins')

    is_none(G.pick_best([G.Detection((0, 0, 5, 5), 0.99)]), note='all filtered by area')
    is_none(G.pick_best([]), note='empty detections')
    is_none(G.pick_best(dets, min_score=0.95), note='min_score filters everything')


def test_detector_stub():
    det = G.StubDetector([G.Detection((1, 2, 3, 4), 0.8, 'box')])
    out = det.detect(None, 'red box')
    eq(len(out), 1, note='stub returns preset')
    out.clear()
    eq(len(det.detect(None, 'x')), 1, note='stub returns fresh copy each call')
    raises(lambda: G.make_detector('nope'), SystemExit, note='unknown backend rejected')


# ---------------------------------------------------------------- 落点
def test_standoff():
    eq_list(G.standoff_point((5, 0), (0, 0), 0.8), [4.2, 0.0], note='normal standoff')
    eq_list(G.standoff_point((0, 0), (0, 0), 0.8), [-0.8, 0.0], note='robot on target, no div0')
    eq_list(G.standoff_point((0.3, 0), (0, 0), 0.8), [0.0, 0.0], note='closer than standoff, stay put')
    eq(abs(G.yaw_towards((0, 0), (0, 3)) - math.pi / 2) < 1e-9, 1.0, note='yaw toward +y')


def test_ground_hit():
    hit = G.ground_hit([0, 0, 1.0], [0, 1, -1], 0.0)
    truthy(hit is not None, 'ground_hit intersects')
    eq_list(hit, [0, 1, 0], note='45° down hits at 1 m')
    is_none(G.ground_hit([0, 0, 1.0], [0, 1, 0], 0.0), note='parallel ray')
    is_none(G.ground_hit([0, 0, 1.0], [0, 1, 1], 0.0), note='ray pointing up')


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
    eq(code, 0, note='dry-run exit 0')
    truthy('[MAP]' in out, 'prints map point')
    truthy('[GOAL]' in out, 'prints approach goal')

    # 手算期望: 像素中心 (340,250) → cam(0.0667, 0.0333, 2.0) → map(1.0667, 2.0333, 2.3)
    obj = [1 + (340 - 320) * 2.0 / 600, 2 + (250 - 240) * 2.0 / 600, 0.3 + 2.0]
    eq_list(obj[0:2], [1.0666667, 2.0333333], tol=1e-6, note='hand-computed object xy')
    goal = G.standoff_point((obj[0], obj[1]), (0, 0), 0.8)
    truthy('[GOAL] approach=(%.3f, %.3f' % (goal[0], goal[1]) in out, 'goal matches hand calc')

    code, out = run_cli(['--dry-run', '--depth', '2.0'])
    eq(code, 2, note='missing args → exit 2')
    truthy('[ERR]' in out, 'explains missing args')


# ---------------------------------------------------------------- main
TESTS = [test_quaternion, test_pose_transform, test_projection, test_median_depth,
         test_boxes, test_detector_stub, test_standoff, test_ground_hit, test_cli_dry_run]

VERBOSE = False


def main():
    global VERBOSE
    VERBOSE = '-v' in sys.argv
    print('grounding.py 几何单测 (%s)' % ('ROS 可用' if G.HAVE_ROS else '纯 python 模式'))
    for fn in TESTS:
        print('- %s' % fn.__name__)
        fn()
    total = CASE[0]
    print('\n断言 %d 条，失败 %d 条' % (total, len(FAILED)))
    for msg in FAILED:
        print('  ' + msg)
    if FAILED:
        print('RESULT: FAIL')
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
