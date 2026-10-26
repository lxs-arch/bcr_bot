#!/usr/bin/env python3
"""test_parse.py —— A1 指令解析层单测（pytest 兼容，也能直接 python3 运行）

覆盖：纯地名 / 别名 / 相对方位(前后左右) / 绝对方位(罗盘) / 距离(数字·中文·厘米·半米·十五米)
     / 多步指令 / 无法解析 / 默认距离 / 裸距离按"往前"处理 / 真实 semantic_map.yaml。

不需要 ROS；有 pyyaml 时额外验证真实 yaml，没有就跳过那条。

    python3 test_parse.py          # 没有 pytest 的机器用这个
    python3 -m pytest test_parse.py -q
"""

import contextlib
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _testkit
import llm_nav

PLACES = {
    '起点': {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'aliases': ['原点', '回家']},
    '走廊尽头': {'x': 4.0, 'y': 0.0, 'yaw': 0.0, 'aliases': ['走到底']},
    'A区': {'x': 1.5, 'y': 0.0, 'yaw': 0.0, 'aliases': ['a区', '一区']},
    'B区': {'x': 2.5, 'y': 0.0, 'yaw': 0.0, 'aliases': ['b区']},
    # 故意给一个 yaw=90° 的地点，用来验证"左右前后相对该地名朝向"
    '东屋': {'x': 5.0, 'y': 5.0, 'yaw': math.pi / 2, 'aliases': ['东边的屋子']},
}

# (说明, 指令, 期望步数, 期望第一步 xy 或 None, 期望错误原因或 None)
CASES = [
    # 1. 纯地名 / 别名
    ('纯地名', '去A区', 1, (1.5, 0.0), None),
    ('别名: 原点', '去原点', 1, (0.0, 0.0), None),
    ('别名: 一区', '去一区', 1, (1.5, 0.0), None),
    ('别名: 回家', '带我回家', 1, (0.0, 0.0), None),
    # 2. 相对方位（相对朝向）
    ('左 = +y', '起点左边两米', 1, (0.0, 2.0), None),
    ('右 = -y', '起点右边1.5米', 1, (0.0, -1.5), None),
    ('前 = +x', 'A区往前走2米', 1, (3.5, 0.0), None),
    ('后 = -x', 'A区往后1米', 1, (0.5, 0.0), None),
    ('只说方向默认1m', 'A区后面', 1, (0.5, 0.0), None),
    # 3. 相对朝向的旋转：东屋 yaw=90°，"东屋左边" 应落在 +x 方向
    ('朝向90°时的左 = +x', '东屋左边2米', 1, (3.0, 5.0), None),
    # 4. 绝对罗盘（与地名朝向无关）
    ('北 = +y', 'A区北边3米', 1, (1.5, 3.0), None),
    ('南 = -y', 'A区南边3米', 1, (1.5, -3.0), None),
    ('东 = +x', 'A区东边3米', 1, (4.5, 0.0), None),
    ('西 = -x', 'A区西边3米', 1, (-1.5, 0.0), None),
    ('西北对角线', 'A区西北2米', 1, (1.5 - 2 / math.sqrt(2), 2 / math.sqrt(2)), None),
    # 5. 距离的各种写法
    ('厘米', 'A区北边200厘米', 1, (1.5, 2.0), None),
    ('半米', 'A区北边半米', 1, (1.5, 0.5), None),
    ('十五米', 'A区北边十五米', 1, (1.5, 15.0), None),
    ('小数', 'A区北边2.5米', 1, (1.5, 2.5), None),
    # 6. 裸距离 → 往前（以起点为基准）
    ('裸距离按往前', '走3米', 1, (3.0, 0.0), None),
    # 7. 无法解析
    ('未知地点', '去厨房', 0, None, 'no-place'),
]


def _parse(text):
    return llm_nav.parse_instruction(text, PLACES, use_llm=False)


def _run_main(argv):
    """临时替换 sys.argv 调 llm_nav.main()（不需要 ROS 的 --dry-run 分支）。"""
    old = sys.argv
    sys.argv = ['llm_nav.py'] + list(argv)
    try:
        return llm_nav.main()
    finally:
        sys.argv = old


def test_parse_table():
    """表驱动：地名/别名、方位叠加、距离写法、默认距离、失败原因。"""
    for desc, text, n_steps, xy, reason in CASES:
        steps, errors = _parse(text)
        assert len(steps) == n_steps, '%s: 步数 %d != %d（%r）' % (
            desc, len(steps), n_steps, text)
        if reason is not None:
            reasons = [e['reason'] for e in errors]
            assert any(reason in r for r in reasons), '%s: 错误原因 %r 不含 %r' % (
                desc, reasons, reason)
        if xy is not None:
            got = (steps[0]['x'], steps[0]['y'])
            assert abs(got[0] - xy[0]) <= 0.02 and abs(got[1] - xy[1]) <= 0.02, (
                '%s: got=(%.3f, %.3f) want=(%.3f, %.3f)' % (
                    desc, got[0], got[1], xy[0], xy[1]))


def test_multistep():
    """多步指令：两段依次发目标，第二步落在 B区。"""
    steps, _ = _parse('先回起点然后去B区')
    assert len(steps) == 2, '多步应拆成 2 步，实际 %d' % len(steps)
    assert steps[0]['label'] == '起点', '第一步应为起点，实际 %r' % steps[0]['label']
    assert steps[1]['label'] == 'B区', '第二步应为B区，实际 %r' % steps[1]['label']


def test_unknown_place_and_available_list():
    """解析失败 → errors 带 no-place，且 CLI 要把"可用地名"打出来（否则没法自助改口）。"""
    steps, errors = _parse('去厨房')
    assert steps == []
    assert errors and errors[0]['reason'] == 'no-place', 'errors=%r' % (errors,)

    here = os.path.dirname(os.path.abspath(__file__))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = _run_main(['去厨房', '--dry-run', '--no-llm',
                          '--map', os.path.join(here, 'semantic_map.yaml')])
    out = buf.getvalue()
    assert code == 2, '无法解析时退出码应为 2，实际 %r' % code
    assert '可用地名' in out and 'A区' in out, 'CLI 未提示可用地名:\n%s' % out


def test_distance_units():
    """同一段距离用 cm / 中文 / 小数写都应落同一处。"""
    expect = (1.5, 2.0)
    for text in ('A区北边2米', 'A区北边200厘米', 'A区北边两米'):
        steps, _ = _parse(text)
        got = (steps[0]['x'], steps[0]['y'])
        assert abs(got[0] - expect[0]) <= 0.02 and abs(got[1] - expect[1]) <= 0.02, (
            '%r -> %r，期望 %r' % (text, got, expect))


def test_real_semantic_map():
    """真实 semantic_map.yaml（有 pyyaml 才跑）：地图里的 A区 应在 (1.5, 0)。"""
    try:
        import yaml  # noqa: F401
    except ImportError:
        print('SKIP 真实 semantic_map.yaml（本机没有 pyyaml）')
        return
    yml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'semantic_map.yaml')
    assert os.path.exists(yml), '找不到 %s' % yml
    _, places_real = llm_nav.load_semantic_map(yml)
    assert places_real, 'yaml 解析出 0 个地名'
    steps, _ = llm_nav.parse_instruction('A区左边1米', places_real, use_llm=False)
    assert len(steps) == 1, '应解析出 1 步，实际 %r' % steps
    got = (steps[0]['x'], steps[0]['y'])
    assert abs(got[0] - 1.5) < 1e-6 and abs(got[1] - 1.0) < 1e-6, (
        'A区左边1米 -> %r，期望 (1.5, 1.0)' % (got,))


if __name__ == '__main__':
    sys.exit(_testkit.run(globals()))
