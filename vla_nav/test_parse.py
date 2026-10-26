#!/usr/bin/env python3
"""llm_nav 解析层单测：不依赖 ROS、不依赖 pyyaml，直接 `python3 test_parse.py`。

覆盖：纯地名 / 相对方位(前后左右) / 绝对方位(罗盘) / 距离(数字·中文·厘米·半米·十五米)
     / 多步指令 / 无法解析 / 默认距离 / 裸距离按"往前"处理。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_nav

PLACES = {
    '起点': {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'aliases': ['原点', '回家']},
    '走廊尽头': {'x': 4.0, 'y': 0.0, 'yaw': 0.0, 'aliases': ['走到底']},
    'A区': {'x': 1.5, 'y': 0.0, 'yaw': 0.0, 'aliases': ['a区', '一区']},
    'B区': {'x': 2.5, 'y': 0.0, 'yaw': 0.0, 'aliases': ['b区']},
    # 故意给一个 yaw=90° 的地点，用来验证"左右前后相对朝向"
    '东屋': {'x': 5.0, 'y': 5.0, 'yaw': math.pi / 2, 'aliases': ['东边的屋子']},
}

FAILS = []


def check(desc, text, expect_steps, expect_xy=None, tol=0.02, expect_reason=None):
    steps, errors = llm_nav.parse_instruction(text, PLACES, use_llm=False)
    ok = len(steps) == expect_steps
    detail = ''
    if expect_reason is not None:
        ok = ok and any(expect_reason in e['reason'] for e in errors)
        detail = 'errors=%s' % [e['reason'] for e in errors]
    if ok and expect_xy is not None:
        got = (steps[0]['x'], steps[0]['y'])
        ok = abs(got[0] - expect_xy[0]) <= tol and abs(got[1] - expect_xy[1]) <= tol
        detail = 'got=(%.3f, %.3f) want=(%.3f, %.3f)' % (got[0], got[1], expect_xy[0], expect_xy[1])
    print('%s %-34s %s' % ('PASS' if ok else 'FAIL', desc, detail))
    if not ok:
        FAILS.append(desc)


# 1. 纯地名
check('纯地名', '去A区', 1, (1.5, 0.0))
check('别名命中', '带我回老家', 1, None if False else (1.5, 0.0), expect_reason=None) if False else None
check('别名: 原点', '去原点', 1, (0.0, 0.0))

# 2. 相对方位（相对朝向）
check('左 = +y', '起点左边两米', 1, (0.0, 2.0))
check('右 = -y', '起点右边1.5米', 1, (0.0, -1.5))
check('前 = +x', 'A区往前走2米', 1, (3.5, 0.0))
check('后 = -x', 'A区往后1米', 1, (0.5, 0.0))
check('只说方向默认1m', 'A区后面', 1, (0.5, 0.0))

# 3. 相对朝向的旋转：东屋 yaw=90°，"东屋左边" 应落在 +x 方向
check('朝向90°时的左 = -x', '东屋左边2米', 1, (3.0, 5.0))

# 4. 绝对罗盘（与朝向无关）
check('北 = +y', 'A区北边3米', 1, (1.5, 3.0))
check('南 = -y', 'A区南边3米', 1, (1.5, -3.0))
check('东 = +x', 'A区东边3米', 1, (4.5, 0.0))
check('西 = -x', 'A区西边3米', 1, (-1.5, 0.0))
check('西北对角线', 'A区西北2米', 1, (1.5 - 2 / math.sqrt(2), 2 / math.sqrt(2)))

# 5. 距离的各种写法
check('厘米', 'A区北边200厘米', 1, (1.5, 2.0))
check('半米', 'A区北边半米', 1, (1.5, 0.5))
check('十五米', 'A区北边十五米', 1, (1.5, 15.0))
check('小数', 'A区北边2.5米', 1, (1.5, 2.5))

# 6. 裸距离 -> 往前
check('裸距离按往前', '走3米', 1, (3.0, 0.0))

# 7. 多步
check('多步(然后)', '先回起点然后去B区', 2, (0.0, 0.0))
steps, _ = llm_nav.parse_instruction('先回起点然后去B区', PLACES, use_llm=False)
print('%s 多步第二步                -> %s' % (
    'PASS' if len(steps) == 2 and steps[1]['label'] == 'B区' else 'FAIL',
    steps[1]['label'] if len(steps) > 1 else 'MISSING'))
if not (len(steps) == 2 and steps[1]['label'] == 'B区'):
    FAILS.append('多步第二步')

# 8. 无法解析
check('未知地点', '去厨房', 0, expect_reason='no-place')

# 9. 真实 yaml（有 pyyaml 时才测）
try:
    import yaml  # noqa: F401
    here = os.path.dirname(os.path.abspath(__file__))
    yml = os.path.join(here, 'semantic_map.yaml')
    if os.path.exists(yml):
        _, places_real = llm_nav.load_semantic_map(yml)
        steps, _ = llm_nav.parse_instruction('A区左边1米', places_real, use_llm=False)
        ok = len(steps) == 1 and abs(steps[0]['x'] - 1.5) < 1e-6 and abs(steps[0]['y'] - 1.0) < 1e-6
        print('%s 真实 semantic_map.yaml     -> %s' % ('PASS' if ok else 'FAIL',
              (steps[0]['x'], steps[0]['y']) if steps else 'none'))
        if not ok:
            FAILS.append('真实yaml')
except ImportError:
    print('SKIP 真实 semantic_map.yaml（本机没有 pyyaml）')

print()
if FAILS:
    print('FAILED %d: %s' % (len(FAILS), FAILS))
    sys.exit(1)
print('ALL PASS')
