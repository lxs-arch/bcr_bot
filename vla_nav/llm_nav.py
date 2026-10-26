#!/usr/bin/env python3
"""
llm_nav.py —— 自然语言指令 -> 语义地图地名/相对位置 -> Nav2 目标点

用法：
    python3 llm_nav.py "去A区"
    python3 llm_nav.py "起点左边两米"
    python3 llm_nav.py "走廊尽头往前走2米"
    python3 llm_nav.py "A区北边3米"
    python3 llm_nav.py "先回起点然后去B区"          # 多步，依次执行
    python3 llm_nav.py --dry-run "A区右边1.5米"     # 只解析不发目标（可离线）
    python3 llm_nav.py --list
    python3 llm_nav.py --raw 1.5 0.0 0.0

解析顺序：
    1) 规则层（离线可用）：
       地名精确/别名命中 → 叠加方位与距离 → 得到坐标
    2) LLM 层（需要 key）：只用来把口语映射到"地名"，方位仍由规则层算
    3) 都失败 → 打印可用地名并退出

方位语义（重要）：
    左/右/前/后 都是**相对该地名自身朝向**（semantic_map.yaml 里的 yaw），
    即"站在 A 区、朝 A 区 yaw 方向看"的左右前后。
    东/南/西/北/东南… 是 **map 帧绝对方向**（+x=东，+y=北），与地名朝向无关。

环境变量：
    SEMANTIC_MAP / OPENAI_API_KEY 或 DEEPSEEK_API_KEY / OPENAI_BASE_URL / LLM_MODEL
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_map import load_semantic_map            # noqa: E402  (同目录模块)

DEFAULT_MAP = os.path.expanduser('~/vla_nav/semantic_map.yaml')

# ------------------------------------------------------------------ 方位词表
# 注意：长词优先，避免 "左" 抢了 "左边"
DIR_EGO = [
    ('左', ['左手边', '左侧', '左边', '左面', '左']),
    ('右', ['右手边', '右侧', '右边', '右面', '右']),
    ('前', ['前面', '前方', '前边', '往前', '向前', '前进', '前']),
    ('后', ['后面', '后方', '后边', '往后', '向后', '后退', '后']),
]

# map 帧绝对方向向量（未归一化）
DIR_COMPASS = {
    '东北': (1, 1), '西北': (-1, 1), '东南': (1, -1), '西南': (-1, -1),
    '东': (1, 0), '西': (-1, 0), '北': (0, 1), '南': (0, -1),
}

CN_NUM = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
          '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '半': 0.5}

DIST_RE = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*(厘米|cm|CM|米|m|M)')
STEP_SPLIT_RE = re.compile(r'(?:然后|接着|之后|再到|再去|再去往|，|,|;|；)')


# ------------------------------------------------------------------ 语义地图
def rule_match(text, places):
    """不依赖 LLM 的兜底匹配：地名精确包含 + aliases 包含（取最长命中）。"""
    low = text.lower()
    hits = [name for name in places if name.lower() in low]
    if hits:
        return max(hits, key=len), 'rule-exact'
    best = None
    for name, p in places.items():
        for alias in (p.get('aliases') or []):
            alias = str(alias).lower()
            if alias and alias in low and (best is None or len(alias) > len(best[1])):
                best = (name, alias)
    if best:
        return best[0], 'rule-alias'
    return None, 'miss'


def llm_match(text, places):
    """可选：用 LLM 把口语指令解析成地名。没有 key 直接跳过。"""
    key = os.environ.get('OPENAI_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        return None, 'no-key'
    try:
        from openai import OpenAI
    except ImportError:
        return None, 'no-openai-pkg'

    base = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    model = os.environ.get('LLM_MODEL', 'gpt-4o-mini')
    client = OpenAI(api_key=key, base_url=base)

    names = list(places.keys())
    prompt = (
        '你是机器人指令解析器。已知地点列表：%s。\n'
        '把用户的话解析成其中一个地点名；如果无法对应，输出 null。\n'
        '只输出 JSON，格式：{"place": "地名或null", "confidence": 0~1}\n'
        '用户说：%s'
    ) % (json.dumps(names, ensure_ascii=False), text)

    try:
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = (r.choices[0].message.content or '').strip()
        raw = raw.replace('```json', '').replace('```', '').strip()
        obj = json.loads(raw)
        place = obj.get('place')
        if place in places:
            return place, 'llm(%.2f)' % float(obj.get('confidence', 0.0))
        return None, 'llm-no-match'
    except Exception as exc:                       # 网络/配额/格式都兜住
        return None, 'llm-error:%s' % type(exc).__name__


# ------------------------------------------------------------------ 方位解析
def parse_distance(text):
    """从文本里取出距离(米)。支持 2米 / 1.5m / 200cm / 两米 / 半米。"""
    m = DIST_RE.search(text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ('厘米', 'cm', 'CM'):
            val /= 100.0
        return val
    # 十位优先：十五米 / 二十米 / 三十米（先于单词匹配，否则"五米"抢先命中）
    m2 = re.search(r'([二两三四五六七八九])?十([一二三四五六七八九])?\s*(?:米|m)', text)
    if m2:
        tens = CN_NUM.get(m2.group(1) or '一', 1) * 10
        ones = CN_NUM.get(m2.group(2) or '零', 0)
        return float(tens + ones)
    # 单个中文数字：左边两米 / 走三米 / 半米
    for cn, num in CN_NUM.items():
        for unit in ('米', 'm'):
            pattern = '%s\\s*%s' % (re.escape(cn), unit)
            if re.search(pattern, text):
                return float(num)
    return None


def parse_direction(text):
    """返回 ('ego', '左'|'右'|'前'|'后') 或 ('compass', '北'|'东南'...) 或 None。"""
    for name in ('东北', '西北', '东南', '西南'):
        if name in text:
            return 'compass', name
    for name, words in DIR_EGO:
        for w in words:
            if w in text:
                return 'ego', name
    for name in ('东', '西', '南', '北'):
        if name in text:
            return 'compass', name
    return None


def rotate(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return dx * c - dy * s, dx * s + dy * c


def resolve_relative(base_name, base, text):
    """在地名 base 基础上叠加方位/距离。返回 (x, y, yaw, why) 或 (None,...,原因)。"""
    direction = parse_direction(text)
    dist = parse_distance(text)
    if direction is None and dist is None:
        return base['x'], base['y'], base.get('yaw', 0.0), 'place'

    if direction is None:
        # 只说了距离没说方向，按"往前"理解（"走三米" ≈ "往前走三米"）
        direction = ('ego', '前')
    if dist is None:
        dist = 1.0                                   # 只说方向不说距离，默认 1 m
        note_default = '(默认1m)'
    else:
        note_default = ''

    kind, name = direction
    bx, by, byaw = base['x'], base['y'], base.get('yaw', 0.0)

    if kind == 'ego':
        local = {'前': (dist, 0.0), '后': (-dist, 0.0),
                 '左': (0.0, dist), '右': (0.0, -dist)}[name]
        dx, dy = rotate(local[0], local[1], byaw)
        why = 'relative(%s,%.2fm%s)' % (name, dist, note_default)
    else:
        vx, vy = DIR_COMPASS[name]
        norm = math.hypot(vx, vy)
        dx, dy = vx / norm * dist, vy / norm * dist
        why = 'relative(%s,%.2fm%s)' % (name, dist, note_default)

    return bx + dx, by + dy, byaw, why


def find_base(text, places):
    """找出这句话的基准地点。优先地名/别名；都没有但出现方位+距离时退回'起点'。"""
    name, why = rule_match(text, places)
    if name and why in ('rule-exact', 'rule-alias'):
        return name, why
    # 没提地名时，只有出现方位/距离才退回"起点"（"走3米" 可以，"去厨房" 不行）
    if parse_direction(text) is None and parse_distance(text) is None:
        return None, why
    for fallback in ('起点', '原点', 'home'):
        if fallback in places:
            return fallback, 'base-default'
    return None, why


def parse_one(fragment, places, use_llm=True):
    """解析单步指令，返回 dict 或 None。"""
    fragment = fragment.strip()
    if not fragment:
        return None

    base_name, why = find_base(fragment, places)
    if base_name is None and use_llm:
        base_name, why = llm_match(fragment, places)

    # 纯地名，没有方位/距离
    direction = parse_direction(fragment)
    dist = parse_distance(fragment)

    if base_name is None:
        # why 可能是 miss / no-key / llm-no-match，对用户统一说"没认出地名"
        return {'ok': False, 'text': fragment, 'reason': 'no-place',
                'hint': '(触发详情: %s)' % why}

    if direction is None and dist is None:
        p = places[base_name]
        return {'ok': True, 'text': fragment, 'label': base_name, 'why': why,
                'x': p['x'], 'y': p['y'], 'yaw': p.get('yaw', 0.0)}

    x, y, yaw, rel_why = resolve_relative(base_name, places[base_name], fragment)
    if x is None:
        return {'ok': False, 'text': fragment, 'reason': rel_why,
                'hint': '说了距离但没说方向（左/右/前/后/东/西/南/北）'}
    short = rel_why.replace('relative(', '').rstrip(')')
    return {'ok': True, 'text': fragment, 'label': '%s(%s)' % (base_name, short),
            'why': rel_why, 'x': x, 'y': y, 'yaw': yaw}


def parse_instruction(text, places, use_llm=True):
    """把整句拆成多步，返回 (steps, errors)。"""
    fragments = [f for f in STEP_SPLIT_RE.split(text) if f and f.strip()]
    if not fragments:
        fragments = [text]
    steps, errors = [], []
    for frag in fragments:
        r = parse_one(frag, places, use_llm=use_llm)
        if r is None:
            continue
        (steps if r.get('ok') else errors).append(r)
    return steps, errors


# ------------------------------------------------------------------ ROS 导航
def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def goto(node, ac, action_type, frame_id, x, y, yaw, timeout_sec=300.0):
    if not ac.wait_for_server(timeout_sec=10.0):
        print('NO_ACTION_SERVER: /navigate_to_pose 不在，nav2 没起？')
        return 'NO_SERVER'

    goal = action_type.Goal()
    goal.pose.header.frame_id = frame_id
    goal.pose.header.stamp = node.get_clock().now().to_msg()
    goal.pose.pose.position.x = float(x)
    goal.pose.pose.position.y = float(y)
    goal.pose.pose.position.z = 0.0
    _, _, qz, qw = quat_from_yaw(yaw)
    goal.pose.pose.orientation.z = qz
    goal.pose.pose.orientation.w = qw

    import rclpy
    send_future = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=15.0)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print('GOAL_REJECTED  (目标可能在障碍/未知区，或定位未就绪)')
        return 'REJECTED'

    print('GOAL_ACCEPTED  (%.2f, %.2f, yaw=%.2f)' % (x, y, yaw))
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    if not result_future.done():
        print('TIMEOUT  (%.0fs 内没结束，可能需要取消)' % timeout_sec)
        return 'TIMEOUT'

    status = result_future.result().status
    # action_msgs/GoalStatus: 4=SUCCEEDED 5=CANCELED 6=ABORTED
    label = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(status, 'STATUS_%s' % status)
    print('RESULT %s' % label)
    return label


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('instruction', nargs='*', help='自然语言指令，例如 "起点左边两米"')
    ap.add_argument('--list', action='store_true', help='列出所有地名')
    ap.add_argument('--raw', nargs=3, type=float, metavar=('X', 'Y', 'YAW'), help='直接发坐标')
    ap.add_argument('--map', default=os.environ.get('SEMANTIC_MAP', DEFAULT_MAP))
    ap.add_argument('--no-llm', action='store_true', help='只用规则匹配，不调 LLM')
    ap.add_argument('--dry-run', action='store_true', help='只解析不发目标（无需 ROS）')
    args = ap.parse_args()

    frame_id, places = load_semantic_map(args.map)
    print('[MAP] %s  frame=%s  places=%d' % (args.map, frame_id, len(places)))

    if args.list:
        for name, p in places.items():
            al = '/'.join(p.get('aliases') or []) or '-'
            print('  %-10s (%6.2f, %6.2f, %5.2f)  aliases=%s' % (name, p['x'], p['y'], p.get('yaw', 0.0), al))
        return 0

    if args.raw:
        targets = [{'label': 'raw', 'x': args.raw[0], 'y': args.raw[1],
                    'yaw': args.raw[2], 'why': 'raw'}]
    else:
        text = ' '.join(args.instruction).strip()
        if not text:
            ap.print_help()
            return 1
        print('[INSTRUCTION] %s' % text)
        steps, errors = parse_instruction(text, places, use_llm=not args.no_llm)
        for e in errors:
            print('[FAIL] %r -> %s %s' % (e['text'], e['reason'], e.get('hint', '')))
        if not steps:
            print('[FAIL] 无法解析。可用地名：%s' % ', '.join(places))
            return 2
        for i, s in enumerate(steps, 1):
            print('[STEP %d] %s  (%s)  target=(%.2f, %.2f, %.2f)' % (
                i, s['label'], s['why'], s['x'], s['y'], s['yaw']))
        targets = steps

    if args.dry_run:
        print('[DRY-RUN] 解析完成，未发送目标（steps=%d）' % len(targets))
        return 0

    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose

    rclpy.init()
    node = rclpy.create_node('llm_nav')
    ac = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    rc = 0
    try:
        for i, t in enumerate(targets, 1):
            print('--- step %d/%d ---' % (i, len(targets)))
            label = goto(node, ac, NavigateToPose, frame_id, t['x'], t['y'], t['yaw'])
            if label != 'SUCCEEDED':
                print('[STOP] step %d 结果 %s，后续步骤不再执行' % (i, label))
                rc = 3
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
