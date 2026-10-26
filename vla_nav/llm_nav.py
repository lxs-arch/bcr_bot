#!/usr/bin/env python3
"""
llm_nav.py —— 自然语言指令 -> 语义地图地名 -> Nav2 目标点

用法：
    python3 llm_nav.py "去A区"
    python3 llm_nav.py "带我回起点"
    python3 llm_nav.py --list                 # 列出所有地名
    python3 llm_nav.py --raw 1.5 0.0 0.0      # 直接发坐标，跳过语义解析

匹配顺序：
    1) 地名精确包含 / aliases 命中（无需网络，离线可用）
    2) 若设置了 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，则调用 LLM 做语义解析
    3) 都不行 -> 打印可用地名并退出

环境变量：
    SEMANTIC_MAP     语义地图路径，默认 ~/vla_nav/semantic_map.yaml
    OPENAI_API_KEY   或 DEEPSEEK_API_KEY（可选）
    OPENAI_BASE_URL  默认 https://api.openai.com/v1
    LLM_MODEL        默认 gpt-4o-mini
"""
import argparse
import json
import math
import os
import sys

import yaml

DEFAULT_MAP = os.path.expanduser('~/vla_nav/semantic_map.yaml')


# ---------------------------------------------------------------- 语义地图
def load_semantic_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    places = data.get('places') or {}
    if not places:
        raise SystemExit('[ERR] 语义地图里没有任何 places: %s' % path)
    return data.get('frame_id', 'map'), places


def rule_match(text, places):
    """不依赖 LLM 的兜底匹配：地名精确包含 + aliases 包含（取最长命中）。"""
    low = text.lower()
    hits = [name for name in places if name.lower() in low]
    if hits:
        return max(hits, key=len), 'rule-exact'
    for name, p in places.items():
        for alias in (p.get('aliases') or []):
            if str(alias).lower() in low:
                return name, 'rule-alias'
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


# ---------------------------------------------------------------- ROS 导航
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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('instruction', nargs='*', help='自然语言指令，例如 "去A区"')
    ap.add_argument('--list', action='store_true', help='列出所有地名')
    ap.add_argument('--raw', nargs=3, type=float, metavar=('X', 'Y', 'YAW'), help='直接发坐标')
    ap.add_argument('--map', default=os.environ.get('SEMANTIC_MAP', DEFAULT_MAP))
    ap.add_argument('--no-llm', action='store_true', help='只用规则匹配，不调 LLM')
    args = ap.parse_args()

    frame_id, places = load_semantic_map(args.map)
    print('[MAP] %s  frame=%s  places=%d' % (args.map, frame_id, len(places)))

    if args.list:
        for name, p in places.items():
            al = '/'.join(p.get('aliases') or []) or '-'
            print('  %-10s (%6.2f, %6.2f, %5.2f)  aliases=%s' % (name, p['x'], p['y'], p.get('yaw', 0.0), al))
        return 0

    if args.raw:
        x, y, yaw = args.raw
        target = (x, y, yaw)
        print('[TARGET] raw (%.2f, %.2f, %.2f)' % target)
    else:
        text = ' '.join(args.instruction).strip()
        if not text:
            ap.print_help()
            return 1
        print('[INSTRUCTION] %s' % text)

        name, why = rule_match(text, places)
        if name is None and not args.no_llm:
            name, why = llm_match(text, places)
        if name is None:
            print('[FAIL] 无法解析(%s)。可用地名：%s' % (why, ', '.join(places)))
            return 2

        p = places[name]
        target = (p['x'], p['y'], p.get('yaw', 0.0))
        print('[PARSE] -> %s   (%s)   target=(%.2f, %.2f, %.2f)' % (name, why, *target))

    import rclpy
    from rclpy.action import ActionClient
    from nav2_msgs.action import NavigateToPose

    rclpy.init()
    node = rclpy.create_node('llm_nav')
    ac = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    try:
        label = goto(node, ac, NavigateToPose, frame_id, *target)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0 if label == 'SUCCEEDED' else 3


if __name__ == '__main__':
    sys.exit(main())
