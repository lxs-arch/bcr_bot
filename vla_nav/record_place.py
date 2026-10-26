#!/usr/bin/env python3
"""
record_place.py —— 在 RViz 里点一下地图，把那个位置记成"地名"

用法（推荐两终端）：
    终端1: python3 record_place.py            # 等待点位
    RViz : 工具栏 Publish Point -> 在地图上点一下
    终端1: 提示输入地名（例如 A区），回车 -> 写入 semantic_map.yaml

也可以不交互：
    python3 record_place.py --name A区 --yaw 0.0
"""
import argparse
import os
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import yaml

DEFAULT_MAP = os.path.expanduser('~/vla_nav/semantic_map.yaml')


class Picker(Node):
    def __init__(self):
        super().__init__('record_place')
        self.point = None
        self.create_subscription(PointStamped, '/clicked_point', self.cb, 10)

    def cb(self, msg):
        self.point = (msg.point.x, msg.point.y, msg.header.frame_id)
        self.get_logger().info('收到点位 x=%.3f y=%.3f frame=%s'
                               % (msg.point.x, msg.point.y, msg.header.frame_id))


def load(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {'frame_id': 'map', 'places': {}}


def save(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=os.environ.get('SEMANTIC_MAP', DEFAULT_MAP))
    ap.add_argument('--name', help='地名；不给则收到点位后交互输入')
    ap.add_argument('--yaw', type=float, default=0.0, help='到达朝向(rad)，默认 0')
    ap.add_argument('--timeout', type=float, default=0.0, help='超时秒数，0=一直等')
    args = ap.parse_args()

    rclpy.init()
    node = Picker()
    print('[WAIT] 在 RViz 里用 Publish Point 点一下目标位置…')
    saved = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.point is None:
                continue
            x, y, frame = node.point
            name = args.name
            if not name:
                name = input('地名（例如 A区）: ').strip()
            if not name:
                print('[SKIP] 没输入名字，继续等下一个点位')
                node.point = None
                continue

            data = load(args.map)
            data.setdefault('places', {})[name] = {
                'x': round(float(x), 3),
                'y': round(float(y), 3),
                'yaw': float(args.yaw),
                'aliases': [],
            }
            save(args.map, data)
            saved = name
            print('[OK] 已写入 %s -> %s  (%s, %.3f, %.3f)' % (args.map, name, frame, x, y))
            break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if saved is None:
        print('[END] 没有记录任何点位')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
