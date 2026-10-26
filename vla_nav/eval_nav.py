#!/usr/bin/env python3
"""批量评测 Nav2 导航:成功率 / 用时 / 终点误差 / recovery 次数。

前提:nav2 已经在跑(stack 起好、AMCL 有位姿)。
本脚本只负责"反复发目标 + 记录数据",不负责起栈。

用法::

    # 直接给一组目标点,每个点重复 5 次
    python3 scripts/eval_nav.py --runs 5 --goal 1.5 0.0 --goal 3.6 0.0

    # 从语义地图按地名取目标(复用 vla_nav/semantic_map.yaml)
    python3 scripts/eval_nav.py --runs 5 --map ~/vla_nav/semantic_map.yaml --place A区

    # 输出 CSV 供后续画表
    python3 scripts/eval_nav.py --runs 3 --goal 1.5 0.0 --csv ~/eval_$(date +%m%d).csv

输出示例::

    [1/5] (1.50,0.00) SUCCEEDED 12.4s err=0.08m recov=0
    ...
    === SUMMARY ===  目标 (1.50,0.00)
      runs=5 success=5 (100%)  mean=12.6s  p90=15.1s  err=0.09m  recoveries=0

约定:成功 = action 返回 STATUS_SUCCEEDED;失败也记录原因与耗时,不要偷偷丢掉。
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from semantic_map import load_semantic_map as load_places  # noqa: E402
except Exception:  # pragma: no cover
    load_places = None

try:  # ROS 依赖延迟导入:没有 ROS 也能 --help / 单测纯函数
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from nav2_msgs.action import NavigateToPose
    from geometry_msgs.msg import PoseStamped

    HAVE_ROS = True
except ImportError:  # pragma: no cover - 只在非 ROS 环境走这里
    HAVE_ROS = False


def yaw_to_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


if HAVE_ROS:

  class Evaluator(Node):
        def __init__(self, frame):
            super().__init__('eval_nav')
            self.frame = frame
            self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
            self.recoveries = 0
            # recovery 次数从行为树日志里数(取不到就退化为 0)
            try:
                from nav2_msgs.msg import BehaviorTreeLog
                self.create_subscription(BehaviorTreeLog, 'behavior_tree_log', self._bt_cb, 10)
                self._bt_ok = True
            except Exception:
                self._bt_ok = False

        def _bt_cb(self, msg):
            for ev in msg.event_log:
                name = getattr(ev, 'node_name', '') or ''
                status = getattr(ev, 'current_status', '') or ''
                if 'recovery' in name.lower() and 'SUCCESS' not in status.upper():
                    self.recoveries += 1

        def wait_server(self, timeout=60.0):
            return self.client.wait_for_server(timeout_sec=timeout)

        def run_once(self, x, y, yaw, timeout=180.0):
            """发一个目标并等待结束,返回 dict。"""
            goal = NavigateToPose.Goal()
            p = PoseStamped()
            p.header.frame_id = self.frame
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            qz, qw = yaw_to_quat(yaw)
            p.pose.orientation.z = qz
            p.pose.orientation.w = qw
            goal.pose = p

            self.recoveries = 0
            best = {'dist': float('inf')}

            def feedback_cb(fb):
                d = getattr(fb.feedback, 'distance_remaining', None)
                if d is not None and 0.0 <= d < best['dist']:
                    best['dist'] = d

            t0 = time.time()
            send = self.client.send_goal_async(goal, feedback_callback=feedback_cb)
            rclpy.spin_until_future_complete(self, send, timeout_sec=30.0)
            handle = send.result()
            if handle is None or not handle.accepted:
                return {'status': 'REJECTED', 'seconds': round(time.time() - t0, 2),
                        'err': None, 'recoveries': 0, 'note': 'goal rejected'}

            res = handle.get_result_async()
            deadline = time.time() + timeout
            while not res.done() and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
            seconds = round(time.time() - t0, 2)

            if not res.done():
                handle.cancel_goal_async()
                return {'status': 'TIMEOUT', 'seconds': seconds, 'err': None,
                        'recoveries': self.recoveries, 'note': f'>{timeout:.0f}s'}

            code = res.result().status
            status = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(code, f'CODE{code}')
            return {'status': status, 'seconds': seconds,
                    'err': None if best['dist'] == float('inf') else round(best['dist'], 3),
                    'recoveries': self.recoveries, 'note': ''}


def summarize(rows):
    ok = [r for r in rows if r['status'] == 'SUCCEEDED']
    ts = sorted(r['seconds'] for r in rows if r['seconds'])
    p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))] if ts else 0
    errs = [r['err'] for r in ok if r['err'] is not None]
    rec = sum(r['recoveries'] for r in rows)
    print('  runs=%d success=%d (%.0f%%)  mean=%.1fs  p90=%.1fs  err=%s  recoveries=%d' % (
        len(rows), len(ok), 100.0 * len(ok) / max(len(rows), 1),
        (sum(r['seconds'] for r in rows) / max(len(rows), 1)), p90,
        ('%.2fm' % (sum(errs) / len(errs))) if errs else 'n/a', rec))


def main():
    ap = argparse.ArgumentParser(description='Nav2 导航批量评测')
    ap.add_argument('--goal', nargs=3, action='append', metavar=('X', 'Y', 'YAW'),
                    type=float, help='目标点,可重复;YAW 弧度(可写 0)')
    ap.add_argument('--map', help='语义地图 yaml')
    ap.add_argument('--place', action='append', help='地名(配合 --map),可重复')
    ap.add_argument('--runs', type=int, default=3, help='每个目标重复次数')
    ap.add_argument('--timeout', type=float, default=180.0, help='单次超时(秒)')
    ap.add_argument('--csv', help='把每次结果追加写入该 CSV')
    ap.add_argument('--settle', type=float, default=2.0, help='两次之间等待(秒)')
    args = ap.parse_args()

    targets = []
    if args.goal:
        targets += [(x, y, yaw, '(%.2f,%.2f)' % (x, y)) for x, y, yaw in args.goal]
    if (args.map or args.place) and load_places is None:
        sys.exit('找不到 semantic_map.py（需与 eval_nav.py 同目录）')

    if args.map and args.place:
        frame, places = load_places(args.map)
        for name in args.place:
            if name not in places:
                sys.exit(f'语义地图里没有地名: {name}(现有: {", ".join(places)})')
            v = places[name]
            targets.append((v['x'], v['y'], v.get('yaw', 0.0), name))
    if not targets:
        sys.exit('至少给一个 --goal x y yaw 或 --map+--place')

    frame = 'map'
    if args.map:
        try:
            frame, _ = load_places(args.map)
        except SystemExit:
            pass

    if not HAVE_ROS:
        sys.exit('需要在 ROS2 环境运行(先 source /opt/ros/<distro>/setup.bash)')

    rclpy.init()
    node = Evaluator(frame)
    print('[INFO] 等待 navigate_to_pose action ...')
    if not node.wait_server(60.0):
        rclpy.shutdown()
        sys.exit('找不到 navigate_to_pose —— nav2 没起,或 namespace 不对')

    all_rows = []
    for x, y, yaw, label in targets:
        rows = []
        print(f'\n=== 目标 {label}  (x={x:.2f} y={y:.2f} yaw={yaw:.2f}) ===')
        for i in range(args.runs):
            r = node.run_once(x, y, yaw, timeout=args.timeout)
            r.update({'target': label, 'x': x, 'y': y,
                      'time': datetime.now().isoformat(timespec='seconds')})
            rows.append(r)
            all_rows.append(r)
            print('[%d/%d] %-10s %6.1fs err=%-7s recov=%d %s' % (
                i + 1, args.runs, r['status'], r['seconds'],
                ('%.2fm' % r['err']) if r['err'] is not None else 'n/a',
                r['recoveries'], r['note']))
            time.sleep(args.settle)
        summarize(rows)

    print('\n=== TOTAL ===')
    summarize(all_rows)

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['time', 'target', 'x', 'y', 'status',
                                              'seconds', 'err', 'recoveries', 'note'])
            if new:
                w.writeheader()
            w.writerows(all_rows)
        print(f'[CSV] {args.csv}')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
