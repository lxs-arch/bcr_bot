#!/usr/bin/env python3
"""run_tests.py —— 一条命令跑完 vla_nav 的所有测试

做三件事:

1. 语法检查：把所有 .py 编译一遍（`compile()`），抓低级错误；
2. 跑 `test_*.py`：每个文件单独子进程执行，互不污染；
3. 汇总：表格 + 退出码（CI 友好）。

    python3 run_tests.py            # 全跑
    python3 run_tests.py -v         # 透传 -v 给子测试

设计说明：不引入 pytest，因为机器人机上经常没有装；子进程隔离避免
某个测试改了 sys.path / 全局状态影响后面的用例。
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 120  # 单个测试文件最长 2 分钟（正常都是秒级）


def py_files():
    return sorted(f for f in os.listdir(HERE)
                  if f.endswith('.py') and not f.startswith('__'))


def test_files():
    return sorted(f for f in os.listdir(HERE)
                  if f.startswith('test_') and f.endswith('.py'))


def check_syntax(files):
    """逐个 compile，返回 (ok_list, failed_list)。"""
    ok, bad = [], []
    for name in files:
        path = os.path.join(HERE, name)
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                compile(fh.read(), path, 'exec')
            ok.append(name)
        except SyntaxError as exc:
            bad.append((name, 'SyntaxError line %s: %s' % (exc.lineno, exc.msg)))
        except Exception as exc:                          # noqa: BLE001
            bad.append((name, '%s: %s' % (type(exc).__name__, exc)))
    return ok, bad


def run_one(name, verbose=False):
    cmd = [sys.executable, os.path.join(HERE, name)]
    if verbose:
        cmd.append('-v')
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=TIMEOUT)
        dur = time.time() - t0
        tail = [ln for ln in (proc.stdout + proc.stderr).strip().splitlines() if ln.strip()]
        return proc.returncode == 0, dur, tail[-1] if tail else '(no output)'
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, 'timeout > %ds' % TIMEOUT


def main():
    verbose = '-v' in sys.argv
    print('=' * 62)
    print('vla_nav 统一测试  dir=%s' % HERE)
    print('=' * 62)

    files = py_files()
    print('\n[1/2] 语法检查 (%d 个文件)' % len(files))
    ok_files, bad_files = check_syntax(files)
    print('  通过 %d / %d' % (len(ok_files), len(files)))
    for name, err in bad_files:
        print('  FAIL %-24s %s' % (name, err))

    tests = test_files()
    print('\n[2/2] 单元测试 (%d 个文件)' % len(tests))
    rows, failures = [], 0
    for name in tests:
        passed, dur, last = run_one(name, verbose)
        rows.append((name, passed, dur, last))
        if not passed:
            failures += 1
        print('  %-6s %-22s %5.2fs  %s' % ('PASS' if passed else 'FAIL', name, dur, last))

    print('\n' + '-' * 62)
    total = len(tests) + len(files)
    bad = failures + len(bad_files)
    print('结果: %d/%d 通过' % (total - bad, total))
    if bad:
        print('RESULT: FAIL')
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
