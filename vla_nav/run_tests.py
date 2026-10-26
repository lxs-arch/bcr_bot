#!/usr/bin/env python3
"""run_tests.py —— 一条命令跑完 vla_nav 的所有测试

做三件事:

1. 语法检查：把所有 .py 编译一遍（`compile()`），抓低级错误；
2. 跑单元测试：**有 pytest 就用 pytest**（一次收集全部 test_*.py），
   没有 pytest 就逐文件子进程跑（每个测试文件都自带 standalone 入口）；
3. 汇总：用例数 + 退出码（CI 友好）。

    python3 run_tests.py              # 自动选 runner（推荐）
    python3 run_tests.py --pytest     # 强制 pytest（没装就报错，CI 用）
    python3 run_tests.py --standalone # 强制不用 pytest（模拟机器人机）
    python3 run_tests.py -v           # 打印每条用例

为什么两套：机器人机（~/vla_nav/）常常没有 pytest，但 CI / 开发机用 pytest 最方便。
测试本身统一写普通 `assert` + `test_*` 函数，两种 runner 跑同一份断言（见 `_testkit.py`）。
"""

import importlib.util
import os
import re
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


def have_pytest():
    return importlib.util.find_spec('pytest') is not None


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


# ---------------------------------------------------------------- runner 1: pytest
def run_pytest(verbose=False):
    """返回 (ok, 用时, 通过用例数, 跳过数, 摘要行列表)。"""
    cmd = [sys.executable, '-m', 'pytest', '-q'] + (['-v'] if verbose else [])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=TIMEOUT)
    dur = time.time() - t0
    out = (proc.stdout + proc.stderr).strip()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    passed = sum(int(m) for m in re.findall(r'(\d+) passed', out))
    skipped = sum(int(m) for m in re.findall(r'(\d+) skipped', out))
    return proc.returncode == 0, dur, passed, skipped, lines[-12:]


# ---------------------------------------------------------------- runner 2: 逐文件
def run_one(name, verbose=False):
    cmd = [sys.executable, os.path.join(HERE, name)]
    if verbose:
        cmd.append('-v')
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=TIMEOUT)
        dur = time.time() - t0
        out = proc.stdout + proc.stderr
        tail = [ln for ln in out.strip().splitlines() if ln.strip()]
        m = re.search(r'(\d+) 个用例，失败 (\d+) 个', out)
        cases = int(m.group(1)) if m else 0
        return proc.returncode == 0, dur, cases, tail[-1] if tail else '(no output)'
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, 0, 'timeout > %ds' % TIMEOUT


def run_standalone(verbose=False):
    tests = test_files()
    rows, failures, cases = [], 0, 0
    for name in tests:
        passed, dur, n, last = run_one(name, verbose)
        rows.append((name, passed, dur, last))
        cases += n
        if not passed:
            failures += 1
        print('  %-6s %-22s %5.2fs  %s' % ('PASS' if passed else 'FAIL', name, dur, last))
    return failures == 0, rows, cases


def main():
    argv = sys.argv[1:]
    verbose = '-v' in argv
    force_pytest = '--pytest' in argv
    force_standalone = '--standalone' in argv

    if force_pytest and force_standalone:
        print('--pytest 与 --standalone 不能同时用')
        return 2

    use_pytest = (not force_standalone) and (force_pytest or have_pytest())
    if force_pytest and not have_pytest():
        print('[ERR] 指定了 --pytest 但当前解释器没装：%s' % sys.executable)
        print('      pip3 install pytest   或改用 --standalone')
        return 2

    print('=' * 66)
    print('vla_nav 统一测试  dir=%s' % HERE)
    print('runner=%s  python=%s' % ('pytest' if use_pytest else 'standalone',
                                    sys.version.split()[0]))
    print('=' * 66)

    files = py_files()
    print('\n[1/2] 语法检查 (%d 个文件)' % len(files))
    ok_files, bad_files = check_syntax(files)
    print('  通过 %d / %d' % (len(ok_files), len(files)))
    for name, err in bad_files:
        print('  FAIL %-24s %s' % (name, err))

    print('\n[2/2] 单元测试 (%d 个测试文件)' % len(test_files()))
    if use_pytest:
        ok, dur, passed, skipped, tail = run_pytest(verbose)
        for line in tail:
            print('  ' + line)
        print('  用时 %.2fs' % dur)
        total_cases = passed
    else:
        ok, rows, total_cases = run_standalone(verbose)

    print('\n' + '-' * 66)
    bad = len(bad_files) + (0 if ok else 1)
    print('语法 %d/%d 通过，单元测试 %s（%d 个用例）' % (
        len(ok_files), len(files), '通过' if ok else '失败', total_cases))
    if bad:
        print('RESULT: FAIL')
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired:
        print('RESULT: FAIL（测试超时）')
        sys.exit(1)
