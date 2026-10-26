#!/usr/bin/env python3
"""_testkit.py —— 让同一份测试既能被 pytest 收，也能在没有 pytest 的机器上直接跑

为什么要有它：机器人机上（`~/vla_nav/`）经常没有 pytest，但 CI 和开发机用 pytest
最方便。所以测试一律写成**普通 `assert` + 模块级 `test_*` 函数**：

* pytest：原生收集 `test_*`，零改造；
* 没有 pytest：文件末尾 `sys.exit(_testkit.run(globals()))` 自己跑一遍。

这样永远不会出现"脚本里一套断言、pytest 里另一套断言"的两份真相。
"""

import sys
import traceback


def run(namespace, argv=None):
    """跑 namespace 里所有 `test_*` 可调用对象，返回进程退出码（0=全过）。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = '-v' in argv
    names = sorted(name for name, obj in namespace.items()
                   if name.startswith('test_') and callable(obj))
    if not names:
        print('没有找到 test_* 函数（namespace=%s）' % namespace.get('__name__'))
        return 1

    failed = []
    for name in names:
        fn = namespace[name]
        if verbose:
            print('- %s' % name)
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc) or 'assert failed'))
            print('FAIL %s: %s' % (name, exc))
        except Exception as exc:                            # noqa: BLE001
            failed.append((name, '%s: %s' % (type(exc).__name__, exc)))
            print('ERROR %s: %s' % (name, exc))
            if verbose:
                traceback.print_exc()

    total = len(names)
    print('\n%d 个用例，失败 %d 个' % (total, len(failed)))
    for name, msg in failed:
        print('  %-28s %s' % (name, msg))
    if failed:
        print('RESULT: FAIL')
        return 1
    print('RESULT: PASS')
    return 0


def approx(actual, expect, tol=1e-6, note=''):
    """浮点比较；失败时把差值和上下文写进断言消息。"""
    assert abs(actual - expect) <= tol, '%s: got %r want %r (tol=%g, diff=%g)' % (
        note or 'approx', actual, expect, tol, abs(actual - expect))


def approx_list(actual, expect, tol=1e-6, note=''):
    actual, expect = list(actual), list(expect)
    assert len(actual) == len(expect), '%s: length %d != %d' % (note, len(actual), len(expect))
    for i, (a, e) in enumerate(zip(actual, expect)):
        approx(a, e, tol, '%s[%d]' % (note or 'list', i))
