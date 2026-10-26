"""让 pytest 从任何工作目录调用都能 import vla_nav 下的模块。

vla_nav 不是安装包（没有 setup.py / pyproject），测试文件靠 `sys.path.insert(0, 本目录)`
导入 `grounding` / `llm_nav`。pytest 默认只把 rootdir 放进 sys.path（importmode=prepend
时会加测试文件所在目录），加这个 conftest 后行为与 `python3 test_x.py` 完全一致。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
