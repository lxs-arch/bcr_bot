#!/usr/bin/env python3
"""语义地图读取：优先 PyYAML；没有 PyYAML 时退化为内置极小解析器。

只服务本项目 `semantic_map.yaml` 这一种结构：

    frame_id: map
    places:
      A区:
        x: 1.5
        y: 0.0
        yaw: 0.0
        aliases: [a区, 一区]

这样拆成独立模块后，`llm_nav.py` / `eval_nav.py` / `test_parse.py`
都不再各自复制一份解析逻辑，也不会因为缺 pyyaml 直接不可用。
"""


def _fallback(path):
    places, current, frame = {}, None, 'map'
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if line.startswith('frame_id:'):
                frame = line.split(':', 1)[1].strip().strip('"\'')
                continue
            if line.startswith('places:'):
                continue
            if line.startswith('  ') and not line.startswith('    '):
                places[line.strip().rstrip(':')] = {}
                current = line.strip().rstrip(':')
                continue
            if current and line.startswith('    '):
                k, _, v = line.strip().partition(':')
                v = v.strip()
                if k == 'aliases':
                    places[current][k] = [x.strip().strip('"\'')
                                          for x in v.strip('[]').split(',') if x.strip()]
                elif k in ('x', 'y', 'yaw'):
                    try:
                        places[current][k] = float(v)
                    except ValueError:
                        pass
    return frame, places


def load_semantic_map(path):
    """返回 (frame_id, places)。places[name] = {x, y, yaw, aliases?}"""
    try:
        import yaml
    except ImportError:
        frame, places = _fallback(path)
    else:
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        frame = data.get('frame_id', 'map')
        places = data.get('places') or {}
    if not places:
        raise SystemExit('[ERR] 语义地图里没有任何 places: %s' % path)
    for name, p in places.items():
        for key in ('x', 'y'):
            if key not in p:
                raise SystemExit('[ERR] 地点 %s 缺少 %s 字段' % (name, key))
    return frame, places
