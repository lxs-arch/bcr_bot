#!/usr/bin/env python3
"""test_detector.py —— 检测器后端契约 + 可选依赖降级 + GroundingDINO 适配层

三件事必须钉死：

1. `stub` 后端：无依赖、可预测，真机联调几何链时用它。
2. 可选依赖缺失时**不能**抛 ImportError 栈：要在构造阶段给出能照抄执行的 pip 提示，
   并且 `missing_deps()` 能提前告知 CI/文档（torch/transformers 是可选依赖）。
3. `GroundingDINO.detect()` 的适配逻辑（prompt 补句点、阈值透传、target_sizes、
   输出转 Detection）可以用假 processor/model 完全离线验证，不需要真装 torch。

    python3 test_detector.py
    python3 -m pytest test_detector.py -q
"""

import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _testkit
import grounding as G


# ---------------------------------------------------------------- 依赖隔离工具
@contextlib.contextmanager
def hidden_modules(*names):
    """把 sys.modules[name] 置 None，模拟"这台机器没装这个包"。"""
    old = {}
    for name in names:
        old[name] = sys.modules.get(name, '<absent>')
        sys.modules[name] = None
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev == '<absent>':
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@contextlib.contextmanager
def fake_modules(*names):
    """塞入空的假模块，模拟"装了但没真功能"。"""
    old = {}
    for name in names:
        old[name] = sys.modules.get(name)
        sys.modules[name] = types.ModuleType(name)
    try:
        yield
    finally:
        for name, prev in old.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _Raises(object):
    def __init__(self, exc):
        self.exc = exc
        self.text = ''

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None, '期望抛 %s，实际没抛' % self.exc.__name__
        assert issubclass(et, self.exc), '抛了 %s，期望 %s' % (et.__name__, self.exc.__name__)
        self.text = str(ev)
        return True                     # 吞掉异常，外层继续断言


# ---------------------------------------------------------------- stub 后端
def test_stub_detector_passthrough():
    dets = [G.Detection((0, 0, 20, 20), 0.5)]
    stub = G.StubDetector(dets)
    out = stub.detect(None, '任意提示')
    assert len(out) == 1 and out[0] is dets[0], 'stub 应原样返回预置检测结果'
    assert G.StubDetector().detect(None, 'x') == [], '无预置时应返回空列表'


def test_make_detector_stub():
    det = G.make_detector('stub', detections=[G.Detection((0, 0, 20, 20), 0.5)])
    assert isinstance(det, G.StubDetector), 'make_detector("stub") 应返回 StubDetector'
    assert len(det.detect(None, 'x')) == 1, '预置结果应透传'


def test_make_detector_unknown_backend():
    """未知后端给出可选项，而不是 KeyError/traceback。"""
    with _Raises(SystemExit) as ctx:
        G.make_detector('yolo9000')
    assert 'stub' in ctx.text and 'dino' in ctx.text, '错误信息应列出可选后端: %r' % ctx.text


# ---------------------------------------------------------------- 可选依赖
def test_missing_deps_detects_absence():
    with hidden_modules('torch', 'transformers'):
        missing = G.GroundingDinoDetector.missing_deps()
    assert set(missing) == {'torch', 'transformers'}, 'missing_deps=%r' % (missing,)


def test_missing_deps_detects_presence():
    with fake_modules('torch', 'transformers'):
        missing = G.GroundingDinoDetector.missing_deps()
    assert missing == [], '假模块在 sys.modules 里应视为"已装"，实际 %r' % (missing,)


def test_construct_without_deps_gives_actionable_hint():
    """没装 torch 时构造检测器：SystemExit + 能照抄的 pip 命令，不是 ImportError 栈。"""
    with hidden_modules('torch', 'transformers'):
        with _Raises(SystemExit) as ctx:
            G.GroundingDinoDetector()
    assert 'pip3 install' in ctx.text, '提示里应有安装命令: %r' % ctx.text
    assert 'torch' in ctx.text and 'transformers' in ctx.text, (
        '提示里应点名缺的依赖: %r' % ctx.text)
    assert 'stub' in ctx.text, '应告诉用户可以先拿 stub 联调: %r' % ctx.text


def test_make_detector_dino_propagates_dependency_hint():
    """CLI 走 make_detector('dino') 时，缺依赖也要是 SystemExit 而不是崩栈。"""
    with hidden_modules('torch', 'transformers'):
        with _Raises(SystemExit) as ctx:
            G.make_detector('dino')
    assert 'pip3 install' in ctx.text


def test_make_detector_filters_kwargs():
    """make_detector 只把 dino 认识的参数传下去，None 不覆盖默认值。"""
    recorded = {}

    class RecordingDetector(object):
        def __init__(self, **kw):
            recorded.update(kw)

    real = G.GroundingDinoDetector
    G.GroundingDinoDetector = RecordingDetector
    try:
        G.make_detector('groundingdino', model_id='local/weights',
                        box_threshold=0.42, text_threshold=None, device=None,
                        detections=[1, 2], color_topic='ignored')
    finally:
        G.GroundingDinoDetector = real

    assert recorded == {'model_id': 'local/weights', 'box_threshold': 0.42}, (
        '应只透传 model_id/box_threshold，实际 %r' % (recorded,))


# ---------------------------------------------------------------- detect() 适配层
class _FakeTensor(list):
    def tolist(self):
        return list(self)


class _FakeInputs(dict):
    """BatchEncoding 的最小替身：既是 mapping（可 **inputs），又带 input_ids/to()。"""

    def __init__(self):
        super().__init__(pixel_values=[[0.0]], input_ids=_FakeTensor([1, 2, 3]))

    def to(self, device):
        self.device = device
        return self

    def __getattr__(self, name):
        try:
            return self[name]           # 真 BatchEncoding 也支持 inputs.input_ids 这种访问
        except KeyError:
            raise AttributeError(name)


class _FakeProcessor(object):
    def __init__(self):
        self.calls = []

    def __call__(self, images, text, return_tensors=None):
        self.calls.append({'text': text, 'return_tensors': return_tensors})
        return _FakeInputs()

    def post_process_grounded_object_detection(self, outputs, input_ids,
                                               box_threshold=None, text_threshold=None,
                                               target_sizes=None):
        self.post = {'box_threshold': box_threshold,
                     'text_threshold': text_threshold,
                     'target_sizes': target_sizes}
        return [{'boxes': _FakeTensor([[10.0, 20.0, 60.0, 90.0]]),
                 'scores': _FakeTensor([0.87]),
                 'labels': ['红箱子']}]


class _FakeTorch(object):
    @contextlib.contextmanager
    def no_grad(self):
        yield


class _FakeModel(object):
    def __call__(self, **kw):
        self.kw = kw
        return 'outputs'


class _Image(object):
    size = (640, 480)          # PIL 的 size 是 (width, height)


def _bare_detector(box_threshold=0.3, text_threshold=0.25):
    """不走 __init__（否则要真 torch），只把手动搭的假后端塞进去。"""
    det = G.GroundingDinoDetector.__new__(G.GroundingDinoDetector)
    det._torch = _FakeTorch()
    det.processor = _FakeProcessor()
    det.model = _FakeModel()
    det.device = 'cpu'
    det.model_id = 'fake/weights'
    det.box_threshold = box_threshold
    det.text_threshold = text_threshold
    return det


def test_dino_detect_adapts_output_to_detections():
    det = _bare_detector(box_threshold=0.42, text_threshold=0.17)
    out = det.detect(_Image(), '红箱子')

    assert len(out) == 1, '应返回 1 个 Detection，实际 %r' % (out,)
    d = out[0]
    assert isinstance(d, G.Detection), '应转换成 Detection'
    assert tuple(d.box) == (10.0, 20.0, 60.0, 90.0), 'box 应保持像素坐标: %r' % (d.box,)
    _testkit.approx(d.score, 0.87, note='score')
    assert d.label == '红箱子', 'label 应透传，实际 %r' % (d.label,)


def test_dino_detect_prompt_and_thresholds():
    det = _bare_detector(box_threshold=0.42, text_threshold=0.17)
    det.detect(_Image(), '红箱子')
    assert det.processor.calls[-1]['text'] == '红箱子.', '提示词应补一个句点: %r' % (
        det.processor.calls[-1],)
    assert det.processor.calls[-1]['return_tensors'] == 'pt', '应按 batch 返回张量'
    assert det.processor.post['box_threshold'] == 0.42, '框阈值应透传'
    assert det.processor.post['text_threshold'] == 0.17, '文本阈值应透传'
    assert det.processor.post['target_sizes'] == [(480, 640)], (
        'target_sizes 应为 [(高, 宽)]，实际 %r' % (det.processor.post['target_sizes'],))

    det.detect(_Image(), 'a red box.')
    assert det.processor.calls[-1]['text'] == 'a red box.', '已带句点不要重复加: %r' % (
        det.processor.calls[-1]['text'],)


def test_dino_detect_empty_result():
    """模型什么都没检出 → 空列表（上层据此判"没找到"），不能崩。"""
    det = _bare_detector()

    def empty(**kw):
        return [{'boxes': _FakeTensor([]), 'scores': _FakeTensor([]), 'labels': []}]

    det.processor.post_process_grounded_object_detection = (
        lambda *a, **kw: [{'boxes': _FakeTensor([]), 'scores': _FakeTensor([]), 'labels': []}])
    assert det.detect(_Image(), 'nothing here') == [], '空结果应返回 []'


if __name__ == '__main__':
    sys.exit(_testkit.run(globals()))
