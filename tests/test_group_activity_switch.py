"""「群里有人说话才推送」必须是一个可关的开关，而且关掉后真的不再看群活跃度。

背景
----
`_target_readiness()` 是 myrss 所有主动投递路径的总闸门：

- `_cron_cb_url()`：拉取前过滤订阅者、分发前再确认一次；
- `run_delivery_diagnostic()` / 面板 `delivery_ready`；
- `_run_natural_subscribe()`：自然语言订阅的入口检查。

对 QQ 官方群，它要求本次启动后群内有人发过言（`_ready_group_sessions`）或适配器
`_session_scene` 里已有该群场景，否则目标被跳过（日志 `RSS目标未就绪，拉取前跳过`）。
Bot 重启后安静的群要等下一条群消息才恢复。

v1.4.1 新增配置项 `push_requires_group_activity`（默认 true，行为不变）：

- true  ：保持原样；
- false ：不再检查群活跃度，返回 `(True, "ready_activity_check_disabled")`。

关掉开关**不等于**跳过主动消息权限：订阅探针照旧真发一条消息，推送被 QQ 明确拒绝
「主动消息失败/无权限」时照旧自动退订当前源。这里只钉住"活跃度"这一层。

实现说明：main.py 顶层要 import astrbot / apscheduler / lxml / bs4 / PIL，测试环境未必
装得齐，因此这里用 ast 只把 `_target_readiness` 这一个方法抽出来执行，测的是真实代码
而不是字符串匹配。
"""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

G1 = "qq_official:GroupMessage:AAAA1111"


@pytest.fixture(scope="module")
def method():
    tree = ast.parse((ROOT / "main.py").read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_target_readiness":
            namespace = {}
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), "main.py", "exec"), namespace)
            return namespace["_target_readiness"]
    raise AssertionError("main.py 里找不到 _target_readiness")


@pytest.fixture(scope="module")
def config_bool():
    tree = ast.parse((ROOT / "main.py").read_text("utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_config_bool")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "main.py", "exec"), namespace)
    return namespace["_config_bool"]


def test_string_false_really_disables_group_activity(config_bool):
    assert config_bool("false", True) is False
    assert config_bool("关闭", True) is False
    assert config_bool("true", False) is True



class FakePlugin:
    """只带 _target_readiness 所需属性的最小替身。"""

    def __init__(self, require_activity, platform_name=None, observed=(), platform_id="qq_official"):
        self.require_group_activity = require_activity
        self._ready_group_sessions = set(observed)
        self.platform_name = platform_name
        self.platform_id = platform_id

    def _loaded_platforms(self):
        if self.platform_name is None:
            return {}
        instance = type("Adapter", (), {"_session_scene": {}})()
        return {self.platform_id: {"name": self.platform_name, "instance": instance}}


def _plugin(method, **kwargs):
    plugin = FakePlugin(**kwargs)
    plugin._target_readiness = method.__get__(plugin, FakePlugin)
    return plugin


def test_the_switch_defaults_to_the_existing_behaviour():
    """默认必须是 true：升级不该悄悄改变别人的推送时机。"""
    schema = json.loads((ROOT / "_conf_schema.json").read_text("utf-8"))
    option = schema["push_requires_group_activity"]
    assert option["type"] == "bool"
    assert option["default"] is True


def test_the_plugin_reads_the_switch_from_the_config():
    """开关要真的接上线，而不是只写在 schema 里。"""
    source = (ROOT / "main.py").read_text("utf-8")
    assert 'config.get("push_requires_group_activity", True)' in source


def test_the_switch_is_documented_in_the_readme():
    readme = (ROOT / "README.md").read_text("utf-8")
    assert "push_requires_group_activity" in readme


def test_gate_on_still_waits_for_a_group_message(method):
    """开关为 true 时维持原行为：没人说话就不算就绪。"""
    plugin = _plugin(method, require_activity=True, platform_name="qq_official")
    assert plugin._target_readiness(G1) == (False, "qq_official_waiting_group_message")


def test_gate_on_accepts_a_group_already_observed_this_process(method):
    plugin = _plugin(method, require_activity=True, platform_name="qq_official", observed=[G1])
    assert plugin._target_readiness(G1) == (True, "ready")


def test_gate_off_is_ready_without_any_group_message(method):
    """关掉后：没人说话也直接算可投递 —— 这就是这个开关的全部意义。"""
    plugin = _plugin(method, require_activity=False, platform_name="qq_official")
    assert plugin._target_readiness(G1) == (True, "ready_activity_check_disabled")


def test_gate_off_does_not_fake_the_observed_sessions(method):
    """不写 `_ready_group_sessions`：那是"真实观察到消息"的记录，面板的群列表也读它，
    运行中把开关改回 true 时不能留下假记录。"""
    plugin = _plugin(method, require_activity=False, platform_name="qq_official")
    for _ in range(3):
        plugin._target_readiness(G1)
    assert plugin._ready_group_sessions == set()


def test_gate_off_still_needs_the_platform_loaded(method):
    """平台没加载时一律不就绪：开关只关活跃度，不关平台检查。"""
    plugin = _plugin(method, require_activity=False, platform_name=None)
    assert plugin._target_readiness(G1) == (False, "platform_not_loaded")


def test_non_qq_official_targets_are_unaffected(method):
    """非 QQ 官方平台本来就恒为就绪，开关不该改变它们。"""
    for require in (True, False):
        plugin = _plugin(method, require_activity=require, platform_name="aiocqhttp", platform_id="aiocqhttp")
        assert plugin._target_readiness("aiocqhttp:GroupMessage:123") == (True, "ready")
