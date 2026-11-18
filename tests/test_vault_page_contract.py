"""面板前端契约：插件页面不是普通网页，接口必须走 AstrBot 桥接层。

这条测试是给"第一次踩坑"留下的护栏——用 fetch("/api/plug/...") 在
AstrBot 的插件页面里会直接 NetworkError，因为页面是嵌在面板里的，
拿不到同源绝对地址。
"""
import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1].joinpath("pages/subscriptions/vault.js").read_text("utf-8")


def test_uses_the_astrbot_page_bridge():
    assert "AstrBotPluginPage" in JS
    assert "apiGet" in JS and "apiPost" in JS


def test_never_calls_the_http_api_directly():
    # 只看代码行，注释里提到 fetch 是在解释"为什么不能这么干"
    code = "\n".join(
        line for line in JS.splitlines() if not line.lstrip().startswith("//")
    )
    assert "fetch(" not in code, "插件页面里不能用裸 fetch，必须走桥接层"
    assert "/api/plug" not in code, "桥接层自己补前缀，写死前缀反而会拼错"


def test_paths_have_no_plugin_prefix():
    paths = re.findall(r'(?:get|post)\("([^"]+)"', JS)
    assert paths, "没找到任何接口调用"
    for p in paths:
        assert not p.startswith("/"), f"{p} 不该以斜杠开头"
        assert "astrbot_plugin_myrss" not in p, f"{p} 不该带插件名"


def test_guard_on_missing_elements():
    """一处 DOM 找不到不能连坐其它按钮。"""
    assert "缺少元素" in JS
    assert "try {" in JS and "catch (err)" in JS
