"""插件侧桥接层测试：不联网、不需要 astrbot、不需要真 token。"""
import asyncio
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 插件根目录
sys.path.insert(0, HERE)

import vault_client  # noqa: E402


class TestToken(unittest.TestCase):
    def test_reads_first_existing_token_file(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "vault_token")
        with open(p, "w", encoding="utf-8") as f:
            f.write("  abc123  \n")
        orig = vault_client.TOKEN_CANDIDATES
        vault_client.TOKEN_CANDIDATES = (p,)
        try:
            self.assertEqual(vault_client.read_token(), "abc123")
        finally:
            vault_client.TOKEN_CANDIDATES = orig

    def test_returns_empty_when_nothing_found(self):
        orig = vault_client.TOKEN_CANDIDATES
        vault_client.TOKEN_CANDIDATES = ("/nonexistent/x", "")
        try:
            self.assertEqual(vault_client.read_token(), "")
        finally:
            vault_client.TOKEN_CANDIDATES = orig


class TestFriendly(unittest.TestCase):
    def test_401_becomes_actionable_message(self):
        out = vault_client._friendly(401, None)
        self.assertFalse(out["ok"])
        self.assertIn("token", out["error"])

    def test_403_mentions_network(self):
        self.assertIn("来源", vault_client._friendly(403, None)["error"])

    def test_500_is_generic(self):
        out = vault_client._friendly(500, None)
        self.assertIn("500", out["error"])

    def test_200_passes_body_through(self):
        self.assertEqual(vault_client._friendly(200, {"ok": True, "platforms": []}),
                         {"ok": True, "platforms": []})


class Recorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, method, path, payload=None, timeout=None):
        self.calls.append({"method": method, "path": path, "payload": payload, "timeout": timeout})
        return {"ok": True}


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()
        self.client = vault_client.VaultClient("http://helper:18100")
        self.client._request = self.rec  # 桩掉网络层

    def run_async(self, coro):
        return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)

    def test_status_is_get(self):
        self.run_async(self.client.status())
        self.assertEqual(self.rec.calls[-1]["method"], "GET")
        self.assertEqual(self.rec.calls[-1]["path"], "/vault/status")

    def test_import_posts_platform_and_uses_long_timeout(self):
        self.run_async(self.client.import_platform("bilibili"))
        c = self.rec.calls[-1]
        self.assertEqual((c["method"], c["path"]), ("POST", "/vault/import/bilibili"))
        self.assertEqual(c["timeout"], vault_client.LONG_TIMEOUT)

    def test_set_sends_value_only_when_kind_given(self):
        self.run_async(self.client.set_value("github", "ghp_x"))
        self.assertEqual(self.rec.calls[-1]["payload"], {"value": "ghp_x"})
        self.run_async(self.client.set_value("javdb", "a=b", "cookie"))
        self.assertEqual(self.rec.calls[-1]["payload"], {"value": "a=b", "kind": "cookie"})

    def test_empty_inputs_short_circuit_without_network(self):
        before = len(self.rec.calls)
        out = self.run_async(self.client.import_platform(""))
        self.assertFalse(out["ok"])
        out2 = self.run_async(self.client.set_value("github", ""))
        self.assertFalse(out2["ok"])
        self.assertEqual(len(self.rec.calls), before, "空参数不该发起请求")

    def test_check_passes_sync_flag(self):
        self.run_async(self.client.check(sync=False))
        self.assertEqual(self.rec.calls[-1]["payload"], {"sync": False})


class TestNoSecretLeakage(unittest.TestCase):
    def test_module_source_never_logs_values(self):
        src = open(os.path.join(HERE, "vault_client.py"), encoding="utf-8").read()
        for bad in ("print(", "logger.info", "logger.debug"):
            self.assertNotIn(bad, src, f"桥接层不该打印任何东西：{bad}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
