"""rsshub-vault 客户端（插件侧）。

只做三件事：
  1. 找到 helper 的地址和 token；
  2. 把请求发过去，把结果原样带回来；
  3. 把各种失败翻译成人话。

硬约束：本模块**不读取、不记录、不返回任何凭据内容**。
凭据内容只存在于 helper 的 secrets.json 和 rsshub 的环境变量里。
"""
from __future__ import annotations

import os
from typing import Any

try:
    import aiohttp
except ImportError:  # pragma: no cover - 容器里一定有
    aiohttp = None


DEFAULT_HELPER = "http://172.18.0.1:18100"
DEFAULT_TIMEOUT = 90.0
# 重建 rsshub 容器会慢一些，导入/设置这两类请求给更长的等待
LONG_TIMEOUT = 240.0

TOKEN_CANDIDATES = (
    os.environ.get("VAULT_TOKEN_FILE", ""),
    "/AstrBot/data/plugin_data/astrbot_plugin_myrss/vault_token",
    "/AstrBot/data/.rsshub_vault_token",
)


def read_token() -> str:
    """按候选路径找 token；找不到返回空串（helper 会回 401，我们翻译成人话）。"""
    for path in TOKEN_CANDIDATES:
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                tok = f.read().strip()
            if tok:
                return tok
        except OSError:
            continue
    return ""


def _friendly(status: int, body: Any) -> dict[str, Any]:
    if status == 401:
        return {
            "ok": False,
            "error": "维护服务鉴权失败：token 对不上（把宿主上的 token 文件复制到插件数据目录，"
                     "或重启一次 rsshub-vault）",
            "code": 401,
        }
    if status == 403:
        return {"ok": False, "error": "维护服务拒绝了本次来源（容器网段不在白名单）", "code": 403}
    if status >= 500:
        return {"ok": False, "error": f"维护服务内部出错（HTTP {status}）", "code": status}
    if isinstance(body, dict):
        return body
    return {"ok": False, "error": f"维护服务返回了非预期内容（HTTP {status}）", "code": status}


class VaultClient:
    """与宿主侧 rsshub-vault 通信。"""

    def __init__(self, base_url: str = "", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = (base_url or DEFAULT_HELPER).rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------ 底层
    async def _request(self, method: str, path: str,
                       payload: dict | None = None,
                       timeout: float | None = None) -> dict[str, Any]:
        if aiohttp is None:
            return {"ok": False, "error": "容器里缺少 aiohttp，无法访问维护服务"}
        token = read_token()
        if not token:
            return {
                "ok": False,
                "error": "还没拿到维护服务的 token：请把宿主上的 "
                         "/mnt/hdd/bot/astrbot/data/.rsshub_vault_token 复制到插件数据目录并命名为 "
                         "vault_token",
                "code": 0,
            }
        url = f"{self.base_url}{path}"
        headers = {"X-Vault-Token": token}
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout or self.timeout)
            ) as sess:
                async with sess.request(method, url, json=payload, headers=headers) as resp:
                    try:
                        body = await resp.json(content_type=None)
                    except Exception:  # noqa: BLE001
                        body = None
                    return _friendly(resp.status, body)
        except aiohttp.ClientConnectorError:
            return {
                "ok": False,
                "error": "维护服务没在运行（rsshub-vault）：在宿主执行 "
                         "systemctl --user start rsshub-vault",
                "code": 0,
            }
        except aiohttp.ClientError as exc:
            return {"ok": False, "error": f"访问维护服务失败：{type(exc).__name__}", "code": 0}
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            name = type(exc).__name__
            if "timeout" in name.lower():
                return {"ok": False, "error": "维护服务响应超时（重建 rsshub 时大约要 20 秒，请稍后刷新）",
                        "code": 0}
            return {"ok": False, "error": f"访问维护服务异常：{name}", "code": 0}

    # ------------------------------------------------------------ 业务
    async def ping(self) -> dict[str, Any]:
        return await self._request("GET", "/vault/health", timeout=10)

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/vault/status")

    async def platforms(self) -> dict[str, Any]:
        return await self._request("GET", "/vault/platforms")

    async def alerts(self) -> dict[str, Any]:
        return await self._request("GET", "/vault/alerts")

    async def import_platform(self, platform: str) -> dict[str, Any]:
        if not platform:
            return {"ok": False, "error": "缺少平台名"}
        return await self._request("POST", f"/vault/import/{platform}", timeout=LONG_TIMEOUT)

    async def set_value(self, platform: str, value: str, kind: str = "") -> dict[str, Any]:
        if not platform:
            return {"ok": False, "error": "缺少平台名"}
        if not value:
            return {"ok": False, "error": "内容为空"}
        payload: dict[str, Any] = {"value": value}
        if kind:
            payload["kind"] = kind
        return await self._request("POST", f"/vault/set/{platform}", payload,
                                   timeout=LONG_TIMEOUT)

    async def verify(self, platform: str) -> dict[str, Any]:
        if not platform:
            return {"ok": False, "error": "缺少平台名"}
        return await self._request("POST", f"/vault/verify/{platform}")

    async def check(self, sync: bool = True) -> dict[str, Any]:
        return await self._request("POST", "/vault/check", {"sync": bool(sync)},
                                   timeout=LONG_TIMEOUT)
