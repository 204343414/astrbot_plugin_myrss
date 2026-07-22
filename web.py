from __future__ import annotations

from typing import Any, Callable, Awaitable, cast

from astrbot.api import logger
from astrbot.api.star import Context

try:
    from quart import jsonify as quart_jsonify
except ImportError:
    quart_jsonify = None


PLUGIN_NAME = "astrbot_plugin_myrss"


class MyRssWebController:
    """只读订阅页面 API；不提供写接口，避免网页误改生产断点。"""

    def __init__(self, context: Context, plugin: Any):
        self.context = context
        self.plugin = plugin

    def register_routes(self) -> None:
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/subscriptions/bootstrap",
            self._wrap(self.bootstrap),
            ["GET"],
            "MyRSS subscribed groups and feed status",
        )

    def _wrap(self, handler: Callable[[], Awaitable]):
        async def wrapped():
            if quart_jsonify is None:
                raise RuntimeError("Web framework is unavailable")
            try:
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": True, "data": await handler()}
                )
            except Exception as exc:
                logger.exception("MyRSS page request failed")
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": False, "message": str(exc)}
                ), 500

        wrapped.__name__ = handler.__name__
        return wrapped

    async def bootstrap(self) -> dict[str, Any]:
        async with self.plugin._data_lock:
            # 始终从当前正式库读一份快照，页面不持有 dh.data 的可变引用。
            disk_data = self.plugin.dh._read_json(self.plugin.dh.get_data_path())
            data = disk_data if isinstance(disk_data, dict) else self.plugin.dh.data
            groups: dict[str, dict[str, Any]] = {}
            for url, feed in data.items():
                if url in ("rsshub_endpoints", "settings") or not isinstance(feed, dict):
                    continue
                info = feed.get("info", {})
                for origin, sub in feed.get("subscribers", {}).items():
                    if "GroupMessage" not in origin or not isinstance(sub, dict):
                        continue
                    group = groups.setdefault(
                        origin,
                        {
                            "origin": origin,
                            "platform": origin.split(":", 1)[0],
                            "group_id": origin.split(":")[-1],
                            "feeds": [],
                        },
                    )
                    group["feeds"].append(
                        {
                            "title": info.get("title", url),
                            "url": url,
                            "cron_expr": sub.get("cron_expr", ""),
                            "last_update": int(sub.get("last_update", 0) or 0),
                            "latest_link": sub.get("latest_link", ""),
                            "seen_count": len(sub.get("seen_links", [])),
                        }
                    )
            result = sorted(groups.values(), key=lambda item: item["group_id"])
            for group in result:
                group["feeds"].sort(key=lambda item: item["title"])
            return {
                "data_path": self.plugin.dh.get_data_path(),
                "group_count": len(result),
                "subscription_count": sum(len(group["feeds"]) for group in result),
                "groups": result,
            }
