from __future__ import annotations

from typing import Any, Callable, Awaitable, cast
import os

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
        self.registered_routes: list[str] = []
        self.last_error = ""

    def register_routes(self) -> None:
        routes = [
            ("/subscriptions/ping", self.ping, "MyRSS page health check"),
            ("/subscriptions/bootstrap", self.bootstrap, "MyRSS subscribed groups and feed status"),
        ]
        for path, handler, description in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}{path}", self._wrap(handler), ["GET"], description
            )
            self.registered_routes.append(path)

    async def ping(self) -> dict[str, Any]:
        return {"message": "pong", "data_path": self.plugin.dh.get_data_path()}

    def _wrap(self, handler: Callable[[], Awaitable]):
        async def wrapped():
            if quart_jsonify is None:
                raise RuntimeError("Web framework is unavailable")
            try:
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": True, "data": await handler()}
                )
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("MyRSS page request failed")
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": False, "message": self.last_error}
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
                info = feed.get("info") if isinstance(feed.get("info"), dict) else {}
                subscribers = feed.get("subscribers")
                if not isinstance(subscribers, dict):
                    continue
                for raw_origin, sub in subscribers.items():
                    origin = str(raw_origin or "")
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
                    preview = feed.get("last_item_preview")
                    if not isinstance(preview, dict) or preview.get("safety_status") != "SAFE":
                        preview = None
                    group["feeds"].append(
                        {
                            "title": info.get("title", url),
                            "description": info.get("description", ""),
                            "avatar": info.get("avatar", ""),
                            "url": url,
                            "cron_expr": sub.get("cron_expr", ""),
                            "last_update": int(sub.get("last_update", 0) or 0),
                            "latest_link": sub.get("latest_link", ""),
                            "seen_count": len(sub.get("seen_links", [])),
                            "delivery_status": sub.get("delivery_status") if isinstance(sub.get("delivery_status"), dict) else None,
                            "preview": preview,
                        }
                    )
            result = sorted(groups.values(), key=lambda item: item["group_id"])
            for group in result:
                group["feeds"].sort(key=lambda item: item["title"])
                ready, reason = self.plugin._target_readiness(group["origin"])
                group["delivery_ready"] = ready
                group["delivery_reason"] = reason
            raw_events = data.get("settings", {}).get("safety_events", []) if isinstance(data.get("settings"), dict) else []
            safety_events = []
            for event in raw_events[:20] if isinstance(raw_events, list) else []:
                if not isinstance(event, dict):
                    continue
                # API 白名单字段，绝不向页面返回原始正文、图片或动态链接。
                safety_events.append({key: event.get(key) for key in (
                    "id", "status", "source", "reason", "blocked_at", "content_fingerprint"
                )})
            return {
                "data_path": self.plugin.dh.get_data_path(),
                "data_mtime": int(os.path.getmtime(self.plugin.dh.get_data_path())) if os.path.exists(self.plugin.dh.get_data_path()) else 0,
                "group_count": len(result),
                "subscription_count": sum(len(group["feeds"]) for group in result),
                "safety_events": safety_events,
                "groups": result,
            }
