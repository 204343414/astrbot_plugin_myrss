from __future__ import annotations

from typing import Any, Callable, Awaitable, cast
import os

from astrbot.api import logger
from astrbot.api.star import Context

try:
    from quart import jsonify as quart_jsonify
    from quart import request as quart_request
except ImportError:
    quart_jsonify = None
    quart_request = None


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
            ("/subscriptions/ping", self.ping, ["GET"], "MyRSS page health check"),
            ("/subscriptions/bootstrap", self.bootstrap, ["GET"], "MyRSS subscribed groups and feed status"),
            ("/subscriptions/test-delivery", self.test_delivery, ["POST"], "Test RSS GET and proactive delivery"),
            ("/subscriptions/add", self.add_subscription, ["POST"], "Add a safety-reviewed subscription"),
            ("/subscriptions/remove", self.remove_subscription, ["POST"], "Remove one group subscription"),
            ("/moderation/resolve", self.resolve_moderation, ["POST"], "Resolve a severe moderation review"),
        ]
        for path, handler, methods, description in routes:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}{path}", self._wrap(handler), methods, description
            )
            self.registered_routes.append(path)

    async def ping(self) -> dict[str, Any]:
        return {"message": "pong", "data_path": self.plugin.dh.get_data_path()}

    async def test_delivery(self) -> dict[str, Any]:
        if quart_request is None:
            raise RuntimeError("Web request framework is unavailable")
        payload = await quart_request.get_json(force=True, silent=True) or {}
        origin = str(payload.get("origin", ""))
        feed_url = str(payload.get("feed_url", ""))
        return await self.plugin.run_delivery_diagnostic(origin, feed_url)

    async def add_subscription(self) -> dict[str, Any]:
        if quart_request is None:
            raise RuntimeError("Web request framework is unavailable")
        payload = await quart_request.get_json(force=True, silent=True) or {}
        return await self.plugin.add_subscription_from_ui(
            str(payload.get("origin", "")), str(payload.get("url", ""))
        )

    async def remove_subscription(self) -> dict[str, Any]:
        if quart_request is None:
            raise RuntimeError("Web request framework is unavailable")
        payload = await quart_request.get_json(force=True, silent=True) or {}
        return await self.plugin.remove_subscription_from_ui(
            str(payload.get("origin", "")), str(payload.get("feed_url", ""))
        )

    async def resolve_moderation(self) -> dict[str, Any]:
        if quart_request is None:
            raise RuntimeError("Web request framework is unavailable")
        payload = await quart_request.get_json(force=True, silent=True) or {}
        return await self.plugin.resolve_moderation_review(
            str(payload.get("review_id", "")),
            str(payload.get("action", "")),
        )

    def _wrap(self, handler: Callable[[], Awaitable]):
        async def wrapped():
            if quart_jsonify is None:
                raise RuntimeError("Web framework is unavailable")
            try:
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": True, "data": await handler()}
                )
            except ValueError as exc:
                # 冷却、目标未就绪、参数不匹配都属于可预期的用户态诊断结果。
                self.last_error = f"ValueError: {exc}"
                return cast(Callable[[dict], Any], quart_jsonify)(
                    {"ok": False, "message": str(exc)}
                ), 400
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
                            "created_by": sub.get("created_by") if isinstance(sub.get("created_by"), dict) else {"source": "legacy"},
                            "paused_by_moderation": bool(sub.get("paused_by_moderation", False)),
                            "preview": preview,
                        }
                    )
            # 已观察但当前零订阅的群也保留在 UI，便于退订最后一个源后重新新增。
            for origin in self.plugin._ready_group_sessions:
                if "GroupMessage" not in origin or origin in groups:
                    continue
                groups[origin] = {
                    "origin": origin,
                    "platform": origin.split(":", 1)[0],
                    "group_id": origin.split(":")[-1],
                    "feeds": [],
                }
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
            settings = data.get("settings", {}) if isinstance(data.get("settings"), dict) else {}
            reviews = [
                review for review in settings.get("moderation_reviews", [])
                if isinstance(review, dict) and review.get("state") == "pending"
            ][:100]
            bans = [
                value for value in settings.get("subscription_bans", {}).values()
                if isinstance(value, dict) and value.get("banned")
            ]
            return {
                "data_path": self.plugin.dh.get_data_path(),
                "data_mtime": int(os.path.getmtime(self.plugin.dh.get_data_path())) if os.path.exists(self.plugin.dh.get_data_path()) else 0,
                "group_count": len(result),
                "subscription_count": sum(len(group["feeds"]) for group in result),
                "safety_events": safety_events,
                "moderation_reviews": reviews,
                "subscription_bans": bans,
                "groups": result,
            }
