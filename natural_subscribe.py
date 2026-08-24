"""
自然语言订阅模块（v1.4.0 新增）。

核心思路（与 NATURAL_LANGUAGE_SUBSCRIBE.md v0.2 一致）：
  - 插件命令 `/myrss + 自然语言` 触发后，临时起一次 `ctx.tool_loop_agent()` 调用
  - LLM 在这次临时 run 里只能调 5 个 myrss_* tool + 可选 TavilyWebSearchTool
  - LLM 不能直接说话给用户；所有用户可见输出都来自 tool handler
  - 跑完即销毁，不持久化对话历史

模块导出：
  - IntentCard                待确认订阅的结构化数据
  - NLIntentStore             内存中的 _nl_pending[origin] = IntentCard
  - build_nl_tool_set         第一次 agent run 的 ToolSet
  - build_confirm_tool_set    第二次 agent run 的 ToolSet
  - NAT_LANG_SYSTEM_PROMPT    第一次 agent run 的 system prompt
  - NAT_LANG_CONFIRM_PROMPT   第二次 agent run 的 system prompt
  - TOOL_LOOKUP / TOOL_PREVIEW_CARD / TOOL_EMIT_RESULT /
    TOOL_CONFIRM_SUBSCRIBE / TOOL_TERMINATE / TOOL_IGNORE  6 个 tool 名常量
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image as MsgImage, Plain
from astrbot.core.agent.tool import FunctionTool, ToolSet, ToolExecResult

try:
    # AstrBot v4 builtin websearch tool. 我们把它当作"LLM 可调的辅助能力"。
    from astrbot.core.tools.web_search_tools import TavilyWebSearchTool
    _HAS_WEBSEARCH = True
except Exception:  # pragma: no cover - 旧版 AstrBot 兼容
    TavilyWebSearchTool = None
    _HAS_WEBSEARCH = False


logger = logging.getLogger("astrbot")


# ============================================================
# 6 个 tool 的固定名称
# ============================================================
TOOL_LOOKUP = "myrss_lookup"
TOOL_PREVIEW_CARD = "myrss_preview_card"
TOOL_EMIT_RESULT = "myrss_emit_result"
TOOL_CONFIRM_SUBSCRIBE = "myrss_confirm_subscribe"
TOOL_TERMINATE = "myrss_terminate"
TOOL_IGNORE = "myrss_ignore"


# ============================================================
# 数据结构
# ============================================================
@dataclass
class IntentCard:
    """LLM 调 myrss_preview_card 后，Bot 解析的"待确认订阅"结构化数据。"""

    feed_url: str
    route: str
    platform: str
    handle: str
    title: str
    description: str
    image_b64: str
    pub_date: str
    card_b64: str
    # 用于"用户引用卡片触发二次 LLM"时把卡片摘要带回去
    summary: str = ""
    creator_openid: str = ""
    creator_name: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0


class NLIntentStore:
    """_nl_pending[origin] = IntentCard。纯内存，插件重载即清空。"""

    def __init__(self, ttl_seconds: int = 600):
        self._by_origin: dict[str, IntentCard] = {}
        self._ttl = ttl_seconds

    def put(self, origin: str, card: IntentCard, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self._ttl
        card.expires_at = time.time() + ttl
        self._by_origin[origin] = card

    def peek(self, origin: str) -> Optional[IntentCard]:
        card = self._by_origin.get(origin)
        if card is None:
            return None
        if card.expires_at > 0 and time.time() > card.expires_at:
            self._by_origin.pop(origin, None)
            return None
        return card

    def pop(self, origin: str) -> Optional[IntentCard]:
        return self._by_origin.pop(origin, None)

    def gc(self) -> int:
        """删除过期 entry，返回删除数。"""
        now = time.time()
        stale = [o for o, c in self._by_origin.items() if c.expires_at > 0 and now > c.expires_at]
        for o in stale:
            self._by_origin.pop(o, None)
        return len(stale)

    def __len__(self) -> int:
        return len(self._by_origin)


# ============================================================
# System prompt
# ============================================================
NAT_LANG_SYSTEM_PROMPT = """\
你是一个临时性的"订阅意图识别 + 预览确认"助手，仅在 QQ 群用户发送 `/myrss + 自然语言` 时被启动。
你的 umo / 群号已隐含在事件上下文中，不要再去获取。

你的全部工作（按顺序）：
  1. 调 myrss_lookup 查 RSSHub 路由（传入用户原文）
  2. 如果返回候选，调 myrss_preview_card 渲染预览卡片（这一步会真正发到群里）
  3. 调 myrss_terminate 结束，等待用户回复

⚠️ 非常重要：调完 myrss_terminate 后，你必须在 final response（不是 tool call）里输出至少一句话。
   AstrBot 框架在 tool loop 结束后会强制要求 LLM 输出 final text，content=None / 空字符串都会导致整个流程报错 EmptyModelOutputError。
   简短即可，例如"完成"、"已发送预览卡片，请等待用户确认"、"未找到博主，已提示用户提供 URL"。
   这一句话**不会**作为 Bot 消息发给用户（Bot 消息只来自 tool handler），但框架需要它。

绝对禁止：
  - 跟用户闲聊、解释自己是什么、回应问候
  - 在 myrss_preview_card 之后再调任何其他 tool（除了 myrss_terminate）
  - 调本对话以外的任何 AstrBot 内置 tool（myrss_* 之外都视为越界，调用就视为失败，调 myrss_ignore）

异常分支：
  - myrss_lookup 返回"未找到"或"需要联网搜"：
      优先调联网搜索工具（工具名形如 web_search_tavily / web_search_bocha / web_search_brave，如果可用）联网查，查完再次调 myrss_lookup
      仍找不到就调 myrss_emit_result(action="not_found", free_text="< 80 字自然语言解释>")，然后 myrss_terminate
  - 用户说的是非订阅请求（闲聊/提问/调试）：
      调 myrss_ignore，然后 myrss_terminate
  - 任何 tool 内部报错：
      调 myrss_emit_result(action="not_found", free_text="服务器暂忙，请稍后再试。")，然后 myrss_terminate
"""

NAT_LANG_CONFIRM_PROMPT = """\
上一条自然语言订阅请求触发了预览卡片（标题/平台/handle/route 见 prompt），用户刚刚在群里引用了那条卡片（或发送了含同意/拒绝关键词的消息），原文如下。

请只做一件事：
  解析用户意图，调对应 tool：
    - 同意 / 订阅 / 对 / 没错 / 就这个 / y / yes / ok
        → myrss_confirm_subscribe(decision="approve")
    - 不要 / 拒绝 / 取消 / 算了 / n / no
        → myrss_confirm_subscribe(decision="reject")
    - 不是这个 / 错了 / 换一个 / 那个是 X 我要 Y
        → myrss_confirm_subscribe(decision="not_this")
    - 其他无关/含糊/聊天/调试
        → myrss_ignore

调完上述任一 tool 后，调 myrss_terminate 结束。
不要发任何文字给用户（myrss_confirm_subscribe 和 myrss_ignore 内部会处理）。
不要解释自己做了什么。

⚠️ 非常重要：调完 myrss_terminate 后，你必须在 final response（不是 tool call）里输出至少一句话，例如"完成"。
   AstrBot 框架在 tool loop 结束后会强制要求 LLM 输出 final text，content=None / 空字符串会报 EmptyModelOutputError。
   这一句话不会作为 Bot 消息发给用户（用户可见内容只来自 tool handler），但框架需要它。
"""


# ============================================================
# 工具函数：抽 URL 里的 handle / 平台
# ============================================================
_PLATFORM_HINTS = [
    # 顺序很重要：先匹配 twitter 域名 (含 x.com)，再匹配 youtube
    (re.compile(r"^https?://(?:www\.)?(?:twitter|x)\.com/(@?)(?P<handle>[A-Za-z0-9_]+)", re.I), "X", "/twitter/user/{handle}"),
    (re.compile(r"^https?://(?:www\.)?youtube\.com/(?P<handle>@[A-Za-z0-9_.-]+)", re.I), "YouTube", "/youtube/user/{handle}"),
    (re.compile(r"^https?://(?:www\.)?youtube\.com/channel/(?P<id>[A-Za-z0-9_-]+)", re.I), "YouTube", "/youtube/channel/{id}"),
    (re.compile(r"^https?://space\.bilibili\.com/(?P<uid>\d+)", re.I), "B站", "/bilibili/user/dynamic/{uid}"),
    (re.compile(r"^https?://(?:www\.)?zhihu\.com/people/(?P<id>[A-Za-z0-9_-]+)", re.I), "知乎", "/zhihu/people/activities/{id}"),
    (re.compile(r"^https?://(?:www\.)?weibo\.com/u/(?P<uid>\d+)", re.I), "微博", "/weibo/user/{uid}"),
    (re.compile(r"^https?://t\.me/(?P<handle>[A-Za-z0-9_]+)", re.I), "Telegram", "/telegram/channel/{handle}"),
    (re.compile(r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)", re.I), "GitHub", "/github/commits/{owner}/{repo}"),
]


def _parse_user_query_to_candidates(query: str) -> List[dict]:
    """
    把用户的自然语言拆成 (platform, handle, candidate_url) 列表。
    不调 LLM，纯规则。LLM 会在 myrss_lookup 里给我们 "OpenAI 的推特" 这种，
    我们先抽出可能的 URL/handle 模式。
    """
    query = (query or "").strip()
    candidates: list[dict] = []
    if not query:
        return candidates

    # 显式 URL
    url_m = re.search(r"https?://[^\s,，]+", query)
    if url_m:
        url = url_m.group(0).rstrip("。.,，)）]】>")
        for pat, platform, tpl in _PLATFORM_HINTS:
            m = pat.match(url)
            if m:
                gd = m.groupdict()
                route = tpl.format(**gd)
                candidates.append({
                    "platform": platform,
                    "handle": gd.get("handle") or gd.get("uid") or gd.get("id") or gd.get("owner", ""),
                    "candidate_url": url,
                    "route": route,
                })
                break
        return candidates

    # 没有显式 URL：让 LLM 拼出 "X 叫 xxx" 这种自然语言模式。
    # 这里我们只做轻量级处理：把"X/推特" → 平台，把"叫 xxx / 是 xxx / OpenAI / sama" → handle。
    q_low = query.lower()
    platform = None
    route_tpl = None
    if any(k in query for k in ("推特", "推 X", "Twitter", "X 上", "X 上面", "x.com", "x 上", " X ")):
        platform = "X"
        route_tpl = "/twitter/user/{handle}"
    elif any(k in query for k in ("YouTube", "youtube", "油管", "YouTube 频道", "YT")):
        platform = "YouTube"
        route_tpl = "/youtube/user/@{handle}"
    elif any(k in query for k in ("B 站", "B站", "bilibili", "哔哩哔哩")):
        platform = "B站"
        route_tpl = "/bilibili/user/dynamic/{handle}"  # 需要数字 uid
    elif "微博" in query:
        platform = "微博"
        route_tpl = "/weibo/user/{handle}"  # 需要数字 uid
    elif "知乎" in query:
        platform = "知乎"
        route_tpl = "/zhihu/people/activities/{handle}"
    elif "GitHub" in query or "github" in query:
        platform = "GitHub"
        route_tpl = "/github/repos/{handle}"
    elif "Telegram" in query or "telegram" in query or "TG" in query or "tg" in q_low:
        platform = "Telegram"
        route_tpl = "/telegram/channel/{handle}"

    if not platform:
        return candidates

    # 提取 handle：尝试 "叫 X" / "是 X" / "用户 X" / "账号 X" 模式
    handle = ""
    patterns = [
        r"叫\s*([A-Za-z0-9_@.\-]+)",
        r"是\s*([A-Za-z0-9_@.\-]+)",
        r"账号\s*([A-Za-z0-9_@.\-]+)",
        r"用户\s*([A-Za-z0-9_@.\-]+)",
        r"博主\s*([A-Za-z0-9_@.\-]+)",
        r"频道\s*([A-Za-z0-9_@.\-]+)",
        r"主页\s*([A-Za-z0-9_@.\-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, query)
        if m:
            handle = m.group(1).rstrip("的,.，。!！?？)）]】")
            if handle:
                break

    # 兜底：取最后一个英文/数字/下划线串
    if not handle:
        m = re.search(r"([A-Za-z][A-Za-z0-9_]{2,})", query)
        if m:
            handle = m.group(1)

    if not handle:
        return candidates

    handle = handle.lstrip("@")
    route = route_tpl.format(handle=handle)
    # X / YouTube / GitHub / Telegram / 知乎：直接拼出 candidate_url 模式（用于 _resolve_feed_url 的 match）
    if platform == "X":
        candidate_url = f"https://x.com/{handle}"
    elif platform == "YouTube":
        candidate_url = f"https://youtube.com/@{handle}"
    elif platform == "B站":
        # B 站需要数字 uid；如果 handle 不是数字，标记 low confidence
        candidate_url = f"https://space.bilibili.com/{handle}"
    elif platform == "微博":
        candidate_url = f"https://weibo.com/u/{handle}"
    elif platform == "知乎":
        candidate_url = f"https://zhihu.com/people/{handle}"
    elif platform == "GitHub":
        candidate_url = f"https://github.com/{handle}"
    elif platform == "Telegram":
        candidate_url = f"https://t.me/{handle}"
    else:
        candidate_url = ""

    candidates.append({
        "platform": platform,
        "handle": handle,
        "candidate_url": candidate_url,
        "route": route,
        # 标记 handle 是否像是数字 uid（B 站 / 微博必须）
        "needs_numeric_uid": platform in ("B站", "微博"),
        "handle_is_numeric": handle.isdigit(),
    })
    return candidates


# ============================================================
# 6 个 tool handler 工厂函数
# ============================================================
def _make_lookup_tool(plugin) -> FunctionTool:
    """myrss_lookup: 把用户自然语言 → 候选订阅源列表。"""

    async def handler(event, **kwargs) -> str:
        query = str(kwargs.get("query", "")).strip()
        if plugin.nl_debug_log:
            logger.info("[NL][lookup] query=%r", query)
        if not query:
            return "未找到。请提供更具体的描述。"

        # 1) 显式 URL 优先
        url_m = re.search(r"https?://[^\s,，]+", query)
        if url_m:
            url = url_m.group(0).rstrip("。.,，)）]】>")
            # plugin 是 main.py 里的 MyRssPlugin 实例，URLMapper 是 main.py 模块级类
            try:
                from . import main as _main_mod
                url_mapper = _main_mod.URLMapper
            except Exception:
                url_mapper = None
            matched = url_mapper.match(url) if url_mapper else None
            if matched:
                route, platform = matched
                return (
                    f"找到 1 个候选订阅源:\n"
                    f"  1. platform={platform}, candidate_url={url}, route={route}\n"
                    f"请用 candidate_url 调 myrss_preview_card。"
                )

        # 2) 规则解析
        candidates = _parse_user_query_to_candidates(query)
        if not candidates:
            # 没匹配上平台。提示 LLM 用 web_search 兜底 / 让用户提供 URL。
            # 不直接判 not_found，避免 LLM 误以为流程结束。
            return (
                "未识别到平台关键词（X/推特/YouTube/B站/微博/知乎/GitHub/Telegram）。\n"
                "请按以下顺序尝试：\n"
                "  1) 如果联网搜索工具可用(工具名形如 web_search_tavily/web_search_bocha/web_search_brave), 调它搜 '"
                + query[:30]
                + " site:x.com 官方账号' 或 '"
                + query[:30]
                + " 官方 账号 平台', 然后再调 myrss_lookup 传入搜到的 URL\n"
                "  2) 仍找不到, 调 myrss_emit_result(action=\"not_found\", free_text=\"< 80 字提示用户给 URL>\") + myrss_terminate\n"
                "  3) 用户已经贴了 URL, 直接用 URL 调 myrss_lookup 再调 myrss_preview_card\n"
                "用户原话: " + query
            )

        # 3) 检查 handle 合法性
        c = candidates[0]
        if c.get("needs_numeric_uid") and not c.get("handle_is_numeric"):
            return (
                f"平台 {c['platform']} 需要数字 uid，但用户给的是 '{c['handle']}'（非数字）。"
                f"请用户明确提供该博主的主页 URL，例如：\n"
                f"  /myrss + https://space.bilibili.com/123456\n"
                f"  /myrss + https://weibo.com/u/123456"
            )

        return (
            f"找到 1 个候选订阅源:\n"
            f"  1. platform={c['platform']}, handle={c['handle']}, "
            f"candidate_url={c['candidate_url']}, route={c['route']}\n"
            f"请用 candidate_url 调 myrss_preview_card。"
        )

    tool = FunctionTool(
        name=TOOL_LOOKUP,
        description=(
            "在 RSSHub 支持的平台里查找某博主/频道。"
            "传入用户原文描述，返回候选订阅源 URL 列表。"
            "调用前不要自己拼 URL；这个工具会校验 URL 合法性。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户原话，例如 'OpenAI 的推特' / 'B站科技区' / '帮我订阅 https://x.com/OpenAI'",
                }
            },
            "required": ["query"],
        },
        handler=handler,
    )
    return tool


def _make_preview_card_tool(plugin) -> FunctionTool:
    """myrss_preview_card: 拉 RSS + 渲染预览 + 写入 pending + 发卡片。"""

    async def handler(event, **kwargs) -> str:
        feed_url = str(kwargs.get("feed_url", "")).strip()
        if plugin.nl_debug_log:
            logger.info("[NL][preview_card] feed_url=%r", feed_url)
        origin = event.unified_msg_origin
        if "GroupMessage" not in origin:
            return "错误：仅在群聊中可用。"

        if not feed_url:
            return await _send_err(plugin, origin, "feed_url 不能为空。")

        # 1) 解析成完整 RSS URL
        full_url, route, error = plugin._resolve_feed_url(feed_url)
        if not full_url:
            return await _send_err(plugin, origin, f"{error}（可直接发送 /myrss + 主页链接或 / 开头路由）")

        # 2) 拉 RSS
        raw = await plugin._fetch(full_url)
        if not raw:
            return await _send_err(plugin, origin, "无法访问该 RSS 源，请稍后再试或换个源。")

        # 3) 解析频道信息
        try:
            title, description, avatar = plugin.dh.parse_channel_info(raw)
        except Exception as exc:
            return await _send_err(plugin, origin, f"频道解析失败: {exc}")

        # 3.5) 先写入临时 dh.data entry（必须在 _poll 之前！）。
        # 之前写在 _poll 之后，导致 _poll 时 full_url 尚不在 dh.data，
        # item.chan_title 变成 "未知"，_make_card_b64 因此跳过头像下载、
        # _get_avatar_url 也匹配不到 -> 预览卡片缺头像/名字（而 /myrss eye
        # 是先写库再 poll，所以正常）。
        old_entry = plugin.dh.data.get(full_url)
        if old_entry is None:
            plugin.dh.data[full_url] = {
                "subscribers": {},
                "info": {"title": title, "description": description, "avatar": avatar},
            }

        try:
            # 4) 拉最新一条
            items = await plugin._poll(full_url, num=1)
            if not items:
                return await _send_err(plugin, origin, "该源目前没有可审核的动态。")

            # 5) 全量内容审核
            status = await plugin._check_content_safe(items[0])
            if status != "SAFE":
                vision = plugin._vision_cache.get(plugin._item_cache_key(items[0]), {})
                reason = vision.get("description", "") if isinstance(vision, dict) else ""
                detail = (reason or f"内容安全审核未通过（{status}）")[:80]
                # 直接发消息，不依赖 LLM 用 myrss_emit_result 转达。
                # 否则 LLM 若直接调 myrss_terminate 会导致"没发任何消息就静默结束"。
                await plugin._send_message_guarded(
                    origin,
                    _plain_chain(f"⛔ 该订阅源最新动态未通过内容安全审核，未创建订阅。\n（{detail}）"),
                )
                return (
                    f"action=rejected_preview|reason={status}|detail={detail}。已直接向群发送审核未通过提示。"
                )

            # 6) 渲染预览卡片
            card_b64 = await plugin._make_card_b64(items[0])
            if not card_b64:
                return await _send_err(plugin, origin, "卡片渲染失败（browserless 不可用），无法生成预览。")

            # 7) 计算 summary (供二次 LLM 用)
            summary = (
                f"title={title}\nplatform={_platform_from_route(route)}\n"
                f"handle={_handle_from_url_or_route(feed_url, route)}\nroute={route}"
            )

            # 8) 写入 store
            try:
                creator_openid = str(event.get_sender_id())
            except Exception:
                creator_openid = ""
            try:
                creator_name = str(event.get_sender_name() or "")
            except Exception:
                creator_name = ""

            card = IntentCard(
                feed_url=full_url,
                route=route,
                platform=_platform_from_route(route),
                handle=_handle_from_url_or_route(feed_url, route),
                title=title or "",
                description=description or "",
                image_b64=card_b64,
                pub_date=items[0].pubDate or "",
                card_b64=card_b64,
                summary=summary,
                creator_openid=creator_openid,
                creator_name=creator_name,
            )
            plugin.nl_pending.put(origin, card, ttl=plugin.nl_pending_ttl)

            # 9) 发卡片到群
            send_ok = await plugin._send_message_guarded(
                origin,
                _build_preview_message(card, plugin),
            )
            if not send_ok:
                plugin.nl_pending.pop(origin)
                error_text = plugin._target_last_send_error.get(origin, "发送失败")
                return await _send_err(plugin, origin, f"卡片发送失败 ({error_text})。请确认本群 Bot 已开启主动消息权限。")

            if plugin.nl_debug_log:
                logger.info("[NL][preview_card] sent, summary=%s", summary)

            return "预览卡片已发送到当前群。等用户引用本卡片回复'同意/拒绝/不是这个'。"
        finally:
            # 清理：仅当这是我们临时创建的 entry 且尚未产生订阅关系时移除，
            # 避免残留空 subscribers 的孤儿 entry。
            if old_entry is None and not plugin.dh.data.get(full_url, {}).get("subscribers"):
                plugin.dh.data.pop(full_url, None)

    tool = FunctionTool(
        name=TOOL_PREVIEW_CARD,
        description=(
            "对候选订阅源做完整的安全预审 + 渲染预览卡片，写入待确认列表，"
            "并向当前群发送预览卡片。**调用此工具后，必须立即调 myrss_terminate 结束本次对话**，"
            "不要再调其他 tool。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "feed_url": {
                    "type": "string",
                    "description": "完整的 RSS 源 URL，来自 myrss_lookup 的输出",
                }
            },
            "required": ["feed_url"],
        },
        handler=handler,
    )
    return tool


def _make_emit_result_tool(plugin) -> FunctionTool:
    """myrss_emit_result: 终态，LLM 决定发什么。"""

    async def handler(event, **kwargs) -> str:
        action = str(kwargs.get("action", "")).strip()
        free_text = str(kwargs.get("free_text", "")).strip()
        if plugin.nl_debug_log:
            logger.info("[NL][emit_result] action=%r free_text=%r", action, free_text)
        origin = event.unified_msg_origin

        if action == "not_found":
            text = (free_text or "未找到该博主，可再使用指令并加上具体 URL 再次尝试。")[:200]
            await plugin._send_message_guarded(origin, _plain_chain(text))
            return "OK: 已发送 not_found 文案。"

        if action == "rejected_preview":
            text = (free_text or "该订阅源最新动态未通过内容安全审核，已停止。")[:200]
            await plugin._send_message_guarded(origin, _plain_chain(text))
            return "OK: 已发送 rejected_preview 文案。"

        # subscribed / cancelled / irrelevant 不应走这个 tool
        return "警告：action 不适用于此 tool。请用 myrss_confirm_subscribe 或 myrss_ignore。"

    tool = FunctionTool(
        name=TOOL_EMIT_RESULT,
        description=(
            "发送一个固定结果给当前群。仅用于 not_found（找不到博主）和 rejected_preview（内容未过审）场景。"
            "subscribed / cancelled / irrelevant 不应走这个 tool。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["not_found", "rejected_preview"],
                    "description": "结果类型",
                },
                "free_text": {
                    "type": "string",
                    "description": "要发送的短句（≤ 200 字）。LLM 自由发挥（not_found 场景建议 ≤ 80 字）。",
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )
    return tool


def _make_confirm_subscribe_tool(plugin, pending_card: IntentCard) -> FunctionTool:
    """myrss_confirm_subscribe: 一次性 tool, 落订 / 取消 / 换一个。"""

    async def handler(event, **kwargs) -> str:
        decision = str(kwargs.get("decision", "")).strip()
        if plugin.nl_debug_log:
            logger.info("[NL][confirm] decision=%r card=%s", decision, pending_card.summary)
        origin = event.unified_msg_origin

        if decision == "approve":
            # 走 add_subscription_from_ui 复用全部安全不变量。
            # 关键：必须传 route（/twitter/user/OpenAI），而不是 pending_card.feed_url。
            # feed_url 在 preview_card 里已被解析成完整 RSSHub URL（https://rsshub.app/...），
            # 而 add_subscription_from_ui -> _resolve_feed_url 只认原始平台域名或 / 开头路由，
            # 传 RSSHub URL 会让 URLMapper.match 失败 -> "未收录此平台"。
            try:
                result = await plugin.add_subscription_from_ui(
                    origin=origin,
                    value=pending_card.route,
                    creator_openid=pending_card.creator_openid or str(event.get_sender_id() or ""),
                    creator_name=pending_card.creator_name or str(event.get_sender_name() or ""),
                    creator_source="natural_language",
                    require_active_probe=True,
                )
                if not result.get("confirmation_sent"):
                    await plugin._send_message_guarded(
                        origin,
                        _plain_chain(
                            f"✅ {result.get('message', '已订阅')}\n源: {result.get('title') or pending_card.title}"
                        ),
                    )
                plugin.nl_pending.pop(origin)
                return f"OK: 已落订 {pending_card.title}"
            except Exception as exc:
                # 落订失败（很可能群没开主动消息权限 / 群订阅数已达上限 / 黑名单）
                # 让用户看到原因
                await plugin._send_message_guarded(
                    origin,
                    _plain_chain(f"❌ 订阅失败: {exc}"),
                )
                return f"ERROR: {exc}"

        if decision == "reject":
            plugin.nl_pending.pop(origin)
            await plugin._send_message_guarded(
                origin,
                _plain_chain("好的，已取消。可再发送 /myrss + 新的描述。"),
            )
            return "OK: 已取消。"

        if decision == "not_this":
            plugin.nl_pending.pop(origin)
            await plugin._send_message_guarded(
                origin,
                _plain_chain("好的，已取消本次订阅请求。\n可再发送 /myrss + 新的描述，或直接贴 URL。"),
            )
            return "OK: 已丢弃本次订阅。"

        return "警告：decision 必须是 approve / reject / not_this。"

    tool = FunctionTool(
        name=TOOL_CONFIRM_SUBSCRIBE,
        description=(
            "把上一条预览卡片对应的订阅源真正落到当前群。"
            "必须先看到用户明确同意/拒绝/换一个才能调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "reject", "not_this"],
                    "description": "approve=用户同意, reject=用户拒绝, not_this=用户要换",
                }
            },
            "required": ["decision"],
        },
        handler=handler,
    )
    return tool


def _make_terminate_tool(plugin) -> FunctionTool:
    """myrss_terminate: 主动结束 agent run。"""
    async def handler(event, **kwargs) -> str:
        if plugin.nl_debug_log:
            logger.info("[NL][terminate] called by LLM")
        return "OK: 本次对话结束。"

    tool = FunctionTool(
        name=TOOL_TERMINATE,
        description="结束本次 LLM 介入。不再调其他 tool，不向用户发送任何内容。",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    return tool


def _make_ignore_tool(plugin) -> FunctionTool:
    """myrss_ignore: 用户的请求跟订阅无关，Bot 不回复。"""
    async def handler(event, **kwargs) -> str:
        if plugin.nl_debug_log:
            logger.info("[NL][ignore] user request is irrelevant to subscribe")
        return "OK: 已忽略。"

    tool = FunctionTool(
        name=TOOL_IGNORE,
        description=(
            "用户的请求跟订阅无关（闲聊/问题/调试/任何非订阅意图）。"
            "调用此 tool 不会向用户发送任何内容。"
        ),
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
    return tool


# ============================================================
# ToolSet 工厂
# ============================================================
def build_nl_tool_set(plugin) -> ToolSet:
    """第一次 agent run 的 ToolSet：6 个 myrss_* tool + 可选 TavilyWebSearchTool。"""
    ts = ToolSet()
    ts.add_tool(_make_lookup_tool(plugin))
    ts.add_tool(_make_preview_card_tool(plugin))
    ts.add_tool(_make_emit_result_tool(plugin))
    ts.add_tool(_make_terminate_tool(plugin))
    ts.add_tool(_make_ignore_tool(plugin))

    # 注入 AstrBot 内置 websearch（如果用户开了）
    if plugin.nl_enable_websearch and _HAS_WEBSEARCH:
        try:
            tool_mgr = plugin.ctx.get_llm_tool_manager()
            ts.add_tool(tool_mgr.get_builtin_tool(TavilyWebSearchTool))
        except Exception as exc:
            logger.warning("[NL] 注入 TavilyWebSearchTool 失败: %s", exc)

    return ts


def build_confirm_tool_set(plugin, pending_card: IntentCard) -> ToolSet:
    """第二次 agent run 的 ToolSet：3 个 tool（无 lookup / preview_card / emit_result）。"""
    ts = ToolSet()
    ts.add_tool(_make_confirm_subscribe_tool(plugin, pending_card))
    ts.add_tool(_make_terminate_tool(plugin))
    ts.add_tool(_make_ignore_tool(plugin))
    return ts


# ============================================================
# 辅助：消息链构造
# ============================================================
def _build_preview_message(card: IntentCard, plugin) -> "MessageChain":
    """预览卡片 + 提示用户引用回复的纯文本。"""
    from astrbot.api.event import MessageChain
    chain = MessageChain()
    if card.card_b64:
        chain.chain.append(MsgImage.fromBase64(card.card_b64))
    chain.chain.append(Plain(
        f"📡 {card.title or card.handle or card.route}\n"
        f"如需订阅该动态，请**引用本图片**回复「同意」或「拒绝」或「不是这个」。\n"
        f"（{plugin.nl_pending_ttl} 秒内有效，超时静默失效，可重发 /myrss + 重新发起）"
    ))
    return chain


def _plain_chain(text: str) -> "MessageChain":
    from astrbot.api.event import MessageChain
    chain = MessageChain()
    chain.chain.append(Plain(text))
    return chain


async def _send_err(plugin, origin: str, text: str) -> str:
    """把错误直接发给当前群并返回给 LLM 的字符串。
    不依赖 LLM 用 myrss_emit_result 转达——否则 LLM 直接 terminate 时
    用户收不到任何报错原因，表现为"静默失败"。
    """
    await plugin._send_message_guarded(origin, _plain_chain(f"❌ {text}"))
    return f"ERROR: {text}"


# ============================================================
# 辅助：route / url 解析
# ============================================================
def _platform_from_route(route: str) -> str:
    if route.startswith("/twitter"):
        return "X"
    if route.startswith("/youtube"):
        return "YouTube"
    if route.startswith("/bilibili"):
        return "B站"
    if route.startswith("/weibo"):
        return "微博"
    if route.startswith("/zhihu"):
        return "知乎"
    if route.startswith("/github"):
        return "GitHub"
    if route.startswith("/telegram"):
        return "Telegram"
    return ""


def _handle_from_url_or_route(feed_url: str, route: str) -> str:
    """尽力从 route 末尾抽出 handle。仅用于摘要展示。"""
    m = re.search(r"/([A-Za-z0-9_@.\-]+)/?$", route)
    if m:
        return m.group(1).lstrip("@")
    return ""


# ============================================================
# 关键词检测（简化版：用户引用了 bot 卡片 / 含同意/拒绝关键词）
# ============================================================
_NL_CONFIRM_KEYWORDS = (
    "同意", "确认", "订阅", "好的", "没错", "就这个", "对",
    "y", "yes", "ok",
    "拒绝", "不要", "算了", "取消",
    "不是", "错了", "换一个",
)


def looks_like_nl_confirm_reply(event: AstrMessageEvent) -> bool:
    """
    简化方案：消息文本中含"同意/拒绝/订阅/取消/不是这个"等关键词。
    不做精确"引用 bot 上一条消息"检测（v0.2 §9.2 提到的简化）。
    """
    text = (event.message_str or "").strip().lower()
    if not text:
        return False
    # 去掉空白
    text_norm = re.sub(r"\s+", "", text)
    for kw in _NL_CONFIRM_KEYWORDS:
        if kw.lower() in text_norm or kw.lower() in text:
            return True
    return False
