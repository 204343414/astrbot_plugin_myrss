# 自然语言订阅设计文档（v0.2，已通过用户 review，待实现）

> **状态**: v0.2。用户已在两轮 ask_user 中对齐了所有关键点。
> **下一步**: 拆 8 个小步分步实现，每步单独 review。
> **不在范围**: 私聊、多群全局推送、推荐系统。

---

## 0. 跟 v0.1 的差异

| 项 | v0.1 | v0.2 | 原因 |
|----|------|------|------|
| LLM 介入方式 | `ctx.llm_generate()` 裸调 | `ctx.tool_loop_agent()` 走 agent 循环 | 用户需要 LLM 自己调 web_search 等内置 tool |
| LLM 工具集 | 0 个 (LLM 仅输出 JSON) | 5 个 (`myrss_lookup`, `myrss_preview_card`, `myrss_emit_result`, `myrss_terminate`, `myrss_ignore`) | 用户说"LLM 继承联网和预览动态并输出卡片和结束当前聊天" |
| 临时上下文 | "新开一个真实 conversation" | 一次性的 `tool_loop_agent` 调用，无 conversation 持久化 | 用户说"临时一个聊天记录" = 一次性，不持久化 |
| 关键词匹配 | 写死正则 (`同意/拒绝/不是这个`) | 引用 bot 上一条卡片才触发 LLM，LLM 判 approve/reject/not_this/irrelevant | 用户原话"关键词直接让 LLM 来理解得了" |
| 找不到博主文案 | 固定模板 | LLM 自由短句 (< 80 字) | 用户原话"让 LLM 输出没找到和为什么没找到" |
| 主动消息权限 | 多种方案选 `plain_text_for_unready` | 同上（明文拒绝，不探针） | 用户 Q3 勾选（与"参考 qqhub 探针"的口头说法冲突，按 Q3 终局） |
| 复用 | `TWO_TOOL_AUDIT.md` 的 `myrss_preview` / `myrss_manage` | **不沿用** | master 实际没实现，且新方案是 tool_loop_agent + 自定义 tool，不冲突 |
| 配置开关 | (无) | `enable_natural_language_subscribe` bool 默认 false | 用户原话"json 加 bool" |
| 私聊 | 默认关闭 | 默认关闭 | 与原命令行为一致 |
| 联网 tool 来源 | "可选增强" | 走 `plugin_context.get_llm_tool_manager().get_builtin_tool(TavilyWebSearchTool)` | 框架有 `tool_mgr.get_builtin_tool(XXX)` API |

---

## 1. 用户的核心需求（一次说清，避免偏移）

| # | 需求 |
|---|------|
| 1 | `/myrss + 帮我订阅 OpenAI 的推特` 这种自然语言能直接订阅 |
| 2 | 插件"临时起一条聊天 LLM"（一次性 `tool_loop_agent` 调用）让 LLM 识别意图 |
| 3 | LLM 能调联网 tool（`web_search_tavily` 等 builtin），预览最新动态，发卡片，结束对话，"已读不回" |
| 4 | 成功 → 推送预览卡片 → 用户**引用该卡片**回 "同意/拒绝/不是这个" 才落订（LLM 判定） |
| 5 | 找不到博主 → LLM 自由短句解释（< 80 字），Bot 原样发 |
| 6 | LLM 故障/不在线 → 固定文案"服务器暂忙" |
| 7 | 配置项 `enable_natural_language_subscribe` bool, 关闭时 `/myrss + xxx` 走纯指令识别（仅 URL / 路由） |
| 8 | 防滥用：用户借 LLM 玩闲聊 → LLM 调 `myrss_ignore` tool 已读不回 |
| 9 | 群未开主动消息 → 明文拒绝，不激活订阅功能 |
| 10 | 复用全量内容审核（违禁词 + 多模态 + LLM 文本三态） |

**不实现**：

- 私聊自然语言订阅
- LLM 自己写卡片文案（卡片是 Bot 模板渲染）
- 把 LLM `completion_text` 直接转发给用户
- 关键词的正则分类（只做"引用 bot 卡片"的触发器）

---

## 2. 整体流程（ASCII 时序图）

```
用户                                Bot (myrss)                            LLM (tool_loop_agent)
 │                                    │                                       │
 │ /myrss + 帮我订阅 OpenAI 的推特    │                                       │
 │───────────────────────────────────>│                                       │
 │                                    │ 1. _target_readiness → True ?         │
 │                                    │ 2. 限流 / 黑名单 / 订阅数上限         │
 │                                    │ 3. ctx.tool_loop_agent(                │
 │                                    │      event=event,                      │
 │                                    │      chat_provider_id=...             │
 │                                    │      prompt="用户原始自然语言订阅请求"│
 │                                    │      system_prompt=NAT_LANG_PROMPT    │
 │                                    │      tools=ToolSet([                  │
 │                                    │        TavilyWebSearchTool,           │
 │                                    │        myrss_lookup,                  │
 │                                    │        myrss_preview_card,            │
 │                                    │        myrss_emit_result,              │
 │                                    │        myrss_terminate,                │
 │                                    │        myrss_ignore,                  │
 │                                    │      ]),                              │
 │                                    │      max_steps=8)                     │
 │                                    │──────────────────────────────────────>│
 │                                    │                                       │ (LLM 自己决定调谁)
 │                                    │<──────────────────────────────────────│
 │                                    │ { "action": "emit_preview_card",       │
 │                                    │   "title": "OpenAI",                  │
 │                                    │   "platform": "X",                    │
 │                                    │   "handle": "OpenAI",                 │
 │                                    │   "route": "/twitter/user/OpenAI",    │
 │                                    │   "feed_url": "https://...",         │
 │                                    │   "card_b64": "..." }                 │
 │                                    │ 4. Bot 拿到 emit_preview_card action │
 │                                    │ 5. 写入 _nl_pending[origin]          │
 │                                    │ 6. _send_message_guarded 推卡片      │
 │                                    │                                       │
 │ [预览卡片 + "如需订阅请引用本卡片回复 同意 / 拒绝 / 不是这个"]            │
 │<───────────────────────────────────│                                       │
 │                                    │                                       │
 │ 用户引用刚才的卡片, 说"同意"        │                                       │
 │───────────────────────────────────>│                                       │
 │                                    │ 7. _on_message_for_nl_confirm 钩子:   │
 │                                    │    检测"用户引用了 bot 上一条"        │
 │                                    │    进入 LLM 二次判定                 │
 │                                    │ 8. ctx.tool_loop_agent(               │
 │                                    │      event=event,                     │
 │                                    │      prompt=f"上一条推送的卡片是 \n   │
 │                                    │               {last_card_summary}\n  │
 │                                    │               用户本次消息: \n       │
 │                                    │               {user_text}",          │
 │                                    │      tools=ToolSet([                  │
 │                                    │        myrss_emit_result (落订用),    │
 │                                    │        myrss_terminate,               │
 │                                    │        myrss_ignore,                  │
 │                                    │      ]),                             │
 │                                    │      max_steps=4)                    │
 │                                    │──────────────────────────────────────>│
 │                                    │<──────────────────────────────────────│
 │                                    │ { "action": "approve",                │
 │                                    │   "feed_url": "...",                  │
 │                                    │   "creator_openid": "..." }           │
 │                                    │ 9. add_subscription_from_ui(...)     │
 │                                    │ 10. 清除 _nl_pending[origin]         │
 │                                    │ 11. _send_message_guarded "✅ 已订阅" │
 │                                    │                                       │
 │ ✅ MyRSS 主动推送测试通过, 已订阅  │                                       │
 │<───────────────────────────────────│                                       │
```

### 错误分支

- **LLM 故障** (provider 找不到 / 超时 / tool_loop_agent 抛错) → 走 `plain_text_for_unready` 风格的固定文案 "服务器暂忙, 请稍后再试"
- **找不到博主** → LLM 调 `myrss_emit_result` 传 `action="not_found"` + `free_text="<80字内解释>"` → Bot 原样发
- **闲聊** → LLM 调 `myrss_ignore` → Bot 不说话
- **超时** → 推送卡片后 10 分钟内用户没引用 → _nl_pending[origin] 过期; 下次 `/myrss +` 重新走
- **群没开主动消息** → 直接明文拒绝, 不调 LLM

---

## 3. 模块拆分

新建 1 个文件 `natural_subscribe.py`, 2 个工具类, 5 个 tool, 1 个 prompt 模块。

### 3.1 文件: `natural_subscribe.py`

```python
# 伪代码展示, 不写最终实现

@dataclass
class IntentCard:
    """LLM 调 myrss_preview_card 后, Bot 解析的"待确认订阅"结构化数据。"""
    feed_url: str
    route: str
    platform: str
    handle: str
    title: str
    description: str
    image_b64: str
    pub_date: str
    card_b64: str
    error: str = ""


class NLIntentStore:
    """_nl_pending[origin] = IntentCard 或 None, 含 expires_at."""
    def put(self, origin, card: IntentCard, creator_openid, ttl=600): ...
    def peek(self, origin) -> Optional[IntentCard]: ...
    def pop(self, origin) -> Optional[IntentCard]: ...
    def gc(self) -> int: ...


def build_nl_tool_set(plugin) -> ToolSet:
    """组装一次 tool_loop_agent 用的 ToolSet。"""
    ts = ToolSet()
    if plugin.enable_websearch:
        ts.add_tool(plugin.ctx.get_llm_tool_manager().get_builtin_tool(TavilyWebSearchTool))
    ts.add_tool(_myrss_lookup(plugin))           # 查找 RSSHub 路由
    ts.add_tool(_myrss_preview_card(plugin))     # 拉 RSS + 渲染预览卡片 + 存 pending
    ts.add_tool(_myrss_emit_result(plugin))      # 终态: 落订/未找到/超时错误/确认落订
    ts.add_tool(_myrss_terminate(plugin))       # LLM 主动结束本次 agent run
    ts.add_tool(_myrss_ignore(plugin))           # LLM 主动已读不回
    return ts


# 5 个 tool, 都是 FunctionTool 实例, 每个 handler 接收 event, **kwargs
# 详细 schema / 参数在 §4
```

### 3.2 文件: `main.py` (改动)

```python
class MyRssPlugin(Star):
    def __init__(...):
        # ...
        self.nl_enabled = config.get("enable_natural_language_subscribe", False)
        self.nl_provider_id = config.get("nl_provider_id", "")
        self.enable_websearch = config.get("nl_enable_websearch", True)
        self.nl_pending = NLIntentStore(ttl_seconds=600)
        # 启动时挂一个 gc 协程
        asyncio.create_task(self._nl_pending_gc_loop())

    @myrss.command("+", alias={"add"})
    async def cmd_add_current_group(self, event, url=""):
        raw = (event.message_str or "")
        value = (extract_after_plus(raw) or url or "").strip()
        if not self.nl_enabled:
            # 旧逻辑, 走 URL 解析
            async for r in self._cmd_add_explicit_url(event, value):
                yield r
            return
        origin = event.unified_msg_origin
        if "GroupMessage" not in origin:
            yield event.plain_result("此命令只能在群内使用。")
            return
        # 路径 1: 显式 URL / 路由, 不调 LLM
        if value.startswith(("http://", "https://", "/")):
            async for r in self._cmd_add_explicit_url(event, value):
                yield r
            return
        # 路径 2: 自然语言
        # 1) 主动消息权限检查
        ready, reason = self._target_readiness(origin)
        if not ready:
            yield event.plain_result(
                "本群 Bot 尚未具备主动推送条件, 请先在 QQ 群设置中开启 "
                "『机器人主动在群聊内发言』并由群内任意成员发送一条消息激活。"
            )
            return
        # 2) 限流 / 黑名单 / 订阅数上限 (复用现有)
        if not self._check_nl_rate_limit(origin):
            yield event.plain_result("本群自然语言订阅请求过于频繁, 请 30 秒后再试。")
            return
        # 3) 调 tool_loop_agent
        self._ready_group_sessions.add(origin)
        tools = build_nl_tool_set(self)
        try:
            resp = await self.ctx.tool_loop_agent(
                event=event,
                chat_provider_id=self.nl_provider_id or await self._get_provider_id(),
                prompt=f"用户在群 ({origin}) 发送的自然语言订阅请求: \n>>> {value} <<<",
                system_prompt=NAT_LANG_SYSTEM_PROMPT,
                tools=tools,
                max_steps=8,
                tool_call_timeout=60,
            )
        except Exception as exc:
            self.logger.warning("[MyRSS][NL] tool_loop_agent failed: %s", exc)
            yield event.plain_result("服务器暂忙, 请稍后再试。")
            return
        # LLM 自己在 agent 循环里调 myrss_preview_card / myrss_emit_result / myrss_ignore
        # 全部副作用在 tool handler 里完成, 这里的 resp 几乎不会被 Bot 直接使用
        # 但仍记录到审计日志
        self.logger.info("[MyRSS][NL] tool_loop_agent finished: text=%r", (resp.completion_text or "")[:200])
        # 不再 yield 任何东西, 卡片已在 tool 内发送

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def _on_message_for_nl_confirm(self, event):
        """监听群消息, 触发 LLM 二次判定 (用户引用了 bot 卡片时)."""
        origin = getattr(event, "unified_msg_origin", "")
        if "GroupMessage" not in origin or not origin:
            return
        # 不处理自己发的消息
        if getattr(event, "is_self", False) or (event.get_sender_id() == self.bot_qq):
            return
        pending = self.nl_pending.peek(origin)
        if not pending:
            return
        # 触发条件: 消息中包含 Reply 组件 (引用) 且 reply 的是 bot 上一条
        # 或者: message_str 里出现 "同意" / "拒绝" / "不是这个" / "确认" 等
        if not self._looks_like_nl_reply(event):
            return
        # 进入 LLM 二次判定
        user_text = (event.message_str or "").strip()
        last_summary = f"title={pending.title} platform={pending.platform} handle={pending.handle} route={pending.route}"
        tools = ToolSet()
        tools.add_tool(_myrss_confirm_subscribe(self, pending))  # 一次性 tool, 落订
        tools.add_tool(_myrss_terminate(self))
        tools.add_tool(_myrss_ignore(self))
        try:
            resp = await self.ctx.tool_loop_agent(
                event=event,
                chat_provider_id=self.nl_provider_id or await self._get_provider_id(),
                prompt=(
                    f"用户上一条自然语言订阅请求触发的预览卡片: \n{last_summary}\n"
                    f"用户本次回复: \n>>> {user_text} <<<\n\n"
                    f"请根据用户回复, 调 myrss_confirm_subscribe 落订, 或 myrss_terminate 结束, 或 myrss_ignore 已读不回。"
                ),
                system_prompt=NAT_LANG_CONFIRM_PROMPT,
                tools=tools,
                max_steps=4,
                tool_call_timeout=30,
            )
        except Exception as exc:
            self.logger.warning("[MyRSS][NL] confirm LLM failed: %s", exc)
            yield event.plain_result("服务器暂忙, 请稍后再试。")
        else:
            self.logger.info("[MyRSS][NL] confirm LLM finished: text=%r", (resp.completion_text or "")[:200])
        # 不 yield, LLM 工具自己负责发消息 / 落订
        return
```

---

## 4. 5 个自定义 tool 的 schema

### 4.1 `myrss_lookup` (第一次 agent run 用)

**目的**: 查 RSSHub 路由表, 给出"X 上叫 OpenAI 的账号"对应的 `/twitter/user/OpenAI` 路由。

**为什么不直接让 LLM 编 route**: LLM 幻觉可能产出 `/twiter/user/OpneAI` 这种非法路由。`URLMapper` 已经有权威的 URL→route 规则。但 `URLMapper` 只认**完整 URL**,不认"X 上叫 sama"。所以需要 LLM 帮我们拼出 URL, 然后 `URLMapper.match` 校验。

```python
async def _myrss_lookup(plugin, event, **kwargs):
    """根据用户的文字描述, 返回候选订阅源 URL 列表 (供 LLM 挑选)。"""
    query: str = kwargs.get("query", "")
    # 1) 调一次 LLM 解析 → (platform, handle, candidate_url)
    # 2) URLMapper.match(candidate_url) 校验
    # 3) 失败就回 "需要数字 uid" / "暂不支持该平台"
    # 4) 成功回 [{platform, handle, candidate_url, route}, ...]
    ...
```

Tool schema:

```json
{
  "name": "myrss_lookup",
  "description": "在 RSSHub 支持的平台里查找某博主/频道。传入用户原文描述, 返回候选订阅源 URL 列表。**调用前不要自己拼 URL**, 这个工具会校验 URL 合法性。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "用户原话, 例如 'OpenAI 的推特' / 'B站科技区' / '知乎用户张三'"
      }
    },
    "required": ["query"]
  }
}
```

Handler 返回值 (LLM 看到的是 string):
```
找到 1 个候选订阅源:
  1. platform=X, handle=OpenAI, url=https://x.com/OpenAI, route=/twitter/user/OpenAI

未找到匹配的博主。可能的平台: twitter/x, youtube, 微博, 知乎, github, telegram, 小红书。
```

### 4.2 `myrss_preview_card` (第一次 agent run 用)

**目的**: LLM 选定某个候选后, Bot 拉 RSS + 渲染卡片 + 写入 _nl_pending + 发卡片。

```json
{
  "name": "myrss_preview_card",
  "description": "对候选订阅源做完整的安全预审 + 渲染预览卡片, 写入待确认列表, 并向当前群发送预览卡片。**调用此工具后, 必须立即调 myrss_terminate 结束本次对话**, 不要继续闲聊。",
  "parameters": {
    "type": "object",
    "properties": {
      "feed_url": {
        "type": "string",
        "description": "完整的 RSS 源 URL, 来自 myrss_lookup 的输出"
      }
    },
    "required": ["feed_url"]
  }
}
```

Handler 流程:
1. `_resolve_feed_url(feed_url)` 解析成 (full_url, route, error)
2. `_fetch(full_url)` 拉 RSS
3. `parse_channel_info` 读 title/description/avatar
4. `_poll(num=1)` 拉最新动态
5. `_check_content_safe(latest_item)` 走全量内容审核
   - **REJECT / MALICIOUS** → 卡片发送 `action="rejected_preview"`, `free_text="内容未通过安全审核: ..."` (走 myrss_emit_result 等价路径)
6. `_make_card_b64(latest_item)` 渲染卡片
7. `_nl_pending.put(origin, card, creator_openid, ttl=600)`
8. `_send_message_guarded(origin, MessageChain([Image, Plain("如需订阅请引用本卡片回复'同意'或'拒绝'或'不是这个'")]))`
9. 返回字符串: `"预览卡片已发送, 等待用户引用本卡片回复。"`

### 4.3 `myrss_emit_result` (两次 agent run 都用)

**目的**: 终态。LLM 表达"我决定就这样了", Bot 据此发固定回复。

```json
{
  "name": "myrss_emit_result",
  "description": "结束本次 LLM 介入并向用户发送一个固定结果。不再调其他 tool。",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["not_found", "rejected_preview", "subscribed", "cancelled", "irrelevant"],
        "description": "not_found=没找到博主, rejected_preview=内容未通过安全审核, subscribed=用户已确认订阅, cancelled=用户拒绝, irrelevant=用户说的是其他事"
      },
      "free_text": {
        "type": "string",
        "description": "可选: 自由短句 (≤ 80 字), Bot 原样发给用户。仅在 not_found / rejected_preview 场景使用。"
      }
    },
    "required": ["action"]
  }
}
```

Handler:
- `action=not_found` / `rejected_preview` → 推 `free_text` (有) 或固定文案 (无)
- `action=subscribed` / `cancelled` / `irrelevant` → **不应**走这个 tool, 应该走更专用的 tool (落订 / 取消 / 已读不回)。**LLM 选错了就走 `myrss_ignore` 兜底**

### 4.4 `myrss_confirm_subscribe` (第二次 agent run 用, 一次性)

**目的**: 用户回复"同意"后, LLM 调它, 走完整订阅流程 + 主动消息探针 + 内容审核。

```json
{
  "name": "myrss_confirm_subscribe",
  "description": "把上一条预览卡片对应的订阅源真正落到当前群。**必须先看到用户明确同意才能调用**。",
  "parameters": {
    "type": "object",
    "properties": {
      "decision": {
        "type": "string",
        "enum": ["approve", "reject", "not_this"],
        "description": "approve=用户同意订阅, reject=用户明确拒绝, not_this=用户说不是这个/换一个"
      }
    },
    "required": ["decision"]
  }
}
```

Handler:
- 闭包 capture 上一个 `IntentCard` (由 `_on_message_for_nl_confirm` 创建 tool 时传进去)
- `decision=approve` → `add_subscription_from_ui(origin, feed_url, creator_openid, creator_name="natural_language", creator_source="natural_language", require_active_probe=True)`
- `decision=reject` / `not_this` → `_nl_pending.pop(origin)`, Bot 推 "好的, 已取消。可再发送 /myrss + 新的描述。"
- **LLM 调错 action**: 静默忽略, 当 irrelevant

### 4.5 `myrss_terminate` (两次 agent run 都用)

**目的**: LLM 主动结束本次 agent run。

```json
{
  "name": "myrss_terminate",
  "description": "结束本次 LLM 介入。不再调其他 tool, 不向用户发送任何内容。",
  "parameters": {"type": "object", "properties": {}}
}
```

Handler: 返回 `"OK"`, LLM agent 循环会因为"LLM 不再调 tool 且未拒绝"而自然结束。

### 4.6 `myrss_ignore` (两次 agent run 都用)

**目的**: LLM 决定"用户说的是其他事, 我不参与"。

```json
{
  "name": "myrss_ignore",
  "description": "用户的请求跟订阅无关 (闲聊/问题/调试/任何非订阅意图)。Bot 不回复。",
  "parameters": {"type": "object", "properties": {}}
}
```

Handler: 写一条审计日志, 返回 `"OK"`, 啥也不发。

---

## 5. 两个 system prompt

### 5.1 `NAT_LANG_SYSTEM_PROMPT` (第一次 agent run)

```
你是一个临时性的"订阅意图识别 + 预览确认"助手, 仅在用户发送 `/myrss + 自然语言` 时被启动, 服务对象是 QQ 群。

你的全部工作:
1. 理解用户想订阅谁/哪个频道
2. 调 myrss_lookup 查 RSSHub 路由
3. 调 myrss_preview_card 渲染预览卡片 (这一步会真正发到群里)
4. 调 myrss_terminate 结束, 等待用户引用卡片回复

绝对禁止:
- 跟用户闲聊, 解释自己是什么, 回应问候
- 输出 JSON / Markdown / 任何"我理解你的需求是..."之类的话
- 在 myrss_preview_card 之后再调其他 tool (除了 myrss_terminate)
- 调本对话以外的任何 AstrBot 内置 tool (myrss_* 之外都视为越界, 调了就 myrss_ignore)

如果 myrss_lookup 返回"未找到", 必须调 myrss_emit_result(action="not_found", free_text="< 80 字解释, 例如 'OpenAI 在微博没有官方号, 请给 X 上的 URL'"), 然后 myrss_terminate。

如果用户说的不是订阅请求 (闲聊/提问/调试), 调 myrss_ignore, 然后 myrss_terminate。
```

### 5.2 `NAT_LANG_CONFIRM_PROMPT` (第二次 agent run)

```
上一条自然语言订阅请求触发了预览卡片 (标题/平台/handle/route 见 prompt), 用户刚刚在群里引用了那条卡片, 原文如下。

请只做三件事之一:
1. 用户的回复明确表示"同意 / 订阅 / 对 / 没错 / 就这个" → 调 myrss_confirm_subscribe(decision="approve")
2. 用户的回复明确表示"不要 / 拒绝 / 取消 / 算了" → 调 myrss_confirm_subscribe(decision="reject")
3. 用户的回复说"不是这个 / 错了 / 换一个 / 那个是 X 我要 Y" → 调 myrss_confirm_subscribe(decision="not_this")
4. 任何无关/含糊/聊天/调试 → 调 myrss_ignore

调完上述任一 tool 后, 调 myrss_terminate 结束。
不要发任何文字给用户 (myrss_confirm_subscribe 和 myrss_ignore 内部会发/不发)。
不要解释自己做了什么。
```

---

## 6. 复用与不复用

### 6.1 复用

| 现有函数 | 怎么用 |
|---------|-------|
| `_target_readiness(origin)` | 路径 2 入口检查 |
| `_check_content_safe(item)` | myrss_preview_card handler 内部 |
| `_fetch`, `_poll`, `parse_channel_info` | myrss_preview_card handler 内部 |
| `_make_card_b64(item)` | myrss_preview_card handler 内部 (但**不**走 `_generate_comment` 锐评, 节省 LLM 调用) |
| `_send_message_guarded` | 卡片发送 |
| `URLMapper.match`, `URLMapper.suggest`, `_resolve_feed_url` | myrss_lookup handler 内部 |
| `add_subscription_from_ui` | myrss_confirm_subscribe handler 内部, 复用"最新动态预审 + 主动消息探针 + 黑名单 + 订阅数上限"全套不变量 |
| `_subscription_count`, `_creator_ban`, `_creator_key` | 限流 / 黑名单 (路径 2 入口) |
| `command_group myrss`, `cmd_add_current_group` | 入口 |
| `cmd_remove_current_group`, `cmd_list`, `cmd_eye` | 完全不动 |

### 6.2 不复用

- `_generate_comment` (锐评): 预览阶段不开, 节省 LLM 调用
- `_last_preview` 单槽状态 (如果存在): 改成 `NLIntentStore` 多 origin map
- `TWO_TOOL_AUDIT.md` 提到的 `myrss_preview` / `myrss_manage`: **不沿用** (理由见 §0)

---

## 7. 安全不变量 (必查清单)

- [x] **不绕过内容审核**: 预览卡片发送前必须过 `_check_content_safe`; 落订前 `add_subscription_from_ui` 也会再过一遍 (复用现有)
- [x] **LLM 故障 ≠ 用户违规**: tool_loop_agent 抛错 / 超时 / provider 找不到 → "服务器暂忙" 文案, 不记 strike
- [x] **不持久化 LLM 输入**: 纯内存 `NLIntentStore`, 插件重载即清空
- [x] **不转发 LLM 原始输出**: Bot 回复永远是固定模板/卡片/固定文案, LLM 输出只走 `mycss_emit_result.free_text` (LLM 自己说要发, Bot 仅作 channel)
- [x] **不新增全局 LLM Tool**: 5 个 myrss_* tool 都是 FunctionTool 实例, 通过 ToolSet 注入, 不进 `add_llm_tools` 全局
- [x] **tool 集合有界**: 第一次 agent run 最多 8 step, 第二次最多 4 step, `tool_call_timeout=60s`
- [x] **限流**: 每群 30 秒内最多 1 次自然语言订阅 (`_nl_rate_limit` 单独计数, 不影响 `_eye_cooldown`)
- [x] **群订阅数上限** (5/群): 与现有 `max_subscriptions_per_group` 一致
- [x] **黑名单机制**: 复用 `_creator_ban`, 落订时强制走 `add_subscription_from_ui` 的黑名单检查
- [x] **接入点不绕过 `_data_lock`**: 写 `dh.data` 的地方都按现有锁策略
- [x] **新增 provider 概念**: 复用 `nl_provider_id` (用户配置, 默认空 → 用当前会话 provider), 跟现有 `filter_provider_id` / `comment_provider_id` 互不干扰
- [x] **审计日志**: 每次 tool_loop_agent 调用 + 每个 tool handler 都记 `level=info, intent={...}, error=...`
- [x] **闲聊/调试拦截**: myrss_ignore tool + system prompt 双重保护
- [x] **超时自动拒绝**: `_nl_pending.gc()` 5 分钟扫描一次, 过期 entry 删除; 不发"已超时"提示, 静默 (用户重发即可)

---

## 8. 实现分步 (确认设计后逐项落地)

| Step | 范围 | 验证 |
|------|------|------|
| 1 | 新建 `natural_subscribe.py`: `IntentCard` / `NLIntentStore` / 5 个 tool handler | `python -m py_compile` |
| 2 | `_conf_schema.json` 加 3 项: `enable_natural_language_subscribe` (bool, false), `nl_provider_id` (select_provider), `nl_enable_websearch` (bool, true) | 配置项可见 |
| 3 | `MyRssPlugin.__init__` 初始化 `nl_enabled` / `nl_provider_id` / `nl_pending` / gc 协程 | 不影响现有逻辑 |
| 4 | `cmd_add_current_group` 改写: URL 走旧逻辑, 自然语言走 `tool_loop_agent` | 旧命令兼容, 新命令能调 LLM |
| 5 | `_on_message_for_nl_confirm` 钩子: 引用卡片触发 LLM 二次判定 | 引用命中能落订/取消, 引用非卡片不触发 |
| 6 | 5 个 tool 接入: `build_nl_tool_set` + 各 handler 内部审计 | 单元测试或 dry-run 脚本 |
| 7 | 限流 / 黑名单 / 群订阅数上限接入 | 边界 case |
| 8 | 文档同步: `README.md` 加一行 + `metadata.yaml` version bump | 文档 |
| 9 | 校验脚本: 解析 6 条典型输入 (OpenAI/微博需 uid/微信公众号/闲聊/无效 JSON/超时) | dry-run 脚本 |

**不在范围内**:

- 私聊自然语言订阅
- LLM 自己写卡片文案
- 让 LLM 解释自己做了什么
- "已超时" 提示 (静默即可, 用户重发)

---

## 9. 风险与已知坑

1. **`_target_readiness` 漏检**: 插件刚启动时 `_ready_group_sessions` 为空, 第一次发 `/myrss +` 该群从未发言 → 返回 False → 拒绝。这是**预期行为**, 提示用户"先发一条消息激活"。**不主动探针** (用户 Q3 选 plain_text_refuse)。

2. **"引用 bot 卡片" 的检测**: AstrBot 的 `event.message_obj.message` 里 `Reply` 组件的 `id` 字段是引用消息的 ID, 我们没存消息 ID → 只能用 `Reply.chain` 内容匹配上次推送的卡片 hash。**简化方案**: 不做精确"引用 bot 上一条"检测, 而是:
   - 检测群里**任何一条消息**含"同意 / 拒绝 / 不是这个 / 订阅 / 取消" 关键词
   - 且当前群有未过期 `_nl_pending`
   - 就**起 LLM** 二次判定
   - **LLM 自己**决定"用户原意是不是回应刚才的卡片", 错就 myrss_ignore 兜底
   - 这个简化方案能处理 80% 场景, 复杂场景 ("用户回 同意 的同时讨论别的事") 由 LLM 判断

3. **多人群并发**: 两个用户几乎同时发 `/myrss + ...`, `_nl_pending[origin]` 群级 key 会互相覆盖。**简单方案**: 写入时检查已有 pending → 拒绝第二条, 要求"先确认上一个"。

4. **LLM 工具调用的副作用**: `myrss_preview_card` 实际**会发卡片到群** (有副作用)。如果 LLM 调它之后没调 `myrss_terminate` 而是继续, 整个 agent 循环结束时会**自动**收尾, 但可能产生奇怪的中间回复。system prompt 强约束"调完 preview_card 必须立即 terminate"。

5. **provider 不存在**: 配置改了但 provider 被删。`ctx.tool_loop_agent` 会抛错, 被 catch 后走 "服务器暂忙"。

6. **插件热重载时 `_nl_pending` 内存清空**: 用户得重发一次。**接受** (QQ空间也是这么做的)。

7. **多模态审核 LLM 调用**: 第一次 agent run 中, myrss_preview_card handler 调 `_check_content_safe` → 可能再调一次多模态 LLM (因为最新动态有图)。**单次自然语言订阅成本 = 1~3 次 LLM 调用** (tool_loop_agent 1~2 次 + 内容审核 1 次)。**可接受**, 日志清晰打点。

8. **AstrBot v4 装饰器钩子顺序**: `@filter.event_message_type(ALL)` 同时被现有 `_observe_group_session` / `_on_group_del_robot` / 新 `_on_message_for_nl_confirm` 监听, **三个都会触发**。新钩子早期判断"是否有 pending 卡片" + 关键词, 没就快速 return, 不影响现有逻辑。

9. **"自己回自己" 的消息**: Bot 发的"预览卡片"和"已订阅"消息**会**进入 `_on_message_for_nl_confirm` 钩子。需要过滤 sender_id == bot_qq。已在 §3.2 钩子里用 `is_self` / `get_sender_id() == bot_qq` 过滤。

10. **websearch 失败**: Tavily key 配错 / API 限流 / 网络问题 → tool 返回 `"Error: ..."`, LLM 看到错误, 应调 `myrss_ignore` 或 `myrss_emit_result(not_found)` 兜底。system prompt 提示了。

11. **与 qqhub 插件的"参考"关系**: 用户原话"参考 qqhub 思路", 我在 `astrbot_plugin_qqofficial_hub` 仓库**没找到**对应的"主动消息探针"实现。**Q3 选了 `plain_text_refuse`, 与用户口头说法有冲突, 按 Q3 终局**。如果用户后续反悔想加探针, 需要重做 §3.2 钩子第一步, 不影响其他流程。

---

## 10. 与既有文档的关系

- `TWO_TOOL_AUDIT.md`: 描述的方案没实现, **不沿用**那两个 Tool 名 (`myrss_preview` / `myrss_manage`)。新方案用 5 个新 tool (`myrss_lookup` / `myrss_preview_card` / `myrss_emit_result` / `myrss_confirm_subscribe` / `myrss_terminate` / `myrss_ignore`)。
- `DEAD_CODE_AUDIT.md`: 不引入新的全局推送/推荐系统代码, **遵守**。
- `SUBSCRIPTION_UI_AND_MIGRATION.md`: **不修改** WebUI。
- `PERSISTENCE_MIGRATION.md`: **不修改** 持久化路径。
- `交接文档.md`: **不修改** `_save` 写入不变量、URL 模式拉取逻辑。

---

## 11. 待用户在确认前最后核对 (已 review 过)

| 已确认 | 来源 |
|------|------|
| 走 `ctx.tool_loop_agent`, 不走 `@filter.llm_tool` | §2 流程图 |
| 5 个 myrss_* tool + 可选 TavilyWebSearchTool | §4 |
| 关键词靠 LLM 判定, 不写死正则 | §1.4 |
| 找不到博主: LLM 自由短句 < 80 字 | Q2 选 `llm_free_text_short` |
| 主动消息权限: plain_text 拒绝, 不探针 | Q3 选 `plain_text_refuse` |
| 私聊默认不开 | 原命令行为一致 |
| 配置项 `enable_natural_language_subscribe` bool 默认 false | 用户原话"json 加 bool" |
| 复用现有 `add_subscription_from_ui` 做落订 | §6.1 |
| 复用全量内容审核 | §7 |

| 还没拍板 (默认按文档) | 备注 |
|------|------|
| GC 周期 5 分钟 | 默认值, 可调 |
| tool_loop_agent max_steps (8 / 4) | 默认值, 可调 |
| `_nl_pending` TTL 600 秒 | 默认值, 可调 |
| 限流 30 秒/群 | 默认值, 可调 |
| websearch 默认开 | 用户原话"LLM 继承联网", 默认开; 加 `nl_enable_websearch` 开关 |
| `is_hide_url` 等现有配置在自然语言路径下生效 | 跟 URL 路径一致 |

---

**设计稿 v0.2 完成。等用户最终 OK 后开始 Step 1。**
