# astrbot_plugin_myrss

AstrBot RSS 订阅插件（LLM 增强版）。详见仓库内：
- [NATURAL_LANGUAGE_SUBSCRIBE.md](NATURAL_LANGUAGE_SUBSCRIBE.md) — v1.4.0 自然语言订阅设计
- [TWO_TOOL_AUDIT.md](TWO_TOOL_AUDIT.md) / [DEAD_CODE_AUDIT.md](DEAD_CODE_AUDIT.md) / [交接文档.md](交接文档.md) — 历史记录

## 关掉「群内有人说话才推送」的限制（v1.4.1 新增）

QQ 官方群的主动推送有一条前置条件：`_target_readiness()` 要求本次启动后**群内有人发过言**
（或适配器 `_session_scene` 里已有该群场景），否则定时任务拉取前就把该目标跳过并打日志
`RSS目标未就绪，拉取前跳过`。Bot 重启后，安静的群要等到下一条群消息才会恢复推送。

配置项 `push_requires_group_activity` 控制这道闸门，默认 `true`（行为不变）：

| 取值 | 行为 |
| --- | --- |
| 默认 `true` | 保持原样：没观察到消息的群视为未就绪，跳过推送 |
| 设为 `false` | 不再检查群活跃度：QQ 官方平台在跑且该群已订阅，就直接推送，无需等群友发言、重启后也照常推 |

关掉开关**不会**跳过主动消息权限，两者是独立的：

- 订阅时 `require_active_probe` 仍会真发一条「MyRSS 主动推送测试通过」；
- 推送时若 QQ 明确返回「主动消息失败/无权限」，仍会把这**一个**源从该群退订（网络/HTTP
  类临时失败只记 `FAILED`，保留订阅）；
- 面板里的 `delivery_ready` 会随之变为就绪，原因码为 `ready_activity_check_disabled`。

唯一的例外是「插件从未观察过该群」这道安全检查（`add_subscription_from_ui` 里
`目标群尚未被 MyRSS 观察到`）：它防的是从面板构造任意群号，和推送时机无关，因此不受这个开关影响。
在群内发 `/myrss +` 订阅时该群自然会被记为已观察，不受影响。

启动日志会打印当前模式：`MyRSS: 群活跃检查: 开启/关闭`。
