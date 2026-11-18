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

## 凭据维护面板（1.5.0）

插件页面顶部多了一个折叠块「🔐 凭据维护」。它的作用是：**RSSHub 各平台登录态（cookie / token）不用再手抄，点一下就从本机 Firefox 同步进去。**

### 为什么需要宿主侧组件

RSSHub 的凭据是**容器的环境变量**，插件既读不到也改不了，而且 astrbot 容器没有 docker socket、读不到 `~/.mozilla`。所以真正的读写由一个宿主侧小服务负责：

```
plugins/astrbot_plugin_myrss  ──HTTP──▶  rsshub-vault（宿主，user systemd）
   （只做 UI 和转发）                       · 读 Firefox cookies.sqlite（热拷贝）
                                           · 先验证再落地（验证不过绝不覆盖旧凭据）
                                           · 写 ~/rsshub.env + 重建 rsshub
                                           · 每天 03:17 自检，浏览器里的更新就自动同步
```

服务地址默认 `http://172.18.0.1:18100`，可在配置项 `vault_helper_url` 改。

### 面板能做什么

- **一键导入**：22 个"浏览器里就有 cookie"的平台（B站 / 微博 / 知乎 / 小红书 / 抖音 / 豆瓣 / 网易云 / 什么值得买 / 米游社 / 知识星球 / 雪球 / 贴吧 / LOFTER / Twitter 等）
- **粘贴导入**：16 个需要 OAuth / PAT / API Key 的平台，给步骤 + 输入框
- 每行显示状态（已生效 / 未配置 / 临期 / 已失效）、过期日期、剩余天数
- 「立即自检」手动跑一遍，浏览器里的凭据比 rsshub 里的新就自动同步

### 几点刻意设计

- **前端永远拿不到完整凭据**：只显示「几段 / 多长 / 到期时间」
- **验证不过不写盘**：不会用一份坏 cookie 覆盖掉现在能用的那份
- **塌陷时零请求**：不展开面板就不打扰维护服务
- **单行出错不连坐**：每个平台一张卡片，一个失败不影响其他

### 已知依赖

需要宿主上跑着 `rsshub-vault` 服务；没跑的话面板会明确显示「维护服务：连不上」，其余功能不受影响。

