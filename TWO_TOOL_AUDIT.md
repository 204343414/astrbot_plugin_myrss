# 两工具自然语言订阅改造审计

## 仅保留的 LLM 工具

1. `myrss_preview`
   - 解析链接/路由，读取频道资料与最新一条动态。
   - 强制执行一次内容安全判断；只有 `SAFE` 才展示。
   - 生成头像、最新动态标题/摘要/首图组成的卡片。
   - 预览卡片不生成 AI 锐评，避免额外 LLM 调用。
   - 生成绑定当前会话、10 分钟有效、一次性消费的 `preview_id`。
   - 不建立订阅、不发送到目标群、不修改 seen_links。

2. `myrss_manage`
   - `list`：沿用当前订阅数据并保留原列表中的 cron 展示。
   - `subscribe`：必须有同会话有效 preview_id、`confirm=true` 和用户明确确认。
   - `unsubscribe`：按唯一编号或关键词取关。
   - 指定其他群仅 AstrBot 管理员可操作。

## 安全不变量

- 内容审核配置开关已移除，运行时强制开启。
- 不再按国内平台免审。
- 没有审核 Provider 或审核异常时默认 `REJECT`。
- 预览结果非 `SAFE` 时不展示正文、不产生确认状态。
- 修复测试命令把三态字符串当布尔值的问题。
- 订阅仍复用 `_add()`；不会在确认时直接发送最新动态。

## 防刷屏与防重复

没有新增调度器、发送循环或后台任务。以下原有核心函数与上游源码保持一致：

- `_cron_cb_url`
- `_cron_cb_inner`
- `_cron_cb_inner_impl`
- `_generate_comment`
- `_make_card_b64`
- `_make_comps`

因此每 URL 单 job、每订阅者锁、推送前保存 seen_links、卡片/锐评缓存等机制未重写。

## 同步修复

- 原 `myrss_subscribe` 当前会话路径 `ret` 未赋值问题：旧工具整体由两工具状态机替代。
- 删除无消费者的 `_last_preview` 单槽状态。
- 修复 `cmd_unsub` 无匹配时错误使用循环变量 `k`。
- 修复测试命令对 `SAFE/REJECT/MALICIOUS` 的判断。
- 之前已移除全局推送和推荐系统残留。

## 校验

- AST 检查：LLM 工具恰好 2 个。
- `python -m py_compile main.py`：通过。
- `_conf_schema.json` 解析：通过。
- `git diff --check`：通过。

未执行真实 AstrBot/RSSHub/Browserless/QQ 集成测试，不能声称完成线上推送验证。上线前应在隔离测试群验证一次预览、确认订阅、产生新动态、取关四条路径。
