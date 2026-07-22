# MyRSS 死代码与全局推送移除审计

## 本次已移除

### 全局推送整条链路
- 全局源配置、调度注册、指数退避、全局 seen/cooldown 状态。
- 活跃群捕获与 `_groups.json` 读写。
- 群级全局源屏蔽/恢复工具。
- 全局推送安全模式、测试群、冷却命令、resetglobal 命令。
- 全局发送锁、bot-ready 状态、aiocqhttp 全局直连发送兜底。
- 仅为全局推送存在的无效群清理与 enable 状态。

### 可静态确认的死代码
- `calendar` 未使用导入。
- `filter_prompt`：读取后从未使用（审核函数使用硬编码 prompt）。
- `image_caption_provider_id`：读取后从未使用。
- `_load_recs` / `_save_recs` / `myrss_cancel_recommend`：推荐系统残留，且引用从未初始化的 `_recs_file`、`_pending_recs`。
- `_check_content_safe_bool`：无调用。
- `_cron_cb`：无调用；调度器实际入口是 `_cron_cb_url`。
- `_fetch.is_local_url`：定义后从未调用。
- `tool_sub` 中无条件 `return` 后的不可达代码。

## 明确保留且未改动的防重推链路

以下函数与仓库原版逐函数源码比对一致：
- `_cron_cb_url`
- `_cron_cb_inner`
- `_cron_cb_inner_impl`
- `_generate_comment`
- `_check_content_safe`
- `_make_card_b64`
- `_make_comps`

因此个人/群定向订阅的以下机制未改：
- 每 URL 单 job、每订阅者锁。
- `seen_links` 归一化去重。
- 推送前先保存去重记录。
- LLM 审核缓存与锐评缓存。
- 多条卡片合并与发送路径。
- 热更新残留调度器清理。

## 发现但本次未修的确定问题

为避免把“删除功能”扩大成行为重构，下列问题仅记录：

1. `tool_sub` 在 `target_group` 为空时会在赋值前读取 `ret`，普通“订阅当前会话”路径可能抛 `UnboundLocalError`。
2. `cmd_unsub` 与 `tool_batch_unsub` 的指定群分支在无匹配项时仍使用循环变量 `k`，可能抛 `UnboundLocalError`，也可能错误记录上一次匹配。
3. `cmd_test` 把 `_check_content_safe()` 返回的字符串当布尔值判断；`"REJECT"` / `"MALICIOUS"` 也是真值，所以测试命令的拦截判断不符合函数三态返回约定。
4. `tool_preview` 中 `b'' in raw[:10000]` 对任意 bytes 恒为真，条件没有实际校验作用。

这些问题不在定时防重推核心函数中，本次没有顺手修改。

## 已执行校验

- `python -m py_compile main.py`：通过。
- `_conf_schema.json` JSON 解析：通过。
- `git diff --check`：通过。
- 全局推送相关标识静态残留搜索：无结果。
- 防重推/LLM/卡片关键函数与原版源码比对：一致。

> 未执行 AstrBot 容器集成测试：当前工作区没有 AstrBot 运行时、RSSHub、Browserless 和真实消息平台连接，不能声称完成真实推送验证。
