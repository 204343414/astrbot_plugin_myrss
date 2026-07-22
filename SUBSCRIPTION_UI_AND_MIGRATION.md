# 订阅只读页面、迁移命令与热重载修复

## 只读页面

插件页面 `pages/subscriptions/` 只展示正式数据文件中确实存在订阅关系的群。

页面字段：

- 正式 `_data.json` 路径；
- 订阅群数量、订阅关系数量；
- 平台与群号；
- 动态源标题和 URL；
- cron；
- last_update；
- seen_links 数量；
- latest_link。

页面只注册 GET API，不提供新增、编辑、重置或删除接口。

## 两阶段迁移

预检：

```text
/myrss migrate
```

执行：

```text
/myrss migrate confirm
```

确认仅在预检后 10 分钟内有效。预检指纹包含旧文件路径及 SHA-256；旧文件变化后必须重新预检。

冲突合并：

- RSSHub endpoints：并集；
- 同 URL、同群：保留新库 cron；
- last_update：取较大值；
- latest_link：采用 last_update 较新一侧；
- seen_links：原值和去 query/fragment 归一化值的去重并集；
- 黑名单等列表设置：并集。

执行顺序：原子写新库 → 读回完全一致校验 → 逐个备份旧文件 → 删除旧 `_data.json`。任一旧 JSON 损坏时拒绝合并和删除。

## 订阅消失竞态修复

旧代码只在 `_load()` 时持有 `_data_lock`，随后释放锁。另一个定时 job 可以替换 `dh.data`，导致前一个 job 修改旧对象、却保存新对象，形成丢更新。

现在锁覆盖完整的：

```text
读库 → 更新断点/seen_links → 保存
```

新增订阅也与该事务互斥，并在获得锁后再次检查是否已经订阅。

## 重装但不重启导致双调度器

模块级全局变量在插件模块重新加载后可能分叉。现在额外使用进程级 `builtins._ASTRBOT_MYRSS_RUNTIME` 登记唯一实例和 generation：

- 新实例启动先停止登记的旧调度器；
- 旧 generation 的 job 在入口和分发前主动退出；
- destroy 仅清理属于自己的登记。

第一次从不含此机制的旧版本升级后，仍建议完整重启 AstrBot 一次，因为旧实例没有登记到新注册表，无法保证被新代码定位。
