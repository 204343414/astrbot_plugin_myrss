# 持久化目录修复说明

## 根因

旧版默认将 `_data.json` 写入插件源码目录：

```text
data/plugins/astrbot_plugin_myrss/_data/_data.json
```

AstrBot 更新或重装插件会替换插件源码目录，因此订阅者、`last_update`、`latest_link` 和 `seen_links` 会一起丢失。随后旧动态会被当作未推送内容重新处理。

## 新默认目录

```text
<AstrBot data>/plugin_data/astrbot_plugin_myrss/_data.json
```

AstrBot 4.9.2 及以上通过 `get_astrbot_data_path()` 定位 data 根目录；旧版本回退到工作目录下的 `data/plugin_data/astrbot_plugin_myrss`。

## 自动迁移

仅当新目录不存在 `_data.json` 时，按顺序寻找：

1. 当前插件目录 `_data/_data.json`；
2. 早期 `data/astrbot_plugin_myrss/_data.json`。

迁移使用临时文件、`fsync` 和 `os.replace`。旧文件不会删除，并尝试创建 `.migrated_bak`。

如果新目录已经有数据，以新目录为准，绝不被旧文件覆盖。

## 自定义目录保护

如果 `custom_data_dir` 指向插件源码目录或其子目录，插件会拒绝该路径并改用官方 `plugin_data` 目录。

## Docker 注意

必须把 AstrBot 的整个 `data` 目录挂载到宿主机。若容器重建时连 `data` 卷也被删除，任何插件代码都无法恢复订阅数据。

## 已执行测试

自动测试覆盖：

- 从插件内部旧 `_data.json` 迁移；
- 订阅结构保持；
- `seen_links` 保持；
- 保存后重新创建 DataHandler，数据仍能读取；
- 配置 JSON 和 Python 语法检查。
