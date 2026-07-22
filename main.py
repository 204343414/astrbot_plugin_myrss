import os
import json
import shutil
import re
import time
import random
import base64
import logging
import asyncio
import aiohttp
import builtins
import copy
import hashlib
from io import BytesIO
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime
from typing import List

from lxml import etree
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageDraw
from jinja2 import Environment, BaseLoader
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig
import astrbot.api.message_components as Comp

from .web import MyRssWebController

# [防冲突] 模块级变量追踪当前活跃的调度器
# 插件热更新时新实例先通过此引用杀掉老调度器，避免新老并行双推
_ACTIVE_SCHED = None
_ALL_SCHEDS = set() # 同一模块对象内追踪调度器

# builtins 在同一 Python 进程的插件模块重载之间保持共享。
# 用它登记唯一活跃实例，弥补模块级全局变量在“重新安装但不重启”时可能分叉的问题。
if not hasattr(builtins, "_ASTRBOT_MYRSS_RUNTIME"):
    builtins._ASTRBOT_MYRSS_RUNTIME = {"generation": 0, "instance": None, "scheduler": None}

@dataclass
class RSSItem:
    chan_title: str
    title: str
    link: str
    description: str
    pubDate: str
    pubDate_timestamp: int
    pic_urls: list

class DataHandler:
    """数据管理（大幅改进版）：

    解决 Docker/容器 vs 主机路径混乱问题。

    路径优先级（从高到低）：
    1. 插件配置里的 custom_data_dir
    2. 环境变量 MYRSS_DATA_DIR
    3. AstrBot 官方持久化目录 data/plugin_data/astrbot_plugin_myrss/

    插件源码目录下的 _data 仅作为旧数据迁移来源，绝不再作为默认写入位置。
    每次启动都会在日志里明确打印实际使用的完整路径。
    """

    def __init__(self, plugin_dir=None, seen_links_max_days=365, custom_data_dir=None):
        self.logger = logging.getLogger("astrbot")
        self.seen_links_max_days = seen_links_max_days
        self.plugin_dir = os.path.abspath(plugin_dir) if plugin_dir else None

        self.data_dir = self._resolve_data_dir(plugin_dir, custom_data_dir)
        self.config_path = os.path.join(self.data_dir, "_data.json")

        is_container = self._is_running_in_container()
        env_type = "容器环境 (Docker/AstrBot容器)" if is_container else "主机/本地文件系统"
        self.logger.info(f"[MyRSS] 运行环境检测: {env_type}")
        self.logger.info(f"[MyRSS] 数据目录已解析: {self.data_dir}")
        self.logger.info(f"[MyRSS] 数据文件路径: {self.config_path}")

        self.data = self._load()

        if not custom_data_dir:
            self.logger.info("[MyRSS] 未设置 custom_data_dir，已使用 AstrBot 官方 plugin_data 持久化目录")

    def _is_running_in_container(self) -> bool:
        """自动检测是否在容器（Docker / AstrBot 常见容器）内运行。
        这样代码能自己判断“容器里面”还是“本地主机文件系统”，
        并在日志中明确报告，方便用户排查路径问题。
        """
        if os.path.exists("/.dockerenv"):
            return True
        try:
            with open("/proc/1/cgroup", "r") as f:
                c = f.read().lower()
                if any(x in c for x in ["docker", "kubepods", "containerd", "lxc"]):
                    return True
        except Exception:
            pass
        cwd = os.getcwd().lower()
        if cwd.startswith("/astrbot") or "/astrbot/" in cwd:
            return True
        if os.environ.get("ASTRBOT_DOCKER") or os.environ.get("IN_DOCKER"):
            return True
        if os.path.exists("/.containerenv") or os.path.exists("/run/.containerenv"):
            return True
        return False

    def _resolve_data_dir(self, plugin_dir: str | None, custom_data_dir: str | None) -> str:
        # 1. 最高优先级：用户在 AstrBot 配置界面填的 custom_data_dir
        if custom_data_dir and str(custom_data_dir).strip():
            d = os.path.abspath(str(custom_data_dir).strip())
            plugin_root = os.path.abspath(plugin_dir) if plugin_dir else ""
            try:
                inside_plugin = bool(plugin_root and os.path.commonpath([d, plugin_root]) == plugin_root)
            except ValueError:
                inside_plugin = False
            if inside_plugin:
                self.logger.error("[MyRSS] custom_data_dir 指向插件源码目录，重装会丢数据；已拒绝并改用官方 plugin_data 目录: %s", d)
            else:
                os.makedirs(d, exist_ok=True)
                self.logger.info("[MyRSS] 使用用户自定义数据目录 (custom_data_dir)")
                return d

        # 2. 环境变量（方便 Docker compose 覆盖）
        env_dir = os.environ.get("MYRSS_DATA_DIR", "").strip()
        if env_dir:
            d = os.path.abspath(env_dir)
            os.makedirs(d, exist_ok=True)
            self.logger.info("[MyRSS] 使用环境变量 MYRSS_DATA_DIR")
            return d

        # 3. AstrBot 官方插件持久化目录。
        # v4.9.2+ 使用官方路径 API；旧版本回退到 AstrBot 工作目录下的 data。
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            data_root = os.path.abspath(str(get_astrbot_data_path()))
        except (ImportError, AttributeError, TypeError):
            data_root = os.path.abspath("data")
            self.logger.warning("[MyRSS] AstrBot 路径 API 不可用，回退到 data/plugin_data")
        d = os.path.join(data_root, "plugin_data", "astrbot_plugin_myrss")
        os.makedirs(d, exist_ok=True)
        self.logger.info("[MyRSS] 使用 AstrBot 官方插件数据目录 data/plugin_data/astrbot_plugin_myrss")
        return d

    def get_data_path(self) -> str:
        """返回当前实际使用的数据文件完整路径（给命令显示用）"""
        return self.config_path

    def get_data_dir(self) -> str:
        return self.data_dir

    def get_legacy_data_paths(self) -> list[str]:
        paths = []
        if self.plugin_dir:
            paths.append(os.path.join(self.plugin_dir, "_data", "_data.json"))
        paths.append(os.path.abspath("data/astrbot_plugin_myrss/_data.json"))
        current = os.path.abspath(self.config_path)
        result = []
        for path in paths:
            absolute = os.path.abspath(path)
            if absolute != current and absolute not in result:
                result.append(absolute)
        return result

    @staticmethod
    def _normalize_seen_link(link: str) -> str:
        value = str(link or "").strip()
        return value.split("#", 1)[0].split("?", 1)[0] if value else ""

    @classmethod
    def merge_persistent_data(cls, current: dict, legacy: dict) -> dict:
        """合并旧新库；断点取较新值，seen_links 做原值+归一化并集。"""
        merged = copy.deepcopy(current if isinstance(current, dict) else {})
        old = legacy if isinstance(legacy, dict) else {}
        merged["rsshub_endpoints"] = list(dict.fromkeys(
            list(merged.get("rsshub_endpoints", [])) + list(old.get("rsshub_endpoints", []))
        ))
        if isinstance(old.get("settings"), dict):
            settings = merged.setdefault("settings", {})
            for key, value in old["settings"].items():
                if isinstance(value, list) and isinstance(settings.get(key), list):
                    settings[key] = list(dict.fromkeys(settings[key] + value))
                elif key not in settings:
                    settings[key] = copy.deepcopy(value)

        for url, old_feed in old.items():
            if url in ("rsshub_endpoints", "settings") or not isinstance(old_feed, dict):
                continue
            if url not in merged or not isinstance(merged[url], dict):
                merged[url] = copy.deepcopy(old_feed)
                continue
            feed = merged[url]
            if not feed.get("info") and old_feed.get("info"):
                feed["info"] = copy.deepcopy(old_feed["info"])
            subscribers = feed.setdefault("subscribers", {})
            for subscriber, old_sub in old_feed.get("subscribers", {}).items():
                if subscriber not in subscribers or not isinstance(subscribers[subscriber], dict):
                    subscribers[subscriber] = copy.deepcopy(old_sub)
                    continue
                new_sub = subscribers[subscriber]
                old_ts = int(old_sub.get("last_update", 0) or 0)
                new_ts = int(new_sub.get("last_update", 0) or 0)
                combined = []
                for raw in list(new_sub.get("seen_links", [])) + list(old_sub.get("seen_links", [])):
                    raw = str(raw or "").strip()
                    normalized = cls._normalize_seen_link(raw)
                    for candidate in (raw, normalized):
                        if candidate and candidate not in combined:
                            combined.append(candidate)
                new_sub["seen_links"] = combined[:1000]
                if old_ts > new_ts:
                    new_sub["last_update"] = old_ts
                    if old_sub.get("latest_link"):
                        new_sub["latest_link"] = old_sub["latest_link"]
                else:
                    new_sub["last_update"] = new_ts
                if not new_sub.get("cron_expr") and old_sub.get("cron_expr"):
                    new_sub["cron_expr"] = old_sub["cron_expr"]
        return merged

    def migration_snapshot(self) -> dict:
        sources = []
        for path in self.get_legacy_data_paths():
            if not os.path.exists(path):
                continue
            data = self._read_json(path)
            with open(path, "rb") as source_file:
                file_hash = hashlib.sha256(source_file.read()).hexdigest()
            sources.append({
                "path": path,
                "valid": isinstance(data, dict),
                "sha256": file_hash,
                "sources": len([k for k in data or {} if k not in ("rsshub_endpoints", "settings")]),
                "subscriptions": sum(len(v.get("subscribers", {})) for k, v in (data or {}).items() if k not in ("rsshub_endpoints", "settings") and isinstance(v, dict)),
            })
        raw = json.dumps(sources, ensure_ascii=False, sort_keys=True)
        return {"current": self.config_path, "legacy": sources, "fingerprint": hashlib.sha256(raw.encode()).hexdigest()[:16]}

    def migrate_and_delete_legacy(self) -> dict:
        """原子写新库、读回校验、备份旧库后才删除旧文件。"""
        merged = copy.deepcopy(self.data)
        valid_paths = []
        for path in self.get_legacy_data_paths():
            if not os.path.exists(path):
                continue
            legacy = self._read_json(path)
            if not isinstance(legacy, dict):
                raise ValueError(f"旧数据文件无法解析，未删除任何文件: {path}")
            merged = self.merge_persistent_data(merged, legacy)
            valid_paths.append(path)
        if not valid_paths:
            return {"merged": 0, "deleted": [], "current": self.config_path}

        original = self.data
        self.data = merged
        try:
            self._save()
            verified = self._read_json(self.config_path)
            if verified != merged:
                raise RuntimeError("新数据写入后的读回校验不一致")
        except Exception:
            self.data = original
            raise

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backups = []
        for path in valid_paths:
            backup = path + f".pre_delete_{timestamp}.bak"
            shutil.copy2(path, backup)
            backups.append(backup)
        deleted = []
        for path in valid_paths:
            os.remove(path)
            deleted.append(path)
        self.data = merged
        return {"merged": len(valid_paths), "deleted": deleted, "backups": backups, "current": self.config_path}

    def _load(self):
        """加载数据，支持从旧路径迁移，并自动备份已有数据"""
        if os.path.exists(self.config_path):
            d = self._read_json(self.config_path)
            if d is not None:
                if len(d) > 1 or d.get("rsshub_endpoints"):
                    self._backup_data(d)
                return d

        # 仅当新目录尚无数据时迁移旧路径。优先迁移插件内部 _data，
        # 其次兼容早期 data/astrbot_plugin_myrss。旧文件只读并保留备份，不删除。
        legacy_paths = []
        if self.plugin_dir:
            legacy_paths.append(os.path.join(self.plugin_dir, "_data", "_data.json"))
        legacy_paths.append(os.path.abspath("data/astrbot_plugin_myrss/_data.json"))
        for old_path in legacy_paths:
            if os.path.abspath(old_path) == os.path.abspath(self.config_path):
                continue
            if not os.path.exists(old_path):
                continue
            old_data = self._read_json(old_path)
            if old_data is None:
                self.logger.error("[MyRSS] 发现旧数据但 JSON 无法读取，拒绝覆盖: %s", old_path)
                continue
            os.makedirs(self.data_dir, exist_ok=True)
            tmp_path = self.config_path + ".migrate_tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(old_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.config_path)
                try:
                    shutil.copy2(old_path, old_path + ".migrated_bak")
                except Exception as backup_error:
                    self.logger.warning("[MyRSS] 旧数据备份失败但迁移已完成: %s", backup_error)
                self.logger.warning("[MyRSS] 已迁移持久化数据: %s -> %s", old_path, self.config_path)
                return old_data
            except Exception as migrate_error:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                self.logger.error("[MyRSS] 旧数据迁移失败，拒绝初始化空库: %s", migrate_error)
                raise

        # 初始化
        d = {"rsshub_endpoints": []}
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return d

    def _backup_data(self, data: dict):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.data_dir, f"_data_backup_{ts}.json")
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _read_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                try:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                except (ImportError, OSError):
                    pass
                return json.load(f)
        except Exception:
            return None

    def save(self):
        self._save()

    def _save(self):
        """保存数据，清理超大 seen_links（仅当配置了较小 max_days 时才做时间清理）。
        写入操作始终执行，保证 reset/clear 等命令能真正更新文件。
        """
        if self.seen_links_max_days < 365:
            max_age_seconds = self.seen_links_max_days * 86400
            now = time.time()
            for url, info in list(self.data.items()):
                if url in ("rsshub_endpoints", "settings"):
                    continue
                subscribers = info.get("subscribers", {})
                for sub_id, sub_data in list(subscribers.items()):
                    last_update = sub_data.get("last_update", 0)
                    seen = sub_data.get("seen_links", [])
                    if last_update > 0 and (now - last_update) > max_age_seconds and len(seen) > 500:
                        sub_data["seen_links"] = []

        # 始终执行原子写入，保证数据持久化
        tmp_path = self.config_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.config_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.logger.error(f"[MyRSS] 保存数据失败: {e}")
            raise e

    def get_subs(self, user_id):
        urls = []
        for url, info in self.data.items():
            if url in ("rsshub_endpoints", "settings"):
                continue
            if user_id in info.get("subscribers", {}):
                urls.append(url)
        return urls

    def parse_channel_info(self, text):
        root = etree.fromstring(text)
        title = root.xpath("//title")[0].text
        desc_nodes = root.xpath("//description")
        desc = desc_nodes[0].text if desc_nodes else ""
        avatar = ""
        img_nodes = root.xpath("//channel/image/url")
        if img_nodes and img_nodes[0].text:
            avatar = img_nodes[0].text
        return title, desc or "", avatar



    def strip_html_pic(self, html):
        """从HTML中提取所有图片URL，包含暴力正则匹配YouTube封面"""
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and src not in urls:
                urls.append(src)
                
        for vid in soup.find_all("video"):
            poster = vid.get("poster")
            if poster and poster not in urls:
                urls.append(poster)
                
        patterns = [
            r'youtube\.com/watch\?v=([\w-]{11})',
            r'youtu\.be/([\w-]{11})',
            r'youtube\.com/embed/([\w-]{11})',
            r'youtube\.com/v/([\w-]{11})'
        ]
        
        found_ids = set()
        for a in soup.find_all("a", href=True):
            for pat in patterns:
                m = re.search(pat, a["href"])
                if m: found_ids.add(m.group(1))

        for pat in patterns:
            for vid_id in re.findall(pat, html):
                found_ids.add(vid_id)

        for vid_id in found_ids:
            u1 = f"https://i.ytimg.com/vi/{vid_id}/maxresdefault.jpg"
            u2 = f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
            if u1 not in urls: urls.append(u1)
            if u2 not in urls: urls.append(u2)
        
        return urls

    def strip_html(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return re.sub(r"\n+", "\n", soup.get_text())

    def get_root_url(self, url):
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"

class PicHandler:
    def __init__(self, adjust=False):
        self.adjust = adjust

    async def to_base64(self, image_url):
        try:
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(trust_env=True, connector=conn) as s:
                async with s.get(image_url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status != 200:
                        return None
                    raw = BytesIO(await r.read())
                    if self.adjust:
                        img = Image.open(raw).convert("RGB")
                        w, h = img.size
                        px = img.load()
                        cx, cy = random.choice([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
                        px[cx, cy] = (255, 255, 255)
                        buf = BytesIO()
                        img.save(buf, format="JPEG")
                        buf.seek(0)
                        return base64.b64encode(buf.read()).decode()
                    else:
                        return base64.b64encode(raw.getvalue()).decode()
        except Exception:
            return None

class URLMapper:
    RULES = [
        (r"space\.bilibili\.com/(\d+)/dynamic", "/bilibili/user/dynamic/{0}", "B站UP主动态"),
        (r"space\.bilibili\.com/(\d+)", "/bilibili/user/dynamic/{0}", "B站UP主动态"),
        (r"bilibili\.com/bangumi/media/md(\d+)", "/bilibili/bangumi/media/{0}", "B站番剧"),
        (r"live\.bilibili\.com/(\d+)", "/bilibili/live/room/{0}", "B站直播间"),
        (r"manga\.bilibili\.com/detail/mc(\d+)", "/bilibili/manga/update/{0}", "B站漫画"),
        (r"youtube\.com/channel/([\\w-]+)", "/youtube/channel/{0}", "YouTube频道"),
        # [修复] 优先匹配 YouTube 的动态(community/posts)、Shorts、直播等特定页面
        # 必须放在通用的 @user 规则之前，否则会被通用规则拦截
        (r"youtube\.com/@([\\w.-]+)/(?:posts|community)", "/youtube/community/@{0}", "YouTube动态"),
        (r"youtube\.com/@([\\w.-]+)/shorts", "/youtube/user/@{0}/shorts", "YouTube Shorts"),
        (r"youtube\.com/@([\\w.-]+)/streams", "/youtube/user/@{0}/live", "YouTube直播记录"),
        # [原规则] 通用用户规则放在最后作为兜底
        (r"youtube\.com/@([\\w.-]+)", "/youtube/user/@{0}", "YouTube用户"),
        (r"youtube\.com/playlist\\?list=([\\w-]+)", "/youtube/playlist/{0}", "YouTube播放列表"),
        # 专门匹配 x.com（优先级高于通用的 twitter|x 正则）
        (r"x\.com/(?:@)?([A-Za-z0-9_]+)(?:/(?:posts|media|with_replies|likes|followers|following)?(?:$|\?))?", "/twitter/user/{0}", "X.com"),
        (r"(?:twitter|x)\.com/(?:@)?([A-Za-z0-9_]+)(?:/(?:posts|media|with_replies|likes|followers|following)?(?:$|\?))?", "/twitter/user/{0}", "Twitter/X"),
        (r"weibo\.com/u/(\d+)", "/weibo/user/{0}", "微博"),
        (r"zhihu\.com/people/([\\w-]+)", "/zhihu/people/activities/{0}", "知乎"),
        (r"zhihu\.com/column/([\\w-]+)", "/zhihu/zhuanlan/{0}", "知乎专栏"),
        (r"xiaohongshu\.com/user/profile/([\\w]+)", "/xiaohongshu/user/{0}/notes", "小红书"),
        (r"github\.com/([\\w.-]+)/([\\w.-]+)/releases", "/github/release/{0}/{1}", "GitHub Release"),
        (r"github\.com/([\\w.-]+)/([\\w.-]+)(?:$|[/?#])", "/github/commits/{0}/{1}", "GitHub仓库"),
        (r"github\.com/([\\w.-]+)(?:$|[/?#])", "/github/repos/{0}", "GitHub用户"),
        (r"t\.me/s?/?([\\w]+)", "/telegram/channel/{0}", "Telegram"),
        (r"douyin\.com/user/([\\w]+)", "/douyin/user/{0}", "抖音"),
        (r"instagram\.com/([\\w.]+)(?:$|[/?#])", "/instagram/user/{0}", "Instagram"),
        (r"pixiv\.net/users/(\d+)", "/pixiv/user/{0}", "Pixiv"),
        (r"sspai\.com/u/([\\w]+)", "/sspai/author/{0}", "少数派"),
        (r"okjike\.com/u/([\\w-]+)", "/jike/user/{0}", "即刻"),
        (r"podcasts\.apple\.com/.*/id(\d+)", "/apple/podcast/{0}", "Apple Podcast"),
    ]

    HINTS = {
        "bilibili": (
            "B站可用路由(uid在space.bilibili.com/{uid}找):\\n"
            " UP主视频: /bilibili/user/video/{uid}\\n"
            " UP主动态: /bilibili/user/dynamic/{uid}\\n"
            " 所有视频: /bilibili/user/video-all/{uid}\\n"
            " UP主图文: /bilibili/user/article/{uid}\\n"
            " UP主合集: /bilibili/user/collection/{uid}/{sid}\\n"
            " 综合热门: /bilibili/popular/all\\n"
            " 每周必看: /bilibili/weekly\\n"
            " 排行榜: /bilibili/ranking/all\\n"
            " 热搜: /bilibili/hot-search\\n"
            " 番剧: /bilibili/bangumi/media/{mediaid}\\n"
            " 直播: /bilibili/live/room/{roomID}\\n"
            " 搜索: /bilibili/vsearch/{keyword}"
        ),
        "youtube": "YouTube路由:\\n 频道: /youtube/channel/{id}\\n 用户: /youtube/user/@{name}\\n 播放列表: /youtube/playlist/{id}",
        "twitter": "Twitter/X路由:\\n 用户: /twitter/user/{name}\\n 媒体: /twitter/media/{name}\\n 搜索: /twitter/keyword/{kw}",
        "x.com": "Twitter/X路由:\\n 用户: /twitter/user/{name}\\n 媒体: /twitter/media/{name}",
        "weibo": "微博路由:\\n 用户: /weibo/user/{uid}\\n 热搜: /weibo/search/hot",
        "zhihu": "知乎路由:\\n 用户: /zhihu/people/activities/{id}\\n 专栏: /zhihu/zhuanlan/{id}\\n 热榜: /zhihu/hot",
        "github": "GitHub路由:\\n Release: /github/release/{owner}/{repo}\\n Commits: /github/commits/{owner}/{repo}",
        "xiaohongshu": "小红书路由:\\n 用户笔记: /xiaohongshu/user/{id}/notes",
        "douyin": "抖音路由:\\n 用户: /douyin/user/{uid}",
        "instagram": "Instagram路由:\\n 用户: /instagram/user/{name}",
        "telegram": "Telegram路由:\\n 频道: /telegram/channel/{name}",
        "pixiv": "Pixiv路由:\\n 用户: /pixiv/user/{uid}\\n 排行: /pixiv/ranking/{mode}",
    }

    @classmethod
    def match(cls, url):
        for pat, tpl, name in cls.RULES:
            m = re.search(pat, url)
            if m:
                return tpl.format(*m.groups()), name
        return None

    @classmethod
    def suggest(cls, url):
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return "无法解析，请提供http开头的链接或/开头的路由。"
        for kw, hint in cls.HINTS.items():
            if kw in netloc:
                return hint
        return "未收录此平台。请到 https://docs.rsshub.app 查找路由后用/开头调用。"

class CardGen:
    """HTML模板 + Browserless 截图的卡片生成器（替代 Pillow）

    优势：
    - Emoji 原生彩色渲染
    - CSS 排版，不用手算像素坐标
    - 图片/头像用 data URI，不怕防盗链
    """
    CARD_HTML = r"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, "Segoe UI", sans-serif; width: {{width}}px; background: #fff; min-width: {{width}}px; }
    .card { padding: 16px; border-bottom: 1px solid #e5e7eb; min-width: 0; }
    .header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; min-width: 0; }
    .avatar { width: 36px; height: 36px; border-radius: 50%; background: #667eea; display: flex; align-items: center; justify-content: center; color: white; font-size: 16px; font-weight: 600; overflow: hidden; flex-shrink: 0; }
    .avatar img { width: 100%; height: 100%; object-fit: cover; }
    .chan-wrap { min-width: 0; flex: 1; overflow: hidden; }
    .chan { font-size: 14px; font-weight: 600; color: #1f2937; }
    .time { font-size: 12px; color: #9ca3af; }
    .title { font-size: 15px; font-weight: 600; color: #111827; margin-bottom: 6px; line-height: 1.4; }
    .desc { font-size: 13px; color: #6b7280; line-height: 1.5; margin-bottom: 8px; }
    .thumb { width: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; margin-bottom: 8px; }
    .link { font-size: 12px; color: #3b82f6; margin-bottom: 8px; }
    .divider { border-top: 1px dashed #e5e7eb; margin: 12px 0; }
    .comment { background: #f9fafb; border-radius: 8px; padding: 10px; margin-top: 8px; }
    .comment-header { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 6px; }
    .bot-avatar { width: 24px; height: 24px; border-radius: 50%; background: #764ba2; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; overflow: hidden; flex-shrink: 0; margin-top: 1px; }
    .bot-avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .comment-text { font-size: 13px; color: #374151; line-height: 1.5; }
    .comment-provider { font-size: 11px; color: #9ca3af; margin-top: 4px; }
</style></head><body>
<div class="card">
    <div class="header">
        <div class="avatar" style="flex-shrink:0">{% if avatar_b64 %}<img src="data:image/jpeg;base64,{{avatar_b64}}" />{% else %}{{avatar_char}}{% endif %}</div>
        <div class="chan-wrap"><div class="chan">{{channel}}</div>{% if time_str %}<div class="time">· {{time_str}}</div>{% endif %}</div>
    </div>
    {% if title %}<div class="title">{{title}}</div>{% endif %}
    {% if desc %}<div class="desc">{{desc}}</div>{% endif %}
    {% if thumb_b64 %}<img class="thumb" src="data:image/jpeg;base64,{{thumb_b64}}" />{% endif %}
    <div class="divider"></div>
    {% if link %}<div class="link">🔗 {{link_display}}</div>{% endif %}
    {% if comment %}
    <div class="comment">
        <div class="comment-header">
            <div class="bot-avatar">{% if bot_avatar_b64 %}<img src="data:image/jpeg;base64,{{bot_avatar_b64}}" />{% else %}B{% endif %}</div>
            <div class="comment-text">{{comment}}</div>
        </div>
        {% if bot_provider_name %}<div class="comment-provider">via {{bot_provider_name}}</div>{% endif %}
    </div>
    {% endif %}
</div>
</body></html>"""

    def __init__(self, width=480, browserless_url="http://browserless:3000"):
        self.w = width
        self.browserless_url = browserless_url.rstrip("/")
        self._env = Environment(loader=BaseLoader(), autoescape=True)
        self._tpl = self._env.from_string(self.CARD_HTML)
        self.logger = logging.getLogger("astrbot")
        self._sema = asyncio.Semaphore(2) # 最多同时 2 个截图请求

    def _format_time(self, ts_str):
        """把RSS时间字符串简化为 YYYY-MM-DD HH:MM"""
        if not ts_str:
            return ""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(ts_str)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]:
            try:
                dt = datetime.strptime(ts_str.replace("Z", "+0000"), fmt)
                return dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                continue
        return ts_str[:25] if len(ts_str) > 25 else ts_str

    async def make(self, channel="", title="", desc="", link="", ts="",
                   thumb=None, avatar=None, comment="", bot_avatar=None,
                   bot_provider_name=""):
        """渲染 HTML → browserless 截图 → 返回 base64 PNG"""

        display_name = (channel or "未知频道")
        display_name = display_name.replace(" - Community Posts - YouTube", "").replace(" - YouTube", "")
        time_str = self._format_time(ts)

        # 头像 → base64 data URI
        avatar_b64 = ""
        avatar_char = "?"
        if avatar and isinstance(avatar, bytes) and len(avatar) > 100:
            avatar_b64 = base64.b64encode(avatar).decode()
            for c in (channel or ""):
                if c.strip():
                    avatar_char = c
                    break

        # 缩略图
        thumb_b64 = ""
        if thumb and isinstance(thumb, bytes) and len(thumb) > 100:
            thumb_b64 = base64.b64encode(thumb).decode()

        # Bot 头像
        bot_avatar_b64 = ""
        if bot_avatar and isinstance(bot_avatar, bytes) and len(bot_avatar) > 100:
            bot_avatar_b64 = base64.b64encode(bot_avatar).decode()

        # 链接截断
        link_display = link if len(link) <= 50 else link[:50] + "..."

        # 去重：desc 与 title 相同则不重复显示
        desc_clean = (desc or "").strip()
        if title and desc_clean == (title or "").strip():
            desc_clean = ""
        show_title = title and title not in ("无标题", "")

        html = self._tpl.render(
            width=self.w,
            channel=display_name,
            time_str=time_str,
            avatar_b64=avatar_b64,
            avatar_char=avatar_char,
            title=title if show_title else "",
            desc=desc_clean,
            thumb_b64=thumb_b64,
            link=link,
            link_display=link_display,
            comment=comment,
            bot_avatar_b64=bot_avatar_b64,
            bot_provider_name=bot_provider_name,
        )

        try:
            return await self._screenshot(html)
        except Exception as e:
            self.logger.error("[CardGen] browserless 截图失败: %s (%s)", type(e).__name__, e)
            return ""

    async def _screenshot(self, html: str) -> str:
        payload = {
            "html": html,
            "options": {
                "fullPage": True,
                "type": "png",
            },
            "viewport": {
                "width": self.w,
                "height": 1,
                "deviceScaleFactor": 2,
            },
            "gotoOptions": {
                "waitUntil": "domcontentloaded",
            },
        }

        endpoints = [
            f"{self.browserless_url}/screenshot",
            f"{self.browserless_url}/chromium/screenshot",
        ]

        async with self._sema:
            conn = aiohttp.TCPConnector(ssl=False)
            timeout = aiohttp.ClientTimeout(total=30)

            # Explicitly set trust_env=False to bypass global proxy for internal browserless container
            async with aiohttp.ClientSession(trust_env=False, connector=conn, timeout=timeout) as session:
                for ep in endpoints:
                    # 429 重试（最多 3 次，间隔递增）
                    for attempt in range(3):
                        try:
                            async with session.post(
                                ep, json=payload,
                                headers={"Content-Type": "application/json"},
                            ) as resp:
                                if resp.status == 200:
                                    ct = resp.headers.get("Content-Type", "")
                                    data = await resp.read()
                                    if len(data) > 500 and (
                                        "image" in ct or data[:4] == b'\x89PNG'
                                    ):
                                        return base64.b64encode(data).decode()
                                elif resp.status == 429:
                                    wait = 2 ** attempt
                                    self.logger.warning("[CardGen] 429, retry in %ds (attempt %d/3)", wait, attempt + 1)
                                    await asyncio.sleep(wait)
                                    continue
                                body = await resp.text()
                                self.logger.warning("[CardGen] %s -> HTTP %d: %s", ep, resp.status, body[:200])
                        except aiohttp.ClientError as e:
                            self.logger.warning("[CardGen] %s 连接失败: %s", ep, e)
                            break # 非429错误不重试

        raise RuntimeError("browserless 不可用")



@register("astrbot_plugin_myrss", "MyRSS", "RSS订阅插件(LLM增强版)", "1.1.0", "")
class MyRssPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.ctx = context
        self.cfg = config
        
        # 插件目录仅用于发现并迁移旧版 _data；新数据写入 AstrBot data/plugin_data。
        _plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化 DataHandler（插件目录 + 迁移 + 清理）
        seen_links_max_days = max(config.get("seen_links_max_days", 365), 1)
        custom_data_dir = config.get("custom_data_dir", "") or ""
        self.dh = DataHandler(
            plugin_dir=_plugin_dir, 
            seen_links_max_days=seen_links_max_days,
            custom_data_dir=custom_data_dir
        )

        self.title_max = config.get("title_max_length", 60)
        self.desc_max = config.get("description_max_length", 200)
        self.max_poll = config.get("max_items_per_poll", 5)
        self.t2i = config.get("t2i", False)
        self.hide_url = config.get("is_hide_url", False)
        self.read_pic = config.get("is_read_pic", True)
        self.adjust_pic = config.get("is_adjust_pic", False)
        self.max_pic = config.get("max_pic_item", 3)
        self.compose = config.get("compose", True)
        self.enable_comment = config.get("enable_comment", True)
        self.comment_provider_id = config.get("comment_provider_id", "")
        self.comment_persona = config.get("comment_persona", "")
        self.comment_max_length = config.get("comment_max_length", 80)
        self.bot_qq = config.get("bot_qq", "")
        self.bot_provider_name = config.get("bot_provider_name", "")
        # 安全审核为强制生产不变量：不能通过配置绕过。
        self.content_filter = True
        self._comment_cache = {} # key=item_link, value=comment_text
        self._safe_cache = {} # key=item_link, value=bool(safe)
        self._vision_cache = {} # key=item_link, value=融合图单次识别结果（失败也缓存）
        self.image_caption_provider_id = config.get("image_caption_provider_id", "")
        self.max_vision_images = 9
        self._preview_states = {}  # key=(origin, preview_id), value=一次性安全预览状态
        self._preview_ttl_seconds = 600
        self._ready_group_sessions = set()  # 本次进程启动后实际观察到消息的群会话
        self._eye_cooldown = {}
        self._target_send_locks = {}
        self._target_last_send = {}
        self._delivery_test_cooldown = {}
        self._official_send_delay_min = 5.0
        self._official_send_delay_max = 10.0
        
        self._last_fetch_error = None  # 拉取错误追踪（新版本 _fetch 使用）
        self.push_delay_min = config.get("push_delay_min", 5.0)
        self.push_delay_max = config.get("push_delay_max", 8.0)
        self.filter_provider_id = config.get("filter_provider_id", "")
        raw_operators = config.get("subscription_operator_ids", "") or ""
        if isinstance(raw_operators, list):
            self.subscription_operator_ids = {str(value).strip() for value in raw_operators if str(value).strip()}
        else:
            self.subscription_operator_ids = {value.strip() for value in re.split(r"[,，\n]+", str(raw_operators)) if value.strip()}
        self.pic = PicHandler(self.adjust_pic)
        self.browserless_url = config.get("browserless_url", "http://browserless:3000")
        self.card = CardGen(browserless_url=self.browserless_url)
        ark_config = config.get("ark", {}) or {}
        self.ark_enabled = bool(ark_config.get("enabled", False))
        self.ark_template_id = int(ark_config.get("template_id", 23))
        self.ark_dashboard_url = str(ark_config.get("public_dashboard_url", "") or "").rstrip("/")

        # 防并发锁，key = (url, user)
        self._locks: dict = {}
        self._data_lock = asyncio.Lock() # 保护 dh.data 读写
        self.web = MyRssWebController(self.ctx, self)
        self.web.register_routes()
        # 推荐系统已移除
        # [防冲突] 在创建新调度器前，先杀掉模块级残留的老调度器
        # 场景：插件热更新时框架直接创建新实例，老实例的destroy()可能未被调用
        # 如果不杀，老调度器继续运行老代码的job，和新调度器同时推送→双推
        global _ACTIVE_SCHED, _ALL_SCHEDS
        runtime = builtins._ASTRBOT_MYRSS_RUNTIME
        stale_schedulers = list(_ALL_SCHEDS)
        process_old = runtime.get("scheduler")
        if process_old is not None and process_old not in stale_schedulers:
            stale_schedulers.append(process_old)
        for old_sched in stale_schedulers:
            try:
                if old_sched.running:
                    old_sched.shutdown(wait=False)
                    self.logger.warning("MyRSS: 停止跨重载残留调度器 id=%s", id(old_sched))
            except Exception as exc:
                self.logger.warning("MyRSS: 停止残留调度器失败: %s", exc)
        _ALL_SCHEDS.clear()
        _ACTIVE_SCHED = None

        runtime["generation"] = int(runtime.get("generation", 0)) + 1
        self._runtime_generation = runtime["generation"]
        self.sched = AsyncIOScheduler()
        runtime["instance"] = self
        runtime["scheduler"] = self.sched
        _ACTIVE_SCHED = self.sched
        _ALL_SCHEDS.add(self.sched)
        self.sched.start()
        self._reload_jobs()
    
    def _is_current_runtime(self) -> bool:
        runtime = builtins._ASTRBOT_MYRSS_RUNTIME
        return runtime.get("instance") is self and runtime.get("generation") == self._runtime_generation

    def _loaded_platforms(self) -> dict:
        result = {}
        manager = getattr(self.ctx, "platform_manager", None)
        instances = []
        try:
            instances = manager.get_insts() if manager and hasattr(manager, "get_insts") else getattr(manager, "platform_insts", [])
        except Exception:
            instances = []
        for instance in instances or []:
            try:
                meta = instance.meta()
                result[str(meta.id)] = {"name": str(meta.name), "instance": instance}
            except Exception:
                continue
        return result

    def _is_qq_official_origin(self, origin: str) -> bool:
        platform_id = str(origin).split(":", 1)[0]
        platform = self._loaded_platforms().get(platform_id)
        return bool(platform and platform.get("name") == "qq_official")

    def _target_readiness(self, origin: str) -> tuple[bool, str]:
        platform_id = str(origin).split(":", 1)[0]
        platform = self._loaded_platforms().get(platform_id)
        if not platform:
            return False, "platform_not_loaded"
        if platform.get("name") == "qq_official" and "GroupMessage" in str(origin):
            if origin in self._ready_group_sessions:
                return True, "ready"
            # 插件热重载会清空自身集合，但 QQ Official 平台实例通常仍保留已观察群场景。
            # 读取 AstrBot v4.26.x 适配器维护的 _session_scene，避免误判必须重新发言。
            session_id = str(origin).split(":", 2)[-1]
            adapter = platform.get("instance")
            scenes = getattr(adapter, "_session_scene", {})
            if isinstance(scenes, dict) and scenes.get(session_id) == "group":
                self._ready_group_sessions.add(origin)
                return True, "ready_from_platform_scene"
            return False, "qq_official_waiting_group_message"
        return True, "ready"

    def _get_target_send_lock(self, origin: str) -> asyncio.Lock:
        if origin not in self._target_send_locks:
            self._target_send_locks[origin] = asyncio.Lock()
        return self._target_send_locks[origin]

    async def _wait_target_send_slot(self, origin: str) -> None:
        """QQ 官方同群主动发送间隔 5~10 秒；不丢消息、不做小时级冷却。"""
        if not self._is_qq_official_origin(origin):
            return
        minimum = random.uniform(self._official_send_delay_min, self._official_send_delay_max)
        elapsed = time.time() - self._target_last_send.get(origin, 0)
        if elapsed < minimum:
            await asyncio.sleep(minimum - elapsed)

    async def _send_message_guarded(self, origin: str, chain: MessageChain) -> bool:
        async with self._get_target_send_lock(origin):
            await self._wait_target_send_slot(origin)
            result = await self.ctx.send_message(origin, chain)
            if result:
                self._target_last_send[origin] = time.time()
            return bool(result)

    async def run_delivery_diagnostic(self, origin: str, feed_url: str) -> dict:
        """UI/命令共用的单次诊断：GET RSS + 合成卡片主动发送，不调用 LLM。"""
        origin = str(origin or "")
        feed_url = str(feed_url or "")
        if "GroupMessage" not in origin:
            raise ValueError("目标不是群会话")
        feed = self.dh.data.get(feed_url)
        if not isinstance(feed, dict) or origin not in feed.get("subscribers", {}):
            raise ValueError("所选 RSS 源不属于该订阅群")
        now = time.time()
        remaining = 60 - (now - self._delivery_test_cooldown.get(origin, 0))
        if remaining > 0:
            raise ValueError(f"同群诊断冷却中，请 {int(remaining) + 1} 秒后再试")
        ready, reason = self._target_readiness(origin)
        if not ready:
            raise ValueError(f"目标群尚未具备主动投递条件: {reason}")
        self._delivery_test_cooldown[origin] = now

        raw = await self._fetch(feed_url)
        fetch_ok = False
        item_count = 0
        fetch_error = self._last_fetch_error or ""
        if raw:
            try:
                root = etree.fromstring(raw)
                item_count = len(root.xpath("//item"))
                fetch_ok = item_count > 0
                if not fetch_ok:
                    fetch_error = "RSS_XML_HAS_NO_ITEM"
            except Exception as exc:
                fetch_error = f"XML_PARSE_{type(exc).__name__}"

        title = feed.get("info", {}).get("title", feed_url) if isinstance(feed.get("info"), dict) else feed_url
        diagnostic_text = (
            f"RSS GET: {'成功' if fetch_ok else '失败'}\n"
            f"解析条目: {item_count}\n"
            f"主动目标: {origin}\n"
            "本测试未调用 LLM、未修改 seen_links、未推送历史动态。"
        )
        b64 = await self.card.make(
            channel="MyRSS 诊断", title=str(title)[:100], desc=diagnostic_text,
            link="", ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            thumb=None, avatar=None, comment="", bot_avatar=None, bot_provider_name="",
        )
        if not b64:
            return {"fetch_ok": fetch_ok, "item_count": item_count, "send_ok": False,
                    "fetch_error": fetch_error, "send_error": "CARD_RENDER_FAILED"}
        send_ok = await self._send_message_guarded(
            origin, MessageChain(chain=[Comp.Image.fromBase64(b64)], use_t2i_=self.t2i)
        )
        return {"fetch_ok": fetch_ok, "item_count": item_count, "send_ok": send_ok,
                "fetch_error": fetch_error if not fetch_ok else "",
                "send_error": "" if send_ok else "SEND_RETURNED_FALSE"}

    async def add_subscription_from_ui(self, origin: str, value: str) -> dict:
        """Dashboard 安全新增：预审最新动态后固定 15 分钟订阅，不推历史内容。"""
        origin = str(origin or "")
        if "GroupMessage" not in origin:
            raise ValueError("目标不是群会话")
        if origin not in {sub for feed_url, feed in self.dh.data.items() if feed_url not in ("rsshub_endpoints", "settings") and isinstance(feed, dict) for sub in feed.get("subscribers", {})}:
            # UI 只管理已经被插件观察/登记过的群，不能构造任意 origin。
            known_in_web = any(origin == group_origin for group_origin in self._ready_group_sessions)
            if not known_in_web:
                raise ValueError("目标群尚未被 MyRSS 观察到；请先在群内使用一次 Bot")
        full_url, route, error = self._resolve_feed_url(value)
        if not full_url:
            raise ValueError(error)
        raw = await self._fetch(full_url)
        if not raw:
            raise ValueError("无法访问该 RSS 源")
        try:
            title, description, avatar = self.dh.parse_channel_info(raw)
        except Exception as exc:
            raise ValueError(f"频道解析失败: {exc}") from exc
        old_entry = self.dh.data.get(full_url)
        if old_entry is None:
            self.dh.data[full_url] = {
                "subscribers": {},
                "info": {"title": title, "description": description, "avatar": avatar},
            }
        try:
            items = await self._poll(full_url, num=1)
            if not items:
                raise ValueError("该源没有可审核的最新动态")
            status = await self._check_content_safe(items[0])
            if status != "SAFE":
                raise ValueError(f"最新动态未通过安全审核: {status}")
            async with self._data_lock:
                self.dh.data = self.dh._load()
                if origin in self.dh.data.get(full_url, {}).get("subscribers", {}):
                    return {"created": False, "title": title, "route": route, "message": "本群已经订阅该源"}
                # 预审期间的临时 feed 不一定在磁盘，正式写入前补齐。
                self.dh.data.setdefault(full_url, {
                    "subscribers": {},
                    "info": {"title": title, "description": description, "avatar": avatar},
                })
                self.dh.data[full_url].setdefault("subscribers", {})[origin] = {
                    "cron_expr": "*/15 * * * *",
                    "last_update": items[0].pubDate_timestamp,
                    "latest_link": items[0].link,
                    "seen_links": [self._item_cache_key(item) for item in items if self._item_cache_key(item)][:200],
                    "created_by": {"source": "dashboard"},
                    "created_at": int(time.time()),
                }
                self.dh.save()
                self._reload_jobs()
            return {"created": True, "title": title, "route": route, "message": "订阅成功，固定每15分钟检查"}
        finally:
            if old_entry is None and not self.dh.data.get(full_url, {}).get("subscribers"):
                self.dh.data.pop(full_url, None)

    async def send_ark_panel_from_ui(self, origin: str) -> dict:
        """向当前 QQ 官方群主动发送一张只读 ARK 管理入口卡。"""
        origin = str(origin or "")
        if not self.ark_enabled:
            raise ValueError("ARK 管理卡未启用，请先在插件配置中开启 ark.enabled")
        if not self.ark_dashboard_url.startswith("https://"):
            raise ValueError("ARK 管理入口必须配置稳定的 HTTPS public_dashboard_url")
        platform_id = origin.split(":", 1)[0]
        platform = self._loaded_platforms().get(platform_id)
        if not platform or platform.get("name") != "qq_official":
            raise ValueError("目标不是当前已加载的 QQ Official 群")
        ready, reason = self._target_readiness(origin)
        if not ready:
            raise ValueError(f"目标群未就绪: {reason}")
        group_openid = origin.split(":", 2)[-1]
        subscriptions = []
        success_count = 0
        failed_count = 0
        for url, feed in self.dh.data.items():
            if url in ("rsshub_endpoints", "settings") or not isinstance(feed, dict):
                continue
            sub = feed.get("subscribers", {}).get(origin)
            if not isinstance(sub, dict):
                continue
            subscriptions.append(feed.get("info", {}).get("title", url))
            delivery = sub.get("delivery_status", {}) if isinstance(sub.get("delivery_status"), dict) else {}
            success_count += delivery.get("status") == "SUCCESS"
            failed_count += delivery.get("status") == "FAILED"
        blocked_count = len(self.dh.data.get("settings", {}).get("safety_events", []))
        ark = {
            "template_id": self.ark_template_id,
            "kv": [
                {"key": "#DESC#", "value": "MyRSS 订阅管理"},
                {"key": "#PROMPT#", "value": f"本群已关注 {len(subscriptions)} 个动态源"},
                {"key": "#LIST#", "obj": [
                    {"obj_kv": [{"key": "desc", "value": f"订阅源：{len(subscriptions)} 个"}]},
                    {"obj_kv": [{"key": "desc", "value": f"最近成功：{success_count}，失败：{failed_count}"}]},
                    {"obj_kv": [{"key": "desc", "value": f"近期安全拦截：{blocked_count}"}]},
                    {"obj_kv": [
                        {"key": "desc", "value": "打开 AstrBot 管理面板"},
                        {"key": "link", "value": self.ark_dashboard_url},
                    ]},
                ]},
            ],
        }
        adapter = platform.get("instance")
        client = adapter.get_client() if hasattr(adapter, "get_client") else getattr(adapter, "client", None)
        if not client or not getattr(client, "api", None):
            raise RuntimeError("无法取得 QQ Official botpy client")
        try:
            await client.api.post_group_message(
                group_openid=group_openid,
                msg_type=3,
                ark=ark,
                msg_seq=random.randint(1, 10000),
            )
        except Exception as exc:
            raise RuntimeError(f"ARK 发送失败: {type(exc).__name__}: {exc}") from exc
        return {"sent": True, "subscription_count": len(subscriptions)}

    async def remove_subscription_from_ui(self, origin: str, feed_url: str) -> dict:
        """Dashboard 精确退订一个群-源关系，不清空其他群与源级缓存。"""
        origin = str(origin or "")
        feed_url = str(feed_url or "")
        async with self._data_lock:
            self.dh.data = self.dh._load()
            feed = self.dh.data.get(feed_url)
            if not isinstance(feed, dict) or origin not in feed.get("subscribers", {}):
                raise ValueError("订阅关系不存在或已被删除")
            title = feed.get("info", {}).get("title", feed_url) if isinstance(feed.get("info"), dict) else feed_url
            del feed["subscribers"][origin]
            self.dh.save()
            self._reload_jobs()
        return {"removed": True, "title": title}

    def _record_delivery_status(self, url: str, origin: str, status: str, category: str = "") -> None:
        sub = self.dh.data.get(url, {}).get("subscribers", {}).get(origin)
        if not isinstance(sub, dict):
            return
        now = int(time.time())
        previous = sub.get("delivery_status") if isinstance(sub.get("delivery_status"), dict) else {}
        sub["delivery_status"] = {
            "status": status,
            "attempted_at": now,
            "delivered_at": now if status == "SUCCESS" else int(previous.get("delivered_at", 0) or 0),
            "error_category": category[:80],
        }

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def _observe_group_session(self, event: AstrMessageEvent):
        """只记录本次启动后真实出现过消息的群；不回复、不修改订阅。"""
        origin = getattr(event, "unified_msg_origin", "")
        if origin and "GroupMessage" in origin:
            self._ready_group_sessions.add(origin)

    async def destroy(self):
        """插件卸载/禁用时停止调度器"""
        global _ACTIVE_SCHED
        try:
            if self.sched.running:
                # [防冲突] wait=True：等正在执行的job跑完再关，避免推送到一半被掐断
                # 之前用wait=False会导致job还在跑但调度器已标记关闭，行为未定义
                self.sched.shutdown(wait=True)
                self.logger.info("MyRSS: 调度器已停止")
                # [防冲突] 清除全局引用，防止下次init误杀已关闭的对象
                if _ACTIVE_SCHED is self.sched:
                    _ACTIVE_SCHED = None
                _ALL_SCHEDS.discard(self.sched)
                runtime = builtins._ASTRBOT_MYRSS_RUNTIME
                if runtime.get("instance") is self:
                    runtime["instance"] = None
                    runtime["scheduler"] = None
        except Exception as e:
            self.logger.error("MyRSS: 停止调度器失败: %s", e)

    def _get_lock(self, url: str, user: str) -> asyncio.Lock:
        key = (url, user)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _cron(self, expr: str) -> dict:
        f = expr.split(" ")
        return {"minute": f[0], "hour": f[1], "day": f[2], "month": f[3], "day_of_week": f[4]}

    def _reload_jobs(self) -> None:
        self.sched.remove_all_jobs()
        for url, info in self.dh.data.items():
            if url in ("rsshub_endpoints", "settings"):
                continue
            subs = info.get("subscribers", {})
            if not subs:
                continue
            # 取所有订阅者中间隔最大的cron（最保守，减少拉取频率）
            def cron_to_minutes(expr: str) -> int:
                """支持 */15 * * * * 和 0 */1 * * * 两种格式"""
                try:
                    f = expr.split(" ")
                    if f[0].startswith("*/"):
                        return int(f[0][2:])
                    if f[1].startswith("*/"):
                        return int(f[1][2:]) * 60
                    return 60
                except Exception:
                    return 60

            max_minutes = max(cron_to_minutes(si["cron_expr"]) for si in subs.values())

            if max_minutes < 60:
                merged_cron = f"*/{max_minutes} * * * *"
            else:
                merged_cron = f"0 */{max_minutes // 60} * * *"
            # 每个URL只注册一个job，拉取后分发给所有订阅者
            # [防冲突] id + replace_existing 保证同一个url在调度器里只有一个job
            # 没有id时APScheduler会自动生成随机id，reload_jobs就无法识别"已存在"
            # replace_existing=True：如果id已存在就替换而非报错，适配热更新场景
            # misfire_grace_time=120：job错过触发时间后120秒内还可以补执行，超时则跳过
            # 防止调度器shutdown/restart期间堆积的job全部同时涌入
            job_id = f"myrss_{url}"
            self.sched.add_job(
                self._cron_cb_url, "cron",
                **self._cron(merged_cron),
                args=[url],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=120,
                jitter=30,
            )
            if max_minutes < 60:
                self.logger.info("RSS调度: %s 每%d分钟拉取，%d个订阅者", url, max_minutes, len(subs))
            else:
                self.logger.info("RSS调度: %s 每%d小时拉取，%d个订阅者", url, max_minutes // 60, len(subs))

    async def _fetch(self, url: str):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        to = aiohttp.ClientTimeout(total=30, connect=10)

        async def _try(u: str):
            try:
                conn = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(trust_env=True, connector=conn, timeout=to, headers=headers) as s:
                    async with s.get(u) as r:
                        if r.status != 200:
                            self.logger.warning(f"[MyRSS] HTTP {r.status} when fetching {u}")
                            self._last_fetch_error = f"HTTP {r.status} from {u}"
                            return None
                        return await r.read()
            except Exception as e:
                self.logger.error(f"[MyRSS] Error fetching {u}: {e}")
                self._last_fetch_error = f"{type(e).__name__}: {e}"
                return None

        data = None
        for attempt in range(3):
            data = await _try(url)
            if data is not None:
                # 检查是否返回了HTML错误页而非XML (exact from old working commit)
                if data[:5] == b'<?xml' or (data[:1] == b'<' and b'<item>' in data[:5000]):
                    return data
                if b'<html>' not in data[:500].lower():
                    return data
                # 拿到HTML错误页，等几秒重试（等RSSHub内部缓存刷新）
                if attempt < 2:
                    await asyncio.sleep(3)
                    continue
                return data  # 第3次不管什么都返回
            if attempt < 2:
                await asyncio.sleep(3)

        if data is not None:
            return data

        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            self._last_fetch_error = "No rsshub_endpoints configured"
            return None

        parsed = urlparse(url)
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        cur_root = f"{parsed.scheme}://{parsed.netloc}"
        norm_eps = [(e[:-1] if e.endswith("/") else e) for e in eps]

        for ep in norm_eps:
            if ep == cur_root:
                continue
            alt = ep + path
            data = await _try(alt)
            if data is not None:
                self.logger.warning("rss: 端点不可用，已自动切换 %s -> %s", url, alt)
                return data

        return None

    def _parse_pubdate(self, pd: str):
        """解析各种日期格式，失败返回None"""
        if not pd:
            return None
        pd = pd.strip()

        # 优先用标准库的RFC2822解析器（最稳，不受locale影响）
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pd)
            return int(dt.timestamp())
        except Exception:
            pass

        # 补充ISO8601等格式
        pd_clean = pd.replace("GMT", "+0000").replace("Z", "+0000")
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(pd_clean, fmt)
                return int(dt.timestamp())
            except Exception:
                continue
        return None

    async def _poll(self, url: str, num: int = -1, after_ts: int = 0, after_link: str = "") -> List[RSSItem]:
        text = await self._fetch(url)
        if text is None:
            return []
        try:
            root = etree.fromstring(text)
        except Exception:
            try:
                root = etree.fromstring(
                    text.replace(b'encoding="gb2312"', b'')
                    .replace(b'encoding="GB2312"', b'')
                )
            except Exception as e:
                self.logger.error(f"[MyRSS] XML syntax error when parsing feed from {url}: {e}")
                return []

        items = root.xpath("//item")
        ns = {"media": "http://search.yahoo.com/mrss/"}
        result = []
        cnt = 0

        for it in items:
            try:
                ch = self.dh.data[url]["info"]["title"] if url in self.dh.data else "未知"

                tn = it.xpath("title")
                title = tn[0].text if tn else "无标题"
                if len(title) > self.title_max:
                    title = title[:self.title_max] + "..."

                ln = it.xpath("link")
                link = (ln[0].text or "").strip() if ln else ""
                if link and not re.match(r"^https?://", link):
                    link = self.dh.get_root_url(url) + link

                dn = it.xpath("description")
                raw = dn[0].text if dn else ""
                pics = self.dh.strip_html_pic(raw) if raw else []
                desc = self.dh.strip_html(raw) if raw else ""
                if len(desc) > self.desc_max:
                    desc = desc[:self.desc_max] + "..."

                # [增强] 从多种XML标签提取图片URL
                # media:thumbnail → RSS标准缩略图（视频路由常用）
                # media:content → 有些源把封面图放这里（YouTube等）
                # enclosure → RSS附件
                # local-name()通配 → 兼容不同命名空间写法
                for u in (
                    it.xpath("media:thumbnail/@url", namespaces=ns)
                    + it.xpath("media:content/@url", namespaces=ns)
                    + it.xpath("media:content/media:thumbnail/@url", namespaces=ns)
                    + it.xpath(".//*[local-name()='thumbnail']/@url")
                    + it.xpath(".//*[local-name()='content']/@url")
                    + it.xpath("enclosure[contains(@type,'image')]/@url")
                    + it.xpath("enclosure/@url")
                ):
                    if u and u not in pics:
                        # 过滤掉视频/音频文件，只保留图片
                        low = u.lower()
                        if not any(low.endswith(e) for e in ('.mp4', '.webm', '.mp3', '.m4a', '.ogg')):
                            pics.append(u)

                pub_nodes = it.xpath("pubDate")
                if pub_nodes:
                    pd = pub_nodes[0].text or ""
                    pts = self._parse_pubdate(pd)

                    if pts is None:
                        # 解析失败，用 link 兜底去重
                        if link and link != after_link:
                            result.append(RSSItem(ch, title, link, desc, pd, 0, pics))
                            cnt += 1
                    elif pts > after_ts:
                        result.append(RSSItem(ch, title, link, desc, pd, pts, pics))
                        cnt += 1
                    else:
                        break
                else:
                    if link and link != after_link:
                        result.append(RSSItem(ch, title, link, desc, "", 0, pics))
                        cnt += 1
                    else:
                        break

                if num != -1 and cnt >= num:
                    break

            except Exception as e:
                self.logger.error("rss: 解析条目失败 %s: %s", url, e)
                break

        return result

    async def _add(self, url: str, cron_expr: str, event: AstrMessageEvent, target_user: str = None):
        user = target_user if target_user else event.unified_msg_origin

        async def poll_with_retry(u: str, retries: int = 3, sleep_s: int = 5):
            last = []
            for i in range(retries):
                last = await self._poll(u)
                if last:
                    return last
                if i < retries - 1:
                    await asyncio.sleep(sleep_s)
            return last

        # 已存在订阅源：只加订阅者
        if url in self.dh.data:
            items = await poll_with_retry(url)
            if not items:
                return event.plain_result("连续3次无法从该源获取内容，源可能暂时不可用，请稍后重试。")

            self.dh.data[url].setdefault("subscribers", {})
            self.dh.data[url]["subscribers"][user] = {
                "cron_expr": cron_expr,
                "last_update": items[0].pubDate_timestamp,
                "latest_link": items[0].link,
                "seen_links": [it.link for it in items if it.link][:200],
            }

            self.dh.save()
            return self.dh.data[url]["info"]

        # 新订阅源：先解析频道信息
        text = await self._fetch(url)
        if text is None:
            return event.plain_result("无法访问: " + url + "\\n请检查RSSHub端点是否可用。")

        try:
            title, desc, avatar = self.dh.parse_channel_info(text)
        except Exception as e:
            return event.plain_result("解析失败: " + str(e))

        items = await poll_with_retry(url)
        if not items:
            return event.plain_result("源可访问但连续3次获取不到内容，可能是该平台接口不稳定，请稍后重试。")

        self.dh.data[url] = {
            "subscribers": {
                user: {
                    "cron_expr": cron_expr,
                    "last_update": items[0].pubDate_timestamp,
                    "latest_link": items[0].link,
                    "seen_links": [it.link for it in items if it.link][:200],
                }
            },
            "info": {"title": title, "description": desc, "avatar": avatar},
        }
        self.dh.save()
        return self.dh.data[url]["info"]

    async def _get_provider_id(self) -> str:
        """获取锐评用的provider ID"""
        if self.comment_provider_id:
            return self.comment_provider_id
        # 自动获取默认provider
        try:
            cfg = self.ctx.get_config()
            default_id = cfg.get("provider_settings", {}).get("default_provider_id", "")
            if default_id:
                return default_id
        except Exception:
            pass
        return ""

    async def _generate_comment(self, item: RSSItem) -> str:
        """调用LLM生成锐评，带缓存"""
        norm_link = item.link.split("#", 1)[0].split("?", 1)[0] if item.link else ""
        cache_key = norm_link or (item.title + "|" + str(item.pubDate_timestamp))

        # 命中缓存直接返回
        if cache_key in self._comment_cache:
            return self._comment_cache[cache_key]

        provider_id = self.comment_provider_id if self.comment_provider_id else await self._get_provider_id()
        if not provider_id:
            self.logger.warning("[MyRSS] no provider for comment")
            return ""

        # 构造prompt
        content_summary = item.title
        if item.description:
            desc_short = item.description[:200]
            content_summary += "\\n" + desc_short
        vision = await self._analyze_item_images(item)
        if vision.get("status") == "SAFE" and vision.get("description"):
            content_summary += "\\n图片内容：" + vision["description"][:600]

        # 获取人格设定（v4 正统：PersonaManager）
        system_prompt = None
        if self.comment_persona:
            try:
                persona = self.ctx.persona_manager.get_persona(self.comment_persona)
                if persona:
                    system_prompt = persona.system_prompt
            except Exception:
                system_prompt = None

        prompt = (
            f"你正在看一条来自「{item.chan_title}」的动态更新，内容如下：\\n"
            f"---\\n{content_summary}\\n---\\n"
            f"请用你的人设风格，对这条动态写一句简短锐评（{self.comment_max_length}字以内）。"
            f"要求：自然、有个性、可以吐槽或夸奖。不要加引号。如果是推特消息和外语，尽可能通俗易懂的转为中文并简单讲讲发生了什么。"
        )

        try:
            resp = await self.ctx.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            comment = (resp.completion_text or "").strip()
            # 截断
            if len(comment) > self.comment_max_length:
                comment = comment[:self.comment_max_length] + "..."
            # 过滤锐评本身
            if self.content_filter and comment:
                unsafe_words = ["习近平", "共产党", "六四", "天安门", "法轮", "台独",
                                "藏独", "疆独", "反共", "颠覆", "推翻", "操你", "傻逼"]
                for w in unsafe_words:
                    if w in comment:
                        self.logger.warning("[MyRSS] comment contains unsafe word '%s', discarding", w)
                        comment = ""
                        break
            # 缓存
            if comment:
                self._comment_cache[cache_key] = comment
            # 限制缓存大小
            if len(self._comment_cache) > 500:
                keys = list(self._comment_cache.keys())
                for k in keys[:200]:
                    del self._comment_cache[k]
            return comment
        except Exception as e:
            self.logger.error("[MyRSS] comment generation failed: %s", e)
            return ""

    def _item_cache_key(self, item: RSSItem) -> str:
        return (item.link.split("#", 1)[0].split("?", 1)[0] if item.link else "") or f"{item.title}|{item.pubDate_timestamp}"

    def _build_contact_sheet(self, images: list[bytes]) -> bytes:
        """把同一动态的图片本地融合为一张带编号联系表，不调用外部服务。"""
        count = len(images)
        if count <= 0:
            return b""
        if count == 1:
            cols, rows = 1, 1
        elif count == 2:
            cols, rows = 2, 1
        elif count <= 4:
            cols, rows = 2, 2
        elif count <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 3, 3
        cell = 512
        canvas = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for index, raw in enumerate(images):
            with Image.open(BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell - 16, cell - 16), Image.LANCZOS)
                x0 = (index % cols) * cell
                y0 = (index // cols) * cell
                x = x0 + (cell - image.width) // 2
                y = y0 + (cell - image.height) // 2
                canvas.paste(image, (x, y))
                draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), outline=(190, 190, 190), width=2)
                draw.rectangle((x0 + 8, y0 + 8, x0 + 70, y0 + 42), fill=(0, 0, 0))
                draw.text((x0 + 16, y0 + 13), f"IMG {index + 1}", fill=(255, 255, 255))
        out = BytesIO()
        canvas.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()

    async def _analyze_item_images(self, item: RSSItem) -> dict:
        """全部图片融合后只调用一次多模态 LLM；任何失败状态也缓存，禁止反复调用。"""
        key = self._item_cache_key(item)
        if key in self._vision_cache:
            return self._vision_cache[key]
        if not item.pic_urls:
            result = {"status": "NO_IMAGE", "description": "", "image_count": 0}
            self._vision_cache[key] = result
            return result
        if len(item.pic_urls) > self.max_vision_images:
            result = {"status": "REJECT", "description": "图片数量超过安全审核上限", "image_count": len(item.pic_urls)}
            self._vision_cache[key] = result
            return result

        downloaded = []
        try:
            async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=False)) as session:
                for image_url in item.pic_urls:
                    try:
                        async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                            raw = await resp.read() if resp.status == 200 else b""
                        if len(raw) <= 100:
                            raise ValueError(f"empty image HTTP {resp.status}")
                        # 在融合前解码一次，拒绝伪图片或损坏文件。
                        with Image.open(BytesIO(raw)) as check:
                            check.verify()
                        downloaded.append(raw)
                    except Exception as exc:
                        # RSS 常同时给出候选封面（例如 YouTube maxres 404 + hq 可用），
                        # 单个候选失败只跳过；最终一张都拿不到时才 fail closed。
                        self.logger.warning("[MyRSS] image candidate download/decode failed, skip: %s", exc)
                        continue
            if not downloaded:
                result = {"status": "REJECT", "description": "所有图片候选均无法下载或解码", "image_count": 0}
                self._vision_cache[key] = result
                return result
            contact_sheet = self._build_contact_sheet(downloaded)
        except Exception as exc:
            self.logger.error("[MyRSS] contact sheet failed: %s", exc)
            result = {"status": "REJECT", "description": "图片融合失败", "image_count": len(item.pic_urls)}
            self._vision_cache[key] = result
            return result

        provider_id = self.image_caption_provider_id or self.filter_provider_id or await self._get_provider_id()
        if not provider_id:
            result = {"status": "REJECT", "description": "没有可用的多模态审核模型", "image_count": len(downloaded)}
            self._vision_cache[key] = result
            return result
        data_uri = "data:image/jpeg;base64," + base64.b64encode(contact_sheet).decode()
        prompt = (
            f"你正在审核一张由同一条动态的 {len(downloaded)} 张原图按编号融合而成的联系表。"
            "请查看所有编号区域，概括人物、场景、可读文字及图片之间的关系。"
            "若任何区域含政治敏感、色情、暴力、血腥、仇恨、违法内容，或小到无法可靠判断，必须拒绝。"
            "只返回严格 JSON，不要代码块："
            '{"status":"SAFE|REJECT|MALICIOUS|UNCERTAIN","description":"完整中文描述","uncertain":false}'
        )
        try:
            # 唯一的视觉 LLM 调用点；不在本函数内重试。
            response = await self.ctx.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=[data_uri],
                request_max_retries=1,
            )
            text = (response.completion_text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
            parsed = json.loads(text)
            status = str(parsed.get("status", "UNCERTAIN")).upper()
            description = str(parsed.get("description", "")).strip()
            uncertain = bool(parsed.get("uncertain", False))
            if status not in {"SAFE", "REJECT", "MALICIOUS", "UNCERTAIN"} or not description:
                raise ValueError("invalid vision JSON fields")
            if uncertain or status == "UNCERTAIN":
                status = "REJECT"
            result = {"status": status, "description": description, "image_count": len(downloaded)}
        except Exception as exc:
            self.logger.error("[MyRSS] single vision call failed; reject and cache: %s", exc)
            result = {"status": "REJECT", "description": "多模态识图失败", "image_count": len(downloaded)}
        self._vision_cache[key] = result
        if len(self._vision_cache) > 500:
            for old_key in list(self._vision_cache)[:200]:
                del self._vision_cache[old_key]
        return result

    async def _check_content_safe(self, item: RSSItem) -> str:
        """
        增强版内容审核（三种状态）：
        返回值：
          - "SAFE"       正常内容
          - "REJECT"     驳回（不严重但可能导致 bot 封号的内容）
          - "MALICIOUS"  违规（故意抹黑、恶俗政治、色情暴力等）
        """
        # 安全审核不可通过配置绕过，也不按平台免审。
        norm_link = item.link.split("#", 1)[0].split("?", 1)[0] if item.link else ""
        cache_key = norm_link or (item.title + "|" + str(item.pubDate_timestamp))
        if cache_key in self._safe_cache:
            cached = self._safe_cache[cache_key]
            if isinstance(cached, str):
                return cached
            return "SAFE" if cached else "REJECT"

        # 硬编码关键词兜底
        check_text = (item.title + " " + (item.description or "")).lower()
        malicious_keywords = [
            "习近平", "共产党", "六四", "天安门", "法轮", "台独", "藏独", "疆独",
            "反共", "颠覆", "推翻政权", "轮子功", "习近平", "毛泽东", "共产主义",
            "色情", "裸露", "暴力", "血腥", "虐待", "毒品", "赌博"
        ]
        for w in malicious_keywords:
            if w in check_text:
                self.logger.warning("[MyRSS] MALICIOUS hard-filter hit '%s': %s", w, item.title[:30])
                self._safe_cache[cache_key] = "MALICIOUS"
                return "MALICIOUS"

        vision = await self._analyze_item_images(item)
        if vision.get("status") in {"REJECT", "MALICIOUS"}:
            status = vision["status"]
            self.logger.warning("[MyRSS] image review %s: %s", status, vision.get("description", "")[:80])
            self._safe_cache[cache_key] = status
            return status

        provider_id = self.filter_provider_id if self.filter_provider_id else await self._get_provider_id()
        if not provider_id:
            self.logger.error("[MyRSS] no provider for mandatory content review; reject by default")
            self._safe_cache[cache_key] = "REJECT"
            return "REJECT"

        image_description = vision.get("description", "")
        content = (item.title + " " + (item.description or "") + "\n图片内容：" + image_description)[:1200]

        # 增强提示词：三种状态
        prompt = (
            "你是内容安全审核员。请严格审核以下内容，返回三种状态之一：\n\n"
            "1. SAFE      —— 正常内容（科技、娱乐、商业、日常等）\n"
            "2. REJECT    —— 可能导致 bot 封号的内容（轻微政治敏感、争议话题、边缘擦边球）\n"
            "3. MALICIOUS —— 绝对违规内容（故意抹黑中国政治、恶俗政治攻击、色情暴力、恐怖主义等）\n\n"
            "判断标准：\n"
            "- 只要涉及中国政治领导人（习近平等）、六四、天安门、台独、反共、颠覆等 → MALICIOUS\n"
            "- 涉及敏感政治但不极端 → REJECT\n"
            "- 正常内容 → SAFE\n\n"
            f"待审核内容：\n{content}\n\n"
            "请只返回 SAFE / REJECT / MALICIOUS 其中一个词，不要解释。"
        )

        try:
            resp = await self.ctx.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            result = (resp.completion_text or "").strip().upper()
            if "MALICIOUS" in result:
                status = "MALICIOUS"
            elif "REJECT" in result:
                status = "REJECT"
            else:
                status = "SAFE"

            self._safe_cache[cache_key] = status
            if status != "SAFE":
                self.logger.warning("[MyRSS] content %s: %s", status, item.title[:50])
            return status
        except Exception as e:
            self.logger.error("[MyRSS] content filter failed, treat as REJECT: %s", e)
            self._safe_cache[cache_key] = "REJECT"
            return "REJECT"

    def _record_safety_event(self, url: str, item: RSSItem, status: str, reason: str = "") -> None:
        """记录不含违规正文/图片的安全事件；同一源同一动态只保留一条。"""
        if status not in ("REJECT", "MALICIOUS"):
            return
        item_key = self._item_cache_key(item)
        event_id = hashlib.sha256(f"{url}|{item_key}|{status}".encode()).hexdigest()[:16]
        settings = self.dh.data.setdefault("settings", {})
        events = settings.setdefault("safety_events", [])
        if any(isinstance(event, dict) and event.get("id") == event_id for event in events):
            return
        events.insert(0, {
            "id": event_id,
            "status": status,
            "source": item.chan_title or self.dh.data.get(url, {}).get("info", {}).get("title", "未知源"),
            "reason": (reason or "综合内容审核未通过")[:160],
            "blocked_at": int(time.time()),
            "content_fingerprint": hashlib.sha256(item_key.encode()).hexdigest()[:12],
        })
        settings["safety_events"] = events[:100]

    def _store_safe_item_preview(self, url: str, item: RSSItem) -> None:
        """保存 UI 所需的最新安全内容，不增加抓取或 LLM 调用。"""
        feed = self.dh.data.get(url)
        if not isinstance(feed, dict):
            return
        feed["last_item_preview"] = {
            "title": (item.title or "")[:160],
            "description": (item.description or "")[:500],
            "link": item.link or "",
            "pub_date": item.pubDate or "",
            "pub_timestamp": int(item.pubDate_timestamp or 0),
            "image_url": item.pic_urls[0] if item.pic_urls else "",
            "safety_status": "SAFE",
            "updated_at": int(time.time()),
        }

    def _get_avatar_url(self, item: RSSItem) -> str:
        """从存储的订阅数据里获取频道头像URL"""
        for url, info in self.dh.data.items():
            if url in ("rsshub_endpoints", "settings"):
                continue
            if info.get("info", {}).get("title") == item.chan_title:
                return info.get("info", {}).get("avatar", "")
        return ""

    async def _make_card_b64(self, item: RSSItem) -> str:
        # 下载频道头像
        avt_data = None
        if item.chan_title and item.chan_title != "未知":
            avt_url = self._get_avatar_url(item)
            if avt_url:
                try:
                    conn = aiohttp.TCPConnector(ssl=False)
                    async with aiohttp.ClientSession(trust_env=True, connector=conn) as s:
                        async with s.get(avt_url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                            if r.status == 200:
                                avt_data = await r.read()
                except Exception:
                    pass
        tb = None
        if self.read_pic and item.pic_urls:
            # [修改] 遍历图片列表尝试下载，直到成功一个
            # 解决YouTube封面可能是404的问题
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(trust_env=True, connector=conn) as s:
                for pu in item.pic_urls:
                    try:
                        async with s.get(pu, timeout=aiohttp.ClientTimeout(total=5)) as r:
                            if r.status == 200:
                                data = await r.read()
                                # 简单校验数据长度，防止下载到空的
                                if len(data) > 100:
                                    tb = data
                                    break
                    except Exception:
                        continue
        # 生成锐评
        comment = ""
        bot_avt = None
        if self.enable_comment:
            comment = await self._generate_comment(item)

        # 下载bot头像
        if self.bot_qq and comment:
            bot_avt_url = f"https://q1.qlogo.cn/g?b=qq&nk={self.bot_qq}&s=640"
            try:
                conn3 = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(trust_env=True, connector=conn3) as s3:
                    async with s3.get(bot_avt_url, timeout=aiohttp.ClientTimeout(total=5)) as r3:
                        if r3.status == 200:
                            bot_avt = await r3.read()
            except Exception:
                pass

        return await self.card.make(
            channel=item.chan_title,
            title=item.title,
            desc=item.description,
            link="" if self.hide_url else item.link,
            ts=item.pubDate or "",
            thumb=tb,
            avatar=avt_data,
            comment=comment,
            bot_avatar=bot_avt,
            bot_provider_name=self.bot_provider_name,
        )

    def _merge_cards_b64(self, cards_b64: list) -> str:
        imgs = []
        for b64 in cards_b64:
            raw = base64.b64decode(b64)
            imgs.append(Image.open(BytesIO(raw)).convert("RGB"))

        if not imgs:
            return ""

        width = max(im.width for im in imgs)
        # [修改] 间距设为0，让每条卡片底部自带的分割线直接充当
        # 条目之间的分隔，拼出来就像推特时间线一样无缝衔接
        pad = 0
        resized = []
        total_h = 0
        for im in imgs:
            if im.width != width:
                nh = int(im.height * (width / im.width))
                im = im.resize((width, nh), Image.LANCZOS)
            resized.append(im)
            total_h += im.height

        # [修改] 白底画布，间距为0紧密拼接
        canvas = Image.new("RGB", (width, total_h), (255, 255, 255))
        y = 0
        for im in resized:
            canvas.paste(im, (0, y))
            y += im.height

        out = BytesIO()
        canvas.save(out, format="PNG")
        out.seek(0)
        return base64.b64encode(out.read()).decode("utf-8")

    async def _make_comps(self, item: RSSItem, include_extra_images: bool = True) -> list:
        comps = []
        tb = None
        if self.read_pic and item.pic_urls:
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(trust_env=True, connector=conn) as s:
                for pu in item.pic_urls:
                    try:
                        async with s.get(pu, timeout=aiohttp.ClientTimeout(total=5)) as r:
                            if r.status == 200:
                                data = await r.read()
                                if len(data) > 100:
                                    tb = data
                                    break
                    except Exception:
                        continue
        # 下载频道头像
        avt_data = None
        avt_url = self._get_avatar_url(item)
        if avt_url:
            try:
                conn2 = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(trust_env=True, connector=conn2) as s2:
                    async with s2.get(avt_url, timeout=aiohttp.ClientTimeout(total=5)) as r2:
                        if r2.status == 200:
                            avt_data = await r2.read()
            except Exception:
                pass
        # 生成锐评
        comment = ""
        bot_avt = None
        if self.enable_comment:
            comment = await self._generate_comment(item)

        if self.bot_qq and comment:
            bot_avt_url = f"https://q1.qlogo.cn/g?b=qq&nk={self.bot_qq}&s=640"
            try:
                conn3 = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(trust_env=True, connector=conn3) as s3:
                    async with s3.get(bot_avt_url, timeout=aiohttp.ClientTimeout(total=5)) as r3:
                        if r3.status == 200:
                            bot_avt = await r3.read()
            except Exception:
                pass

        try:
            b64 = await self.card.make(
                channel=item.chan_title, title=item.title, desc=item.description,
                link="" if self.hide_url else item.link, ts=item.pubDate or "", thumb=tb,
                avatar=avt_data,
                comment=comment,
                bot_avatar=bot_avt,
                bot_provider_name=self.bot_provider_name,
            )
            if b64:
                comps.append(Comp.Image.fromBase64(b64))
            else:
                comps.append(Comp.Plain("📡 " + item.chan_title + "\\n📝 " + item.title + "\\n" + item.description))
        except Exception as e:
            self.logger.error("卡片生成失败: %s", e)
            comps.append(Comp.Plain("📡 " + item.chan_title + "\\n📝 " + item.title + "\\n" + item.description))

        if include_extra_images and self.read_pic and item.pic_urls:
            mx = len(item.pic_urls) if self.max_pic == -1 else self.max_pic
            for pu in item.pic_urls[1:mx]:
                try:
                    b = await self.pic.to_base64(pu)
                    if b:
                        comps.append(Comp.Image.fromBase64(b))
                except Exception:
                    pass
        return comps

    async def _cron_cb_url(self, url: str) -> None:
        """每个URL只拉取一次，结果分发给所有订阅者"""
        if not self._is_current_runtime():
            self.logger.warning("RSS跳过失效插件实例: instance=%s url=%s", id(self), url)
            return
        # [诊断] 打印实例ID和调度器ID，如果日志里同一url出现两个不同的id就是双实例并行
        self.logger.info("RSS拉取开始: instance=%s sched=%s url=%s", id(self), id(self.sched), url)
        if url not in self.dh.data:
            return
        subs = self.dh.data[url].get("subscribers", {})
        if not subs:
            return
        ready_users = []
        for user in list(subs):
            ready, reason = self._target_readiness(user)
            if ready:
                ready_users.append(user)
            else:
                self.logger.warning("RSS目标未就绪，拉取前跳过: target=%s reason=%s", user, reason)
        if not ready_users:
            return

        self.logger.info("RSS公共拉取: %s -> %d/%d个目标已就绪", url, len(ready_users), len(subs))

        # 只根据可投递目标计算断点；离线旧平台不消耗 RSS/LLM。
        min_ts = min(subs[user].get("last_update", 0) for user in ready_users)
        min_link = "" # 公共拉取不用after_link过滤，靠seen_links去重

        items = await self._poll(url, num=self.max_poll, after_ts=min_ts, after_link=min_link)
        if not items:
            return

        # 分发给每个订阅者（各自独立去重）
        for i, user in enumerate(ready_users):
            if not self._is_current_runtime():
                self.logger.warning("RSS分发中止：插件实例已被重载替换")
                return
            ready, reason = self._target_readiness(user)
            if not ready:
                self.logger.warning("RSS分发前目标失去就绪状态: target=%s reason=%s", user, reason)
                continue
            lock = self._get_lock(url, user)
            async with lock:
                await self._cron_cb_inner(url, user, prefetched_items=items)
            # 多个订阅者间随机延迟防风控
            if i < len(ready_users) - 1:
                delay = random.uniform(self.push_delay_min, self.push_delay_max)
                await asyncio.sleep(delay)

    async def _cron_cb_inner(self, url: str, user: str, prefetched_items=None) -> None:
        # 必须覆盖“读库→更新 seen_links→保存”整个事务。
        # 旧代码只锁住 _load，另一个 job 随后替换 dh.data，会让前一个 job 保存错对象。
        async with self._data_lock:
            await self._cron_cb_inner_impl(url, user, prefetched_items)

    async def _cron_cb_inner_impl(self, url: str, user: str, prefetched_items=None) -> None:
        self.dh.data = self.dh._load()

        if url not in self.dh.data or user not in self.dh.data[url].get("subscribers", {}):
            return

        self.logger.info("RSS定时触发: %s -> %s", url, user)
        si = self.dh.data[url]["subscribers"][user]

        if prefetched_items is not None:
            # 使用公共拉取的结果，再按该用户的断点过滤一次
            items = [
                it for it in prefetched_items
                if it.pubDate_timestamp > si.get("last_update", 0)
                or (it.pubDate_timestamp == 0 and it.link != si.get("latest_link", ""))
            ]
        else:
            items = await self._poll(
                url,
                num=self.max_poll,
                after_ts=si["last_update"],
                after_link=si["latest_link"],
            )
        if not items:
            # [修复] 无新内容也更新last_update到当前时间，防止seen_links被时间清理后重推
            si["last_update"] = int(time.time())
            self.dh.save()
            return

        def item_key(it: RSSItem) -> str:
            if it.link:
                return it.link.split("#", 1)[0].split("?", 1)[0]
            return f"{it.title}|{it.pubDate_timestamp}"

        # 去重
        seen = set(si.get("seen_links", []))
        new_items = [it for it in items if item_key(it) not in seen]

        if not new_items:
            # [修复] 即使没有新条目，也更新last_update，防止seen_links被时间清理后重推
            si["latest_link"] = items[0].link
            si["last_update"] = max(si.get("last_update", 0), int(time.time()))
            self.dh.save()
            return

        # 先更新去重记录再发送，防止并发重推
        new_keys = [item_key(it) for it in new_items]
        si["seen_links"] = (new_keys + si.get("seen_links", []))[:200]
        si["latest_link"] = items[0].link
        ts_candidates = [it.pubDate_timestamp for it in new_items if it.pubDate_timestamp > 0]
        if ts_candidates:
            si["last_update"] = max(ts_candidates)
        self.dh.save()
        # 内容过滤（增强版）
        if self.content_filter:
            filtered = []
            metadata_changed = False
            for it in new_items:
                status = await self._check_content_safe(it)
                if status == "SAFE":
                    filtered.append(it)
                else:
                    vision = self._vision_cache.get(self._item_cache_key(it), {})
                    vision_status = vision.get("status") if isinstance(vision, dict) else None
                    reason = (
                        "图片审核未通过或无法可靠识别"
                        if vision_status in ("REJECT", "MALICIOUS")
                        else "正文与图片综合审核未通过"
                    )
                    self._record_safety_event(url, it, status, reason)
                    metadata_changed = True
                    if status == "MALICIOUS":
                        self.logger.warning("[MyRSS] MALICIOUS filtered: %s", it.title[:30])
                    else:
                        self.logger.info("[MyRSS] REJECT filtered: %s", it.title[:30])
            new_items = filtered
            if new_items:
                self._store_safe_item_preview(url, new_items[0])
                metadata_changed = True
            if metadata_changed:
                self.dh.save()
        if not new_items:
            return
        pn = user.split(":")[0]
        official_target = self._is_qq_official_origin(user)
        merge_limit = 5
        batch = new_items[:merge_limit]

        send_ok = True
        if len(batch) > 1:
            cards_raw = [await self._make_card_b64(it) for it in batch]
            cards = [c for c in cards_raw if c] # 过滤掉被内容审核拦截的空卡片
            if not cards:
                self.logger.info("[MyRSS] all items filtered, skip push")
                return
            merged = self._merge_cards_b64(cards)
            if not merged:
                for it in batch:
                    comps = await self._make_comps(it, include_extra_images=not official_target)
                    result = await self._send_message_guarded(user, MessageChain(chain=comps, use_t2i_=self.t2i))
                    send_ok = send_ok and bool(result)
            else:
                comps = [Comp.Image.fromBase64(merged)]
                if pn == "aiocqhttp" and self.compose:
                    node = Comp.Node(uin=0, name="Astrbot", content=comps)
                    result = await self._send_message_guarded(user, MessageChain(chain=[node], use_t2i_=self.t2i))
                else:
                    result = await self._send_message_guarded(user, MessageChain(chain=comps, use_t2i_=self.t2i))
                send_ok = bool(result)
        else:
            it = batch[0]
            comps = await self._make_comps(it, include_extra_images=not official_target)
            if pn == "aiocqhttp" and self.compose:
                node = Comp.Node(uin=0, name="Astrbot", content=comps)
                result = await self._send_message_guarded(user, MessageChain(chain=[node], use_t2i_=self.t2i))
            else:
                result = await self._send_message_guarded(user, MessageChain(chain=comps, use_t2i_=self.t2i))
            send_ok = bool(result)

        if send_ok:
            self._record_delivery_status(url, user, "SUCCESS")
            self.logger.info("RSS推送完成: %s -> %s (%d条)", url, user, len(batch))
        else:
            self._record_delivery_status(url, user, "FAILED", "SEND_RETURNED_FALSE")
            self.logger.error("RSS投递失败（seen_links 已保留以防循环重推）: %s -> %s (%d条)", url, user, len(batch))
        self.dh.save()

        # ============================================================
        # LLM 工具
        # ============================================================
    # ============================================================
    # 手动命令
    # ============================================================

    def _resolve_feed_url(self, value: str):
        """把路由或已知平台链接解析为 (full_url, route, platform)。"""
        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            return None, None, "未配置 RSSHub 端点"
        value = (value or "").strip()
        if value.startswith("/"):
            route, platform = value, "RSSHub"
        elif value.startswith("http"):
            matched = URLMapper.match(value)
            if not matched:
                return None, None, URLMapper.suggest(value)
            route, platform = matched
        else:
            return None, None, "请提供 http 开头的链接或 / 开头的 RSSHub 路由"
        return eps[0].rstrip("/") + route, route, platform

    async def _make_safe_preview_card(self, item: RSSItem) -> str:
        """预览专用卡片：不调用锐评 LLM，只下载头像和首张图片。"""
        avatar_data = None
        avatar_url = self._get_avatar_url(item)
        if avatar_url:
            try:
                async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=False)) as session:
                    async with session.get(avatar_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            avatar_data = await resp.read()
            except Exception:
                pass
        thumb_data = None
        if self.read_pic:
            for image_url in item.pic_urls[:3]:
                try:
                    async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=False)) as session:
                        async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            data = await resp.read() if resp.status == 200 else b""
                            if len(data) > 100:
                                thumb_data = data
                                break
                except Exception:
                    continue
        return await self.card.make(
            channel=item.chan_title, title=item.title, desc=item.description,
            link="" if self.hide_url else item.link, ts=item.pubDate or "",
            thumb=thumb_data, avatar=avatar_data, comment="",
            bot_avatar=None, bot_provider_name="",
        )

    def _explicit_subscribe_confirmation(self, event: AstrMessageEvent) -> bool:
        """以用户原始消息为准，不信任 LLM 自行填写的 confirm 参数。"""
        text = re.sub(r"[\s，,。.!！?？:：]", "", getattr(event, "message_str", "") or "")
        return any(word in text for word in ("确认订阅", "确认关注", "确认添加订阅", "认订阅"))

    def _target_group_from_request(self, event: AstrMessageEvent, target_group: str = "") -> str:
        """优先使用工具参数；缺失时从用户原始消息提取明确群号。"""
        if str(target_group or "").strip().isdigit():
            return str(target_group).strip()
        text = getattr(event, "message_str", "") or ""
        match = re.search(r"(?:群号|群)\s*[:：]?\s*(\d{5,})", text)
        return match.group(1) if match else ""

    def _latest_preview_state(self, origin: str, preview_id: str = ""):
        """按当前会话取预览；LLM 抄错/漏填编号时回退到该会话最新状态。"""
        exact = self._preview_states.get((origin, preview_id)) if preview_id else None
        if exact:
            return exact
        candidates = [v for (state_origin, _), v in self._preview_states.items() if state_origin == origin]
        return max(candidates, key=lambda x: x.get("created_at", 0)) if candidates else None

    def _subscription_target(self, event: AstrMessageEvent, target_group: str):
        """返回 (target_origin, error)。跨群只能由 AstrBot 管理员操作。"""
        origin = event.unified_msg_origin
        if not target_group:
            return origin, ""
        current_gid = origin.split(":")[-1] if "GroupMessage" in origin else ""
        if target_group != current_gid and not self._is_astrbot_admin(event):
            return "", "⚠️ 指定其他群仅 AstrBot 管理员可用。"
        return f"{origin.split(':')[0]}:GroupMessage:{target_group}", ""

    async def _confirm_preview_subscription(self, event: AstrMessageEvent, state: dict, target_group: str):
        """幂等确认订阅；绝不对已有订阅再次调用 _add，避免重置 seen_links。"""
        permission = self._require_subscription_operator(event, "确认新增订阅")
        if permission:
            return permission
        target_origin, error = self._subscription_target(event, target_group)
        if error:
            return error
        if time.time() - state.get("created_at", 0) > self._preview_ttl_seconds:
            return "❌ 最近一次预览已过期，请重新预览。"
        feed_url = state["url"]
        if target_origin in self.dh.data.get(feed_url, {}).get("subscribers", {}):
            state["consumed"] = True
            state["target_origin"] = target_origin
            return f"ℹ️「{state.get('title', feed_url)}」已经订阅到 {target_origin}，未重复写入。"
        if state.get("consumed"):
            return f"❌ 最近一次预览已经用于 {state.get('target_origin', '其他目标')}，如需换群请重新预览。"
        async with self._data_lock:
            # 订阅落库与定时 job 的读改写事务互斥，防止刚新增的群订阅被旧快照覆盖。
            self.dh.data = self.dh._load()
            # 锁内再次检查，避免等待锁期间其他请求已经完成同一订阅。
            if target_origin in self.dh.data.get(feed_url, {}).get("subscribers", {}):
                state["consumed"] = True
                state["target_origin"] = target_origin
                return f"ℹ️「{state.get('title', feed_url)}」已经订阅到 {target_origin}，未重复写入。"
            ret = await self._add(feed_url, "*/15 * * * *", event, target_user=target_origin)
            notice_groups = self.dh.data.setdefault("settings", {}).setdefault("proactive_notice_groups", [])
            show_proactive_notice = "GroupMessage" in target_origin and target_origin not in notice_groups
            if show_proactive_notice:
                notice_groups.append(target_origin)
                self.dh.save()
        if isinstance(ret, MessageEventResult):
            return ret
        state["consumed"] = True
        state["target_origin"] = target_origin
        self._reload_jobs()
        message = f"✅ 已订阅「{ret['title']}」\n目标：{target_origin}\n预览编号已消费。"
        if show_proactive_notice:
            message += (
                "\n\n⚠️ 为确保定时动态能主动送达，请群主在 QQ 群设置 → 机器人 → 当前 Bot 中开启："
                "\n1. 获取群内全部消息\n2. 机器人主动在群聊内发言"
            )
        return message

    @filter.llm_tool(name="myrss_preview")
    async def tool_preview(self, event: AstrMessageEvent, url: str = "",
                           target_group: str = "", confirm: bool = False):
        """生成“可订阅动态源”的安全预览卡片，并为后续确认订阅创建 preview_id。

        这是 RSS 订阅工作流工具，不是通用联网搜索、新闻搜索或事实查询工具。
        它通过已配置的 RSSHub 读取某个账号/UP主/频道的 RSS，审核最新一条动态，
        然后生成包含动态主头像、频道名、最新动态标题、摘要、图片和时间的订阅预览卡片。

        应当调用本工具的情况：
        - 用户想关注、订阅、推荐某个 B站UP主、Twitter/X 账号、YouTube 频道等动态源；
        - 用户想先看看某个账号是否适合订阅；
        - 用户要求生成订阅卡片、频道卡片或订阅前预览；
        - 用户说“把某人的动态推荐到群”，此时应先调用本工具生成安全预览，不能直接订阅。

        不应调用本工具的情况：
        - 用户只是询问“某人最近发生了什么”“搜索最新消息”“查一下新闻”；
        - 用户要求普通网页搜索、资料检索、事实核查或总结实时事件；
        - 用户没有订阅、关注、取关、推荐动态源或生成订阅卡片的意图。

        默认只生成预览，不会建立订阅或向目标群发送历史动态。
        但 AstrBot 管理员若在同一句原始消息中明确说“预览……并确认订阅到群号”，
        本工具会在安全预览通过后直接完成订阅，不要求模型复制 preview_id。
        是否明确确认以用户原始消息为准，不信任模型自行填写的 confirm 参数。

        Args:
            url(string): 用户准备订阅的账号/UP主/频道链接，或 / 开头的 RSSHub 路由。不是搜索关键词。
            target_group(string): 仅在用户同一句话明确要求确认订阅时填写目标群号。
            confirm(bool): 用户同一句话明确说确认订阅时为 true；插件仍会复核原始消息。
        """
        full_url, route, error = self._resolve_feed_url(url)
        if not full_url:
            yield event.plain_result("❌ " + error)
            return
        raw = await self._fetch(full_url)
        if not raw:
            yield event.plain_result("❌ 无法访问该源，本次不生成确认状态。")
            return
        try:
            title, desc, avatar = self.dh.parse_channel_info(raw)
        except Exception as exc:
            yield event.plain_result(f"❌ 频道解析失败：{exc}")
            return
        # 临时注入资料供 _poll / 头像查找使用，不新增订阅者，不保存到磁盘。
        old_entry = self.dh.data.get(full_url)
        if old_entry is None:
            self.dh.data[full_url] = {"subscribers": {}, "info": {"title": title, "description": desc, "avatar": avatar}}
        try:
            items = await self._poll(full_url, num=1)
            if not items:
                yield event.plain_result("❌ 该源没有可预览的最新动态。")
                return
            item = items[0]
            status = await self._check_content_safe(item)
            if status != "SAFE":
                self.logger.warning("[MyRSS] preview blocked: status=%s route=%s", status, route)
                yield event.plain_result("🚫 最新动态未通过安全审核，本次不展示内容，也不能确认订阅。")
                return
            preview_id = f"P{int(time.time() * 1000):x}{random.randint(0, 0xffff):04x}"
            origin = event.unified_msg_origin
            # 每个会话只保留最近一次预览，防止状态堆积或误确认旧目标。
            for key in [k for k in self._preview_states if k[0] == origin]:
                del self._preview_states[key]
            self._preview_states[(origin, preview_id)] = {
                "created_at": time.time(), "url": full_url, "route": route,
                "title": title, "consumed": False,
            }
            card = await self._make_safe_preview_card(item)
            card_sent = False
            if card:
                # LLM 工具的普通 chain_result 可能只作为工具结果交回模型，
                # 不一定直接显示给用户。这里按 AstrBot 事件 API 主动发送一次图片，
                # 随后的 plain_result 只负责把确认信息交给模型，避免重复发卡片。
                try:
                    await event.send(MessageChain(chain=[Comp.Image.fromBase64(card)]))
                    card_sent = True
                except Exception as exc:
                    self.logger.error("[MyRSS] safe preview card send failed: route=%s error=%s", route, exc)
            if not card_sent:
                if not card:
                    self.logger.error("[MyRSS] safe preview card render failed: route=%s", route)
                yield event.plain_result(
                    f"⚠️ 安全审核已通过，但图片预览卡片生成或发送失败。\n"
                    f"频道：{title}\n最新动态：{item.title}\n"
                    "本次仍保留预览编号，但没有发送图片卡片。"
                )
            requested_group = self._target_group_from_request(event, target_group)
            if self._explicit_subscribe_confirmation(event):
                result = await self._confirm_preview_subscription(event, self._preview_states[(origin, preview_id)], requested_group)
                if isinstance(result, MessageEventResult):
                    yield result
                else:
                    yield event.plain_result(result)
            else:
                yield event.plain_result(
                    f"预览编号：{preview_id}\n"
                    "这只是预览，尚未订阅。直接明确说“确认订阅到当前群”或“确认订阅到群号”即可；"
                    "插件会自动使用本会话最近一次预览，不需要引用或复制编号。编号 10 分钟内有效。"
                )
        finally:
            if old_entry is None and not self.dh.data.get(full_url, {}).get("subscribers"):
                self.dh.data.pop(full_url, None)

    @filter.llm_tool(name="myrss_manage")
    async def tool_manage(self, event: AstrMessageEvent, action: str = "list",
                          preview_id: str = "", target_group: str = "",
                          keyword: str = "", confirm: bool = False):
        """执行 RSS 动态订阅管理：确认新增订阅、列出订阅或取关动态源。

        这是订阅数据库管理工具，不是联网搜索、内容预览、即时转发或新闻查询工具。
        新增订阅后，插件只会在该动态源以后出现新内容时按原有防重机制推送；
        本工具不会把刚才的预览卡片或历史最新动态立即重复发送到目标群。

        action 使用规则：
        - list：用户询问“我订阅了什么”“当前关注列表”时使用；不需要 preview_id。
        - subscribe：只能由 AstrBot 管理员或 MyRSS 操作员使用，并且必须先由 myrss_preview
          成功生成安全卡片。用户必须明确说“确认订阅/确认关注”。插件自动读取当前会话最近一次有效预览；preview_id 可省略，
          即使模型漏填或抄错也不会串到其他会话。仅发送群号不算确认。
        - unsubscribe：用户明确要求取消关注、退订、取关某个现有动态源时使用；
          使用订阅列表编号或能唯一匹配的标题/URL关键词。

        不应调用本工具的情况：
        - 用户只是想搜索或了解某账号的最新动态；
        - 用户尚未进行安全预览，却要求直接新增订阅；
        - 用户只是看完预览、发送群号，但没有明确确认订阅；
        - 用户要求立即转发某一条消息，而不是持续订阅未来更新。

        Args:
            action(string): 必须是 list、subscribe 或 unsubscribe。
            preview_id(string): subscribe 可选；插件会回退到当前会话最近一次预览，禁止跨会话匹配。
            target_group(string): 订阅/取关的可选目标群号；遗漏时插件可从原始消息提取。
            keyword(string): unsubscribe 使用的唯一标题、URL关键词或订阅列表编号。
            confirm(bool): 提示模型用；真正授权以用户原始消息中的明确确认措辞为准。
        """
        action = (action or "list").strip().lower()
        origin = event.unified_msg_origin
        if action == "list":
            urls = self.dh.get_subs(origin)
            if not urls:
                yield event.plain_result("当前没有任何订阅。")
                return
            lines = ["📋 当前订阅："]
            for i, feed_url in enumerate(urls):
                info = self.dh.data[feed_url].get("info", {})
                cron = self.dh.data[feed_url]["subscribers"][origin].get("cron_expr", "?")
                lines.append(f" {i}. {info.get('title', feed_url)} [{cron}]")
            yield event.plain_result("\n".join(lines))
            return
        if action == "subscribe":
            if not self._explicit_subscribe_confirmation(event):
                yield event.plain_result("尚未执行：原始消息中必须明确包含“确认订阅/确认关注”，不能仅发送群号。")
                return
            state = self._latest_preview_state(origin, preview_id)
            if not state:
                yield event.plain_result("❌ 当前会话没有可用预览，请先生成订阅预览卡片。")
                return
            requested_group = self._target_group_from_request(event, target_group)
            result = await self._confirm_preview_subscription(event, state, requested_group)
            if isinstance(result, MessageEventResult):
                yield result
            else:
                yield event.plain_result(result)
            return
        if action == "unsubscribe":
            target_origin = origin
            if target_group:
                current_gid = origin.split(":")[-1] if "GroupMessage" in origin else ""
                if str(target_group) != current_gid and not self._is_astrbot_admin(event):
                    yield event.plain_result("⚠️ 指定其他群仅 AstrBot 管理员可用。")
                    return
                target_origin = f"{origin.split(':')[0]}:GroupMessage:{target_group}"
            urls = self.dh.get_subs(target_origin)
            if not urls:
                yield event.plain_result("目标会话当前没有订阅。")
                return
            matches = []
            if str(keyword).isdigit() and int(keyword) < len(urls):
                matches = [urls[int(keyword)]]
            elif keyword:
                low = keyword.lower()
                matches = [u for u in urls if low in u.lower() or low in self.dh.data[u].get("info", {}).get("title", "").lower()]
            if len(matches) != 1:
                yield event.plain_result("请提供唯一匹配的订阅编号或关键词；可先执行订阅列表。")
                return
            feed_url = matches[0]
            title = self.dh.data[feed_url].get("info", {}).get("title", feed_url)
            self.dh.data[feed_url].get("subscribers", {}).pop(target_origin, None)
            self.dh.save()
            self._reload_jobs()
            yield event.plain_result(f"✅ 已取关：{title}")
            return
        yield event.plain_result("action 只支持 list、subscribe、unsubscribe。")

    @filter.command_group("myrss")
    def myrss(self):
        pass

    @myrss.command("eye")
    async def cmd_eye(self, event: AstrMessageEvent):
        """随机查看一条科技/自然动态；不订阅、不修改 seen_links。"""
        origin = event.unified_msg_origin
        now = time.time()
        if now - self._eye_cooldown.get(origin, 0) < 60:
            yield event.plain_result("⏳ /myrss eye 每个会话 60 秒内只能使用一次。")
            return
        self._eye_cooldown[origin] = now
        routes = [
            "/twitter/user/NASA", "/twitter/user/esa", "/twitter/user/CERN",
            "/twitter/user/OpenAI", "/twitter/user/AnthropicAI", "/twitter/user/WHO",
            "/twitter/user/NatGeo", "/twitter/user/ScienceMagazine",
        ]
        route = random.choice(routes)
        full_url, _, error = self._resolve_feed_url(route)
        if not full_url:
            yield event.plain_result("❌ " + error)
            return
        raw = await self._fetch(full_url)
        if not raw:
            yield event.plain_result("本次随机源暂时不可访问，请稍后再试。")
            return
        try:
            title, description, avatar = self.dh.parse_channel_info(raw)
        except Exception:
            yield event.plain_result("本次随机源无法解析，请稍后再试。")
            return
        old_entry = self.dh.data.get(full_url)
        if old_entry is None:
            self.dh.data[full_url] = {"subscribers": {}, "info": {"title": title, "description": description, "avatar": avatar}}
        try:
            items = await self._poll(full_url, num=1)
            if not items:
                yield event.plain_result("本次随机源暂无可展示内容。")
                return
            item = items[0]
            status = await self._check_content_safe(item)
            if status != "SAFE":
                yield event.plain_result("本次随机内容未通过安全审核，已停止；不会继续循环尝试其他源。")
                return
            comps = await self._make_comps(item, include_extra_images=not self._is_qq_official_origin(origin))
            yield event.chain_result(comps).use_t2i(self.t2i)
        finally:
            if old_entry is None:
                self.dh.data.pop(full_url, None)

    @myrss.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """列出当前订阅"""
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if not urls:
            yield event.plain_result("暂无订阅")
            return
        txt = "订阅列表：\\n"
        for i, u in enumerate(urls):
            info = self.dh.data[u]["info"]
            txt += " " + str(i) + ". " + info["title"] + "\\n"
        yield event.plain_result(txt)

    # ============================================================
    # AstrBot 管理员权限 + 全局恶意黑名单
    # ============================================================
    def _is_astrbot_admin(self, event: AstrMessageEvent) -> bool:
        """检查是否为 AstrBot 管理员（bot 主人）"""
        try:
            # 优先使用 AstrBot 官方 is_admin
            if hasattr(event, "is_admin") and event.is_admin():
                return True
            # 其次检查配置中的 admin_ids
            admin_ids = self.cfg.get("admin_ids", []) or []
            sender_id = str(event.get_sender_id())
            if sender_id in [str(a) for a in admin_ids]:
                return True
            # 兜底：私聊或群内发送者是 bot 自己
            if hasattr(event, "message_obj") and event.message_obj and event.message_obj.sender:
                if str(event.message_obj.sender.user_id) == str(self.bot_qq):
                    return True
        except Exception:
            pass
        return False

    def _is_subscription_operator(self, event: AstrMessageEvent) -> bool:
        return self._is_astrbot_admin(event) or str(event.get_sender_id()) in self.subscription_operator_ids

    def _require_subscription_operator(self, event: AstrMessageEvent, action: str):
        if not self._is_subscription_operator(event):
            return event.plain_result(
                f"⚠️ {action}仅 AstrBot 管理员或 MyRSS 操作员可用。"
                f"\n当前 OpenID：{event.get_sender_id()}"
            )
        return None
