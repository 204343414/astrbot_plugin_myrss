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
from io import BytesIO
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime
from typing import List

from lxml import etree
from bs4 import BeautifulSoup
from PIL import Image
from jinja2 import Environment, BaseLoader
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig
import astrbot.api.message_components as Comp

# [防冲突] 模块级变量追踪当前活跃的调度器
# 插件热更新时新实例先通过此引用杀掉老调度器，避免新老并行双推
_ACTIVE_SCHED = None
_ALL_SCHEDS = set() # 追踪所有调度器，防多实例

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
    1. 插件配置里的 custom_data_dir （推荐 Docker 用户直接填绝对路径）
    2. 环境变量 MYRSS_DATA_DIR
    3. 插件源码所在目录下的 _data/ （默认，git pull 安全）
    4. 旧的 data/astrbot_plugin_myrss 兜底

    每次启动都会在日志里明确打印实际使用的完整路径。
    """

    def __init__(self, plugin_dir=None, seen_links_max_days=365, custom_data_dir=None):
        self.logger = logging.getLogger("astrbot")
        self.seen_links_max_days = seen_links_max_days

        self.data_dir = self._resolve_data_dir(plugin_dir, custom_data_dir)
        self.config_path = os.path.join(self.data_dir, "_data.json")

        is_container = self._is_running_in_container()
        env_type = "容器环境 (Docker/AstrBot容器)" if is_container else "主机/本地文件系统"
        self.logger.info(f"[MyRSS] 运行环境检测: {env_type}")
        self.logger.info(f"[MyRSS] 数据目录已解析: {self.data_dir}")
        self.logger.info(f"[MyRSS] 数据文件路径: {self.config_path}")

        self.data = self._load()

        # 强提示：推荐用户配置 custom_data_dir 防止重装丢数据
        if not custom_data_dir:
            self.logger.warning("[MyRSS] ⚠️ 强烈建议在插件配置中设置 custom_data_dir（绝对路径），否则重装/更新插件可能导致 seen_links 丢失，引发重复推送！")

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

        # 3. 插件目录 _data （最推荐的默认方式）
        if plugin_dir:
            d = os.path.join(os.path.abspath(plugin_dir), "_data")
            os.makedirs(d, exist_ok=True)
            if self._is_running_in_container():
                self.logger.info("[MyRSS] 检测到容器环境，使用插件目录下的 _data")
            return d

        # 4. 最后兜底（旧行为）
        d = os.path.abspath("data/astrbot_plugin_myrss")
        os.makedirs(d, exist_ok=True)
        self.logger.warning("[MyRSS] 使用兜底数据目录 data/astrbot_plugin_myrss")
        return d

    def get_data_path(self) -> str:
        """返回当前实际使用的数据文件完整路径（给命令显示用）"""
        return self.config_path

    def get_data_dir(self) -> str:
        return self.data_dir

    def _load(self):
        """加载数据，支持从旧路径迁移，并自动备份已有数据"""
        if os.path.exists(self.config_path):
            d = self._read_json(self.config_path)
            if d is not None:
                if len(d) > 1 or d.get("rsshub_endpoints"):
                    self._backup_data(d)
                return d

        # 迁移旧路径
        old_path = "data/astrbot_plugin_myrss/_data.json"
        if os.path.exists(old_path):
            old_data = self._read_json(old_path)
            if old_data is not None:
                os.makedirs(self.data_dir, exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(old_data, f, indent=2, ensure_ascii=False)
                try:
                    shutil.copy2(old_path, old_path + ".migrated_bak")
                except Exception:
                    pass
                return old_data

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



@register("astrbot_plugin_myrss", "MyRSS", "RSS订阅插件(LLM增强版)", "1.0.0", "")
class MyRssPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.ctx = context
        self.cfg = config
        
        # 插件目录（用于存储数据文件，避免 git pull 时丢失）
        # main.py 在 astrbot_plugin_myrss/main.py，数据目录在 astrbot_plugin_myrss/_data/
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
        self._preview_states = {}  # key=(origin, preview_id), value=一次性安全预览状态
        self._preview_ttl_seconds = 600
        
        self._last_fetch_error = None  # 拉取错误追踪（新版本 _fetch 使用）
        self.push_delay_min = config.get("push_delay_min", 5.0)
        self.push_delay_max = config.get("push_delay_max", 8.0)
        self.filter_provider_id = config.get("filter_provider_id", "")
        self.pic = PicHandler(self.adjust_pic)
        self.browserless_url = config.get("browserless_url", "http://browserless:3000")
        self.card = CardGen(browserless_url=self.browserless_url)

        # 防并发锁，key = (url, user)
        self._locks: dict = {}
        self._data_lock = asyncio.Lock() # 保护 dh.data 读写
        # 推荐系统已移除
        # [防冲突] 在创建新调度器前，先杀掉模块级残留的老调度器
        # 场景：插件热更新时框架直接创建新实例，老实例的destroy()可能未被调用
        # 如果不杀，老调度器继续运行老代码的job，和新调度器同时推送→双推
        global _ACTIVE_SCHED, _ALL_SCHEDS
        # 杀掉所有残留的调度器（不只是上一个）
        for old_sched in list(_ALL_SCHEDS):
            try:
                if old_sched.running:
                    old_sched.shutdown(wait=False)
                    self.logger.warning("MyRSS: 停止残留调度器 id=%s", id(old_sched))
            except Exception:
                pass
        _ALL_SCHEDS.clear()
        _ACTIVE_SCHED = None

        self.sched = AsyncIOScheduler()
        _ACTIVE_SCHED = self.sched
        _ALL_SCHEDS.add(self.sched)
        self.sched.start()
        self._reload_jobs()
    
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

        provider_id = self.filter_provider_id if self.filter_provider_id else await self._get_provider_id()
        if not provider_id:
            self.logger.error("[MyRSS] no provider for mandatory content review; reject by default")
            self._safe_cache[cache_key] = "REJECT"
            return "REJECT"

        content = (item.title + " " + (item.description or ""))[:400]

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

    async def _make_comps(self, item: RSSItem) -> list:
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

        if self.read_pic and item.pic_urls:
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
        # [诊断] 打印实例ID和调度器ID，如果日志里同一url出现两个不同的id就是双实例并行
        self.logger.info("RSS拉取开始: instance=%s sched=%s url=%s", id(self), id(self.sched), url)
        if url not in self.dh.data:
            return
        subs = self.dh.data[url].get("subscribers", {})
        if not subs:
            return

        self.logger.info("RSS公共拉取: %s -> %d个订阅者", url, len(subs))

        # 所有订阅者中最早的 last_update（拉最多内容，再各自过滤）
        min_ts = min(si.get("last_update", 0) for si in subs.values())
        min_link = "" # 公共拉取不用after_link过滤，靠seen_links去重

        items = await self._poll(url, num=self.max_poll, after_ts=min_ts, after_link=min_link)
        if not items:
            return

        # 分发给每个订阅者（各自独立去重）
        for i, user in enumerate(list(subs.keys())):
            lock = self._get_lock(url, user)
            async with lock:
                await self._cron_cb_inner(url, user, prefetched_items=items)
            # 多个订阅者间随机延迟防风控
            if i < len(subs) - 1:
                delay = random.uniform(self.push_delay_min, self.push_delay_max)
                await asyncio.sleep(delay)

    async def _cron_cb_inner(self, url: str, user: str, prefetched_items=None) -> None:
        await self._cron_cb_inner_impl(url, user, prefetched_items)

    async def _cron_cb_inner_impl(self, url: str, user: str, prefetched_items=None) -> None:
        async with self._data_lock:
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
            for it in new_items:
                status = await self._check_content_safe(it)
                if status == "SAFE":
                    filtered.append(it)
                elif status == "MALICIOUS":
                    self.logger.warning("[MyRSS] MALICIOUS filtered: %s", it.title[:30])
                else:
                    self.logger.info("[MyRSS] REJECT filtered: %s", it.title[:30])
            new_items = filtered
        if not new_items:
            return
        pn = user.split(":")[0]
        merge_limit = 5
        batch = new_items[:merge_limit]

        if len(batch) > 1:
            cards_raw = [await self._make_card_b64(it) for it in batch]
            cards = [c for c in cards_raw if c] # 过滤掉被内容审核拦截的空卡片
            if not cards:
                self.logger.info("[MyRSS] all items filtered, skip push")
                return
            merged = self._merge_cards_b64(cards)
            if not merged:
                for it in batch:
                    comps = await self._make_comps(it)
                    await self.ctx.send_message(user, MessageChain(chain=comps, use_t2i_=self.t2i))
            else:
                comps = [Comp.Image.fromBase64(merged)]
                if pn == "aiocqhttp" and self.compose:
                    node = Comp.Node(uin=0, name="Astrbot", content=comps)
                    await self.ctx.send_message(user, MessageChain(chain=[node], use_t2i_=self.t2i))
                else:
                    await self.ctx.send_message(user, MessageChain(chain=comps, use_t2i_=self.t2i))
        else:
            it = batch[0]
            comps = await self._make_comps(it)
            if pn == "aiocqhttp" and self.compose:
                node = Comp.Node(uin=0, name="Astrbot", content=comps)
                await self.ctx.send_message(user, MessageChain(chain=[node], use_t2i_=self.t2i))
            else:
                await self.ctx.send_message(user, MessageChain(chain=comps, use_t2i_=self.t2i))

        self.logger.info("RSS推送完成: %s -> %s (%d条)", url, user, len(batch))

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

    @filter.llm_tool(name="myrss_preview")
    async def tool_preview(self, event: AstrMessageEvent, url: str = ""):
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

        本工具只生成预览，不会建立订阅、不会向目标群发送动态，也不会修改 seen_links。
        审核通过后会返回一次性 preview_id；必须等待用户明确确认，再调用 myrss_manage。

        Args:
            url(string): 用户准备订阅的账号/UP主/频道链接，或 / 开头的 RSSHub 路由。不是搜索关键词。
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
            yield event.plain_result(
                f"预览编号：{preview_id}\n"
                "这只是预览，尚未订阅。若确认，请明确说：确认订阅到当前群；"
                "跨群请由 AstrBot 管理员说：确认订阅到群号。编号 10 分钟内有效。"
            )
        finally:
            if old_entry is None:
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
        - subscribe：只能在 myrss_preview 已成功生成安全卡片后使用。用户必须明确说
          “确认订阅/确认关注”，并提供同一会话内有效的 preview_id。仅发送群号不算确认。
        - unsubscribe：用户明确要求取消关注、退订、取关某个现有动态源时使用；
          使用订阅列表编号或能唯一匹配的标题/URL关键词。

        不应调用本工具的情况：
        - 用户只是想搜索或了解某账号的最新动态；
        - 用户尚未进行安全预览，却要求直接新增订阅；
        - 用户只是看完预览、发送群号，但没有明确确认订阅；
        - 用户要求立即转发某一条消息，而不是持续订阅未来更新。

        Args:
            action(string): 必须是 list、subscribe 或 unsubscribe。
            preview_id(string): subscribe 必填，必须来自当前会话最近一次 myrss_preview。
            target_group(string): 订阅/取关的可选目标群号；跨群仅 AstrBot 管理员可操作。
            keyword(string): unsubscribe 使用的唯一标题、URL关键词或订阅列表编号。
            confirm(bool): 仅当用户明确表达“确认订阅/确认关注”时传 true；不得自行推断。
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
            if not confirm:
                yield event.plain_result("尚未执行：请明确说“确认订阅”，不能仅发送群号。")
                return
            state = self._preview_states.get((origin, preview_id))
            if not state or state.get("consumed") or time.time() - state.get("created_at", 0) > self._preview_ttl_seconds:
                yield event.plain_result("❌ 预览编号无效、已使用或已过期，请重新预览。")
                return
            target_origin = origin
            if target_group:
                current_gid = origin.split(":")[-1] if "GroupMessage" in origin else ""
                if str(target_group) != current_gid:
                    if not self._is_astrbot_admin(event):
                        yield event.plain_result("⚠️ 指定其他群仅 AstrBot 管理员可用。")
                        return
                    platform = origin.split(":")[0]
                    target_origin = f"{platform}:GroupMessage:{target_group}"
            ret = await self._add(state["url"], "*/15 * * * *", event, target_user=target_origin)
            if isinstance(ret, MessageEventResult):
                yield ret
                return
            state["consumed"] = True
            self._reload_jobs()
            yield event.plain_result(f"✅ 已订阅「{ret['title']}」\n目标：{target_origin}\n预览编号已消费。")
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

    @myrss.group("rsshub")
    def rsshub(self, event: AstrMessageEvent):
        pass

    @rsshub.command("add")
    async def rsshub_add(self, event: AstrMessageEvent, url: str):
        """添加RSSHub端点（仅管理员）"""
        admin_check = self._require_admin(event, "添加 RSSHub 端点")
        if admin_check:
            yield admin_check
            return
        if url.endswith("/"):
            url = url[:-1]
        if url in self.dh.data["rsshub_endpoints"]:
            yield event.plain_result("已存在")
            return
        self.dh.data["rsshub_endpoints"].append(url)
        self.dh.save()
        yield event.plain_result("✅ 已添加: " + url)

    @rsshub.command("list")
    async def rsshub_list(self, event: AstrMessageEvent):
        """列出所有RSSHub端点"""
        eps = self.dh.data["rsshub_endpoints"]
        if not eps:
            yield event.plain_result("暂无端点，请先 /myrss rsshub add ")
            return
        txt = "RSSHub端点：\\n"
        for i, x in enumerate(eps):
            txt += " " + str(i) + ": " + x + "\\n"
        yield event.plain_result(txt)

    @rsshub.command("remove")
    async def rsshub_rm(self, event: AstrMessageEvent, idx: int):
        """删除RSSHub端点（仅管理员）"""
        admin_check = self._require_admin(event, "删除 RSSHub 端点")
        if admin_check:
            yield admin_check
            return
        eps = self.dh.data["rsshub_endpoints"]
        if idx < 0 or idx >= len(eps):
            yield event.plain_result("编号越界")
            return
        removed = eps.pop(idx)
        self.dh.save()
        yield event.plain_result("✅ 已删除: " + removed)

    # ============================================================
    # 黑名单管理命令（仅管理员）
    # ============================================================
    @myrss.group("blacklist")
    def blacklist(self, event: AstrMessageEvent):
        pass

    @blacklist.command("list")
    async def blacklist_list(self, event: AstrMessageEvent):
        """查看全局黑名单（仅管理员）"""
        admin_check = self._require_admin(event, "查看黑名单")
        if admin_check:
            yield admin_check
            return
        settings = self.dh.data.setdefault("settings", {})
        bl = settings.get("blacklisted_users", [])
        if not bl:
            yield event.plain_result("当前黑名单为空。")
            return
        txt = "🚫 全局黑名单用户：\n" + "\n".join(f"  - {uid}" for uid in bl)
        yield event.plain_result(txt)

    @blacklist.command("add")
    async def blacklist_add(self, event: AstrMessageEvent, user_id: str):
        """添加用户到全局黑名单（仅管理员）"""
        admin_check = self._require_admin(event, "添加黑名单")
        if admin_check:
            yield admin_check
            return
        if not user_id:
            yield event.plain_result("用法: /myrss blacklist add <用户ID>")
            return
        self._add_to_blacklist(user_id, "管理员手动添加")
        yield event.plain_result(f"✅ 已将 {user_id} 加入全局黑名单。")

    @blacklist.command("remove")
    async def blacklist_remove(self, event: AstrMessageEvent, user_id: str):
        """从全局黑名单移除用户（仅管理员）"""
        admin_check = self._require_admin(event, "移除黑名单")
        if admin_check:
            yield admin_check
            return
        if not user_id:
            yield event.plain_result("用法: /myrss blacklist remove <用户ID>")
            return
        self._remove_from_blacklist(user_id)
        yield event.plain_result(f"✅ 已将 {user_id} 从黑名单移除。")

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

    @myrss.command("remove")
    async def cmd_rm(self, event: AstrMessageEvent, idx: int):
        """取消订阅"""
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if idx < 0 or idx >= len(urls):
            yield event.plain_result("编号越界")
            return
        u = urls[idx]
        t = self.dh.data[u]["info"]["title"]
        self.dh.data[u]["subscribers"].pop(user)
        self.dh.save()
        self._reload_jobs()
        yield event.plain_result("✅ 已取消: " + t)

    @myrss.command("get")
    async def cmd_get(self, event: AstrMessageEvent, idx: int):
        """获取最新内容"""
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if idx < 0 or idx >= len(urls):
            yield event.plain_result("编号越界")
            return
        items = await self._poll(urls[idx])
        if not items:
            yield event.plain_result("暂无内容")
            return
        comps = await self._make_comps(items[0])
        pn = user.split(":")[0]
        if pn == "aiocqhttp" and self.compose:
            yield event.chain_result([Comp.Node(uin=0, name="Astrbot", content=comps)]).use_t2i(self.t2i)
        else:
            yield event.chain_result(comps).use_t2i(self.t2i)
    
    @myrss.command("clearcache")
    async def cmd_clearcache(self, event: AstrMessageEvent):
        """清空过滤缓存和锐评缓存"""
        safe_count = len(self._safe_cache)
        comment_count = len(self._comment_cache)
        self._safe_cache.clear()
        self._comment_cache.clear()
        yield event.plain_result(f"✅ 缓存已清空\\n 过滤缓存: {safe_count} 条已清除\\n 锐评缓存: {comment_count} 条已清除")

    @myrss.command("clear")
    async def cmd_clear(self, event: AstrMessageEvent, target: str = "all"):
        """清空所有/指定源的已推送历史记录，从今天开始重新计数（不会重复推旧内容）。
        用法：
        /myrss clear          清空当前用户所有订阅的推送历史
        /myrss clear all      清空当前用户所有订阅
        /myrss clear 0,2,3    清空指定编号的订阅源
        /myrss clear 推特      清空包含关键词的订阅源
        """
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if not urls:
            yield event.plain_result("当前没有任何订阅，无需清空。")
            return

        target_urls = []
        t = target.strip().lower()

        if t in ("all", "全部", "清空", ""):
            target_urls = urls
        else:
            # 尝试解析为编号列表（逗号/空格分隔）
            nums = [int(x) for x in re.findall(r"\d+", t)]
            if nums and all(0 <= n < len(urls) for n in nums):
                target_urls = [urls[n] for n in nums]
            else:
                # 模糊匹配关键词
                for u in urls:
                    info = self.dh.data.get(u, {}).get("info", {})
                    title = info.get("title", "")
                    if t in title.lower() or t in u.lower():
                        target_urls.append(u)

        if not target_urls:
            yield event.plain_result(f"没找到匹配的订阅源。用 /myrss list 查看当前订阅。")
            return

        cleared = []
        for u in target_urls:
            if u in self.dh.data:
                subs = self.dh.data[u].get("subscribers", {})
                if user in subs:
                    subs[user]["seen_links"] = []
                    subs[user]["latest_link"] = ""
                    # last_update 保持原值，不清空（避免下次全量拉取）
                    cleared.append(self.dh.data[u].get("info", {}).get("title", u))

        self.dh.save()
        yield event.plain_result(
            f"✅ 已清空以下源的推送历史记录：\\n" +
            "\\n".join(f" - {x}" for x in cleared) +
            "\\n\\n从此刻开始的新内容才会被推送，历史旧内容不会重推。"
        )

    @myrss.command("test")
    async def cmd_test(self, event: AstrMessageEvent, route: str = "/twitter/user/AnthropicAI"):
        """测试推送流程：拉取指定源的最新一条，走完整的过滤+锐评+缓存流程。
        用法：
        /myrss test （默认 Anthropic 推特）
        /myrss test /twitter/user/elonmusk （RSSHub 路由）
        /myrss test https://x.com/elonmusk （自动转路由）
        /myrss test https://space.bilibili.com/2267573/dynamic
        """
        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            yield event.plain_result("没有配置 RSSHub 端点，无法测试。")
            return
        # 健壮的URL提取：不管什么格式（[url](url)、++格式、纯URL），先提取真实URL再匹配
        urls = re.findall(r'https?://[^\s)\]]+', route)
        if urls:
            route = urls[0]  # 取第一个提取到的URL

        if not route.startswith("/"):
            matched = URLMapper.match(route)
            if matched:
                converted_route, platform_name = matched
                yield event.plain_result(f"🔄 识别为 {platform_name}，转换路由: {converted_route}")
                route = converted_route
            else:
                suggest = URLMapper.suggest(route)
                yield event.plain_result(f"❌ 无法识别该链接: {route}\n\n{suggest}\n\n请用 /开头的路由重试，例如 /twitter/user/用户名。")
                return

        url = eps[0].rstrip("/") + route

        yield event.plain_result(f"⏳ 开始测试推送流程...\\n源: {route}\\n10秒后拉取（模拟真实延迟）")

        await asyncio.sleep(10)

        # 第1步：拉取
        yield event.plain_result("📡 [1/4] 正在拉取 RSS...")
        # [test] 先抓一次频道信息，写入 dh.data，让 _poll() 能拿到 chan_title（否则显示"未知"）
        try:
            txt = await self._fetch(url)
            if txt:
                t, d, a = self.dh.parse_channel_info(txt)
                self.dh.data[url] = {
                    "info": {"title": t, "description": d, "avatar": a},
                    "subscribers": {},
                    "is_test": True,
                }
            else:
                last_err = getattr(self, "_last_fetch_error", "无响应数据")
                yield event.plain_result(f"⚠️ [1/4] 预抓取频道信息失败，后台报错：{last_err}")
        except Exception as e:
            yield event.plain_result(f"⚠️ [1/4] 预抓取频道信息异常：{type(e).__name__}: {e}")

        items = await self._poll(url, num=1)
        if not items:
            last_err = getattr(self, "_last_fetch_error", "未知错误")
            # Try to get more context from recent logs or the actual exception during this test
            debug_extra = ""
            try:
                # If the last operation raised, it may have been caught higher; show what we have
                if last_err == "未知错误" or "All fetch attempts" in str(last_err):
                    debug_extra = "\n注意：本次失败没有抛出 Python 异常（可能是 HTTP 非200、超时、或返回空/错误页）。请查看 AstrBot 容器日志获取 aiohttp 详细错误。"
            except:
                pass

            yield event.plain_result(f"""❌ 拉取失败，源无内容或不可访问。

🔍 调试排错信息：
 - 请求 URL: {url}
 - 错误详情: {last_err}{debug_extra}

💡 修复建议：
 1. 代理问题（最常见）：为内网 rsshub 设置 NO_PROXY=rsshub,localhost,127.0.0.1 在容器环境变量或 docker run -e。
 2. Docker 网络问题：如果容器是单独 docker run 启动的，"rsshub" 主机名可能无法解析。请使用 docker-compose 把 astrbot 和 rsshub 放在同一个 network，或把端点改成宿主机 IP:1200（如 http://172.17.0.1:1200）。
 3. 本插件已对 rsshub 等内网地址自动 trust_env=False 禁用代理，请确认更新已应用并重启 Bot。
 4. 临时绕过：把 rsshub_endpoints 改成能从 astrbot 容器直达的地址测试（例如宿主机 IP）。
""")
            return
        item = items[0]
        # [Hack] 临时把测试源的信息注入 data，让 _make_card_b64 能查到头像/标题
        if url not in self.dh.data:
            # 尝试再 fetch 一次拿 channel info
            try:
                txt = await self._fetch(url)
                if txt:
                    t, d, a = self.dh.parse_channel_info(txt)
                    self.dh.data[url] = {
                        "info": {"title": t, "description": d, "avatar": a},
                        "subscribers": {}, # 空订阅
                        "is_test": True # 标记为测试
                    }
            except Exception:
                pass
        yield event.plain_result(f"✅ 拉取成功: {item.title[:80]}")

        # 第2步：内容过滤（走真实函数，会用缓存）
        yield event.plain_result("🔍 [2/4] 正在过滤内容（LLM审核）...")
        norm_link = item.link.split("#", 1)[0].split("?", 1)[0] if item.link else ""
        cache_key = norm_link or (item.title + "|" + str(item.pubDate_timestamp))
        was_cached = cache_key in self._safe_cache
        safe = await self._check_content_safe(item)
        if safe != "SAFE":
            yield event.plain_result(
                f"🚫 内容被过滤（判定不安全），不会推送。\\n"
                f" 缓存命中: {was_cached}\\n"
                f" 标题: {item.title[:60]}\\n"
                f" 如果这是误杀，可能需要调整过滤 prompt 或换一个过滤 provider。\\n"
                f" 提示: 可以临时关闭 content_filter 再测试，确认是过滤器问题还是其他问题。"
            )
            return
        yield event.plain_result(f"✅ 内容安全。缓存命中: {was_cached}")

        # 第3步：生成锐评（走真实函数，会用缓存）
        yield event.plain_result("💬 [3/4] 正在生成锐评（LLM评论）...")
        comment = ""
        if self.enable_comment:
            comment_was_cached = cache_key in self._comment_cache
            comment = await self._generate_comment(item)
            if comment:
                yield event.plain_result(f"✅ 锐评: {comment[:80]}\\n 缓存命中: {comment_was_cached}")
            else:
                yield event.plain_result("⚠️ 锐评生成失败或为空")
        else:
            yield event.plain_result("⏭️ 锐评已关闭，跳过")

        # 第4步：生成卡片并发送（走真实函数）
        yield event.plain_result("🎨 [4/4] 正在生成卡片...")
        comps = await self._make_comps(item)

        user = event.unified_msg_origin
        pn = user.split(":")[0]
        if pn == "aiocqhttp" and self.compose:
            yield event.chain_result([Comp.Node(uin=0, name="[测试]Astrbot", content=comps)]).use_t2i(self.t2i)
        else:
            yield event.chain_result(comps).use_t2i(self.t2i)

        yield event.plain_result(
            "✅ 测试完成！\\n"
            f" 过滤缓存大小: {len(self._safe_cache)}\\n"
            f" 锐评缓存大小: {len(self._comment_cache)}\\n"
            "再次执行同样的命令可验证缓存是否命中（应显示 True）"
        )

    @myrss.command("groups")
    async def cmd_groups(self, event: AstrMessageEvent):
        """列出机器人加入的群（需要 aiocqhttp / NapCat）"""
        try:
            if event.get_platform_name() != "aiocqhttp":
                yield event.plain_result("当前平台不支持获取群列表（仅 aiocqhttp/NapCat 支持）。")
                return

            # AstrBot 官方文档要求的调用方式
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
            if not isinstance(event, AiocqhttpMessageEvent):
                yield event.plain_result("事件类型不匹配，无法调用协议端 API。")
                return

            client = event.bot
            if not client:
                yield event.plain_result("无法获取协议端 client。")
                return

            ret = await client.api.call_action('get_group_list')

            # NapCat 返回格式可能是 list 或 dict{"data": list}
            if isinstance(ret, list):
                data = ret
            elif isinstance(ret, dict):
                data = ret.get("data", [])
            else:
                data = []

            if not data:
                yield event.plain_result("群列表为空，或协议端未返回数据。\\n返回值类型: " + str(type(ret).__name__))
                return

            lines = ["📋 机器人所在群列表："]
            for i, g in enumerate(data):
                if isinstance(g, dict):
                    gid = g.get("group_id", "")
                    gname = g.get("group_name", "")
                    lines.append(f" {i}. {gname} ({gid})")
                else:
                    lines.append(f" {i}. {g}")

            yield event.plain_result("\\n".join(lines))
        except Exception as e:
            self.logger.error("[MyRSS] get group list failed: %s", e, exc_info=True)
            yield event.plain_result("获取群列表失败：" + str(e))

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

    def _require_admin(self, event: AstrMessageEvent, action: str = "此操作"):
        """管理员权限检查，失败时直接 yield 回复"""
        if not self._is_astrbot_admin(event):
            return event.plain_result(f"⚠️ {action}仅 AstrBot 管理员可用。\n普通用户请用自然语言对我说「关注 xxx」或「订阅 xxx」。")
        return None

    # 全局黑名单（存在 dh.data["settings"]["blacklisted_users"]）
    def _is_blacklisted(self, user_id: str) -> bool:
        settings = self.dh.data.setdefault("settings", {})
        bl = settings.get("blacklisted_users", [])
        is_bl = str(user_id) in [str(x) for x in bl]
        if is_bl:
            self.logger.info("[MyRSS] 黑名单命中: %s", user_id)
        return is_bl

    def _add_to_blacklist(self, user_id: str, reason: str = ""):
        settings = self.dh.data.setdefault("settings", {})
        bl = settings.setdefault("blacklisted_users", [])
        uid = str(user_id)
        if uid not in bl:
            bl.append(uid)
            self.dh.save()
            self.logger.warning("[MyRSS] 用户 %s 已被加入全局黑名单，原因: %s", uid, reason)

    def _remove_from_blacklist(self, user_id: str):
        settings = self.dh.data.setdefault("settings", {})
        bl = settings.get("blacklisted_users", [])
        uid = str(user_id)
        if uid in bl:
            bl.remove(uid)
            self.dh.save()

    

    

    @myrss.command("subs")
    async def cmd_subs(self, event: AstrMessageEvent):
        """查看所有订阅源及其订阅群列表"""
        lines = ["📋 所有订阅源："]
        idx = 0
        for url, info in self.dh.data.items():
            if url in ("rsshub_endpoints", "settings"):
                continue
            subs = info.get("subscribers", {})
            if not subs:
                continue
            title = info.get("info", {}).get("title", url)
            lines.append(f"\\n{idx}. 📡 {title}")
            lines.append(f" 路由: {url.split(':1200')[-1] if ':1200' in url else url}")
            if subs:
                for sub_id in subs:
                    gid_short = sub_id.split(":")[-1]
                    platform = sub_id.split(":")[0]
                    cron = subs[sub_id].get("cron_expr", "?")
                    lines.append(f" └ {gid_short} ({platform}) [{cron}]")
            else:
                lines.append(f" └ (无订阅者)")
            idx += 1
        if idx == 0:
            yield event.plain_result("当前没有任何订阅源。")
            return
        yield event.plain_result("\\n".join(lines))

    @myrss.command("unbind")
    async def cmd_unbind(self, event: AstrMessageEvent, group_id: str = ""):
        """把指定群从所有订阅源中退订。用法：/myrss unbind 721058477"""
        if not group_id:
            yield event.plain_result("用法: /myrss unbind <群号>\n 先用 /myrss subs 查看所有群号和订阅关系")
            return

        gids = [g.strip() for g in re.split(r'[,，\s]+', group_id) if g.strip()]
        removed_per_group = {}
        for target_gid in gids:
            removed = 0
            for url, info in self.dh.data.items():
                if url in ("rsshub_endpoints", "settings"):
                    continue
                subs = info.get("subscribers", {})
                for key in [k for k in subs if target_gid in k]:
                    del subs[key]
                    removed += 1
            removed_per_group[target_gid] = removed

        total_removed = sum(removed_per_group.values())
        if total_removed:
            self.dh.save()
            self._reload_jobs()
            details = "\n".join(f" 群 {gid}: 退订 {count} 个源" for gid, count in removed_per_group.items() if count)
            yield event.plain_result(f"✅ 已退订，共 {total_removed} 条：\n{details}")
        else:
            yield event.plain_result(f"未发现指定群的订阅：{', '.join(gids)}")

    @myrss.command("reset")
    async def cmd_reset(self, event: AstrMessageEvent):
        """重置所有订阅源的推送基准。
        根据配置 force_reset_without_poll 决定是否依赖 poll 拉取最新内容。
        默认（推荐）使用强制模式：直接把每个订阅的 seen_links 重置为只有当前 latest_link 的一条。
        执行后会回传实际使用的数据文件路径和证据。
        （仅管理员）
        """
        admin_check = self._require_admin(event, "重置推送记录")
        if admin_check:
            yield admin_check
            return
        use_force = self.cfg.get("force_reset_without_poll", True)
        
        mode = "强制模式（不依赖网络，拉当前 latest_link）" if use_force else "智能模式（尝试拉最新 RSS 作为基准）"
        yield event.plain_result(f"⏳ 正在重置...（{mode} + 清理缓存 + 强制保存）")
        
        # 1. 清空内存缓存
        self._safe_cache.clear()
        self._comment_cache.clear()
        self.logger.info("[MyRSS] 内存缓存已清空")
        
        count = 0
        force_count = 0
        
        for url, info in self.dh.data.items():
            if url in ("rsshub_endpoints", "settings"):
                continue
            subs = info.get("subscribers", {})
            if not subs:
                continue
            
            did_update = False
            
            if not use_force:
                # 智能模式：尝试拉最新
                try:
                    items = await self._poll(url, num=1)
                    if items:
                        item = items[0]
                        ik = item.link.split("#", 1)[0].split("?", 1)[0] if item.link else f"{item.title}|{item.pubDate_timestamp}"
                        for user, sub_data in subs.items():
                            if item.pubDate_timestamp > 0:
                                sub_data["last_update"] = item.pubDate_timestamp
                            sub_data["latest_link"] = item.link
                            sub_data["seen_links"] = [ik]
                        did_update = True
                except Exception as e:
                    self.logger.warning(f"[MyRSS] reset poll failed for {url}: {e}")
            
            if not did_update or use_force:
                # 强制模式或 poll 失败时：直接用已有的 latest_link 设为单条
                for user, sub_data in subs.items():
                    latest = sub_data.get("latest_link", "")
                    sub_data["seen_links"] = [latest] if latest else []
                force_count += 1
                did_update = True
            
            if did_update:
                count += 1

        # 2. 强制保存
        self.dh.save()
        self._reload_jobs()
        
        # 3. 读回文件发证据（关键：用 dh.get_data_path() 确保显示真实路径）
        actual_path = self.dh.get_data_path()
        try:
            with open(actual_path, "r", encoding="utf-8") as f:
                content = f.read()
            sample = content[:600]
            msg = (
                f"✅ 重置完成！共影响 {count} 个源（其中 {force_count} 个使用强制单条模式）。\n"
                f"📂 实际数据文件路径：{actual_path}\n"
                f"📋 证据（文件内容片段）：\n{sample}..."
            )
            yield event.plain_result(msg)
        except Exception as e:
            yield event.plain_result(f"✅ 重置完成，但读取文件失败：{e}\n实际路径应为：{actual_path}")
    @myrss.command("datapath")
    async def cmd_datapath(self, event: AstrMessageEvent):
        """显示当前实际使用的数据文件路径和基本统计。强烈建议每次遇到路径问题时先跑这个。"""
        admin_check = self._require_admin(event, "查看数据路径")
        if admin_check:
            yield admin_check
            return
        try:
            path = self.dh.get_data_path()
            data_dir = self.dh.get_data_dir()
            total_sources = len([k for k in self.dh.data if k not in ("rsshub_endpoints", "settings")])
            
            total_subs = 0
            for url, info in self.dh.data.items():
                if url in ("rsshub_endpoints", "settings"):
                    continue
                total_subs += len(info.get("subscribers", {}))
            
            import os, time
            size = os.path.getsize(path) if os.path.exists(path) else 0
            mtime = ""
            if os.path.exists(path):
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
            
            msg = (
                f"📂 当前实际数据文件路径：\n{path}\n\n"
                f"📁 数据目录：{data_dir}\n"
                f"📊 统计：{total_sources} 个订阅源，{total_subs} 个订阅记录\n"
                f"📦 文件大小：{size} bytes\n"
                f"🕒 最后修改：{mtime}\n\n"
                "⚠️ 重要提示：\n"
                "1. 看到这个路径说明数据实际写在这里。\n"
                "2. 重装/更新插件时，如果这个目录被删除，seen_links 会丢失 → 可能重复推送！\n"
                "3. 强烈建议在插件配置中设置 custom_data_dir（填绝对路径）来固定数据位置。"
            )
            yield event.plain_result(msg)
        except Exception as e:
            yield event.plain_result(f"获取数据路径失败: {e}")

    @myrss.command("unsub")
    async def cmd_unsub(self, event: AstrMessageEvent, route: str = "", group_ids: str = ""):
        """从指定源批量退订群
        用法：/myrss unsub /bilibili/user/dynamic/2107422684 721058477,123456
        /myrss unsub /bilibili/user/dynamic/2107422684 all
        """
        if not route:
            yield event.plain_result(
                "用法: /myrss unsub <路由> <群号列表>\\n"
                " 群号用逗号分隔，或填 all 退订所有群\\n"
                " 先用 /myrss subs 查看路由和群号"
            )
            return

        # 找到匹配的URL
        target_url = None
        for url in self.dh.data:
            if url in ("rsshub_endpoints", "settings"):
                continue
            if route in url:
                target_url = url
                break

        if not target_url:
            yield event.plain_result(f"找不到包含 '{route}' 的订阅源\\n用 /myrss subs 查看")
            return

        subs = self.dh.data[target_url].get("subscribers", {})
        if not subs:
            yield event.plain_result("该源没有订阅者。")
            return

        title = self.dh.data[target_url].get("info", {}).get("title", route)

        if not group_ids or group_ids.strip().lower() == "all":
            removed = list(subs.keys())
            subs.clear()
        else:
            gids = [g.strip() for g in re.split(r'[,，\\s]+', group_ids) if g.strip()]
            removed = []
            for gid in gids:
                # 模糊匹配：群号可能只传了数字
                to_del = [k for k in subs if gid in k]
                for key in to_del:
                    del subs[key]
                    removed.append(key.split(":")[-1])

        if removed:
            self.dh.save()
            self._reload_jobs()
            yield event.plain_result(
                f"✅ 已从「{title}」退订 {len(removed)} 个群:\\n" +
                "\\n".join(f" - {g}" for g in removed)
            )
        else:
            yield event.plain_result("没有匹配的群号，请检查输入。")
