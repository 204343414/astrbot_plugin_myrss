import os
import json
import re
import time
import random
import base64
import logging
import asyncio
import aiohttp
import calendar
from io import BytesIO
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime
from typing import List

from lxml import etree
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig
import astrbot.api.message_components as Comp

# [防冲突] 模块级变量追踪当前活跃的调度器
# 插件热更新时新实例先通过此引用杀掉老调度器，避免新老并行双推
_ACTIVE_SCHED = None


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
    def __init__(self, config_path="data/astrbot_plugin_myrss_data.json"):
        self.config_path = config_path
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.config_path):
            d = {"rsshub_endpoints": []}
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            return d
        # [防冲突] 共享读锁，等待排他写锁释放后再读，避免读到写了一半的JSON
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            except (ImportError, OSError):
                pass
            return json.load(f)

    def save(self):
        # [防冲突] 文件排他锁，防止新老实例同时写JSON导致数据丢失
        # 场景：老实例的job推送完更新seen_links写文件，同时新实例也在写→后写的覆盖前面的
        # fcntl仅Linux/Mac可用，Windows环境静默跳过
        with open(self.config_path, "w", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            json.dump(self.data, f, indent=2, ensure_ascii=False)

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
        
        # 1. 常规 <img src="...">
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and src not in urls:
                urls.append(src)
                
        # 2. <video poster="...">
        for vid in soup.find_all("video"):
            poster = vid.get("poster")
            if poster and poster not in urls:
                urls.append(poster)
                
        # 3. [暴力增强] 直接正则扫描整个HTML文本匹配YouTube ID
        # 因为有时候 RSSHub 返回的 description 里只有纯文本链接，没有 <a> 标签
        # 匹配 youtube.com/watch?v=xxx 或 youtu.be/xxx
        patterns = [
            r'youtube\.com/watch\?v=([\w-]{11})',
            r'youtu\.be/([\w-]{11})',
            r'youtube\.com/embed/([\w-]{11})',
            r'youtube\.com/v/([\w-]{11})'
        ]
        
        found_ids = set()
        # 先搜 soup 里的 a 标签
        for a in soup.find_all("a", href=True):
            for pat in patterns:
                m = re.search(pat, a["href"])
                if m: found_ids.add(m.group(1))

        # 再暴力搜全文（兜底）
        for pat in patterns:
            for vid_id in re.findall(pat, html):
                found_ids.add(vid_id)

        # 构造封面地址
        for vid_id in found_ids:
            # 存两个分辨率，优先高清(maxres)，其次中等(hq)，防止maxres不存在
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
        (r"youtube\.com/channel/([\w-]+)", "/youtube/channel/{0}", "YouTube频道"),
        # [修复] 优先匹配 YouTube 的动态(community/posts)、Shorts、直播等特定页面
        # 必须放在通用的 @user 规则之前，否则会被通用规则拦截
        (r"youtube\.com/@([\w.-]+)/(?:posts|community)", "/youtube/community/@{0}", "YouTube动态"),
        (r"youtube\.com/@([\w.-]+)/shorts", "/youtube/user/@{0}/shorts", "YouTube Shorts"),
        (r"youtube\.com/@([\w.-]+)/streams", "/youtube/user/@{0}/live", "YouTube直播记录"),
        # [原规则] 通用用户规则放在最后作为兜底
        (r"youtube\.com/@([\w.-]+)", "/youtube/user/@{0}", "YouTube用户"),
        (r"youtube\.com/playlist\?list=([\w-]+)", "/youtube/playlist/{0}", "YouTube播放列表"),
        (r"(?:twitter|x)\.com/(?!home|explore|search|settings|i/)([\w]+)", "/twitter/user/{0}", "Twitter/X"),
        (r"weibo\.com/u/(\d+)", "/weibo/user/{0}", "微博"),
        (r"zhihu\.com/people/([\w-]+)", "/zhihu/people/activities/{0}", "知乎"),
        (r"zhihu\.com/column/([\w-]+)", "/zhihu/zhuanlan/{0}", "知乎专栏"),
        (r"xiaohongshu\.com/user/profile/([\w]+)", "/xiaohongshu/user/{0}/notes", "小红书"),
        (r"github\.com/([\w.-]+)/([\w.-]+)/releases", "/github/release/{0}/{1}", "GitHub Release"),
        (r"github\.com/([\w.-]+)/([\w.-]+)(?:$|[/?#])", "/github/commits/{0}/{1}", "GitHub仓库"),
        (r"github\.com/([\w.-]+)(?:$|[/?#])", "/github/repos/{0}", "GitHub用户"),
        (r"t\.me/s?/?([\w]+)", "/telegram/channel/{0}", "Telegram"),
        (r"douyin\.com/user/([\w]+)", "/douyin/user/{0}", "抖音"),
        (r"instagram\.com/([\w.]+)(?:$|[/?#])", "/instagram/user/{0}", "Instagram"),
        (r"pixiv\.net/users/(\d+)", "/pixiv/user/{0}", "Pixiv"),
        (r"sspai\.com/u/([\w]+)", "/sspai/author/{0}", "少数派"),
        (r"okjike\.com/u/([\w-]+)", "/jike/user/{0}", "即刻"),
        (r"podcasts\.apple\.com/.*/id(\d+)", "/apple/podcast/{0}", "Apple Podcast"),
    ]

    HINTS = {
        "bilibili": (
            "B站可用路由(uid在space.bilibili.com/{uid}找):\n"
            "  UP主视频: /bilibili/user/video/{uid}\n"
            "  UP主动态: /bilibili/user/dynamic/{uid}\n"
            "  所有视频: /bilibili/user/video-all/{uid}\n"
            "  UP主图文: /bilibili/user/article/{uid}\n"
            "  UP主合集: /bilibili/user/collection/{uid}/{sid}\n"
            "  综合热门: /bilibili/popular/all\n"
            "  每周必看: /bilibili/weekly\n"
            "  排行榜: /bilibili/ranking/all\n"
            "  热搜: /bilibili/hot-search\n"
            "  番剧: /bilibili/bangumi/media/{mediaid}\n"
            "  直播: /bilibili/live/room/{roomID}\n"
            "  搜索: /bilibili/vsearch/{keyword}"
        ),
        "youtube": "YouTube路由:\n  频道: /youtube/channel/{id}\n  用户: /youtube/user/@{name}\n  播放列表: /youtube/playlist/{id}",
        "twitter": "Twitter/X路由:\n  用户: /twitter/user/{name}\n  媒体: /twitter/media/{name}\n  搜索: /twitter/keyword/{kw}",
        "x.com": "Twitter/X路由:\n  用户: /twitter/user/{name}\n  媒体: /twitter/media/{name}",
        "weibo": "微博路由:\n  用户: /weibo/user/{uid}\n  热搜: /weibo/search/hot",
        "zhihu": "知乎路由:\n  用户: /zhihu/people/activities/{id}\n  专栏: /zhihu/zhuanlan/{id}\n  热榜: /zhihu/hot",
        "github": "GitHub路由:\n  Release: /github/release/{owner}/{repo}\n  Commits: /github/commits/{owner}/{repo}",
        "xiaohongshu": "小红书路由:\n  用户笔记: /xiaohongshu/user/{id}/notes",
        "douyin": "抖音路由:\n  用户: /douyin/user/{uid}",
        "instagram": "Instagram路由:\n  用户: /instagram/user/{name}",
        "telegram": "Telegram路由:\n  频道: /telegram/channel/{name}",
        "pixiv": "Pixiv路由:\n  用户: /pixiv/user/{uid}\n  排行: /pixiv/ranking/{mode}",
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
    def __init__(self, width=480):
        self.w = width
        self.pad = 22
        self.font_path = self._find()

    def _find(self):
        base_dir = os.path.dirname(__file__)
        root_fonts = []
        for fn in os.listdir(base_dir):
            lower = fn.lower()
            if lower.endswith((".ttf", ".otf", ".ttc")):
                root_fonts.append(os.path.join(base_dir, fn))
        if root_fonts:
            return root_fonts[0]

        fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
        if os.path.isdir(fonts_dir):
            files = []
            for fn in os.listdir(fonts_dir):
                lower = fn.lower()
                if lower.endswith((".ttf", ".otf", ".ttc")):
                    files.append(fn)

            def score(name: str) -> int:
                n = name.lower()
                s = 0
                if "notosanscjk" in n or "noto sans cjk" in n:
                    s += 100
                if "notosansjp" in n or "noto sans jp" in n:
                    s += 90
                if "notosanssc" in n or "noto sans sc" in n:
                    s += 80
                if "cjk" in n:
                    s += 70
                if "jp" in n or "japan" in n:
                    s += 60
                if "sc" in n or "chinese" in n:
                    s += 50
                if "minecraft" in n:
                    s += 40
                if "中文" in name:
                    s += 30
                return -s

            files.sort(key=score)
            if files:
                return os.path.join(fonts_dir, files[0])

        for p in [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:\\Windows\\Fonts\\msyh.ttc",
        ]:
            if os.path.exists(p):
                return p
        return None

    def _f(self, sz):
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, sz)
            except Exception:
                pass
        return ImageFont.load_default()

    def _wrap(self, txt, font, mw, draw):
        # [修复] 更健壮的换行逻辑，防止某些特殊字符导致崩溃
        if not txt:
            return []
        lines = []
        # 将文本按段落分割，保留空行
        paragraphs = txt.split("\n")
        
        for para in paragraphs:
            # 移除首尾空白，但如果是空行则保留高度
            if not para:
                lines.append("")
                continue
            
            # 逐字扫描
            current_line = ""
            for char in para:
                # 尝试加入字符
                test_line = current_line + char
                # 获取宽度
                w = draw.textlength(test_line, font=font)
                if w > mw:
                    # 如果超宽，且当前行不为空，则推入上一行
                    if current_line:
                        lines.append(current_line)
                        current_line = char
                    else:
                        # 强制切断（针对超长连续字符）
                        lines.append(char)
                        current_line = ""
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
        return lines
    def _round_image(self, img, radius=14):
        """给图片加圆角效果
        原理：画一个圆角矩形白色蒙版，把图片贴进去
        需要 Pillow>=8.2（rounded_rectangle 支持）
        """
        img = img.convert("RGBA")
        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=radius, fill=255)
        white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        white.paste(img, mask=mask)
        return white.convert("RGB")

    def _draw_avatar_circle(self, im, x, y, size, char, color):
        """在图片上绘制一个带文字的圆形头像
        用4x超采样画大圆再缩小，实现抗锯齿的平滑圆形边缘
        char: 圆心里显示的字符（频道名首字）
        color: 圆形的RGB背景色
        """
        scale = 4
        big = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
        bd = ImageDraw.Draw(big)
        bd.ellipse([(0, 0), (size * scale - 1, size * scale - 1)], fill=color + (255,))
        big = big.resize((size, size), Image.LANCZOS)
        im.paste(big, (x, y), big)
        # 在圆心画字
        d = ImageDraw.Draw(im)
        font = self._f(int(size * 0.42))
        try:
            bbox = font.getbbox(char)
            cw = bbox[2] - bbox[0]
            ch = bbox[3] - bbox[1]
            d.text((x + (size - cw) / 2 - bbox[0], y + (size - ch) / 2 - bbox[1]),
                   char, font=font, fill=(255, 255, 255))
        except Exception:
            d.text((x + size // 4, y + size // 4), "?", font=font, fill=(255, 255, 255))

    def _format_time(self, ts_str):
        """把RSS的长时间字符串简化成 YYYY-MM-DD HH:MM 格式
        失败则原样截断返回，保证不崩溃
        """
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
    def make(self, channel="", title="", desc="", link="", ts="", thumb=None, avatar=None, comment="", bot_avatar=None, bot_provider_name=""):
        """生成 Twitter/X 风格的动态卡片

        布局（模仿推特时间线的单条推文）:
        ┌──────────────────────────────────┐
        │  [●]  频道名 · 2025-02-19 19:54  │
        │       正文正文正文正文正文        │
        │       正文正文...                │
        │       ╭────────────────────╮     │
        │       │   图片(圆角14px)    │     │
        │       ╰────────────────────╯     │
        │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
        │       🔗 来源链接                │
        └──────────────────────────────────┘
        多条拼合后底部分割线连成连续时间线。
        """
        # ============ Twitter/X 精确配色 ============
        BG       = (255, 255, 255)       # 背景纯白
        C_NAME   = (15, 20, 25)          # 名字黑 (#0F1419)
        C_BODY   = (15, 20, 25)          # 正文黑
        C_GRAY   = (83, 100, 113)        # 副文字灰 (#536471)
        C_BORDER = (239, 243, 244)       # 分割线 (#EFF3F4)
        C_BLUE   = (29, 155, 240)        # Twitter蓝 (#1D9BF0)

        # ============ 布局常量 ============
        W   = self.w                     # 卡片总宽度(默认480)
        PX  = 16                         # 左右内边距
        PY  = 14                         # 上下内边距
        AVT = 48                         # 头像直径
        GAP = 12                         # 头像和内容之间水平间距
        CX  = PX + AVT + GAP            # 内容区起始X坐标
        CW  = W - CX - PX               # 内容区可用宽度

        # ============ 字体 ============
        # fn: 频道名  ft: 时间  fh: 标题(大)  fb: 正文  fm: 底部链接
        fn = self._f(16)                 # 频道名
        ft = self._f(12)                 # 时间
        fh = self._f(18)                 # 标题（比正文大，视觉层次分明）
        fb = self._f(15)                 # 正文
        fm = self._f(12)                 # 底部链接

        # ============ 1. 预计算文本换行 ============
        tmp = Image.new("RGB", (1, 1))
        d0 = ImageDraw.Draw(tmp)

        # 标题和正文分开处理（而不是合并成一段）
        # 这样标题可以用大字体，正文用小字体，有层次感
        # 如果标题和正文内容重复，则只显示标题
        show_title = title and title not in ("无标题", "")
        title_lines = self._wrap(title, fh, CW, d0) if show_title else []
        if len(title_lines) > 4:
            title_lines = title_lines[:4]
            title_lines[-1] = title_lines[-1].rstrip() + "..."
        TITLE_LH = 30  # 标题行高(px)：18px字体 × 1.67倍 ≈ 30

        # 正文：如果和标题完全相同就不重复显示
        desc_text = (desc or "").strip()
        if show_title and desc_text == title.strip():
            desc_text = ""
        desc_lines = self._wrap(desc_text, fb, CW, d0) if desc_text else []
        if len(desc_lines) > 15:
            desc_lines = desc_lines[:15]
            desc_lines[-1] = desc_lines[-1].rstrip() + "..."
        DESC_LH = 26   # 正文行高(px)：15px字体 × 1.73倍 ≈ 26，不再挤

        # ============ 2. 处理缩略图 ============
        pic = None
        pic_h = 0
        if thumb:
            try:
                src = Image.open(BytesIO(thumb))
                # 统一转RGBA，处理透明PNG
                if src.mode != "RGBA":
                    src = src.convert("RGBA")

                ratio = CW / src.width
                new_h = int(src.height * ratio)
                # 限制最大高度，防竖长图撑爆卡片
                max_h = int(CW * 1.3)
                src = src.resize((CW, min(new_h, max_h)), Image.LANCZOS)
                if new_h > max_h:
                    src = src.crop((0, 0, CW, max_h))
                    new_h = max_h

                # 把透明图合成到白底上（防止透明区域变黑）
                white_bg = Image.new("RGBA", (CW, min(new_h, max_h)), (255, 255, 255, 255))
                try:
                    white_bg.paste(src, mask=src.split()[3])
                except Exception:
                    white_bg.paste(src)
                # 加圆角
                pic = self._round_image(white_bg.convert("RGB"), radius=14)
                pic_h = pic.height
            except Exception:
                pic = None

        # 格式化时间
        time_str = self._format_time(ts)

        # ============ 3. 计算总高度 ============
        # 逐块累加：上边距 → 头像区 → 标题 → 正文 → 图片 → 分割线 → 链接 → 下边距
        H = PY                                                 # 上边距
        H += max(AVT, 24) + 10                                 # 头像/名字区 + 间距
        if title_lines:
            H += len(title_lines) * TITLE_LH + 8              # 标题块 + 底部间距
        if desc_lines:
            H += len(desc_lines) * DESC_LH + 10               # 正文块 + 底部间距
        if pic:
            H += pic_h + 14                                    # 图片 + 底部间距
        H += 1 + 10                                            # 分割线 + 间距
        if link:
            H += 18 + 4                                        # 链接行
        if comment:
            H += 6 + 1 + 10                                    # 锐评分割线
            comment_lines_est = min(3, max(1, len(comment) // 20 + 1))
            H += max(32 + 8, comment_lines_est * 18 + 8)       # 锐评区域
            if bot_provider_name:
                H += 14
        H += PY                                                # 下边距

        # ============ 4. 绘制画布 ============
        im = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(im)
        cy = PY  # 当前Y游标

        # ---- 头像 ----
        if avatar and isinstance(avatar, bytes) and len(avatar) > 100:
            try:
                avt_img = Image.open(BytesIO(avatar)).convert("RGBA")
                avt_img = avt_img.resize((AVT, AVT), Image.LANCZOS)
                # 圆形裁剪
                mask = Image.new("L", (AVT, AVT), 0)
                md = ImageDraw.Draw(mask)
                md.ellipse([(0, 0), (AVT - 1, AVT - 1)], fill=255)
                white = Image.new("RGB", (AVT, AVT), (255, 255, 255))
                white.paste(avt_img.convert("RGB"), mask=mask)
                im.paste(white, (PX, cy))
                # 画圆形边框
                dr.ellipse([(PX, cy), (PX + AVT - 1, cy + AVT - 1)], outline=C_BORDER, width=1)
            except Exception:
                avt_char = (channel or "?")[0] if channel else "?"
                self._draw_avatar_circle(im, PX, cy, AVT, avt_char, C_BLUE)
        else:
            avt_char = "?"
            for c in (channel or ""):
                if c.strip():
                    avt_char = c
                    break
            self._draw_avatar_circle(im, PX, cy, AVT, avt_char, C_BLUE)

        # ---- 频道名 + 时间（同一行，模仿推特 "Name · 2h"） ----
        # [修复] 强制截断超长频道名，防止和时间重叠乱码
        name_y = cy + (AVT - 20) // 2  # 垂直居中于头像
        
        display_name = channel or "未知频道"
        # 去掉RSSHub可能附加的冗余后缀，让名字更短更干净
        display_name = display_name.replace(" - Community Posts - YouTube", "").replace(" - YouTube", "")
        
        if time_str:
            dot = " · "
            # 预留给时间和点的宽度
            time_w = d0.textlength(time_str, font=ft)
            dot_w = d0.textlength(dot, font=ft)
            
            # 计算名字最大允许宽度 = 总宽度 - 时间宽 - 点宽 - 缓冲(10px)
            max_name_w = CW - time_w - dot_w - 10
            
            # 测量当前名字宽度
            current_w = d0.textlength(display_name, font=fn)
            
            # 如果名字太长，就循环截断直到放得下
            if current_w > max_name_w:
                while current_w > max_name_w and len(display_name) > 1:
                    display_name = display_name[:-1]
                    current_w = d0.textlength(display_name + "...", font=fn)
                display_name += "..."
            
            # 绘制名字
            dr.text((CX, name_y), display_name, font=fn, fill=C_NAME)
            
            # 紧接着绘制 · 时间
            final_name_w = d0.textlength(display_name, font=fn)
            dr.text((CX + final_name_w, name_y + 1), dot, font=ft, fill=C_GRAY)
            dr.text((CX + final_name_w + dot_w, name_y + 1), time_str, font=ft, fill=C_GRAY)
        else:
            # 没有时间，直接画名字（也要防止超长）
            current_w = d0.textlength(display_name, font=fn)
            if current_w > CW:
                while current_w > CW and len(display_name) > 1:
                    display_name = display_name[:-1]
                    current_w = d0.textlength(display_name + "...", font=fn)
                display_name += "..."
            dr.text((CX, name_y), display_name, font=fn, fill=C_NAME)

        cy += max(AVT, 24) + 10

        # ---- 标题（大字，深黑） ----
        if title_lines:
            for line in title_lines:
                dr.text((CX, cy), line, font=fh, fill=C_NAME)
                cy += TITLE_LH
            cy += 8

        # ---- 正文（小字，深灰，和标题形成对比） ----
        if desc_lines:
            for line in desc_lines:
                dr.text((CX, cy), line, font=fb, fill=C_GRAY)
                cy += DESC_LH
            cy += 10

        # ---- 图片（圆角） ----
        if pic:
            im.paste(pic, (CX, cy))
            # 加圆角边框线，让图片边缘更清晰
            dr.rounded_rectangle(
                [(CX, cy), (CX + CW - 1, cy + pic_h - 1)],
                radius=14, outline=C_BORDER, width=1
            )
            cy += pic_h + 14

        # ---- 分割线 ----
        dr.line([(PX, cy), (W - PX, cy)], fill=C_BORDER, width=1)
        cy += 10

        # ---- 链接 ----
        if link:
            lk = link if len(link) <= 50 else link[:50] + "..."
            dr.text((CX, cy), "🔗 " + lk, font=fm, fill=C_BLUE)
            cy += 22
        # ---- 锐评栏 ----
        if comment:
            cy += 6
            dr.line([(PX, cy), (W - PX, cy)], fill=C_BORDER, width=1)
            cy += 10

            # bot头像（小圆形，32px）
            bot_avt_size = 32
            bot_avt_x = CX
            if bot_avatar and isinstance(bot_avatar, bytes) and len(bot_avatar) > 100:
                try:
                    bavt = Image.open(BytesIO(bot_avatar)).convert("RGBA")
                    bavt = bavt.resize((bot_avt_size, bot_avt_size), Image.LANCZOS)
                    bmask = Image.new("L", (bot_avt_size, bot_avt_size), 0)
                    bmd = ImageDraw.Draw(bmask)
                    bmd.ellipse([(0, 0), (bot_avt_size - 1, bot_avt_size - 1)], fill=255)
                    bwhite = Image.new("RGB", (bot_avt_size, bot_avt_size), (255, 255, 255))
                    bwhite.paste(bavt.convert("RGB"), mask=bmask)
                    im.paste(bwhite, (bot_avt_x, cy))
                    dr.ellipse([(bot_avt_x, cy), (bot_avt_x + bot_avt_size - 1, cy + bot_avt_size - 1)], outline=C_BORDER, width=1)
                except Exception:
                    self._draw_avatar_circle(im, bot_avt_x, cy, bot_avt_size, "B", (100, 100, 100))
            else:
                self._draw_avatar_circle(im, bot_avt_x, cy, bot_avt_size, "B", (100, 100, 100))

            # 锐评文字
            comment_x = bot_avt_x + bot_avt_size + 8
            comment_w = W - comment_x - PX
            fc_comment = self._f(13)
            comment_lines = self._wrap(comment, fc_comment, comment_w, dr)
            if len(comment_lines) > 3:
                comment_lines = comment_lines[:3]
                comment_lines[-1] = comment_lines[-1][:-2] + "..."

            comment_y = cy + 2
            for cl in comment_lines:
                dr.text((comment_x, comment_y), cl, font=fc_comment, fill=C_GRAY)
                comment_y += 18

            # 服务商标注
            if bot_provider_name:
                provider_font = self._f(10)
                provider_text = "via " + bot_provider_name
                dr.text((comment_x, comment_y + 2), provider_text, font=provider_font, fill=(180, 180, 180))
                comment_y += 14

            cy = max(cy + bot_avt_size + 8, comment_y + 8)

        # 底部边线（多条拼合时充当条目间分隔线，像推特时间线的灰线）
        dr.line([(0, H - 1), (W, H - 1)], fill=C_BORDER, width=1)

        buf = BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

@register("astrbot_plugin_myrss", "MyRSS", "RSS订阅插件(LLM增强版)", "1.0.0", "")
class MyRssPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.logger = logging.getLogger("astrbot")
        self.ctx = context
        self.cfg = config
        self.dh = DataHandler()

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
        self.content_filter = config.get("content_filter", True)
        self._comment_cache = {}  # key=item_link, value=comment_text
        # 活跃群检测（学新闻插件）
        self._active_groups = set()  # 内存中记录有人说话的群
        self._group_data_file = "data/astrbot_plugin_myrss_groups.json"
        self._group_data = self._load_group_data()

        # 全局订阅
        self.global_feeds = [
            line.strip() for line in config.get("global_feeds", "").split("\n")
            if line.strip() and line.strip().startswith("/")
        ]
        self.global_feed_interval = max(config.get("global_feed_interval", 15), 5)
        self.push_delay_min = config.get("push_delay_min", 5.0)
        self.push_delay_max = config.get("push_delay_max", 8.0)
        self.filter_provider_id = config.get("filter_provider_id", "")
        self.image_caption_provider_id = config.get("image_caption_provider_id", "")

        # 注册全局订阅的定时任务
        if self.global_feeds:
            self._setup_global_feeds()
        self.pic = PicHandler(self.adjust_pic)
        self.card = CardGen()

        # 防并发锁，key = (url, user)
        self._locks: dict = {}

        # [防冲突] 在创建新调度器前，先杀掉模块级残留的老调度器
        # 场景：插件热更新时框架直接创建新实例，老实例的destroy()可能未被调用
        # 如果不杀，老调度器继续运行老代码的job，和新调度器同时推送→双推
        global _ACTIVE_SCHED
        if _ACTIVE_SCHED is not None:
            try:
                if _ACTIVE_SCHED.running:
                    _ACTIVE_SCHED.shutdown(wait=True)
                    self.logger.warning("MyRSS: 已强制停止残留的老调度器 id=%s", id(_ACTIVE_SCHED))
            except Exception:
                pass
            _ACTIVE_SCHED = None

        self.sched = AsyncIOScheduler()
        _ACTIVE_SCHED = self.sched  # [防冲突] 注册为全局引用，下次init时可找到并销毁
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
        conn = aiohttp.TCPConnector(ssl=False)
        to = aiohttp.ClientTimeout(total=30, connect=10)

        async def _try(u: str):
            try:
                async with aiohttp.ClientSession(trust_env=True, connector=conn, timeout=to, headers=headers) as s:
                    async with s.get(u) as r:
                        if r.status != 200:
                            return None
                        return await r.read()
            except Exception:
                return None

        data = await _try(url)
        if data is not None:
            return data

        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
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
        except ValueError:
            try:
                root = etree.fromstring(
                    text.replace(b'encoding="gb2312"', b'')
                        .replace(b'encoding="GB2312"', b'')
                )
            except Exception:
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
                # media:thumbnail  → RSS标准缩略图（视频路由常用）
                # media:content    → 有些源把封面图放这里（YouTube等）
                # enclosure        → RSS附件
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

    async def _add(self, url: str, cron_expr: str, event: AstrMessageEvent):
        user = event.unified_msg_origin
        if url in self.dh.data:
            items = await self._poll(url)
            if not items:
                return event.plain_result("无法从该源获取内容，请检查链接。")
            self.dh.data[url]["subscribers"][user] = {
                "cron_expr": cron_expr,
                "last_update": items[0].pubDate_timestamp,
                "latest_link": items[0].link,
                "seen_links": [it.link for it in items if it.link][:200],
            }
        else:
            text = await self._fetch(url)
            if text is None:
                return event.plain_result("无法访问: " + url + "\n请检查RSSHub端点是否可用。")
            try:
                title, desc, avatar = self.dh.parse_channel_info(text)
            except Exception as e:
                return event.plain_result("解析失败: " + str(e))
            items = await self._poll(url)
            if not items:
                return event.plain_result("源可访问但无内容条目。")
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
    # ============================================================
    #  活跃群检测
    # ============================================================

    def _load_group_data(self) -> dict:
        if os.path.exists(self._group_data_file):
            try:
                with open(self._group_data_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"groups": {}, "blocked_feeds": {}}
        # blocked_feeds 结构: { "群unified_id": ["/twitter/user/elonmusk", ...] }

    def _save_group_data(self):
        try:
            with open(self._group_data_file, "w", encoding="utf-8") as f:
                json.dump(self._group_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error("[MyRSS] save group data failed: %s", e)

    def _mark_active(self, unified_id: str):
        """标记某个群有人说话"""
        if "GroupMessage" not in unified_id:
            return
        self._active_groups.add(unified_id)
        if unified_id not in self._group_data["groups"]:
            self._group_data["groups"][unified_id] = {
                "active": True,
                "dormant": False,
                "last_activity": int(time.time()),
            }
        else:
            self._group_data["groups"][unified_id]["active"] = True
            self._group_data["groups"][unified_id]["dormant"] = False
            self._group_data["groups"][unified_id]["last_activity"] = int(time.time())

    def _get_active_groups(self) -> list:
        """获取所有活跃群的unified_id列表"""
        result = []
        for uid, info in self._group_data["groups"].items():
            if info.get("active") and not info.get("dormant"):
                result.append(uid)
        return result

    def _reset_activity(self):
        """推送后重置活跃状态，等下次有人说话"""
        for uid in self._group_data["groups"]:
            self._group_data["groups"][uid]["active"] = uid in self._active_groups
        self._active_groups.clear()
        self._save_group_data()

    @filter.regex(r"[\s\S]*")
    async def _catch_activity(self, event: AstrMessageEvent):
        """捕获所有消息记录活跃度，不产生回复"""
        if hasattr(event, "unified_msg_origin"):
            self._mark_active(event.unified_msg_origin)
    # ============================================================
    #  全局订阅
    # ============================================================

    def _setup_global_feeds(self):
        """为全局订阅源注册定时任务"""
        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            self.logger.warning("[MyRSS] no endpoints for global feeds")
            return

        for route in self.global_feeds:
            url = eps[0].rstrip("/") + route

            # 确保数据里有这个URL的记录
            if url not in self.dh.data:
                self.dh.data[url] = {
                    "subscribers": {},
                    "info": {"title": route, "description": "全局订阅"},
                    "global": True,
                }

            self.dh.data[url]["global"] = True

            # 注册定时任务
            if self.global_feed_interval < 60:
                cron = f"*/{self.global_feed_interval} * * * *"
            else:
                cron = f"0 */{self.global_feed_interval // 60} * * *"

            job_id = f"myrss_global_{url}"
            self.sched.add_job(
                self._global_feed_job, "cron",
                **self._cron(cron),
                args=[url],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=120,
            )
            self.logger.info("[MyRSS] global feed registered: %s every %d min", route, self.global_feed_interval)

        self.dh.save()

    async def _global_feed_job(self, url: str):
        """全局订阅的定时推送"""
        self.logger.info("[MyRSS] global feed job: %s", url)

        # 获取全局seen_links
        if url not in self.dh.data:
            return
        feed_data = self.dh.data[url]
        seen = set(feed_data.get("global_seen_links", []))

        items = await self._poll(url, num=self.max_poll, after_ts=feed_data.get("global_last_update", 0))
        if not items:
            return

        def item_key(it):
            return it.link if it.link else f"{it.title}|{it.pubDate_timestamp}"

        new_items = [it for it in items if item_key(it) not in seen]
        if not new_items:
            return

        # 内容过滤
        safe_items = []
        for it in new_items:
            if self.content_filter:
                if await self._check_content_safe(it):
                    safe_items.append(it)
                else:
                    self.logger.info("[MyRSS] global feed filtered: %s", it.title[:30])
            else:
                safe_items.append(it)

        if not safe_items:
            # 更新seen即使被过滤
            new_keys = [item_key(it) for it in new_items]
            feed_data["global_seen_links"] = (new_keys + feed_data.get("global_seen_links", []))[:200]
            ts_list = [it.pubDate_timestamp for it in new_items if it.pubDate_timestamp > 0]
            if ts_list:
                feed_data["global_last_update"] = max(ts_list)
            self.dh.save()
            return

        # 更新seen_links（推送前，防重复）
        new_keys = [item_key(it) for it in safe_items]
        feed_data["global_seen_links"] = (new_keys + feed_data.get("global_seen_links", []))[:200]
        ts_list = [it.pubDate_timestamp for it in safe_items if it.pubDate_timestamp > 0]
        if ts_list:
            feed_data["global_last_update"] = max(ts_list)
        self.dh.save()

        # 生成卡片（只生成一次，推给所有群）
        batch = safe_items[:5]
        if len(batch) > 1:
            cards = [await self._make_card_b64(it) for it in batch]
            cards = [c for c in cards if c]
            if cards:
                merged = self._merge_cards_b64(cards)
                comps = [Comp.Image.fromBase64(merged)] if merged else None
            else:
                comps = None
        elif batch:
            comps = await self._make_comps(batch[0])
        else:
            return

        if not comps:
            return

        # 推送给所有活跃群（排除屏蔽了该源的群）
        active_groups = self._get_active_groups()
        # 从URL提取路由部分用于匹配屏蔽列表
        eps = self.dh.data.get("rsshub_endpoints", [])
        feed_route = url
        for ep in eps:
            ep = ep.rstrip("/")
            if url.startswith(ep):
                feed_route = url[len(ep):]
                break

        blocked = self._group_data.get("blocked_feeds", {})
        push_groups = [g for g in active_groups if feed_route not in blocked.get(g, [])]
        self.logger.info("[MyRSS] global push %d items to %d groups (skipped %d blocked)",
                        len(batch), len(push_groups), len(active_groups) - len(push_groups))

        for group_id in push_groups:
            try:
                pn = group_id.split(":")[0]
                if pn == "aiocqhttp" and self.compose:
                    node = Comp.Node(uin=0, name="Astrbot", content=comps)
                    await self.ctx.send_message(group_id, MessageChain(chain=[node], use_t2i_=self.t2i))
                else:
                    await self.ctx.send_message(group_id, MessageChain(chain=comps, use_t2i_=self.t2i))

                # 随机延迟防风控
                delay = random.uniform(self.push_delay_min, self.push_delay_max)
                await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error("[MyRSS] global push failed to %s: %s", group_id, e)

        self._reset_activity()
        self.logger.info("[MyRSS] global push done")
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
        cache_key = item.link or (item.title + "|" + str(item.pubDate_timestamp))

        # 命中缓存直接返回
        if cache_key in self._comment_cache:
            return self._comment_cache[cache_key]

        provider_id = self.filter_provider_id if self.filter_provider_id else await self._get_provider_id()
        if not provider_id:
            self.logger.warning("[MyRSS] no provider for comment")
            return ""

        # 构造prompt
        content_summary = item.title
        if item.description:
            desc_short = item.description[:200]
            content_summary += "\n" + desc_short

        # 获取人格设定
        system_prompt = None
        if self.comment_persona:
            try:
                cfg = self.ctx.get_config()
                personas = cfg.get("persona", [])
                for p in personas:
                    if p.get("name") == self.comment_persona:
                        system_prompt = p.get("prompt", "")
                        break
            except Exception:
                pass

        prompt = (
            f"你正在看一条来自「{item.chan_title}」的动态更新，内容如下：\n"
            f"---\n{content_summary}\n---\n"
            f"请用你的人设风格，对这条动态写一句简短锐评（{self.comment_max_length}字以内）。"
            f"要求：自然、有个性、可以吐槽或夸奖。不要复述原文。不要加引号。用中文表达。"
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

    async def _check_content_safe(self, item: RSSItem) -> bool:
        """检查内容是否安全，不安全返回False"""
        if not self.content_filter:
            return True

        provider_id = self.filter_provider_id if self.filter_provider_id else await self._get_provider_id()
        if not provider_id:
            return True  # 没有provider就不过滤，放行

        content = (item.title + " " + (item.description or ""))[:300]

        prompt = (
            "判断以下内容是否包含以下任何一种不当信息：\n"
            "1. 对中国国情比较政治敏感/反动和过激言论\n"
            "2. 暴力血腥\n"
            "3. 色情/露骨性内容\n"
            "4. 其他可能导致社交平台账号被封禁的内容\n\n"
            f"内容：{content}\n\n"
            "只回答 SAFE 或 UNSAFE，不要解释。"
        )

        try:
            resp = await self.ctx.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            result = (resp.completion_text or "").strip().upper()
            if "UNSAFE" in result:
                self.logger.warning("[MyRSS] content filtered: %s", item.title[:50])
                return False
            return True
        except Exception as e:
            self.logger.error("[MyRSS] content filter failed, blocking for safety: %s", e)
            return False  # 过滤出错时拦截，宁可不推也不冒险
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
            if await self._check_content_safe(item):
                comment = await self._generate_comment(item)
            else:
                return ""  # 不安全内容，返回空跳过

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

        return self.card.make(
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
            if await self._check_content_safe(item):
                comment = await self._generate_comment(item)
            else:
                self.logger.warning("[MyRSS] unsafe content skipped: %s", item.title[:50])
                return [Comp.Plain("[内容已过滤]")]

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
            b64 = self.card.make(
                channel=item.chan_title, title=item.title, desc=item.description,
                link="" if self.hide_url else item.link, ts=item.pubDate or "", thumb=tb,
                avatar=avt_data,
                comment=comment,
                bot_avatar=bot_avt,
                bot_provider_name=self.bot_provider_name,
            )
            comps.append(Comp.Image.fromBase64(b64))
        except Exception as e:
            self.logger.error("卡片生成失败: %s", e)
            comps.append(Comp.Plain("📡 " + item.chan_title + "\n📝 " + item.title + "\n" + item.description))

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
        min_link = ""  # 公共拉取不用after_link过滤，靠seen_links去重

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

    async def _cron_cb(self, url: str, user: str) -> None:
        """带锁的定时回调入口，防止同一订阅并发执行"""
        lock = self._get_lock(url, user)
        async with lock:
            await self._cron_cb_inner(url, user)

    async def _cron_cb_inner(self, url: str, user: str, prefetched_items=None) -> None:
        # [防冲突] 每次推送前从磁盘重载数据，拿到最新的seen_links
        # 原因：新老实例各持有独立的内存副本(self.dh.data)，
        # 如果只读内存，老实例看不到新实例写入的seen_links→重复推送
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
            return

        def item_key(it: RSSItem) -> str:
            return it.link if it.link else f"{it.title}|{it.pubDate_timestamp}"

        # 去重
        seen = set(si.get("seen_links", []))
        new_items = [it for it in items if item_key(it) not in seen]

        if not new_items:
            si["latest_link"] = items[0].link
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

        pn = user.split(":")[0]
        merge_limit = 5
        batch = new_items[:merge_limit]

        if len(batch) > 1:
            cards_raw = [await self._make_card_b64(it) for it in batch]
            cards = [c for c in cards_raw if c]  # 过滤掉被内容审核拦截的空卡片
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
    #  LLM 工具
    # ============================================================

    @filter.llm_tool(name="myrss_subscribe")
    async def tool_sub(self, event: AstrMessageEvent, url: str = "https://example.com", interval: int = 15, target_group: str = ""):
        """用户想订阅某个网站/博主更新时调用。

        Args:
            url(string): 链接或路由路径
            interval(int): 间隔分钟数，默认15
            target_group(string): 指定推送到的群号(可选，不填则推到当前会话)
        """
        if not url or url == "https://example.com":
            yield event.plain_result(
                "需要用户提供链接或路由。支持平台：B站、YouTube、Twitter/X、微博、知乎等。\n"
                "路由示例：/youtube/community/@用户名、/twitter/user/用户名、/bilibili/user/dynamic/UID\n"
                "可选参数：interval(分钟)、target_group(指定群号)\n"
                "详见 https://docs.rsshub.app"
            )
            return
        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            yield event.plain_result(
                "尚未配置RSSHub端点，请告诉用户执行以下命令之一：\n"
                "/myrss rsshub add https://rsshub.rssforever.com\n"
                "/myrss rsshub add https://rsshub.app\n"
                "配置后即可订阅。"
            )
            return
        if url.startswith("/"):
            furl = eps[0] + url
        elif url.startswith("http"):
            r = URLMapper.match(url)
            if r:
                route, pn = r
                furl = eps[0] + route
            else:
                yield event.plain_result("无法自动识别该链接。\n\n" + URLMapper.suggest(url) + "\n\n请选择路由后用/开头再次调用。")
                return
        else:
            yield event.plain_result("请提供http开头的链接或/开头的路由。")
            return
        if interval < 15:
            interval = 15

        # 如果已有订阅者，间隔只能取更大值（保护公共源）
        if furl in self.dh.data:
            existing_subs = self.dh.data[furl].get("subscribers", {})
            if existing_subs:
                def cron_to_minutes(expr: str) -> int:
                    try:
                        f = expr.split(" ")
                        # */15 * * * *
                        if f[0].startswith("*/"):
                            return int(f[0][2:])
                        # 0 */1 * * *
                        if f[1].startswith("*/"):
                            return int(f[1][2:]) * 60
                        return 60
                    except Exception:
                        return 60

                max_existing = max(cron_to_minutes(si["cron_expr"]) for si in existing_subs.values())
                if interval < max_existing:
                    interval = max_existing
                    yield event.plain_result(f"⚠️ 已有订阅者使用{max_existing}分钟间隔，为保护公共源已自动调整为{max_existing}分钟。")

        # 分钟制cron
        if interval < 60:
            cron_expr = f"*/{interval} * * * *"
        else:
            cron_expr = f"0 */{interval // 60} * * *"
        # 如果指定了目标群
        if target_group:
            # 构造目标群的unified_msg_origin
            pn = event.unified_msg_origin.split(":")[0]
            target_umo = f"{pn}:GroupMessage:{target_group}"
            # 临时替换event的origin
            original_umo = event.unified_msg_origin
            event._unified_msg_origin = target_umo
            ret = await self._add(furl, cron_expr, event)
            event._unified_msg_origin = original_umo
            if isinstance(ret, MessageEventResult):
                yield ret
                return
            self._reload_jobs()
            yield event.plain_result(
                "✅ 订阅成功！\n📡 " + ret["title"] +
                "\n⏰ 每" + str(show_interval) + unit +
                "\n📍 推送到群 " + target_group +
                "\n🔗 " + furl
            )
            return
        ret = await self._add(furl, cron_expr, event)
        if isinstance(ret, MessageEventResult):
            yield ret
            return
        self._reload_jobs()
        unit = "分钟" if interval < 60 else "小时"
        show_interval = interval if interval < 60 else interval // 60
        yield event.plain_result("✅ 订阅成功！\n📡 " + ret["title"] + "\n📝 " + ret["description"] + "\n⏰ 每" + str(show_interval) + unit + "\n🔗 " + furl)

    @filter.llm_tool(name="myrss_list")
    async def tool_list(self, event: AstrMessageEvent, query: str = "all"):
        """用户问订阅了什么时调用。
    
        Args:
            query(string): 固定传all
        """
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if not urls:
            yield event.plain_result("当前没有任何订阅。")
            return
        txt = "📋 订阅列表：\n"
        for i, u in enumerate(urls):
            info = self.dh.data[u]["info"]
            cr = self.dh.data[u]["subscribers"][user]["cron_expr"]
            txt += "  " + str(i) + ". " + info["title"] + " [" + cr + "]\n"
        yield event.plain_result(txt)
    @filter.llm_tool(name="myrss_block_feed")
    async def tool_block(self, event: AstrMessageEvent, feed_keyword: str = ""):
        """当群友说不想看某个订阅源的推送时调用（如"别发推特了""不要马斯克的"）。

        Args:
            feed_keyword(string): 要屏蔽的源关键词（如elonmusk、flag__chan、bilibili等）
        """
        if not feed_keyword:
            yield event.plain_result(
                "请告诉我要屏蔽哪个源。当前全局订阅源：\n" +
                "\n".join(f"  {i}. {r}" for i, r in enumerate(self.global_feeds)) +
                "\n回复关键词即可屏蔽，如 'elonmusk' 或 'bilibili'"
            )
            return

        group_id = event.unified_msg_origin
        if "GroupMessage" not in group_id:
            yield event.plain_result("此功能仅在群聊中可用。")
            return

        # 模糊匹配
        matched = []
        for route in self.global_feeds:
            if feed_keyword.lower() in route.lower():
                matched.append(route)

        if not matched:
            yield event.plain_result(
                f"没找到包含 '{feed_keyword}' 的订阅源。当前全局源：\n" +
                "\n".join(f"  {r}" for r in self.global_feeds)
            )
            return

        blocked = self._group_data.setdefault("blocked_feeds", {})
        group_blocked = blocked.setdefault(group_id, [])

        newly_blocked = []
        for route in matched:
            if route not in group_blocked:
                group_blocked.append(route)
                newly_blocked.append(route)

        if not newly_blocked:
            yield event.plain_result("这些源在本群已经屏蔽了：\n" + "\n".join(matched))
            return

        self._save_group_data()
        yield event.plain_result(
            "已在本群屏蔽以下推送：\n" +
            "\n".join(f"  ✅ {r}" for r in newly_blocked) +
            "\n其他群不受影响。如需恢复，说'恢复推送xxx'即可。"
        )

    @filter.llm_tool(name="myrss_unblock_feed")
    async def tool_unblock(self, event: AstrMessageEvent, feed_keyword: str = ""):
        """当群友说想恢复某个被屏蔽的推送时调用。

        Args:
            feed_keyword(string): 要恢复的源关键词
        """
        group_id = event.unified_msg_origin
        if "GroupMessage" not in group_id:
            yield event.plain_result("此功能仅在群聊中可用。")
            return

        blocked = self._group_data.get("blocked_feeds", {})
        group_blocked = blocked.get(group_id, [])

        if not group_blocked:
            yield event.plain_result("本群没有屏蔽任何全局推送源。")
            return

        if not feed_keyword:
            yield event.plain_result(
                "本群当前屏蔽的源：\n" +
                "\n".join(f"  {i}. {r}" for i, r in enumerate(group_blocked)) +
                "\n告诉我要恢复哪个即可。"
            )
            return

        matched = [r for r in group_blocked if feed_keyword.lower() in r.lower()]
        if not matched:
            yield event.plain_result(f"没找到包含 '{feed_keyword}' 的已屏蔽源。")
            return

        for r in matched:
            group_blocked.remove(r)
        self._save_group_data()

        yield event.plain_result(
            "已在本群恢复以下推送：\n" +
            "\n".join(f"  ✅ {r}" for r in matched)
        )
    @filter.llm_tool(name="myrss_unsubscribe")
    async def tool_unsub(self, event: AstrMessageEvent, idx: int = 0):
        """取消订阅，先调用myrss_list获取编号。
    
        Args:
            idx(int): 订阅编号
        """
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if idx < 0 or idx >= len(urls):
            yield event.plain_result("编号" + str(idx) + "不存在，有效范围0~" + str(len(urls) - 1))
            return
        u = urls[idx]
        t = self.dh.data[u]["info"]["title"]
        self.dh.data[u]["subscribers"].pop(user)
        self.dh.save()
        self._reload_jobs()
        yield event.plain_result("✅ 已取消: " + t)

    # ============================================================
    #  手动命令
    # ============================================================

    @filter.command_group("myrss")
    def myrss(self):
        pass

    @myrss.group("rsshub")
    def rsshub(self, event: AstrMessageEvent):
        pass

    @rsshub.command("add")
    async def rsshub_add(self, event: AstrMessageEvent, url: str):
        """添加RSSHub端点"""
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
            yield event.plain_result("暂无端点，请先 /myrss rsshub add <url>")
            return
        txt = "RSSHub端点：\n"
        for i, x in enumerate(eps):
            txt += "  " + str(i) + ": " + x + "\n"
        yield event.plain_result(txt)

    @rsshub.command("remove")
    async def rsshub_rm(self, event: AstrMessageEvent, idx: int):
        """删除RSSHub端点"""
        eps = self.dh.data["rsshub_endpoints"]
        if idx < 0 or idx >= len(eps):
            yield event.plain_result("编号越界")
            return
        removed = eps.pop(idx)
        self.dh.save()
        yield event.plain_result("✅ 已删除: " + removed)

    @myrss.command("list")
    async def cmd_list(self, event: AstrMessageEvent):
        """列出当前订阅"""
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if not urls:
            yield event.plain_result("暂无订阅")
            return
        txt = "订阅列表：\n"
        for i, u in enumerate(urls):
            info = self.dh.data[u]["info"]
            txt += "  " + str(i) + ". " + info["title"] + "\n"
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
