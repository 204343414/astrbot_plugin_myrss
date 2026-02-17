import os
import json
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

from lxml import etree
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star, register
from astrbot.api import AstrBotConfig
import astrbot.api.message_components as Comp
from typing import List


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
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
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
        return title, desc or ""

    def strip_html_pic(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return [img.get("src") for img in soup.find_all("img") if img.get("src")]

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
                        cx, cy = random.choice([(0,0),(w-1,0),(0,h-1),(w-1,h-1)])
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
        (r"space\.bilibili\.com/(\d+)", "/bilibili/user/video/{0}", "B站UP主视频"),
        (r"bilibili\.com/bangumi/media/md(\d+)", "/bilibili/bangumi/media/{0}", "B站番剧"),
        (r"live\.bilibili\.com/(\d+)", "/bilibili/live/room/{0}", "B站直播间"),
        (r"manga\.bilibili\.com/detail/mc(\d+)", "/bilibili/manga/update/{0}", "B站漫画"),
        (r"youtube\.com/channel/([\w-]+)", "/youtube/channel/{0}", "YouTube频道"),
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
            "  搜索: /bilibili/vsearch/{关键字}"
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
        for p in [
            os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.ttf"),
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
        if not txt:
            return []
        lines = []
        for para in txt.split("\n"):
            if not para.strip():
                lines.append("")
                continue
            buf = ""
            for ch in para:
                t = buf + ch
                if draw.textbbox((0,0), t, font=font)[2] > mw and buf:
                    lines.append(buf)
                    buf = ch
                else:
                    buf = t
            if buf:
                lines.append(buf)
        return lines

    def make(self, channel="", title="", desc="", link="", ts="", thumb=None):
        pad = self.pad
        cw = self.w - 2*pad
        fc = self._f(13)
        ft = self._f(18)
        fd = self._f(13)
        ff = self._f(11)
        tmp = Image.new("RGB",(1,1))
        d = ImageDraw.Draw(tmp)
        tl = self._wrap(title, ft, cw, d)
        dl = self._wrap((desc or "")[:300], fd, cw, d)
        if len(dl) > 5:
            dl = dl[:5]
            dl[-1] = dl[-1][:-2]+"..."

        th = None
        th_h = 0
        if thumb:
            try:
                th = Image.open(BytesIO(thumb)).convert("RGB")
                r = cw/th.width
                th_h = min(int(th.height*r), 280)
                th = th.resize((cw, th_h), Image.LANCZOS)
            except Exception:
                th = None

        y = 5 + pad + 18 + 14 + len(tl)*26 + 14
        if th:
            y += th_h + 14
        if dl:
            y += len(dl)*20 + 14
        y += 11
        if link:
            y += 20
        if ts:
            y += 16
        y += pad
        h = y

        img = Image.new("RGB", (self.w, h), (255,255,255))
        dr = ImageDraw.Draw(img)
        dr.rectangle([(0,0),(self.w,5)], fill=(66,133,244))

        y = 5 + pad
        dr.text((pad, y), f"📡 {channel}", font=fc, fill=(66,133,244))
        y += 32
        for ln in tl:
            dr.text((pad, y), ln, font=ft, fill=(26,26,46))
            y += 26
        y += 14
        if th:
            dr.rectangle([(pad-1,y-1),(pad+cw,y+th_h)], outline=(224,224,224))
            img.paste(th, (pad, y))
            y += th_h + 14
        if dl:
            for ln in dl:
                dr.text((pad, y), ln, font=fd, fill=(85,85,85))
                y += 20
            y += 14
        dr.line([(pad,y),(self.w-pad,y)], fill=(230,230,230))
        y += 11
        if link:
            lk = link if len(link)<=48 else link[:48]+"..."
            dr.text((pad, y), f"🔗 {lk}", font=ff, fill=(153,153,153))
            y += 20
        if ts:
            dr.text((pad, y), f"🕐 {ts}", font=ff, fill=(153,153,153))
        dr.rectangle([(0,0),(self.w-1,h-1)], outline=(224,224,224))

        buf = BytesIO()
        img.save(buf, format="PNG")
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

        self.pic = PicHandler(self.adjust_pic)
        self.card = CardGen()
        self.sched = AsyncIOScheduler()
        self.sched.start()
        self._reload_jobs()

    def _cron(self, expr):
        f = expr.split(" ")
        return {"minute":f[0],"hour":f[1],"day":f[2],"month":f[3],"day_of_week":f[4]}

    def _reload_jobs(self):
        self.sched.remove_all_jobs()
        for url, info in self.dh.data.items():
            if url in ("rsshub_endpoints","settings"):
                continue
            for user, si in info.get("subscribers",{}).items():
                self.sched.add_job(self._cron_cb, "cron", **self._cron(si["cron_expr"]), args=[url, user])

    async def _fetch(self, url):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        conn = aiohttp.TCPConnector(ssl=False)
        to = aiohttp.ClientTimeout(total=30, connect=10)
        try:
            async with aiohttp.ClientSession(trust_env=True, connector=conn, timeout=to, headers=headers) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        self.logger.error(f"rss: 站点返回 {r.status}: {url}")
                        return None
                    return await r.read()
        except Exception as e:
            self.logger.error(f"rss: 请求失败 {url}: {e}")
            return None

    async def _poll(self, url, num=-1, after_ts=0, after_link=""):
        text = await self._fetch(url)
        if text is None:
            return []
        try:
            root = etree.fromstring(text)
        except ValueError:
            try:
                root = etree.fromstring(text.replace(b'encoding="gb2312"',b'').replace(b'encoding="GB2312"',b''))
            except Exception:
                return []

        items = root.xpath("//item")
        ns = {"media":"http://search.yahoo.com/mrss/"}
        result = []
        cnt = 0

        for it in items:
            try:
                ch = self.dh.data[url]["info"]["title"] if url in self.dh.data else "未知"

                tn = it.xpath("title")
                title = tn[0].text if tn else "无标题"
                if len(title) > self.title_max:
                    title = title[:self.title_max]+"..."

                ln = it.xpath("link")
                link = ln[0].text if ln else ""
                if link and not re.match(r"^https?://", link):
                    link = self.dh.get_root_url(url) + link

                dn = it.xpath("description")
                raw = dn[0].text if dn else ""
                pics = self.dh.strip_html_pic(raw) if raw else []
                desc = self.dh.strip_html(raw) if raw else ""
                if len(desc) > self.desc_max:
                    desc = desc[:self.desc_max]+"..."

                for u in it.xpath("media:thumbnail/@url", namespaces=ns) + it.xpath(".//*[local-name()='thumbnail']/@url") + it.xpath("enclosure[contains(@type,'image')]/@url"):
                    if u not in pics:
                        pics.append(u)

                if it.xpath("pubDate"):
                    pd = it.xpath("pubDate")[0].text
                    try:
                        if "GMT" in pd:
                            pts = int(time.mktime(time.strptime(pd.replace("GMT","+0000"),"%a, %d %b %Y %H:%M:%S %z")))
                        else:
                            pts = int(time.mktime(time.strptime(pd,"%a, %d %b %Y %H:%M:%S %z")))
                    except Exception:
                        pts = int(time.time())
                    if pts > after_ts:
                        result.append(RSSItem(ch, title, link, desc, pd, pts, pics))
                        cnt += 1
                        if num != -1 and cnt >= num:
                            break
                    else:
                        break
                else:
                    if link != after_link:
                        result.append(RSSItem(ch, title, link, desc, "", 0, pics))
                        cnt += 1
                        if num != -1 and cnt >= num:
                            break
                    else:
                        break
            except Exception as e:
                self.logger.error(f"rss: 解析条目失败 {url}: {e}")
                break
        return result

    async def _add(self, url, cron_expr, event):
        user = event.unified_msg_origin
        if url in self.dh.data:
            items = await self._poll(url)
            if not items:
                return event.plain_result("无法从该源获取内容，请检查链接。")
            self.dh.data[url]["subscribers"][user] = {
                "cron_expr": cron_expr,
                "last_update": items[0].pubDate_timestamp,
                "latest_link": items[0].link,
            }
        else:
            text = await self._fetch(url)
            if text is None:
                return event.plain_result(f"无法访问: {url}\n请检查RSSHub端点是否可用。")
            try:
                title, desc = self.dh.parse_channel_info(text)
            except Exception as e:
                return event.plain_result(f"解析失败: {e}")
            items = await self._poll(url)
            if not items:
                return event.plain_result("源可访问但无内容条目。")
            self.dh.data[url] = {
                "subscribers": {
                    user: {
                        "cron_expr": cron_expr,
                        "last_update": items[0].pubDate_timestamp,
                        "latest_link": items[0].link,
                    }
                },
                "info": {"title": title, "description": desc},
            }
        self.dh.save()
        return self.dh.data[url]["info"]

    async def _make_comps(self, item):
        comps = []
        tb = None
        if self.read_pic and item.pic_urls:
            try:
                conn = aiohttp.TCPConnector(ssl=False)
                async with aiohttp.ClientSession(trust_env=True, connector=conn) as s:
                    async with s.get(item.pic_urls[0], timeout=aiohttp.ClientTimeout(total=15)) as r:
                        if r.status == 200:
                            tb = await r.read()
            except Exception:
                pass
        try:
            b64 = self.card.make(
                channel=item.chan_title, title=item.title, desc=item.description,
                link="" if self.hide_url else item.link, ts=item.pubDate or "", thumb=tb,
            )
            comps.append(Comp.Image.fromBase64(b64))
        except Exception as e:
            self.logger.error(f"卡片生成失败: {e}")
            comps.append(Comp.Plain(f"📡 {item.chan_title}\n📝 {item.title}\n{item.description}"))

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

    async def _cron_cb(self, url, user):
        if url not in self.dh.data or user not in self.dh.data[url].get("subscribers",{}):
            return
        self.logger.info(f"RSS定时触发: {url} -> {user}")
        si = self.dh.data[url]["subscribers"][user]
        items = await self._poll(url, num=self.max_poll, after_ts=si["last_update"], after_link=si["latest_link"])
        if not items:
            return
        max_ts = si["last_update"]
        pn = user.split(":")[0]
        if pn == "aiocqhttp" and self.compose:
            nodes = []
            for it in items:
                c = await self._make_comps(it)
                nodes.append(Comp.Node(uin=0, name="Astrbot", content=c))
                max_ts = max(max_ts, it.pubDate_timestamp)
            if nodes:
                await self.ctx.send_message(user, MessageChain(chain=nodes, use_t2i_=self.t2i))
        else:
            for it in items:
                c = await self._make_comps(it)
                await self.ctx.send_message(user, MessageChain(chain=c, use_t2i_=self.t2i))
                max_ts = max(max_ts, it.pubDate_timestamp)
        self.dh.data[url]["subscribers"][user]["last_update"] = max_ts
        self.dh.data[url]["subscribers"][user]["latest_link"] = items[0].link
        self.dh.save()
        self.logger.info(f"RSS推送完成: {url} -> {user}")

    # ============================================================
    #  LLM 工具
    # ============================================================

    @filter.llm_tool(name="rss_subscribe")
    async def tool_sub(self, event: AstrMessageEvent, feed_input: str = "", hours: int = 1):
        """当用户想订阅、关注、追踪某个网站或博主的更新时调用。支持B站/YouTube/Twitter(X)/微博/知乎等链接自动识别，也接受RSSHub路由。

        Args:
            feed_input: 网页链接(http开头)或RSSHub路由(/开头)
            hours: 检查间隔小时数，默认1
        """
        if not feed_input:
            yield event.plain_result("请提供要订阅的链接或路由。")
            return
        eps = self.dh.data.get("rsshub_endpoints", [])
        if not eps:
            yield event.plain_result("尚未配置RSSHub端点，请先执行：/myrss rsshub add https://rsshub.rssforever.com")
            return
        if feed_input.startswith("/"):
            furl = eps[0] + feed_input
        elif feed_input.startswith("http"):
            r = URLMapper.match(feed_input)
            if r:
                route, pn = r
                furl = eps[0] + route
            else:
                yield event.plain_result(f"无法自动识别该链接。\n\n{URLMapper.suggest(feed_input)}\n\n请选择路由后用/开头再次调用。")
                return
        else:
            yield event.plain_result("请提供http开头的链接或/开头的路由。")
            return
        if hours < 1:
            hours = 1
        ret = await self._add(furl, f"0 */{hours} * * *", event)
        if isinstance(ret, MessageEventResult):
            yield ret
            return
        self._reload_jobs()
        yield event.plain_result(f"✅ 订阅成功！\n📡 {ret['title']}\n📝 {ret['description']}\n⏰ 每{hours}小时\n🔗 {furl}")

    @filter.llm_tool(name="rss_list")
    async def tool_list(self, event: AstrMessageEvent, dummy: str = ""):
        """查看当前已订阅的RSS源。用户问"我订阅了什么"时调用。

        Args:
            dummy: 无用参数，忽略即可
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
            txt += f"  {i}. {info['title']} [{cr}]\n"
        yield event.plain_result(txt)

    @filter.llm_tool(name="rss_unsubscribe")
    async def tool_unsub(self, event: AstrMessageEvent, idx: int = 0):
        """取消RSS订阅。先调用rss_list看编号，再传入编号取消。

        Args:
            idx: 订阅编号
        """
        user = event.unified_msg_origin
        urls = self.dh.get_subs(user)
        if idx < 0 or idx >= len(urls):
            yield event.plain_result(f"编号{idx}不存在，范围0~{len(urls)-1}")
            return
        u = urls[idx]
        t = self.dh.data[u]["info"]["title"]
        self.dh.data[u]["subscribers"].pop(user)
        self.dh.save()
        self._reload_jobs()
        yield event.plain_result(f"✅ 已取消: {t}")

    # ============================================================
    #  手动命令
    # ============================================================

    @filter.command_group("myrss")
    def myrss(self):
        """RSS订阅管理"""
        pass

    @myrss.group("rsshub")
    def rsshub(self, event: AstrMessageEvent):
        """RSSHub端点管理"""
        pass

    @rsshub.command("add")
    async def rsshub_add(self, event: AstrMessageEvent, url: str):
        """添加RSSHub端点

        Args:
            url: 端点地址，如 https://rsshub.rssforever.com
        """
        if url.endswith("/"):
            url = url[:-1]
        if url in self.dh.data["rsshub_endpoints"]:
            yield event.plain_result("已存在")
            return
        self.dh.data["rsshub_endpoints"].append(url)
        self.dh.save()
        yield event.plain_result(f"✅ 已添加: {url}")

    @rsshub.command("list")
    async def rsshub_list(self, event: AstrMessageEvent):
        """列出所有RSSHub端点"""
        eps = self.dh.data["rsshub_endpoints"]
        if not eps:
            yield event.plain_result("暂无端点，请先 /myrss rsshub add <url>")
            return
        yield event.plain_result("RSSHub端点：\n" + "\n".join(f"  {i}: {x}" for i, x in enumerate(eps)))

    @rsshub.command("remove")
    async def rsshub_rm(self, event: AstrMessageEvent, idx: int):
        """删除RSSHub端点

        Args:
            idx: 端点编号
        """
        eps = self.dh.data["rsshub_endpoints"]
        if idx < 0 or idx >= len(eps):
            yield event.plain_result("编号越界")
            return
        removed = eps.pop(idx)
        self.dh.save()
        yield event.plain_result(f"✅ 已删除: {removed}")

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
            txt += f"  {i}. {info['title']}\n"
        yield event.plain_result(txt)

    @myrss.command("remove")
    async def cmd_rm(self, event: AstrMessageEvent, idx: int):
        """取消订阅

        Args:
            idx: 订阅编号
        """
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
        yield event.plain_result(f"✅ 已取消: {t}")

    @myrss.command("get")
    async def cmd_get(self, event: AstrMessageEvent, idx: int):
        """获取最新内容

        Args:
            idx: 订阅编号
        """
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
