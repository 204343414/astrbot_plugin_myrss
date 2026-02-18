import asyncio  # 已有，确认引入

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

        # 用于防止并发重复推送的锁，key = (url, user)
        self._locks: dict[tuple, asyncio.Lock] = {}

        self.sched = AsyncIOScheduler()
        self.sched.start()
        self._reload_jobs()

    async def destroy(self):
        """AstrBot 卸载/禁用插件时调用"""
        try:
            if self.sched.running:
                self.sched.shutdown(wait=False)
                self.logger.info("MyRSS: 调度器已停止")
        except Exception as e:
            self.logger.error("MyRSS: 停止调度器失败: %s", e)

    def _get_lock(self, url: str, user: str) -> asyncio.Lock:
        key = (url, user)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    async def _poll(self, url: str, num: int = -1, after_ts: int = 0, after_link: str = "") -> list:
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

                for u in (
                    it.xpath("media:thumbnail/@url", namespaces=ns)
                    + it.xpath(".//*[local-name()='thumbnail']/@url")
                    + it.xpath("enclosure[contains(@type,'image')]/@url")
                ):
                    if u not in pics:
                        pics.append(u)

                pub_nodes = it.xpath("pubDate")
                if pub_nodes:
                    pd = pub_nodes[0].text or ""
                    pts = self._parse_pubdate(pd)

                    # 修复：用 >= 改为严格 > ，且时间戳解析失败时跳过而非 fallback
                    if pts is None:
                        # 解析失败，用 link 去重兜底
                        if link and link != after_link:
                            result.append(RSSItem(ch, title, link, desc, pd, 0, pics))
                            cnt += 1
                        # 不 break，继续处理后续条目
                    elif pts > after_ts:
                        result.append(RSSItem(ch, title, link, desc, pd, pts, pics))
                        cnt += 1
                    else:
                        # 遇到已处理过的时间戳，后续条目更旧，直接停止
                        break
                else:
                    # 无 pubDate，用 link 去重
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

    def _parse_pubdate(self, pd: str):
        """
        解析 pubDate 字符串为 Unix 时间戳。
        解析失败返回 None（而非 fallback 到当前时间）。
        """
        if not pd:
            return None
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
        ]
        pd_clean = pd.strip().replace("GMT", "+0000")
        for fmt in formats:
            try:
                import calendar
                t = time.strptime(pd_clean, fmt)
                # mktime 会用本地时区，这里直接用 calendar.timegm 处理 UTC
                # 但 %z 已经含时区，用 datetime 更准确
                from datetime import datetime
                dt = datetime.strptime(pd_clean, fmt)
                return int(dt.timestamp())
            except Exception:
                continue
        return None

    async def _cron_cb(self, url: str, user: str) -> None:
        # 用锁防止同一 (url, user) 并发执行
        lock = self._get_lock(url, user)
        async with lock:
            await self._cron_cb_inner(url, user)

    async def _cron_cb_inner(self, url: str, user: str) -> None:
        if url not in self.dh.data or user not in self.dh.data[url].get("subscribers", {}):
            return

        self.logger.info("RSS定时触发: %s -> %s", url, user)
        si = self.dh.data[url]["subscribers"][user]

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
            # 全部重复，仅推进指针
            si["latest_link"] = items[0].link
            # 注意：不要更新 last_update，防止时间跳变
            self.dh.save()
            return

        # 先更新 seen_links（在发送前），防止并发重复
        new_keys = [item_key(it) for it in new_items]
        si["seen_links"] = (new_keys + si.get("seen_links", []))[:200]
        si["latest_link"] = items[0].link
        # 只更新时间戳不为 0 的条目的最大值
        ts_candidates = [it.pubDate_timestamp for it in new_items if it.pubDate_timestamp > 0]
        if ts_candidates:
            si["last_update"] = max(ts_candidates)
        self.dh.save()   # ← 先保存，再发送，避免发送失败后数据不一致

        pn = user.split(":")[0]
        merge_limit = 5
        batch = new_items[:merge_limit]

        if len(batch) > 1:
            cards = [await self._make_card_b64(it) for it in batch]
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
