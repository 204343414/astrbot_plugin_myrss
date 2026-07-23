const bridge = window.AstrBotPluginPage;
let payload = { groups: [] };
let selected = "";
const el = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function fmt(timestamp) {
  return timestamp ? new Date(timestamp * 1000).toLocaleString() : "—";
}

function unwrap(response) {
  let value = response;
  // 兼容插件页面桥接层直接返回 JSON、包一层 data，或重复包裹的版本差异。
  for (let depth = 0; depth < 3; depth += 1) {
    if (!value || typeof value !== "object") break;
    if (value.ok === false) throw new Error(value.message || JSON.stringify(value));
    if (value.ok === true && Object.prototype.hasOwnProperty.call(value, "data")) {
      value = value.data;
      continue;
    }
    if (Object.keys(value).length === 1 && Object.prototype.hasOwnProperty.call(value, "data")) {
      value = value.data;
      continue;
    }
    break;
  }
  if (!value || !Array.isArray(value.groups)) {
    throw new Error(`API 响应格式异常：${JSON.stringify(response).slice(0, 800)}`);
  }
  return value;
}

async function load() {
  try {
    if (!bridge?.apiGet) throw new Error("AstrBotPluginPage.apiGet 不可用");
    const response = await bridge.apiGet("subscriptions/bootstrap");
    payload = unwrap(response);
    el("dataPath").textContent = `正式数据文件：${payload.data_path}`;
    el("groupCount").textContent = payload.group_count;
    el("subCount").textContent = payload.subscription_count;
    el("blockedCount").textContent = payload.safety_events?.length || 0;
    renderSafety();
    renderModeration();
    if (!payload.groups.some((group) => group.origin === selected)) {
      selected = payload.groups[0]?.origin || "";
    }
    render();
  } catch (error) {
    const message = error?.message || String(error);
    el("dataPath").textContent = "正式数据文件读取失败";
    el("detail").innerHTML = `<div class="empty">页面读取失败：${escapeHtml(message)}</div>`;
    console.error("MyRSS bootstrap failed", error);
  }
}

function safeHttpUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch { return ""; }
}

function renderSafety() {
  const panel = el("safetyPanel");
  const events = Array.isArray(payload.safety_events) ? payload.safety_events : [];
  if (!events.length) { panel.classList.add("hidden"); panel.innerHTML = ""; return; }
  panel.classList.remove("hidden");
  panel.innerHTML = `<h2>🛡️ 近期未通过安全审核的动态（仅元数据）</h2><div class="safety-list">${events.map((event) => `<div class="safety-event"><span class="${event.status === "MALICIOUS" ? "danger" : "reject"}">${escapeHtml(event.status)}</span><span><b>${escapeHtml(event.source || "未知源")}</b>　${escapeHtml(event.reason || "审核未通过")}</span><span class="muted">${fmt(event.blocked_at)} · ${escapeHtml(event.content_fingerprint || "")}</span></div>`).join("")}</div>`;
}

const moderationConfirmUntil = new Map();
function creatorLabel(creator) {
  if (!creator || creator.source === "legacy") return "历史订阅（创建者未知）";
  if (creator.source === "dashboard") return "Dashboard 管理员";
  const id = String(creator.openid || "");
  return `${creator.name || "OpenID"}${id ? ` · ${id}` : ""}`;
}
function renderModeration() {
  const reviews = Array.isArray(payload.moderation_reviews) ? payload.moderation_reviews : [];
  const reviewPanel = el("reviewPanel");
  if (!reviews.length) { reviewPanel.classList.add("hidden"); reviewPanel.innerHTML = ""; }
  else {
    reviewPanel.classList.remove("hidden");
    reviewPanel.innerHTML = `<h2>⚠️ 严重拦截待复核</h2><div class="review-list">${reviews.map((review) => {
      const image = safeHttpUrl(review.image_url), link = safeHttpUrl(review.link);
      return `<article class="review-item">${image ? `<img src="${escapeHtml(image)}" referrerpolicy="no-referrer">` : ""}<div><h3>${escapeHtml(review.source || "未知源")}</h3><p>${escapeHtml(review.title || "")}</p><p class="muted">${escapeHtml(review.description || "")}</p><p>创建者：${escapeHtml(creatorLabel(review.created_by))}</p><p class="muted">${fmt(review.created_at)} · ${escapeHtml(review.reason || "严重审核未通过")}${link ? ` · <a href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer">查看原链接</a>` : ""}</p><div class="review-actions"><button data-review="${escapeHtml(review.id)}" data-action="restore">误判恢复</button><button data-review="${escapeHtml(review.id)}" data-action="confirm" class="warn-button">确认违规+1</button><button data-review="${escapeHtml(review.id)}" data-action="ban" class="danger-button">立即禁止创建者新增</button></div></div></article>`;
    }).join("")}</div>`;
    reviewPanel.querySelectorAll("[data-review]").forEach((button) => { button.onclick = () => resolveReview(button); });
  }
  const bans = Array.isArray(payload.subscription_bans) ? payload.subscription_bans : [];
  const banPanel = el("banPanel");
  if (!bans.length) { banPanel.classList.add("hidden"); banPanel.innerHTML = ""; }
  else {
    banPanel.classList.remove("hidden");
    banPanel.innerHTML = `<h2>🚫 禁止新增订阅</h2>${bans.map((item) => `<div class="ban-item"><b>${escapeHtml(item.name || "OpenID")}</b> · ${escapeHtml(item.openid || "未知")} · strikes=${Number(item.strikes || 0)}<br><span class="muted">${escapeHtml(item.origin || "")} · ${escapeHtml(item.reason || "")}</span></div>`).join("")}`;
  }
}
async function resolveReview(button) {
  const action = button.dataset.action, id = button.dataset.review, key = `${id}:${action}`;
  if (action !== "restore" && (moderationConfirmUntil.get(key) || 0) < Date.now()) {
    moderationConfirmUntil.set(key, Date.now() + 5000);
    const old = button.textContent; button.textContent = "5秒内再次点击确认";
    setTimeout(() => { if ((moderationConfirmUntil.get(key) || 0) <= Date.now()) button.textContent = old; }, 5100);
    return;
  }
  moderationConfirmUntil.delete(key); button.disabled = true;
  try {
    const response = await bridge.apiPost("moderation/resolve", { review_id: id, action });
    if (response?.ok === false) throw new Error(response.message || "处理失败");
    const result = response?.ok === true ? response.data : (response?.data || response);
    setTestStatus("success", `复核完成：${result.state}${result.creator_openid ? `\n创建者：${result.creator_openid}\nstrikes=${result.strikes} banned=${result.banned}` : ""}`);
    await load();
  } catch (error) { setTestStatus("error", `复核失败：${error?.message || error}`); button.disabled = false; }
}

function renderFeed(feed, origin) {
  const avatar = safeHttpUrl(feed.avatar);
  const preview = feed.preview && feed.preview.safety_status === "SAFE" ? feed.preview : null;
  const image = preview ? safeHttpUrl(preview.image_url) : "";
  const link = safeHttpUrl(preview?.link || feed.latest_link);
  const delivery = feed.delivery_status;
  const deliveryLabel = !delivery ? "⚪ 尚无投递记录" : delivery.status === "SUCCESS" ? `🟢 最近投递成功 ${fmt(delivery.delivered_at)}` : `🔴 最近投递失败 ${fmt(delivery.attempted_at)} · ${escapeHtml(delivery.error_category || "UNKNOWN")}`;
  return `<section class="feed"><div class="source-header">${avatar ? `<img class="avatar" src="${escapeHtml(avatar)}" referrerpolicy="no-referrer" />` : `<div class="avatar avatar-fallback">${escapeHtml((feed.title || "?").slice(0, 1))}</div>`}<div><h3>${escapeHtml(feed.title)}</h3><div class="muted">${escapeHtml(feed.cron_expr || "—")} · 去重 ${feed.seen_count} 条</div></div></div>${feed.description ? `<p class="source-description">${escapeHtml(feed.description)}</p>` : ""}<p class="muted">创建者：${escapeHtml(creatorLabel(feed.created_by))}${feed.paused_by_moderation ? " · ⚠️ 严重审核待复核，已暂停" : ""}</p><p class="muted">${deliveryLabel}</p><div class="grid"><span class="label">路由 / URL</span><span class="value">${escapeHtml(feed.url)}</span><span class="label">最后断点</span><span>${fmt(feed.last_update)}</span></div>${preview ? `<div class="latest-preview ${image ? "has-image" : "no-image"}">${image ? `<img src="${escapeHtml(image)}" referrerpolicy="no-referrer" />` : ""}<div><h4>${escapeHtml(preview.title || "最新安全动态")}</h4><p>${escapeHtml(preview.description || "暂无摘要")}</p><span class="muted">${fmt(preview.pub_timestamp || preview.updated_at)}</span>${link ? `　<a class="open-link" href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer">打开动态 ↗</a>` : ""}</div></div>` : `<p class="muted">暂无安全内容缓存；下一次出现并通过审核的新动态后自动补齐。</p>`}<div class="feed-actions"><button class="test-delivery" data-origin="${escapeHtml(origin)}" data-feed="${escapeHtml(feed.url)}">测试 GET + 主动推送</button><button class="remove-subscription danger-button" data-origin="${escapeHtml(origin)}" data-feed="${escapeHtml(feed.url)}">退订此源</button></div></section>`;
}

function setTestStatus(kind, message) {
  const panel = el("testStatus");
  panel.className = `test-status ${kind}`;
  panel.textContent = message;
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function runDeliveryTest(button) {
  const origin = button.dataset.origin;
  const feedUrl = button.dataset.feed;
  const oldText = button.textContent;
  button.disabled = true;
  button.textContent = "测试中…";
  setTestStatus("running", "正在执行一次 RSS GET，并向目标群主动发送诊断卡片……\n不会调用 LLM，也不会修改 seen_links。");
  try {
    if (!bridge?.apiPost) throw new Error("AstrBotPluginPage.apiPost 不可用");
    const response = await bridge.apiPost("subscriptions/test-delivery", {
      origin, feed_url: feedUrl,
    });
    if (response?.ok === false) throw new Error(response.message || "测试失败");
    const result = response?.ok === true ? response.data : (response?.data || response);
    if (!result || typeof result !== "object") throw new Error(`响应格式异常：${JSON.stringify(response)}`);
    const lines = [
      `RSS GET：${result.fetch_ok ? "成功" : "失败"}`,
      `RSS 条目：${result.item_count ?? 0}`,
      `主动推送：${result.send_ok ? "成功" : "失败"}`,
    ];
    if (result.fetch_error) lines.push(`GET 错误：${result.fetch_error}`);
    if (result.send_error) lines.push(`发送错误：${result.send_error}`);
    if (result.send_ok) lines.push("请同时确认目标群确实出现了诊断卡片。");
    setTestStatus(result.fetch_ok && result.send_ok ? "success" : "error", lines.join("\n"));
  } catch (error) {
    setTestStatus("error", `诊断失败：${error?.message || error}`);
  } finally {
    button.disabled = false;
    button.textContent = oldText;
  }
}

async function addSubscription(origin) {
  const input = el("addSubscriptionUrl");
  const button = el("addSubscriptionBtn");
  const url = input.value.trim();
  if (!url) { setTestStatus("error", "请先输入账号网页 URL 或 RSSHub 路由。"); return; }
  button.disabled = true;
  button.textContent = "预审中…";
  setTestStatus("running", "正在拉取最新动态并执行文字+图片安全审核；只有 SAFE 才会建立订阅。此操作不会推送历史动态。");
  try {
    const response = await bridge.apiPost("subscriptions/add", { origin, url });
    if (response?.ok === false) throw new Error(response.message || "新增失败");
    const result = response?.ok === true ? response.data : (response?.data || response);
    setTestStatus("success", `${result.message || "操作完成"}\n${result.title || ""}`);
    input.value = "";
    await load();
  } catch (error) {
    setTestStatus("error", `新增订阅失败：${error?.message || error}`);
  } finally {
    button.disabled = false;
    button.textContent = "安全预审并订阅";
  }
}

const removeConfirmUntil = new Map();
async function removeSubscription(button) {
  const key = `${button.dataset.origin}|${button.dataset.feed}`;
  const now = Date.now();
  if ((removeConfirmUntil.get(key) || 0) < now) {
    removeConfirmUntil.set(key, now + 5000);
    button.textContent = "5秒内再次点击确认退订";
    setTimeout(() => { if ((removeConfirmUntil.get(key) || 0) <= Date.now()) button.textContent = "退订此源"; }, 5100);
    return;
  }
  removeConfirmUntil.delete(key);
  button.disabled = true;
  button.textContent = "退订中…";
  try {
    const response = await bridge.apiPost("subscriptions/remove", {
      origin: button.dataset.origin, feed_url: button.dataset.feed,
    });
    if (response?.ok === false) throw new Error(response.message || "退订失败");
    const result = response?.ok === true ? response.data : (response?.data || response);
    setTestStatus("success", `已退订：${result.title || button.dataset.feed}`);
    await load();
  } catch (error) {
    setTestStatus("error", `退订失败：${error?.message || error}`);
    button.disabled = false;
    button.textContent = "退订此源";
  }
}

function render() {
  const query = el("search").value.trim().toLowerCase();
  const list = payload.groups.filter((group) =>
    `${group.group_id}${group.platform}`.toLowerCase().includes(query));
  el("groups").innerHTML = list.length
    ? list.map((group) => `<button class="group ${group.origin === selected ? "active" : ""}" data-id="${escapeHtml(group.origin)}"><b>${group.delivery_ready ? "🟢" : "🟡"} 群 ${escapeHtml(group.group_id)}</b><small>${escapeHtml(group.platform)} · ${group.feeds.length} 个动态源 · ${escapeHtml(group.delivery_reason || "unknown")}</small></button>`).join("")
    : '<div class="empty">没有订阅群</div>';
  document.querySelectorAll(".group").forEach((button) => {
    button.onclick = () => { selected = button.dataset.id; render(); };
  });
  const group = payload.groups.find((item) => item.origin === selected);
  if (!group) {
    el("detail").innerHTML = '<div class="empty">当前正式数据文件中没有群订阅</div>';
    return;
  }
  el("detail").innerHTML = `<h2>群 ${escapeHtml(group.group_id)}</h2><p><span class="badge">${escapeHtml(group.platform)}</span>　${group.feeds.length} 个订阅源</p><div class="add-subscription-panel"><input id="addSubscriptionUrl" placeholder="输入账号网页 URL 或 /开头的 RSSHub 路由"/><button id="addSubscriptionBtn">安全预审并订阅</button></div>${group.feeds.map((feed) => renderFeed(feed, group.origin)).join("")}`;
  el("addSubscriptionBtn").onclick = () => addSubscription(group.origin);
  document.querySelectorAll(".test-delivery").forEach((button) => {
    button.onclick = () => runDeliveryTest(button);
  });
  document.querySelectorAll(".remove-subscription").forEach((button) => {
    button.onclick = () => removeSubscription(button);
  });
}

el("refresh").onclick = load;
el("search").oninput = render;
load();
