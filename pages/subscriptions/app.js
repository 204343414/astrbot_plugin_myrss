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

function renderFeed(feed, origin) {
  const avatar = safeHttpUrl(feed.avatar);
  const preview = feed.preview && feed.preview.safety_status === "SAFE" ? feed.preview : null;
  const image = preview ? safeHttpUrl(preview.image_url) : "";
  const link = safeHttpUrl(preview?.link || feed.latest_link);
  const delivery = feed.delivery_status;
  const deliveryLabel = !delivery ? "⚪ 尚无投递记录" : delivery.status === "SUCCESS" ? `🟢 最近投递成功 ${fmt(delivery.delivered_at)}` : `🔴 最近投递失败 ${fmt(delivery.attempted_at)} · ${escapeHtml(delivery.error_category || "UNKNOWN")}`;
  return `<section class="feed"><div class="source-header">${avatar ? `<img class="avatar" src="${escapeHtml(avatar)}" referrerpolicy="no-referrer" />` : `<div class="avatar avatar-fallback">${escapeHtml((feed.title || "?").slice(0, 1))}</div>`}<div><h3>${escapeHtml(feed.title)}</h3><div class="muted">${escapeHtml(feed.cron_expr || "—")} · 去重 ${feed.seen_count} 条</div></div></div>${feed.description ? `<p class="source-description">${escapeHtml(feed.description)}</p>` : ""}<p class="muted">${deliveryLabel}</p><div class="grid"><span class="label">路由 / URL</span><span class="value">${escapeHtml(feed.url)}</span><span class="label">最后断点</span><span>${fmt(feed.last_update)}</span></div>${preview ? `<div class="latest-preview">${image ? `<img src="${escapeHtml(image)}" referrerpolicy="no-referrer" />` : ""}<div><h4>${escapeHtml(preview.title || "最新安全动态")}</h4><p>${escapeHtml(preview.description || "暂无摘要")}</p><span class="muted">${fmt(preview.pub_timestamp || preview.updated_at)}</span>${link ? `　<a class="open-link" href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer">打开动态 ↗</a>` : ""}</div></div>` : `<p class="muted">暂无安全内容缓存；下一次出现并通过审核的新动态后自动补齐。</p>`}<div class="feed-actions"><button class="test-delivery" data-origin="${escapeHtml(origin)}" data-feed="${escapeHtml(feed.url)}">测试 GET + 主动推送</button></div></section>`;
}

async function runDeliveryTest(button) {
  const origin = button.dataset.origin;
  const feedUrl = button.dataset.feed;
  if (!window.confirm("将对该 RSS 执行一次 GET，并向所选群主动发送一张诊断卡片。\n不会调用 LLM、不会修改 seen_links。是否继续？")) return;
  const oldText = button.textContent;
  button.disabled = true;
  button.textContent = "测试中…";
  try {
    const response = await bridge.apiPost("subscriptions/test-delivery", {
      origin, feed_url: feedUrl,
    });
    if (response?.ok === false) throw new Error(response.message || "测试失败");
    const result = response?.ok === true ? response.data : (response?.data || response);
    if (!result || typeof result !== "object") throw new Error(`响应格式异常：${JSON.stringify(response)}`);
    window.alert(
      `RSS GET：${result.fetch_ok ? "成功" : "失败"}\n` +
      `RSS 条目：${result.item_count ?? 0}\n` +
      `主动推送：${result.send_ok ? "成功" : "失败"}\n` +
      `${result.fetch_error ? `GET错误：${result.fetch_error}\n` : ""}` +
      `${result.send_error ? `发送错误：${result.send_error}` : "请确认目标群确实出现诊断卡片。"}`
    );
  } catch (error) {
    window.alert(`诊断失败：${error?.message || error}`);
  } finally {
    button.disabled = false;
    button.textContent = oldText;
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
  el("detail").innerHTML = `<h2>群 ${escapeHtml(group.group_id)}</h2><p><span class="badge">${escapeHtml(group.platform)}</span>　${group.feeds.length} 个订阅源</p>${group.feeds.map((feed) => renderFeed(feed, group.origin)).join("")}`;
  document.querySelectorAll(".test-delivery").forEach((button) => {
    button.onclick = () => runDeliveryTest(button);
  });
}

el("refresh").onclick = load;
el("search").oninput = render;
load();
