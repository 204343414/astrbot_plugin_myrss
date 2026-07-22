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

function render() {
  const query = el("search").value.trim().toLowerCase();
  const list = payload.groups.filter((group) =>
    `${group.group_id}${group.platform}`.toLowerCase().includes(query));
  el("groups").innerHTML = list.length
    ? list.map((group) => `<button class="group ${group.origin === selected ? "active" : ""}" data-id="${escapeHtml(group.origin)}"><b>群 ${escapeHtml(group.group_id)}</b><small>${escapeHtml(group.platform)} · ${group.feeds.length} 个动态源</small></button>`).join("")
    : '<div class="empty">没有订阅群</div>';
  document.querySelectorAll(".group").forEach((button) => {
    button.onclick = () => { selected = button.dataset.id; render(); };
  });
  const group = payload.groups.find((item) => item.origin === selected);
  if (!group) {
    el("detail").innerHTML = '<div class="empty">当前正式数据文件中没有群订阅</div>';
    return;
  }
  el("detail").innerHTML = `<h2>群 ${escapeHtml(group.group_id)}</h2><p><span class="badge">${escapeHtml(group.platform)}</span>　${group.feeds.length} 个订阅源</p>${group.feeds.map((feed) => `<section class="feed"><h3>${escapeHtml(feed.title)}</h3><div class="grid"><span class="label">路由 / URL</span><span class="value">${escapeHtml(feed.url)}</span><span class="label">轮询</span><span>${escapeHtml(feed.cron_expr || "—")}</span><span class="label">最后断点</span><span>${fmt(feed.last_update)}</span><span class="label">去重记录</span><span>${feed.seen_count} 条</span><span class="label">最近链接</span><span class="value">${escapeHtml(feed.latest_link || "—")}</span></div></section>`).join("")}`;
}

el("refresh").onclick = load;
el("search").oninput = render;
load();
