// 凭据维护面板（独立模块，不碰 app.js）
// 设计原则：
//   1. 任何一个元素缺失、任何一个接口失败，都只影响自己这一块，绝不连坐其他按钮
//   2. 前端永远接触不到完整凭据：只显示「几段 / 多长 / 到期时间 / 剩余天数」
//   3. 所有按钮在处理中都要禁用，防止连点把 rsshub 连着重启
//
// 调用方式必须走 AstrBot 的插件页面桥接层（window.AstrBotPluginPage）：
// 插件页面是嵌在 AstrBot 面板里的，不是同源静态页，直接 fetch("/api/plug/...")
// 会因为拿不到绝对地址而抛 NetworkError。app.js 用的就是这套桥接。
// 路径也不带插件名前缀——桥接层自己会补。

const STATE_LABEL = { ok: "✅ 已生效", idle: "⚪ 未配置", warn: "⚠ 临期", bad: "❌ 已失效" };
const NET_ERR = "接口不通（可能是维护服务没启动，或插件需要重载）";

const $ = (id) => document.getElementById(id);
const on = (id, type, handler) => {
  const el = $(id);
  if (!el) {
    console.warn("[vault] 缺少元素", id);
    return;
  }
  el.addEventListener(type, handler);
};

function bridge() {
  const api = window.AstrBotPluginPage;
  if (!api || !api.apiGet || !api.apiPost) {
    throw new Error("AstrBotPluginPage 桥接层不可用（插件页面没有正确挂载）");
  }
  return api;
}

/** 剥掉插件 web.py 的 {ok, data} 外壳，兼容多层包裹。 */
function unwrap(response) {
  let value = response;
  for (let depth = 0; depth < 3; depth += 1) {
    if (!value || typeof value !== "object") break;
    if (value.data !== undefined) {
      value = value.data;
      continue;
    }
    if (value.ok === true && Object.keys(value).length === 2 && value.result !== undefined) {
      value = value.result;
      continue;
    }
    break;
  }
  return value;
}

async function get(path) {
  const response = await bridge().apiGet(path);
  return unwrap(response);
}

async function post(path, payload) {
  const response = await bridge().apiPost(path, payload || {});
  return unwrap(response);
}

function fmtDate(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function fmtDays(days) {
  if (days === null || days === undefined) return "—";
  if (days < 0) return `${Math.abs(Math.round(days))} 天前过期`;
  return `剩 ${Math.round(days)} 天`;
}

function setNotice(text, kind = "info") {
  const el = $("vaultNotice");
  if (!el) return;
  if (!text) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = text;
  el.className = `vault-notice vault-notice-${kind}`;
}

function chip(id, text, kind) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className = `vault-chip vault-chip-${kind}`;
}

// ------------------------------------------------------------ 渲染
function renderRow(item, tier) {
  const row = document.createElement("div");
  row.className = "vault-row";
  row.dataset.platform = item.id;

  const main = document.createElement("div");
  main.className = "vault-row-main";

  const name = document.createElement("b");
  name.textContent = item.label;
  main.appendChild(name);

  const state = document.createElement("span");
  state.className = `vault-state st-${item.state}`;
  state.textContent = STATE_LABEL[item.state] || item.state;
  main.appendChild(state);

  const meta = document.createElement("small");
  if (item.state === "idle") {
    meta.textContent = tier === "B" ? "还没配置" : "点右边导入";
  } else {
    meta.textContent = `过期 ${fmtDate(item.expires_at)} · ${fmtDays(item.days_left)} · ${item.fields} 段 / ${item.length} 字符`;
  }
  main.appendChild(meta);
  row.appendChild(main);

  const actions = document.createElement("div");
  actions.className = "vault-row-actions";

  if (tier === "A") {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.state === "idle" ? "一键导入" : "重新导入";
    btn.addEventListener("click", () => runImport(item, btn));
    actions.appendChild(btn);
  } else {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "设置";
    btn.addEventListener("click", () => openSet(item));
    actions.appendChild(btn);
  }

  if (item.state !== "idle") {
    const test = document.createElement("button");
    test.type = "button";
    test.textContent = "测试";
    test.addEventListener("click", () => runVerify(item, test));
    actions.appendChild(test);
  }

  row.appendChild(actions);

  if (tier === "B") {
    const box = document.createElement("div");
    box.className = "vault-setbox hidden";
    const p = document.createElement("p");
    p.textContent = (item.steps || []).join(" → ") || "按平台文档取一次 token，粘贴到这里。";
    box.appendChild(p);
    if (item.helpUrl) {
      const a = document.createElement("a");
      a.href = item.helpUrl;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = "打开取 token 的页面";
      box.appendChild(a);
    }
    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = "粘贴 token / 整段 cookie（提交后立即清空）";
    input.autocomplete = "off";
    box.appendChild(input);
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "保存";
    save.addEventListener("click", async () => {
      const value = input.value.trim();
      if (!value) return setNotice("内容是空的", "bad");
      save.disabled = true;
      try {
        const res = await post(`vault/set`, { platform: item.id, value });
        input.value = "";
        showResult(item, res);
        await refresh();
      } catch (err) {
        setNotice(`${item.label}：${err.message}`, "bad");
      } finally {
        save.disabled = false;
      }
    });
    box.appendChild(save);
    row.appendChild(box);
    row._setbox = box;
  }

  return row;
}

function openSet(item) {
  const row = document.querySelector(`.vault-row[data-platform="${item.id}"]`);
  const box = row && row._setbox;
  if (!box) return;
  box.classList.toggle("hidden");
  const input = box.querySelector("input");
  if (input && !box.classList.contains("hidden")) input.focus();
}

function showResult(item, res) {
  if (!res) return setNotice(`${item.label}：没有返回内容`, "bad");
  if (res.ok === false || res.error) {
    return setNotice(`${item.label}：${res.error || "失败"}`, "bad");
  }
  const bits = [`${item.label} 已生效`];
  if (res.verify) bits.push(res.verify);
  if (res.probe && res.probe.status) bits.push(`路由 ${res.probe.status}（${res.probe.detail || ""}）`);
  setNotice(bits.join(" · "), "ok");
}

async function runImport(item, btn) {
  const row = document.querySelector(`.vault-row[data-platform="${item.id}"]`);
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "导入中…";
  if (row) row.classList.add("busy");
  setNotice(`${item.label}：正在读取本机 Firefox 并验证（重建 rsshub 约 20 秒）…`);
  try {
    const res = await post(`vault/import`, { platform: item.id });
    showResult(item, res);
    await refresh();
  } catch (err) {
    setNotice(`${item.label}：${err.message}`, "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = old;
    if (row) row.classList.remove("busy");
  }
}

async function runVerify(item, btn) {
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "测试中…";
  try {
    const res = await post(`vault/verify`, { platform: item.id });
    if (res && res.ok) setNotice(`${item.label}：测试通过 · ${res.detail || ""}`, "ok");
    else setNotice(`${item.label}：测试失败 · ${(res && (res.detail || res.error)) || "未知"}`, "bad");
  } catch (err) {
    setNotice(`${item.label}：${err.message}`, "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

async function refresh() {
  const listA = $("vaultListA");
  const listB = $("vaultListB");
  if (!listA || !listB) return;
  try {
    const data = await get("vault/status");
    if (!data) throw new Error(NET_ERR);
    if (data.rsshub) {
      const healthy = data.rsshub.healthy;
      chip("vaultRsshub", `rsshub：${healthy ? "在线" : "异常"}`, healthy ? "ok" : "bad");
    }
    chip("vaultHelper", "维护服务：已连接", "ok");
    const plat = (data.platforms || []).slice();
    const A = plat.filter((p) => p.tier === "A");
    const B = plat.filter((p) => p.tier === "B");
    listA.replaceChildren(...A.map((p) => renderRow(p, "A")));
    listB.replaceChildren(...B.map((p) => renderRow(p, "B")));
    const ready = A.filter((p) => p.state === "ok").length;
    const warn = A.filter((p) => p.state === "warn" || p.state === "bad").length;
    const sum = $("vaultSummary");
    if (sum) {
      sum.textContent = `已配好 ${ready} / ${A.length}${warn ? ` · 有 ${warn} 个要处理` : ""}`;
    }
  } catch (err) {
    chip("vaultHelper", "维护服务：连不上", "bad");
    chip("vaultRsshub", "rsshub：未检测", "idle");
    setNotice(`${err.message}`, "bad");
  }
}

async function runCheck(btn) {
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "自检中…";
  setNotice("正在逐个平台自检，浏览器里的比 rsshub 里的新就自动同步…");
  try {
    const res = await post(`vault/check`, { sync: true });
    const actions = (res && res.actions) || [];
    const alerts = (res && res.alerts) || [];
    const okN = actions.filter((a) => a.ok).length;
    setNotice(
      `自检完成：处理 ${actions.length} 项（成功 ${okN}），待提醒 ${alerts.length} 项`,
      alerts.length ? "warn" : "ok",
    );
    await refresh();
  } catch (err) {
    setNotice(`自检失败：${err.message}`, "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
}

function boot() {
  const section = $("vaultSection");
  if (!section) return;
  on("vaultRefresh", "click", () => refresh());
  const btnCheck = $("vaultCheck");
  if (btnCheck) btnCheck.addEventListener("click", () => runCheck(btnCheck));
  // 展开时才拉数据，省得每次打开页面都去烦维护服务
  const details = section.querySelector("details");
  if (details) {
    details.addEventListener("toggle", () => {
      if (details.open) refresh();
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}

export { refresh, renderRow, get, post };
