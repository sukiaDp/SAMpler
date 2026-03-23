// ── API client ────────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  try {
    const res = await fetch(path, options);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.error || j.detail || msg; } catch {}
      toast(msg, "error");
      throw new Error(msg);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  } catch (e) {
    if (e.message === "Failed to fetch") toast("无法连接服务器", "error");
    throw e;
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function toast(msg, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  container.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Theme ─────────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const isDark = current === "dark" ||
    (!current && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const next = isDark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  document.getElementById("theme-toggle").textContent = next === "dark" ? "☀" : "🌙";
}

// ── SAM status widget ─────────────────────────────────────────────────────────

const SAM_STATUS_LABEL = {
  not_found: "未下载",
  idle:      "未加载",
  loading:   "加载中",
  ready:     "就绪",
  inferring: "推理中",
};
const SAM_DOWNLOAD_URL = "https://huggingface.co/1038lab/sam3";

function initSamStatus() {
  const widget = document.getElementById("sam-status-widget");
  const label  = document.getElementById("sam-status-label");

  function update(status) {
    widget.dataset.status = status;
    label.textContent = SAM_STATUS_LABEL[status] ?? status;
    const clickable = status === "not_found";
    widget.dataset.clickable = clickable;
    widget.title = clickable ? "点击前往下载 sam3.pt" : `SAM3 模型状态：${SAM_STATUS_LABEL[status]}`;
  }

  widget.addEventListener("click", () => {
    if (widget.dataset.clickable === "true") window.open(SAM_DOWNLOAD_URL, "_blank");
  });

  async function poll() {
    try {
      const res = await fetch("/api/sam/status");
      if (res.ok) { const d = await res.json(); update(d.status); }
    } catch {}
  }

  poll();
  setInterval(poll, 3000);
}

// ── Router ────────────────────────────────────────────────────────────────────

const VIEWS = {
  sam:      { label: "标注测试", icon: "✂️" },
  annotate: { label: "自动标注", icon: "🏷" },
  preview:  { label: "预览编辑", icon: "🖼" },
  train:    { label: "YOLO 训练", icon: "⚡" },
  infer:    { label: "推理测试", icon: "🔍" },
};

let currentView = "sam";

function navigate(view) {
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach(el => {
    el.classList.toggle("active", el.id === `view-${view}`);
  });
  document.querySelector("#topbar .page-title").textContent =
    VIEWS[view]?.label ?? view;
  currentView = view;
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  initSamStatus();

  // Build sidebar
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML =
    `<div class="nav-label">功能</div>` +
    Object.entries(VIEWS).map(([key, v]) =>
      `<div class="nav-item${key === currentView ? " active" : ""}"
            data-view="${key}">
         <span class="nav-icon">${v.icon}</span>${v.label}
       </div>`
    ).join("");

  sidebar.querySelectorAll(".nav-item").forEach(el =>
    el.addEventListener("click", () => window.navigate(el.dataset.view))
  );

  document.getElementById("theme-toggle")
    .addEventListener("click", toggleTheme);

  // Lazy-init views
  const modules = {
    annotate: () => import("./views/annotate.js"),
    preview:  () => import("./views/preview.js"),
    train:    () => import("./views/train.js"),
    infer:    () => import("./views/infer.js"),
    sam:      () => import("./views/sam.js"),
  };
  const initialized = new Set();

  async function maybeInit(view) {
    if (!initialized.has(view) && modules[view]) {
      const mod = await modules[view]();
      if (mod.init) mod.init();
      initialized.add(view);
    }
  }

  // Override navigate to lazy-init
  const _navigate = navigate;
  window.navigate = async (view) => {
    _navigate(view);
    await maybeInit(view);
  };

  await maybeInit("sam");
  navigate("sam");
});

export { api, toast };
