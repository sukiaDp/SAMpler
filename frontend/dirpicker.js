import { api } from "./app.js";

/**
 * 打开目录/文件选择弹窗
 * @param {string} initialPath - 初始路径（相对项目根）
 * @param {{ mode?: "dir"|"file", onSelect?: (path: string) => void }} options
 */
export function openDirPicker(initialPath = "", { mode = "dir", onSelect } = {}) {
  let currentPath = initialPath;

  const backdrop = document.createElement("div");
  backdrop.className = "dp-backdrop";
  backdrop.innerHTML = `
    <div class="dp-modal">
      <div class="dp-header">
        <span class="dp-title">${mode === "file" ? "选择文件" : "选择目录"}</span>
        <button class="dp-close" aria-label="关闭">✕</button>
      </div>
      <div class="dp-breadcrumb" id="dp-bc"></div>
      <div class="dp-list" id="dp-list">
        <div class="dp-empty">加载中…</div>
      </div>
      <div class="dp-footer">
        <span class="dp-current-label" id="dp-cur"></span>
        ${mode === "dir"
          ? '<button class="btn btn-primary dp-confirm" id="dp-ok">选择此目录</button>'
          : ""}
      </div>
    </div>
  `;
  document.body.appendChild(backdrop);

  function close() {
    backdrop.classList.add("dp-leaving");
    backdrop.addEventListener("animationend", () => backdrop.remove(), { once: true });
  }

  function renderBreadcrumb(path) {
    const el = backdrop.querySelector("#dp-bc");
    const parts = path ? path.split("/") : [];
    const crumbs = [
      { label: "根目录", path: "" },
      ...parts.map((p, i) => ({ label: p, path: parts.slice(0, i + 1).join("/") })),
    ];
    el.innerHTML = crumbs
      .map((c, i) =>
        `${i > 0 ? '<span class="dp-sep">/</span>' : ""}` +
        `<button class="dp-crumb" data-path="${c.path}">${c.label}</button>`
      )
      .join("");
    el.querySelectorAll(".dp-crumb").forEach((b) =>
      b.addEventListener("click", () => navigate(b.dataset.path))
    );
  }

  async function navigate(path) {
    currentPath = path;
    renderBreadcrumb(path);
    const curEl = backdrop.querySelector("#dp-cur");
    if (curEl) curEl.textContent = path || "项目根目录";

    const listEl = backdrop.querySelector("#dp-list");
    listEl.innerHTML = '<div class="dp-empty">加载中…</div>';

    try {
      const data = await api(`/api/dirs?path=${encodeURIComponent(path)}`);
      const dirItems = data.dirs.map((d) => ({
        name: d,
        type: "dir",
        fullPath: path ? `${path}/${d}` : d,
      }));
      const fileItems = mode === "file"
        ? data.files.map((f) => ({
            name: f,
            type: "file",
            fullPath: path ? `${path}/${f}` : f,
          }))
        : [];
      const items = [...dirItems, ...fileItems];

      if (!items.length) {
        listEl.innerHTML = `<div class="dp-empty">${
          mode === "file" ? "没有子目录或 .pt 文件" : "没有子目录"
        }</div>`;
        return;
      }

      listEl.innerHTML = items
        .map(
          (item) =>
            `<button class="dp-item ${item.type === "file" ? "dp-file" : ""}"
                     data-path="${item.fullPath}" data-type="${item.type}">
               <span class="dp-icon">${item.type === "dir" ? "📁" : "📄"}</span>
               <span class="dp-name">${item.name}</span>
               ${item.type === "dir" ? '<span class="dp-chevron">›</span>' : ""}
             </button>`
        )
        .join("");

      listEl.querySelectorAll(".dp-item").forEach((btn) => {
        btn.addEventListener("click", () => {
          if (btn.dataset.type === "dir") {
            navigate(btn.dataset.path);
          } else {
            onSelect?.(btn.dataset.path);
            close();
          }
        });
      });
    } catch {
      listEl.innerHTML = '<div class="dp-empty">加载失败</div>';
    }
  }

  // Event wiring
  backdrop.querySelector(".dp-close").addEventListener("click", close);
  backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
  const okBtn = backdrop.querySelector("#dp-ok");
  if (okBtn) okBtn.addEventListener("click", () => { onSelect?.(currentPath); close(); });

  navigate(currentPath);
}

/**
 * 在已有 input 旁边插入一个浏览按钮，点击时打开目录/文件选择弹窗
 * @param {string} inputId
 * @param {"dir"|"file"} mode
 */
export function attachDirPicker(inputId, mode = "dir") {
  const input = document.getElementById(inputId);
  if (!input) return;

  // Wrap input in a relative container
  const wrap = document.createElement("div");
  wrap.className = "dp-input-wrap";
  input.parentNode.insertBefore(wrap, input);
  wrap.appendChild(input);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "dp-browse-btn";
  btn.title = mode === "file" ? "浏览文件" : "浏览目录";
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M1 3.5A1.5 1.5 0 012.5 2h3.172a1.5 1.5 0 011.06.44l.828.828A1.5 1.5 0 008.62 3.5H13.5A1.5 1.5 0 0115 5v7a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 011 12.5v-9z"/>
  </svg>`;
  wrap.appendChild(btn);

  btn.addEventListener("click", () => {
    openDirPicker(input.value || "", {
      mode,
      onSelect: (path) => {
        input.value = path;
        // Fire both events so all listeners (input + change) are triggered
        input.dispatchEvent(new Event("input",  { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      },
    });
  });
}
