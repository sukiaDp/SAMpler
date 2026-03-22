import { api, toast } from "../app.js";

let state = {
  files: [],
  idx: 0,
  imagesDir: "dataset/images/train",
  selectedAnns: new Set(),
};

// id → preview API response, cleared when imagesDir changes or annotation deleted
const previewCache = new Map();

async function fetchPreview(imageId) {
  if (previewCache.has(imageId)) return previewCache.get(imageId);
  const data = await api(
    `/api/images/${imageId}/preview?images_dir=${encodeURIComponent(state.imagesDir)}`
  );
  previewCache.set(imageId, data);
  return data;
}

function preloadAround(centerIdx) {
  const files = state.files;
  for (let offset = -5; offset <= 5; offset++) {
    if (offset === 0) continue;
    const i = centerIdx + offset;
    if (i < 0 || i >= files.length) continue;
    const id = files[i].id;
    if (!previewCache.has(id)) {
      // fire-and-forget; errors silently ignored
      fetchPreview(id).catch(() => {});
    }
  }
}

export function init() {
  const container = document.getElementById("view-preview");
  container.innerHTML = `
    <div class="card" style="margin-bottom:12px">
      <div class="form-row" style="align-items:flex-end">
        <div style="flex:3">
          <label>图片目录 (dataset/images/train 或 val)</label>
          <input type="text" id="prev-dir" value="dataset/images/train" />
        </div>
        <button class="btn btn-primary" id="prev-load-btn">加载</button>
      </div>
    </div>

    <div class="preview-toolbar">
      <span class="filename" id="prev-filename">—</span>
      <button class="btn" id="prev-prev-btn" style="padding:6px 10px">◀</button>
      <span id="prev-page" style="font-size:12px;color:var(--text-secondary);min-width:50px;text-align:center">0 / 0</span>
      <button class="btn" id="prev-next-btn" style="padding:6px 10px">▶</button>
      <button class="btn btn-danger" id="prev-del-img-btn">删除此图</button>
    </div>

    <div class="preview-img-wrap" id="prev-img-wrap">
      <span style="color:var(--text-secondary);font-size:13px">请先加载图片目录</span>
    </div>

    <div class="annotation-tags" id="prev-ann-tags"></div>
    <div style="display:flex;justify-content:flex-end;margin-top:8px">
      <button class="btn btn-danger" id="prev-del-ann-btn" disabled>删除选中标注</button>
    </div>
  `;

  document.getElementById("prev-load-btn").addEventListener("click", loadDir);
  document.getElementById("prev-prev-btn").addEventListener("click", () => navigate(-1));
  document.getElementById("prev-next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-del-img-btn").addEventListener("click", deleteCurrentImage);
  document.getElementById("prev-del-ann-btn").addEventListener("click", deleteSelectedAnns);
}

async function loadDir() {
  const dir = document.getElementById("prev-dir").value.trim();
  try {
    const data = await api(`/api/images?dir=${encodeURIComponent(dir)}`);
    previewCache.clear();
    state.files = data.files;
    state.idx = 0;
    state.imagesDir = dir;
    state.selectedAnns = new Set();
    toast(`加载了 ${data.total} 张图片`);
    await showCurrent();
  } catch {}
}

async function navigate(dir) {
  if (!state.files.length) return;
  state.idx = Math.max(0, Math.min(state.idx + dir, state.files.length - 1));
  state.selectedAnns = new Set();
  await showCurrent(dir);
}

async function showCurrent(dir = 0) {
  const files = state.files;
  if (!files.length) return;
  const f = files[state.idx];

  document.getElementById("prev-filename").textContent = f.filename;
  document.getElementById("prev-page").textContent =
    `${state.idx + 1} / ${files.length}`;

  try {
    const data = await fetchPreview(f.id);
    const wrap = document.getElementById("prev-img-wrap");
    const animClass = dir > 0 ? "slide-from-right" : dir < 0 ? "slide-from-left" : "";
    wrap.innerHTML = `<img src="${data.preview_url}?t=${Date.now()}" alt="${f.filename}"
                           class="${animClass}" />`;
    renderTags(data.annotations);
    preloadAround(state.idx);
  } catch {}
}

function renderTags(anns) {
  const container = document.getElementById("prev-ann-tags");
  state.selectedAnns = new Set();
  updateDeleteBtn();

  if (!anns.length) {
    container.innerHTML = `<span style="color:var(--text-secondary);font-size:12px">无标注</span>`;
    return;
  }

  container.innerHTML = anns.map((a, i) => {
    const [r, g, b] = a.color;
    return `
      <div class="ann-tag" data-ann-id="${a.id}">
        <span class="ann-color-swatch" style="background:rgb(${r},${g},${b})"></span>
        <span>${a.class_name}</span>
        <span style="font-size:10px;color:var(--text-secondary)">${a.type}</span>
      </div>`;
  }).join("");

  container.querySelectorAll(".ann-tag").forEach(el => {
    el.addEventListener("click", () => {
      const id = parseInt(el.dataset.annId);
      if (state.selectedAnns.has(id)) {
        state.selectedAnns.delete(id);
        el.classList.remove("selected");
      } else {
        state.selectedAnns.add(id);
        el.classList.add("selected");
      }
      updateDeleteBtn();
    });
  });
}

function updateDeleteBtn() {
  document.getElementById("prev-del-ann-btn").disabled =
    state.selectedAnns.size === 0;
}

async function deleteCurrentImage() {
  if (!state.files.length) return;
  const f = state.files[state.idx];
  try {
    await api(
      `/api/images/${f.id}?images_dir=${encodeURIComponent(state.imagesDir)}`,
      { method: "DELETE" }
    );
    state.files.splice(state.idx, 1);
    if (!state.files.length) {
      document.getElementById("prev-img-wrap").innerHTML =
        `<span style="color:var(--text-secondary)">列表已空</span>`;
      document.getElementById("prev-ann-tags").innerHTML = "";
      return;
    }
    state.idx = Math.min(state.idx, state.files.length - 1);
    state.selectedAnns = new Set();
    toast("已删除");
    await showCurrent();
  } catch {}
}

async function deleteSelectedAnns() {
  if (!state.files.length || !state.selectedAnns.size) return;
  const f = state.files[state.idx];
  // Delete in reverse order to preserve indices
  const ids = [...state.selectedAnns].sort((a, b) => b - a);
  try {
    for (const annId of ids) {
      await api(
        `/api/images/${f.id}/annotations/${annId}?images_dir=${encodeURIComponent(state.imagesDir)}`,
        { method: "DELETE" }
      );
    }
    toast(`已删除 ${ids.length} 个标注`);
    state.selectedAnns = new Set();
    previewCache.delete(f.id);  // force re-fetch after annotation change
    await showCurrent();
  } catch {}
}
