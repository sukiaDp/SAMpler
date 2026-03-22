import { api, toast } from "../app.js";
import { attachDirPicker } from "../dirpicker.js";

export function init() {
  const container = document.getElementById("view-infer");
  container.innerHTML = `
    <div style="display:flex;gap:16px">
      <div style="flex:1;min-width:280px">
        <div class="card">
          <div class="card-title">上传图片</div>
          <input type="file" id="infer-file" accept="image/*"
                 style="font-size:13px;color:var(--text-secondary);width:100%" />
          <div id="infer-thumb-wrap" style="margin-top:10px;display:none">
            <img id="infer-thumb" style="max-width:100%;max-height:160px;border-radius:6px" />
          </div>
        </div>

        <div class="card">
          <div class="card-title">模型</div>
          <label>权重文件路径</label>
          <input type="text" id="infer-weights"
                 value="runs/detect/train/weights/best.pt" />
          <div id="infer-model-info"
               style="margin-top:8px;font-size:11px;
                      color:var(--text-secondary);white-space:pre-line;
                      min-height:40px;background:var(--bg);
                      border-radius:6px;padding:8px;
                      border:1px solid var(--separator)">
            （输入权重路径后自动解析）
          </div>
        </div>

        <div class="card">
          <div class="card-title">参数</div>
          <div class="form-row">
            <div>
              <label>置信度</label>
              <input type="number" id="infer-conf" value="0.25"
                     step="0.05" min="0.05" max="0.95" />
            </div>
            <div>
              <label>ImgSz</label>
              <input type="number" id="infer-imgsz" value="640" min="32" step="32" />
            </div>
          </div>
        </div>

        <button class="btn btn-primary" id="infer-run-btn"
                style="width:100%" disabled>开始推理</button>

        <div id="infer-stats"
             style="margin-top:12px;font-size:12px;
                    color:var(--text-secondary);white-space:pre-line"></div>
      </div>

      <div style="flex:3">
        <div class="preview-img-wrap" id="infer-result-wrap" style="min-height:400px">
          <span style="color:var(--text-secondary);font-size:13px">推理结果将在此显示</span>
        </div>
      </div>
    </div>
  `;

  // File preview
  document.getElementById("infer-file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const thumb = document.getElementById("infer-thumb");
    thumb.src = url;
    document.getElementById("infer-thumb-wrap").style.display = "block";
    document.getElementById("infer-run-btn").disabled = false;
  });

  // Model info on weights path change
  let debounce;
  document.getElementById("infer-weights").addEventListener("input", (e) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => fetchModelInfo(e.target.value.trim()), 600);
  });

  document.getElementById("infer-run-btn").addEventListener("click", runInference);
  attachDirPicker("infer-weights", "file");
}

async function fetchModelInfo(path) {
  const el = document.getElementById("infer-model-info");
  if (!path) { el.textContent = "（未指定路径）"; return; }
  try {
    const data = await api(`/api/model-info?weights_path=${encodeURIComponent(path)}`);
    el.textContent = data.raw || "（无法解析）";
  } catch {}
}

async function runInference() {
  const fileInput = document.getElementById("infer-file");
  const weights = document.getElementById("infer-weights").value.trim();
  const conf = document.getElementById("infer-conf").value;
  const imgsz = document.getElementById("infer-imgsz").value;

  if (!fileInput.files[0]) { toast("请先选择图片", "error"); return; }

  const btn = document.getElementById("infer-run-btn");
  btn.disabled = true;
  btn.textContent = "推理中...";

  const fd = new FormData();
  fd.append("image", fileInput.files[0]);
  fd.append("weights_path", weights);
  fd.append("conf", conf);
  fd.append("imgsz", imgsz);

  try {
    const data = await api("/api/infer", { method: "POST", body: fd });

    // Show result image
    const wrap = document.getElementById("infer-result-wrap");
    wrap.innerHTML = `<img src="${data.result_url}?t=${Date.now()}"
                           style="max-width:100%;max-height:70vh;border-radius:6px" />`;

    // Stats
    const s = data.stats;
    const lines = [`检测到 ${s.total} 个目标`];
    for (const [cls, cnt] of Object.entries(s.classes || {})) {
      lines.push(`  ${cls}: ${cnt}`);
    }
    document.getElementById("infer-stats").textContent = lines.join("\n");
    toast("推理完成");
  } catch {
    // error toast already shown by api()
  } finally {
    btn.disabled = false;
    btn.textContent = "开始推理";
  }
}
