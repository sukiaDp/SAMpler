import { api, toast } from "../app.js";

export function init() {
  const container = document.getElementById("view-sam");
  container.innerHTML = `
    <div style="display:flex;gap:16px;align-items:flex-start">

      <!-- Left: controls -->
      <div style="flex:0 0 260px;display:flex;flex-direction:column;gap:0">

        <div class="card">
          <div class="card-title">上传图片</div>
          <input type="file" id="sam-file" accept="image/*"
                 style="font-size:13px;color:var(--text-secondary);width:100%" />
          <div id="sam-thumb-wrap" style="margin-top:10px;display:none">
            <img id="sam-thumb"
                 style="max-width:100%;max-height:160px;border-radius:6px;display:block" />
          </div>
        </div>

        <div class="card">
          <div class="card-title">提示词</div>
          <label>目标类别（逗号分隔）</label>
          <input type="text" id="sam-prompts" placeholder="person, car, dog" />
          <div style="margin-top:12px;display:flex;flex-direction:column;gap:8px">
            <div style="display:flex;gap:10px">
              <div style="flex:1">
                <label>置信度</label>
                <input type="number" id="sam-conf" value="0.25"
                       min="0.01" max="1" step="0.05" />
              </div>
              <div style="flex:1">
                <label>最大实例数</label>
                <input type="number" id="sam-max" value="7" min="1" max="50" />
              </div>
            </div>
            <div>
              <label>排序方式</label>
              <select id="sam-sort">
                <option value="conf">置信度优先</option>
                <option value="area">面积优先</option>
              </select>
            </div>
          </div>
        </div>

        <button class="btn btn-primary" id="sam-run-btn"
                style="width:100%;margin-bottom:16px">运行 SAM3</button>

        <div class="card" id="sam-stats-card" style="display:none">
          <div class="card-title">结果统计</div>
          <div id="sam-stats" style="font-size:13px;color:var(--text-secondary)"></div>
        </div>

      </div>

      <!-- Right: result -->
      <div style="flex:1;min-width:0">
        <div class="card" style="min-height:300px;display:flex;
                                  align-items:center;justify-content:center">
          <div id="sam-placeholder"
               style="font-size:13px;color:var(--text-secondary);text-align:center">
            上传图片并输入提示词后点击运行
          </div>
          <img id="sam-result"
               style="max-width:100%;border-radius:6px;display:none" />
        </div>
      </div>

    </div>
  `;

  // Thumbnail preview on file select
  document.getElementById("sam-file").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const thumb = document.getElementById("sam-thumb");
    thumb.src = url;
    document.getElementById("sam-thumb-wrap").style.display = "block";
  });

  document.getElementById("sam-run-btn").addEventListener("click", runSam);
}

async function runSam() {
  const fileInput = document.getElementById("sam-file");
  const file = fileInput.files?.[0];
  if (!file) { toast("请先上传图片", "error"); return; }

  const prompts = document.getElementById("sam-prompts").value.trim();
  if (!prompts) { toast("请输入提示词", "error"); return; }

  const btn = document.getElementById("sam-run-btn");
  btn.disabled = true;
  btn.textContent = "推理中…";

  const resultImg   = document.getElementById("sam-result");
  const placeholder = document.getElementById("sam-placeholder");

  try {
    const form = new FormData();
    form.append("image",        file);
    form.append("prompts",      prompts);
    form.append("conf",         document.getElementById("sam-conf").value);
    form.append("max_instances",document.getElementById("sam-max").value);
    form.append("sort_mode",    document.getElementById("sam-sort").value);

    const data = await api("/api/segment", { method: "POST", body: form });

    resultImg.src           = data.preview_url + `?t=${Date.now()}`;
    resultImg.style.display = "block";
    placeholder.style.display = "none";

    // Stats
    const statsCard = document.getElementById("sam-stats-card");
    const statsEl   = document.getElementById("sam-stats");
    statsCard.style.display = "block";
    const { total, classes } = data.stats;
    const classLines = Object.entries(classes)
      .map(([k, v]) => `<span style="color:var(--text-primary)">${k}</span> × ${v}`)
      .join("　");
    statsEl.innerHTML = `共检测到 <strong style="color:var(--text-primary)">${total}</strong> 个实例<br>
      <span style="margin-top:4px;display:inline-block">${classLines || "—"}</span>`;
  } catch {
    // error already toasted by api()
  } finally {
    btn.disabled = false;
    btn.textContent = "运行 SAM3";
  }
}
