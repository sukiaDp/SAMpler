import { api, toast } from "../app.js";
import { attachDirPicker } from "../dirpicker.js";

let pollInterval = null;

export function init() {
  const container = document.getElementById("view-annotate");
  container.innerHTML = `
    <div class="card">
      <div class="card-title">数据源</div>
      <div class="form-row">
        <div>
          <label>图片目录</label>
          <input type="text" id="ann-image-dir" value="rawData" />
        </div>
        <div>
          <label>输出目录</label>
          <input type="text" id="ann-output-dir" value="dataset" />
        </div>
      </div>
      <div>
        <label>提示词（逗号分隔）</label>
        <input type="text" id="ann-prompts" placeholder="person, car, dog" />
      </div>
    </div>

    <div class="card">
      <div class="card-title">标注设置</div>
      <div class="form-row">
        <div>
          <label>标注模式</label>
          <select id="ann-mode">
            <option value="segment">分割 (segment)</option>
            <option value="detect">检测 (detect)</option>
          </select>
        </div>
        <div>
          <label>排序模式</label>
          <select id="ann-sort">
            <option value="conf">置信度</option>
            <option value="area">框面积</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>置信度阈值</label>
          <input type="number" id="ann-conf" value="0.25" step="0.05" min="0.01" max="0.99" />
        </div>
        <div>
          <label>验证集比例</label>
          <input type="number" id="ann-val-ratio" value="0.1" step="0.05" min="0" max="0.5" />
        </div>
        <div>
          <label>最大实例数</label>
          <input type="number" id="ann-max-inst" value="7" min="1" max="100" />
        </div>
      </div>
    </div>

    <button class="btn btn-primary" id="ann-run-btn" style="width:100%">开始标注</button>

    <div id="ann-progress-wrap" style="display:none;margin-top:16px">
      <div class="progress-bar"><div class="progress-bar-fill" id="ann-progress-fill" style="width:0%"></div></div>
      <p style="font-size:12px;color:var(--text-secondary);margin-top:4px" id="ann-progress-text"></p>
    </div>
    <div id="ann-result" style="margin-top:12px;font-size:13px;color:var(--text-secondary)"></div>
  `;

  document.getElementById("ann-run-btn").addEventListener("click", startAnnotation);
  attachDirPicker("ann-image-dir");
  attachDirPicker("ann-output-dir");
}

async function startAnnotation() {
  const req = {
    image_dir:    document.getElementById("ann-image-dir").value.trim(),
    output_dir:   document.getElementById("ann-output-dir").value.trim(),
    prompts:      document.getElementById("ann-prompts").value.trim(),
    mode:         document.getElementById("ann-mode").value,
    sort_mode:    document.getElementById("ann-sort").value,
    conf:         parseFloat(document.getElementById("ann-conf").value),
    val_ratio:    parseFloat(document.getElementById("ann-val-ratio").value),
    max_instances: parseInt(document.getElementById("ann-max-inst").value),
  };

  if (!req.prompts) { toast("请输入提示词", "error"); return; }

  const btn = document.getElementById("ann-run-btn");
  btn.disabled = true;
  btn.textContent = "标注中...";
  document.getElementById("ann-progress-wrap").style.display = "block";
  document.getElementById("ann-result").textContent = "";

  try {
    const { task_id } = await api("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    pollInterval = setInterval(async () => {
      try {
        const task = await api(`/api/tasks/${task_id}`);
        const pct = task.total > 0
          ? Math.round((task.progress / task.total) * 100) : 0;
        document.getElementById("ann-progress-fill").style.width = `${pct}%`;
        document.getElementById("ann-progress-text").textContent =
          `${task.message} (${pct}%)`;

        if (task.status === "done" || task.status === "error") {
          clearInterval(pollInterval);
          btn.disabled = false;
          btn.textContent = "开始标注";
          if (task.status === "done") {
            const r = task.result || {};
            const lines = [`标注完成！总图片: ${r.total}`];
            if (r.class_counts) {
              for (const [k, v] of Object.entries(r.class_counts)) {
                lines.push(`  ${k}: ${v}`);
              }
            }
            document.getElementById("ann-result").textContent = lines.join("\n");
            toast("标注完成！");
          } else {
            toast(task.message || "标注失败", "error");
          }
        }
      } catch {}
    }, 1000);
  } catch {
    btn.disabled = false;
    btn.textContent = "开始标注";
  }
}
