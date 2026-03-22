import { api, toast } from "../app.js";

let eventSource = null;

export function init() {
  const container = document.getElementById("view-train");
  container.innerHTML = `
    <div class="card">
      <div class="card-title">训练配置</div>
      <div class="form-row">
        <div>
          <label>数据集目录</label>
          <input type="text" id="tr-dataset-dir" value="dataset" />
        </div>
        <div>
          <label>任务类型</label>
          <select id="tr-task">
            <option value="segment">segment</option>
            <option value="detect">detect</option>
          </select>
        </div>
      </div>
      <div class="form-row">
        <div>
          <label>YOLO 版本</label>
          <select id="tr-version">
            <option value="YOLOv11">YOLOv11</option>
            <option value="YOLOv8">YOLOv8</option>
            <option value="YOLO26">YOLO26</option>
          </select>
        </div>
        <div>
          <label>模型大小</label>
          <select id="tr-size">
            <option value="n">n (最小)</option>
            <option value="s">s</option>
            <option value="m">m</option>
            <option value="l">l</option>
            <option value="x">x (最大)</option>
          </select>
        </div>
        <div>
          <label>Epochs</label>
          <input type="number" id="tr-epochs" value="100" min="1" max="1000" />
        </div>
        <div>
          <label>ImgSz</label>
          <input type="number" id="tr-imgsz" value="640" min="32" step="32" />
        </div>
      </div>
    </div>

    <button class="btn btn-primary" id="tr-run-btn" style="width:100%">开始训练</button>

    <div style="margin-top:16px">
      <div class="progress-bar">
        <div class="progress-bar-fill" id="tr-progress-fill" style="width:0%"></div>
      </div>
      <div class="log-output" id="tr-log" style="margin-top:10px"></div>
    </div>
  `;

  document.getElementById("tr-run-btn").addEventListener("click", startTraining);
}

async function startTraining() {
  if (eventSource) { eventSource.close(); eventSource = null; }

  const req = {
    dataset_dir:  document.getElementById("tr-dataset-dir").value.trim(),
    task:         document.getElementById("tr-task").value,
    yolo_version: document.getElementById("tr-version").value,
    model_size:   document.getElementById("tr-size").value,
    epochs:       parseInt(document.getElementById("tr-epochs").value),
    imgsz:        parseInt(document.getElementById("tr-imgsz").value),
  };

  const btn = document.getElementById("tr-run-btn");
  btn.disabled = true;
  btn.textContent = "训练中...";
  const logEl = document.getElementById("tr-log");
  logEl.textContent = "";

  try {
    const { task_id } = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    eventSource = new EventSource(`/api/train/${task_id}/logs`);

    eventSource.onmessage = (e) => {
      logEl.textContent += e.data + "\n";
      logEl.scrollTop = logEl.scrollHeight;
    };

    eventSource.addEventListener("done", () => {
      eventSource.close();
      btn.disabled = false;
      btn.textContent = "开始训练";
      document.getElementById("tr-progress-fill").style.width = "100%";
      toast("训练完成！");
    });

    eventSource.addEventListener("error", (e) => {
      eventSource.close();
      btn.disabled = false;
      btn.textContent = "开始训练";
      toast(e.data || "训练失败", "error");
    });

    // Poll progress for progress bar
    const poll = setInterval(async () => {
      try {
        const t = await api(`/api/tasks/${task_id}`);
        if (t.total > 0) {
          const pct = Math.round((t.progress / t.total) * 100);
          document.getElementById("tr-progress-fill").style.width = `${pct}%`;
        }
        if (t.status === "done" || t.status === "error") clearInterval(poll);
      } catch {}
    }, 2000);

  } catch {
    btn.disabled = false;
    btn.textContent = "开始训练";
  }
}
