import { api, toast } from "../app.js";
import { attachDirPicker } from "../dirpicker.js";

let eventSource = null;

// 正则：匹配 epoch 训练行，如 "  1/10   2.37G   2.409   3.352   2.079   110   640: 100%"
const RE_TRAIN = /^\s*(\d+)\/(\d+)\s+([\d.]+G?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+:\s*100%/;
// 正则：匹配 val 汇总行，如 "  all   100   450   0.512   0.489   0.501   0.278"
const RE_VAL   = /^\s*all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/;

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

      <div class="tr-metrics" id="tr-metrics" style="display:none">
        <div class="tr-metric-item">
          <span class="tr-metric-label">Epoch</span>
          <span class="tr-metric-value" id="trm-epoch">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">GPU</span>
          <span class="tr-metric-value" id="trm-gpu">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">box_loss</span>
          <span class="tr-metric-value" id="trm-box">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">cls_loss</span>
          <span class="tr-metric-value" id="trm-cls">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">dfl_loss</span>
          <span class="tr-metric-value" id="trm-dfl">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">Precision</span>
          <span class="tr-metric-value" id="trm-p">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">Recall</span>
          <span class="tr-metric-value" id="trm-r">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">mAP50</span>
          <span class="tr-metric-value" id="trm-map50">—</span>
        </div>
        <div class="tr-metric-item">
          <span class="tr-metric-label">mAP50-95</span>
          <span class="tr-metric-value" id="trm-map">—</span>
        </div>
      </div>

      <div class="log-output" id="tr-log" style="margin-top:10px"></div>
    </div>
  `;

  document.getElementById("tr-run-btn").addEventListener("click", startTraining);
  attachDirPicker("tr-dataset-dir");
}

function updateMetrics(line) {
  let m;
  if ((m = RE_TRAIN.exec(line))) {
    document.getElementById("tr-metrics").style.display = "grid";
    document.getElementById("trm-epoch").textContent = `${m[1]} / ${m[2]}`;
    document.getElementById("trm-gpu").textContent   = m[3];
    document.getElementById("trm-box").textContent   = m[4];
    document.getElementById("trm-cls").textContent   = m[5];
    document.getElementById("trm-dfl").textContent   = m[6];
  } else if ((m = RE_VAL.exec(line))) {
    document.getElementById("trm-p").textContent     = m[1];
    document.getElementById("trm-r").textContent     = m[2];
    document.getElementById("trm-map50").textContent = m[3];
    document.getElementById("trm-map").textContent   = m[4];
  }
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
  document.getElementById("tr-metrics").style.display = "none";

  try {
    const { task_id } = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });

    eventSource = new EventSource(`/api/train/${task_id}/logs`);

    eventSource.onmessage = (e) => {
      updateMetrics(e.data);
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

    // "train_error" is our custom SSE event (avoids conflict with EventSource's
    // built-in "error" event which fires on transient connection issues)
    eventSource.addEventListener("train_error", (e) => {
      eventSource.close();
      btn.disabled = false;
      btn.textContent = "开始训练";
      toast(e.data || "训练失败", "error");
    });

    // Built-in connection error — just log, EventSource auto-reconnects
    eventSource.onerror = () => {
      logEl.textContent += "(连接中断，正在重连...)\n";
      logEl.scrollTop = logEl.scrollHeight;
    };

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
