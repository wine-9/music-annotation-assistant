const form = document.querySelector("#upload-form");
const progress = document.querySelector("#progress");
const progressBar = document.querySelector("#progress-bar");
const stage = document.querySelector("#stage");
const errorBox = document.querySelector("#error");
let currentResult = null;
let reviewStartedAt = null;
let fieldStartedAt = new Map();
let reviewEvents = [];

loadThresholdProfiles();
loadRuntimeCapabilities();

async function loadRuntimeCapabilities() {
  const response = await fetch("/health");
  const health = await response.json();
  const checkbox = document.querySelector('[name="enable_separation"]');
  checkbox.disabled = !(health.separation_enabled && health.separation_available);
  if (checkbox.disabled) checkbox.parentElement.title = "需先安装可选 Demucs 依赖";
}

async function loadThresholdProfiles() {
  const response = await fetch("/api/v1/threshold-profile");
  const payload = await response.json();
  const select = document.querySelector("#profile-select");
  select.innerHTML = payload.available.map(profile =>
    `<option value="${escapeHtml(profile.name)}">${escapeHtml(profile.name)}</option>`
  ).join("");
  select.value = payload.active.name;
  renderProfile(payload.active);
}
function renderProfile(profile) {
  const datasets = profile.applicable_datasets.length ? profile.applicable_datasets.join("、") : "未校准默认值";
  document.querySelector("#profile-details").textContent =
    `名称 ${profile.name} · 版本 ${profile.version} · 生成日期 ${profile.generated_at} · 适用数据集 ${datasets}`;
}
document.querySelector("#profile-select").addEventListener("change", async (event) => {
  const response = await fetch(`/api/v1/threshold-profile/${encodeURIComponent(event.target.value)}`, {method: "PUT"});
  if (!response.ok) return showError("阈值配置切换失败");
  const payload = await response.json();
  renderProfile(payload.active);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  progress.hidden = false;
  const file = document.querySelector("#audio-file").files[0];
  const body = new FormData(form);
  document.querySelector("#player").src = URL.createObjectURL(file);
  document.querySelector("#player-panel").hidden = false;
  try {
    const response = await fetch("/api/v1/analyses", { method: "POST", body });
    if (!response.ok) throw new Error((await response.json()).detail || "无法创建任务");
    const task = await response.json();
    await poll(task.analysis_id);
  } catch (error) {
    showError(error.message);
  }
});

async function poll(id) {
  const response = await fetch(`/api/v1/analyses/${id}`);
  const task = await response.json();
  progressBar.style.width = `${Math.round(task.progress * 100)}%`;
  stage.textContent = task.stage;
  if (task.status === "failed") return showError(task.error || "分析失败");
  if (task.status !== "completed") return setTimeout(() => poll(id), 900);
  const resultResponse = await fetch(`/api/v1/analyses/${id}/result`);
  currentResult = await resultResponse.json();
  reviewStartedAt = performance.now();
  reviewEvents = [];
  render(currentResult, id);
}

function render(result, id) {
  document.querySelector("#spatial-panel").hidden = false;
  document.querySelector("#results-panel").hidden = false;
  const spatial = result.global_spatial;
  const metrics = [
    ["位置", spatial.pan], ["宽度", spatial.width], ["稳定性", spatial.stability],
    ["Pan", format(spatial.pan_value)], ["Width ratio", format(spatial.width_ratio)],
    ["Correlation", format(spatial.correlation)]
  ];
  document.querySelector("#spatial-grid").innerHTML = metrics
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value ?? "unknown"}</strong></div>`).join("");
  document.querySelector("#cards").innerHTML = result.instruments
    .map((item, index) => ({ item, index }))
    .sort((a, b) => b.item.model_score - a.item.model_score)
    .map(({item, index}) => {
      const override = [...result.human_overrides].reverse()
        .find(entry => entry.field_path === `instruments.${index}.decision`);
      return `<article class="card">
      <div class="card-head"><div><h3>${escapeHtml(item.display_name_zh)}</h3>
      <span class="raw">${escapeHtml(item.canonical_label)}</span></div>
      <span class="badge ${item.decision}">${item.decision}</span></div>
      <p>模型分数 <strong>${item.model_score.toFixed(3)}</strong></p>
      <p class="raw">原始标签：${Object.entries(item.raw_model_scores).map(([k,v]) => `${escapeHtml(k)} ${v.toFixed(3)}`).join(" · ")}</p>
      <p class="raw">区间：${item.time_ranges.length ? item.time_ranges.map(r =>
        `<button class="time-link" data-seek="${r.start_seconds}">${r.start_seconds.toFixed(1)}–${r.end_seconds.toFixed(1)}s · ${(r.interval_confidence * 100).toFixed(0)}%</button>`
      ).join(" ") : "无可靠区间"}</p>
      ${override ? `<p class="human-note">人工修订：${escapeHtml(String(override.new_value))}</p>` : ""}
      <div class="review-actions">
        <button data-index="${index}" data-new="confirmed">接受</button>
        <button class="secondary" data-index="${index}" data-new="candidate">改为候选</button>
        <button class="secondary" data-index="${index}" data-new="removed">删除</button>
      </div>
    </article>`;
    }).join("");
  document.querySelector("#csv-link").href = `/api/v1/analyses/${id}/result.csv`;
  document.querySelectorAll(".review-actions button").forEach(button => {
    button.addEventListener("click", () => saveDecision(button.dataset.index, button.dataset.new));
  });
  document.querySelectorAll(".time-link").forEach(button => {
    button.addEventListener("click", () => {
      const player = document.querySelector("#player");
      player.currentTime = Number(button.dataset.seek);
      player.play();
    });
  });
  renderFormPreview(result.target_form);
}

function renderFormPreview(preview) {
  if (!preview) return;
  document.querySelector("#form-preview-panel").hidden = false;
  const summary = preview.summary;
  const values = [
    ["自动确认", summary.confirmed], ["候选", summary.candidate],
    ["未判断", summary.unknown], ["总字段", summary.total],
    ["确认覆盖率", `${(summary.confirmed_coverage * 100).toFixed(1)}%`],
    ["总预填覆盖率", `${(summary.prefill_coverage * 100).toFixed(1)}%`]
  ];
  document.querySelector("#form-summary").innerHTML = values.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
  ).join("");
  document.querySelector("#form-fields").innerHTML = preview.fields.map(field => {
    if (!fieldStartedAt.has(field.field_id)) fieldStartedAt.set(field.field_id, performance.now());
    return `<article class="form-field ${field.decision}">
      <div><strong>${escapeHtml(field.field_name)}</strong>
      <span class="raw">${escapeHtml(field.instrument_label || "global")}</span></div>
      <div class="form-value">${field.value == null ? "—" : escapeHtml(typeof field.value === "object" ? JSON.stringify(field.value) : String(field.value))}</div>
      <span class="badge ${field.decision}">${field.decision}</span>
      <p class="raw">${escapeHtml(field.reason)}</p>
      <div class="review-actions">
        <button data-field="${escapeHtml(field.field_id)}" data-outcome="accept_as_is">接受</button>
        <button class="secondary" data-field="${escapeHtml(field.field_id)}" data-outcome="accept_after_edit">修改</button>
        <button class="secondary" data-field="${escapeHtml(field.field_id)}" data-outcome="reject">删除</button>
        <button class="secondary" data-field="${escapeHtml(field.field_id)}" data-outcome="missing_label">漏标</button>
        <button class="secondary" data-field="${escapeHtml(field.field_id)}" data-outcome="uncertain">不确定</button>
      </div>
    </article>`;
  }).join("");
  document.querySelectorAll("#form-fields [data-outcome]").forEach(button => {
    button.addEventListener("click", () => recordReview(button.dataset.field, button.dataset.outcome, button));
  });
}

function recordReview(fieldId, outcome, button) {
  const field = currentResult.target_form.fields.find(item => item.field_id === fieldId);
  let finalValue = field.value;
  let editDescription = null;
  if (outcome === "accept_after_edit" || outcome === "missing_label") {
    const entered = window.prompt(outcome === "missing_label" ? "输入模型漏掉的最终值" : "输入修改后的值", field.value ?? "");
    if (entered === null) return;
    finalValue = entered;
    editDescription = `${field.value ?? "空"} → ${entered}`;
  }
  reviewEvents.push({
    field_id: fieldId,
    outcome,
    original_prediction: field,
    final_value: finalValue,
    edit_description: editDescription,
    threshold_profile_name: currentResult.pipeline.threshold_profile_name,
    field_elapsed_seconds: Math.max(0, (performance.now() - (fieldStartedAt.get(fieldId) || reviewStartedAt)) / 1000)
  });
  button.closest(".form-field").classList.add("reviewed");
}

document.querySelector("#finish-review").addEventListener("click", async () => {
  if (!currentResult || !reviewStartedAt) return;
  const totalReviewSeconds = (performance.now() - reviewStartedAt) / 1000;
  const baselineText = window.prompt("可选：如果完全手工标注，预计需要多少秒？留空则不计算节省时间。", "");
  const baseline = baselineText && Number(baselineText) > 0 ? Number(baselineText) : null;
  const response = await fetch(`/api/v1/analyses/${currentResult.analysis_id}/review`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      analysis_id: currentResult.analysis_id,
      events: reviewEvents,
      total_review_seconds: totalReviewSeconds,
      baseline_manual_seconds: baseline,
      model_analysis_seconds: currentResult.model_analysis_seconds
    })
  });
  if (!response.ok) return showError("人工复核记录保存失败");
  document.querySelector("#finish-review").textContent = `已保存 ${reviewEvents.length} 个操作`;
});

document.querySelector("#copy-json").addEventListener("click", () => {
  if (currentResult) navigator.clipboard.writeText(JSON.stringify(currentResult, null, 2));
});
document.querySelector("#add-label-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const label = document.querySelector("#manual-label").value.trim();
  if (!label) return;
  await saveOverride("human_labels", null, {canonical_label: label, decision: "confirmed"});
  reviewEvents.push({
    field_id: `human_labels.${label}`,
    outcome: "missing_label",
    original_prediction: null,
    final_value: {canonical_label: label, decision: "confirmed"},
    edit_description: "人工添加模型漏标",
    threshold_profile_name: currentResult.pipeline.threshold_profile_name,
    field_elapsed_seconds: Math.max(0, (performance.now() - reviewStartedAt) / 1000)
  });
  document.querySelector("#manual-label").value = "";
});
async function saveDecision(index, newValue) {
  const item = currentResult.instruments[Number(index)];
  await saveOverride(`instruments.${index}.decision`, item.decision, newValue);
}
async function saveOverride(fieldPath, oldValue, newValue) {
  const response = await fetch(`/api/v1/analyses/${currentResult.analysis_id}/result`, {
    method: "PATCH",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify([{field_path: fieldPath, old_value: oldValue, new_value: newValue}])
  });
  if (!response.ok) return showError("人工修改保存失败");
  currentResult = await response.json();
  render(currentResult, currentResult.analysis_id);
}
function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }
function format(value) { return value == null ? "unknown" : Number(value).toFixed(3); }
function escapeHtml(value) { const e = document.createElement("div"); e.textContent = value; return e.innerHTML; }
