'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => Number(value || 0).toLocaleString();
const percent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const title = value => value.charAt(0).toUpperCase() + value.slice(1).replaceAll('_', ' ');
let state = {categories: [], runs: []}, jobs = [], token = '', selectedRun = '', polling = false, loading = false, submitting = false;
let scanVersion = 0, runVersion = 0, previousJobStatus = '';
let folderCategory = '';
let modelRecordsVersion = 0;
let liveStream = null, liveSession = '', liveActive = false, liveVersion = 0, liveTimer = null;
let liveCaptureVersion = 0, liveResetting = false;
let predictionTestsVersion = 0, editingTest = null, editingPrediction = null, suggestedTestName = '';
const latestPredictions = {sample: null, live: null};
let exportingRun = '';
let automatedTest = '', automatedViewedTest = '', automatedTrial = null, automatedPending = null;
let automatedSaving = false, automatedStarting = false, automatedTimer = null;

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = !message;
}
async function api(path, body, options = {}) {
  const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Dashboard-Token': token}, body: JSON.stringify(body), ...options
  });
  const result = await response.json();
  if (response.status === 404 && ['/api/model-record', '/api/prediction-test', '/api/prediction-test/trial'].includes(path)) {
    throw new Error('Restart dashboard.py and refresh the page to enable saved model records and prediction tests.');
  }
  if (response.status === 404 && path.startsWith('/api/live/')) {
    throw new Error('Live prediction is unavailable in the running server. Finish any running job, restart dashboard.py, then refresh this page.');
  }
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
  return result;
}
function route() {
  const names = {overview: 'Overview', datasets: 'Datasets', training: 'Training', runs: 'Model runs', predict: 'Try a prediction'};
  const view = location.hash.slice(1) in names ? location.hash.slice(1) : 'overview';
  document.querySelectorAll('.view').forEach(el => { el.hidden = el.id !== `view-${view}`; });
  document.querySelectorAll('[data-view]').forEach(el => {
    const active = el.dataset.view === view;
    el.classList.toggle('active', active);
    if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
  });
  $('breadcrumb').textContent = names[view];
  document.title = `${names[view]} · FSL Studio`;
  if (view !== 'predict' && liveActive) stopLive();
}
function stat(label, value, note, icon) {
  return `<article class="stat"><div class="stat-top">${label}<span class="stat-icon" aria-hidden="true">${icon}</span></div><strong>${value}</strong><small>${note}</small></article>`;
}
function empty(text) { return `<div class="empty-state"><p>${escapeHTML(text)}</p></div>`; }
function populate(select, items, placeholder) {
  const previous = select.value;
  select.innerHTML = items.length ? items.map(([value, label]) => `<option value="${escapeHTML(value)}">${escapeHTML(label)}</option>`).join('') : `<option value="">${escapeHTML(placeholder)}</option>`;
  select.value = items.some(([value]) => value === previous) ? previous : (items[0]?.[0] || '');
}
function render() {
  const categories = state.categories;
  const completed = state.runs.filter(r => r.complete);
  const scores = completed.map(r => r.accuracy).filter(v => typeof v === 'number');
  $('stats').innerHTML = stat('Gesture categories', fmt(categories.length), `${fmt(categories.filter(c => c.samples).length)} with source samples`, '▦')
    + stat('Source samples', fmt(categories.reduce((s, c) => s + c.samples, 0)), 'Original images, videos & landmarks', '◈')
    + stat('Model runs', fmt(state.runs.length), `${fmt(completed.length)} with test results`, '⌁')
    + stat('Best test accuracy', scores.length ? percent(Math.max(...scores)) : '—', 'Across completed runs · held-out test', '↗');
  $('nav-categories').textContent = categories.length;
  $('category-list').innerHTML = categories.length ? categories.map(c => `<div class="category-row"><span class="category-icon" aria-hidden="true">${escapeHTML(c.name[0].toUpperCase())}</span><div><div class="category-name">${escapeHTML(title(c.name))}</div><div class="category-meta">${fmt(c.classes.length)} gestures <span>·</span> ${fmt(c.samples)} samples</div></div><span class="tag ${c.prepared ? 'ready' : ''}">${c.prepared ? 'Prepared' : c.samples ? 'Raw samples' : 'Awaiting data'}</span></div>`).join('') : empty('No categories found. Add source folders in your selected source root.');
  document.querySelectorAll('.category-select').forEach(select => populate(select, categories.map(c => [c.name, title(c.name)]), 'No categories found'));
  populate($('run-filter'), [['', 'All categories'], ...[...new Set(state.runs.map(r => r.category))].map(c => [c, title(c)])], 'All categories');
  renderPredictRuns();
  $('environment').textContent = `Python ${state.python} · Local processing`;
  $('recent-runs').innerHTML = runTable(state.runs.slice(0, 4));
  renderSourceFolders(); renderTrainingDatasets(); renderDatasets(); renderRuns(); controls();
}
function preparedChoices() {
  return state.categories.find(c => c.name === $('train-form').elements.category.value)?.datasets || [];
}
function renderTrainingDatasets() {
  populate($('train-dataset'), preparedChoices().map(d => [d.path, d.name]), 'Prepare a dataset first');
  controls();
}
function datasetPaths() {
  const category = $('prepare-form').elements.category.value;
  const name = $('dataset-name').value;
  $('prepare-output').textContent = `Output: ${$('dataset-root').value}/${category || '<category>'}/${name || '<dataset name>'}`;
  const selected = preparedChoices().find(d => d.path === $('train-dataset').value);
  $('training-source').textContent = selected
    ? `Dataset: ${selected.path} · ${protocolDescription(selected.evaluation_protocol)} · Output: runs/${$('train-form').elements.category.value}/<run name>`
    : 'Prepare a dataset, then select it here to train a model.';
}
function protocolDescription(protocol) {
  return protocol?.strategy === 'leave_one_signer_out'
    ? `LOSO test: ${protocol.test_signer}; validation: ${protocol.validation_signer}; train: ${protocol.train_signers.join(', ')}`
    : 'Saved source-group split';
}
function sourceSelection() {
  return {source_folders: $('source-folder-mode').value === 'selected'
    ? [...$('source-folder-options').querySelectorAll('input:checked')].map(input => input.value) : null,
    videos_only: $('source-videos-only').checked === true};
}
function renderSourceFolders() {
  const category = $('prepare-form').elements.category.value;
  const classes = state.categories.find(c => c.name === category)?.classes || [];
  const previous = folderCategory === category
    ? new Map([...$('source-folder-options').querySelectorAll('input')].map(input => [input.value, input.checked])) : new Map();
  folderCategory = category;
  const names = [...new Set(classes.flatMap(c => Object.keys(c.folders || {})))].sort();
  $('source-folder-options').innerHTML = names.map(name => `<label class="checkbox"><input type="checkbox" value="${escapeHTML(name)}" ${previous.get(name) !== false ? 'checked' : ''}> ${escapeHTML(name)}</label>`).join('') || '<span class="hint">No subfolders found beneath the gesture folders.</span>';
  populate($('test-signer'), names.map(name => [name, name]), 'No signer subfolders found');
  populate($('validation-signer'), [['', 'Automatic (next signer)'], ...names.map(name => [name, name])], 'Automatic');
}
function selectedClasses(category) {
  const selection = sourceSelection();
  return (category?.classes || []).map(c => ({name: c.name, samples: selection.source_folders === null
    ? (selection.videos_only ? c.videos : c.samples)
    : selection.source_folders.reduce((sum, name) => sum + (c.folders?.[name]?.[selection.videos_only ? 'videos' : 'samples'] || 0), 0)}));
}
function runTable(rows) {
  if (!rows.length) return empty('No runs here yet. Prepare a dataset and start your first training run.');
  return `<table><thead><tr><th>RUN NAME</th><th>CATEGORY</th><th>TEST ACCURACY</th><th>MACRO F1</th><th>STATUS</th><th><span class="sr-only">Open run</span></th></tr></thead><tbody>${rows.map(r => `<tr><td class="run-name"><button class="run-link" data-run="${escapeHTML(r.id)}">${escapeHTML(r.name)}</button></td><td><span class="mini-category">${escapeHTML(title(r.category))}</span></td><td class="metric">${percent(r.accuracy)}</td><td>${percent(r.f1)}</td><td><span class="tag ${r.complete ? 'completed' : ''}">${r.complete ? 'Completed' : 'Incomplete'}</span></td><td><button class="run-link table-arrow" data-run="${escapeHTML(r.id)}" aria-label="View ${escapeHTML(r.name)}">↗</button></td></tr>`).join('')}</tbody></table>`;
}
function modelRecord(id) {
  return state.model_records?.[id] || {};
}
function recordValue(value) {
  return value === null || value === undefined ? '—' : escapeHTML(value);
}
function recordPercent(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
}
function renderPredictRuns() {
  populate($('predict-run'), state.runs.filter(r => r.checkpoint).map(r => [r.id,
    `${title(r.category)} / ${r.name}${modelRecord(r.id).decision === 'Preferred' ? ' · Preferred' : ''}`]), 'No trained checkpoints found');
}
function modelRecordsTable(rows) {
  if (!rows.length) return empty('No matching saved models. Change the filters or train a model first.');
  return `<table class="model-record-table"><thead><tr><th>Model</th><th>Decision</th><th>Data used</th><th>Training and preparation</th><th>Offline evaluation</th><th>Recorded live settings</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows.map(r => {
    const record = modelRecord(r.id), m = r.metadata || {}, t = m.training || {}, p = m.preparation || {}, counts = m.split_counts || {};
    const folders = Array.isArray(m.source_folders) ? m.source_folders.join(', ') : 'Not recorded';
    const mode = {motion: 'Moving gesture', gesture: 'Perform once', pose: 'Held pose'}[record.mode] || 'Not recorded';
    const liveAccuracy = record.live_trials && record.live_correct != null ? record.live_correct / record.live_trials : null;
    return `<tr><td><button class="run-link" data-run="${escapeHTML(r.id)}">${escapeHTML(r.name)}</button><small>${escapeHTML(title(r.category))}</small><details><summary>${fmt(r.classes)} classes</summary><span>${escapeHTML((r.class_labels || []).join(', ') || 'Class labels unavailable')}</span></details></td>
    <td><span class="tag ${record.decision === 'Preferred' ? 'ready' : ''}">${escapeHTML(record.decision || 'Not tested')}</span></td>
    <td><span class="record-dataset">${escapeHTML(m.dataset || 'Dataset not recorded')}</span><small>${escapeHTML(protocolDescription(m.evaluation_protocol))}</small><small>Folders: ${escapeHTML(folders)}${m.input_kind ? ` · ${escapeHTML(m.input_kind)}` : ''}</small><small>Originals: ${recordValue(m.original_sources)} · Selected: ${recordValue(m.selected_sources)}</small><small>Train / validation / test: ${recordValue(counts.train)} / ${recordValue(counts.validation)} / ${recordValue(counts.test)}</small></td>
    <td><span>Epochs recorded / max: ${recordValue(m.epochs_recorded)} / ${recordValue(t.epochs)}</span><small>Batch: ${recordValue(t.batch_size)} · LR: ${recordValue(t.learning_rate)}</small><small>Patience: ${recordValue(t.patience)} · Seed: ${recordValue(t.seed)}</small><small>Class weights: ${t.class_weights == null ? '—' : t.class_weights ? 'Yes' : 'No'}</small><small>Static variants (images): ${recordValue(p.static_variants)} · Jitter: ${recordValue(p.jitter_std)}</small><small>Video extras: ${recordValue(p.video_extra_samples)} · Warp: ${recordValue(p.video_time_warp)} · Opposite-hand copies: ${p.video_mirror_hands == null ? 'Not recorded' : p.video_mirror_hands ? 'Yes' : 'No'}</small><small>Detection confidence: ${recordValue(p.detection_confidence)}</small>${t.resume_from ? `<small>Resumed: ${escapeHTML(t.resume_from)}</small>` : ''}</td>
    <td><strong>${recordPercent(r.accuracy)}</strong><small>Test accuracy</small><small>Macro F1: ${recordPercent(r.f1)}</small></td>
    <td><span>${escapeHTML(mode)}</span><small>Stillness: ${recordValue(record.settle_ms)} ms</small><small>Hands missing: ${recordValue(record.missing_ms)} ms</small><small>Frame interval: ${recordValue(record.frame_interval_ms)} ms</small><small>Live accuracy: ${recordPercent(liveAccuracy)}</small>${record.live_trials ? `<small>${recordValue(record.live_correct)} / ${recordValue(record.live_trials)} correct</small>` : ''}<small>Response: ${recordValue(record.response_ms)} ms</small>${record.tested_on ? `<small>Tested: ${escapeHTML(record.tested_on)}</small>` : ''}</td>
    <td><div class="record-notes">${escapeHTML(record.notes || 'No notes yet.')}</div>${record.device ? `<small>Device / session: ${escapeHTML(record.device)}</small>` : ''}</td>
    <td><div class="record-actions"><button type="button" class="button secondary" data-record-run="${escapeHTML(r.id)}">Edit record</button><button type="button" class="button secondary" data-test-run="${escapeHTML(r.id)}" ${r.checkpoint ? '' : 'disabled'}>Try model</button><button type="button" class="button secondary" data-history-run="${escapeHTML(r.id)}">Test history (${Object.values(state.prediction_tests || {}).filter(t => t.run === r.id).length})</button><button type="button" class="button secondary" data-download-model="${escapeHTML(r.id)}">Download record</button><button type="button" class="button secondary" data-download-model-zip="${escapeHTML(r.id)}" ${(r.exports?.download ?? r.checkpoint) ? '' : 'disabled'}>Download model ZIP</button><button type="button" class="button secondary export-model-action" data-export-model="${escapeHTML(r.id)}" data-available="${Boolean(r.exports?.hudyat ?? r.checkpoint)}" title="${escapeHTML(r.exports?.reason || 'Create a browser model bundle for HUDYAT')}" ${(r.exports?.hudyat ?? r.checkpoint) ? '' : 'disabled'}>Export for HUDYAT</button><button type="button" class="button danger submit-job" data-delete-kind="model" data-delete-path="${escapeHTML(r.id)}">Delete model</button></div></td></tr>`;
  }).join('')}</tbody></table>`;
}
let editingRecord = '', recordSaving = false;
const recordFields = ['decision', 'mode', 'settle_ms', 'missing_ms', 'frame_interval_ms', 'live_trials', 'live_correct', 'response_ms', 'tested_on', 'device', 'notes'];
const recordNumbers = ['settle_ms', 'missing_ms', 'frame_interval_ms', 'live_trials', 'live_correct', 'response_ms'];
function editModelRecord(id) {
  if (recordSaving) return;
  const run = state.runs.find(r => r.id === id);
  if (!run) return;
  editingRecord = id;
  editingTest = null;
  editingPrediction = null;
  $('record-test-name-field').hidden = true;
  $('record-decision-field').hidden = false;
  $('model-record-save').textContent = 'Save record';
  $('model-record-description').textContent = 'Save your decision, tested slider values, and live observations.';
  $('model-record-hint').textContent = 'Choosing Preferred replaces your previous preferred run for this category. These records remain after restarting the dashboard.';
  const record = modelRecord(id);
  for (const key of ['live_trials', 'live_correct', 'response_ms']) $('record-' + key).readOnly = false;
  $('model-record-title').textContent = `Record: ${title(run.category)} / ${run.name}`;
  for (const name of recordFields) $('record-' + name).value = record[name] ?? (name === 'decision' ? 'Not tested' : '');
  $('model-record-error').hidden = true;
  $('model-record-editor').hidden = false;
  $('model-record-editor').scrollIntoView({behavior: 'smooth', block: 'start'});
  renderTestControls();
}
async function persistModelRecord(payload) {
  if (recordSaving) return false;
  recordSaving = true; controls();
  try {
    const result = await api('/api/model-record', payload);
    state.model_records = result.model_records;
    ++modelRecordsVersion;
    renderRuns(); renderPredictRuns();
    return true;
  } finally { recordSaving = false; controls(); }
}
async function saveModelRecord() {
  if (!editingRecord || recordSaving) return;
  const payload = {run: editingRecord};
  for (const name of recordFields) {
    const value = $('record-' + name).value;
    payload[name] = recordNumbers.includes(name) ? (value === '' ? null : Number(value)) : value;
  }
  $('model-record-error').hidden = true;
  try {
    if (editingTest !== null) {
      const name = $('record-test-name').value;
      if (editingTest || name !== suggestedTestName) payload.name = name;
      payload.prediction = editingPrediction;
      if (editingTest) payload.id = editingTest;
      recordSaving = true; controls();
      try {
        const result = await api('/api/prediction-test', payload);
        state.prediction_tests = result.prediction_tests;
        ++predictionTestsVersion;
        editingTest = result.test.id;
        $('record-test-name').value = result.test.name;
        $('model-record-title').textContent = `${result.test.name} · ${result.test.model.name}`;
        $('run-filter').value = result.test.model.category;
        $('run-preferred-only').checked = false;
        $('test-model-filter').value = result.test.run;
        renderRuns();
        notice(`${result.test.name} saved as a separate prediction test. Download it from Prediction test runs.`);
      } finally { recordSaving = false; controls(); }
    } else if (await persistModelRecord(payload)) notice('Model record saved. Preferred selections and notes will remain after restarting.');
  } catch (error) {
    $('model-record-error').textContent = error.message;
    $('model-record-error').hidden = false;
  }
}
function copyCameraSettings() {
  const settings = {...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value};
  for (const [name, value] of Object.entries(settings)) $('record-' + name).value = value;
}
async function recordCameraSettings() {
  const id = $('predict-run').value;
  if (!id || recordSaving) return;
  try {
    if (await persistModelRecord({run: id, ...modelRecord(id), ...liveTiming(),
      frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value})) notice('Current camera settings recorded for the selected model. Review them under Model runs.');
  } catch (error) { notice(error.message, true); }
}
async function useRecordedRun(id, settings = null) {
  if (automatedTest || automatedStarting) { notice('Finish the current automated test session before switching models.', true); return; }
  if (!state.runs.some(r => r.id === id && r.checkpoint)) return;
  if (liveActive) await stopLive();
  const record = settings || modelRecord(id);
  $('predict-run').value = id;
  if (record.mode) $('live-mode').value = record.mode;
  for (const [name, control] of [['settle_ms','live-settle-ms'], ['missing_ms','live-missing-ms'], ['frame_interval_ms','live-frame-ms']]) {
    if (record[name] != null) $(control).value = record[name];
  }
  renderLiveTiming(); controls();
  location.hash = 'predict';
}
function benchmarkTable(rows) {
  if (!rows.length) return empty('No matching models for the benchmark table.');
  return `<table class="benchmark-table"><thead><tr><th>Model / run</th><th>Signs</th><th>Accuracy (≥85%)</th><th>Macro precision (≥85%)</th><th>Macro recall (≥85%)</th><th>Macro F1 (≥85%)</th><th>Lowest sign recall (≥70%)</th><th>Response time (&lt;1,000 ms)</th><th>Met benchmark?</th><th>Actions</th></tr></thead><tbody>${rows.map(r => {
    const b = r.benchmark || {}, timing = b.timing || {};
    const missing = Object.entries(b.checks || {}).filter(([, value]) => value !== true)
      .map(([key, value]) => `${key.replaceAll('_', ' ')}: ${value === null ? 'missing' : 'failed'}`).join('; ');
    const timed = typeof b.response_time_ms === 'number' && Number.isFinite(b.response_time_ms);
    return `<tr><td><strong>${escapeHTML(title(r.category))}</strong><small>${escapeHTML(r.name)}</small></td>
      <td>${recordValue(b.signs ?? r.classes)}<small>Required: ${recordValue(b.required_signs)}</small></td>
      ${['accuracy', 'macro_precision', 'macro_recall', 'macro_f1'].map(key => `<td>${recordPercent(b[key])}</td>`).join('')}
      <td>${recordPercent(b.lowest_sign_recall)}<small>${escapeHTML((b.lowest_recall_signs || []).join(', '))}</small></td>
      <td>${timed ? b.response_time_ms.toFixed(2) + ' ms' : 'Not measured'}<small>${timed ? 'HUDYAT camera-to-result · median' : 'Run the HUDYAT camera test'}</small>${typeof timing.p95_ms === 'number' ? `<small>p95: ${timing.p95_ms.toFixed(2)} ms · ${fmt(timing.samples)} trials</small>` : ''}</td>
      <td><span class="tag ${b.status === 'Met' ? 'completed' : ''}">${escapeHTML(b.status || 'Incomplete')}</span><small>${escapeHTML(missing)}</small></td>
      <td><button class="button secondary benchmark-model-action" type="button" data-benchmark-run="${escapeHTML(r.id)}" data-available="${Boolean(r.checkpoint && r.complete)}" ${r.checkpoint && r.complete ? '' : 'disabled'}>Measure local inference</button><button class="button secondary benchmark-model-action" type="button" data-hudyat-timing-run="${escapeHTML(r.id)}" data-available="${Boolean(r.exports?.hudyat)}" ${r.exports?.hudyat ? '' : 'disabled'}>HUDYAT camera test</button>${typeof b.local_inference?.median_ms === 'number' ? `<small>Local inference only: ${b.local_inference.median_ms.toFixed(2)} ms</small>` : ''}</td></tr>`;
  }).join('')}</tbody></table>`;
}

function renderRuns() {
  const filter = $('run-filter').value;
  const preferredOnly = $('run-preferred-only').checked;
  const rows = state.runs.filter(r => (!filter || r.category === filter)
    && (!preferredOnly || modelRecord(r.id).decision === 'Preferred'));
  $('all-runs').innerHTML = modelRecordsTable(rows);
  document.querySelectorAll('#all-runs .record-actions').forEach((actions, index) => {
    actions.insertAdjacentHTML('beforeend', paperReportActions(rows[index].id, 'run'));
  });
  $('benchmark-table').innerHTML = benchmarkTable(rows);
  $('run-record-summary').textContent = `${rows.length} saved runs shown. Preferred models are selected manually, one per category.`;
  renderTestHistory();
}
function nextTestNumber(run) {
  return Math.max(0, ...Object.values(state.prediction_tests || {}).filter(t => t.run === run).map(t => t.number)) + 1;
}
function modelDescription(model) {
  const m = model.metadata || {}, t = m.training || {}, p = m.preparation || {}, counts = m.split_counts || {};
  return `<details><summary>Model details saved with this test</summary><p>Dataset: ${escapeHTML(m.dataset || 'Not recorded')}<br>Source folders: ${escapeHTML((m.source_folders || []).join(', ') || 'Not recorded')}<br>Input: ${recordValue(m.input_kind)} · Originals: ${recordValue(m.original_sources)} · Selected: ${recordValue(m.selected_sources)}<br>Train / validation / test: ${recordValue(counts.train)} / ${recordValue(counts.validation)} / ${recordValue(counts.test)}<br>Epochs recorded / max: ${recordValue(m.epochs_recorded)} / ${recordValue(t.epochs)}<br>Batch: ${recordValue(t.batch_size)} · LR: ${recordValue(t.learning_rate)}<br>Patience: ${recordValue(t.patience)} · Seed: ${recordValue(t.seed)} · Class weights: ${t.class_weights == null ? '—' : t.class_weights ? 'Yes' : 'No'}<br>Static variants: ${recordValue(p.static_variants)} · Jitter: ${recordValue(p.jitter_std)}<br>Video extras: ${recordValue(p.video_extra_samples)} · Warp: ${recordValue(p.video_time_warp)} · Opposite-hand copies: ${p.video_mirror_hands == null ? 'Not recorded' : p.video_mirror_hands ? 'Yes' : 'No'}<br>Detection confidence: ${recordValue(p.detection_confidence)}${t.resume_from ? `<br>Resumed: ${escapeHTML(t.resume_from)}` : ''}<br>Offline test accuracy: ${recordPercent(model.accuracy)} · Macro F1: ${recordPercent(model.f1)}<br>Classes: ${escapeHTML((model.class_labels || []).join(', '))}${model.record?.notes ? `<br>Model notes: ${escapeHTML(model.record.notes)}` : ''}</p></details>`;
}
function renderTestHistory() {
  const all = Object.values(state.prediction_tests || {});
  const models = new Map([...state.runs.map(r => [r.id, r]), ...all.map(t => [t.run, t.model])]);
  populate($('test-model-filter'), [['', 'All models'], ...[...models].map(([id, m]) => [id, `${title(m.category)} / ${m.name}`])], 'All models');
  const filter = $('run-filter').value, model = $('test-model-filter').value;
  const rows = all.filter(t => (!model || t.run === model) && (!filter || t.model.category === filter)
    && (!$('run-preferred-only').checked || modelRecord(t.run).decision === 'Preferred'))
    .sort((a, b) => b.created_at.localeCompare(a.created_at) || b.number - a.number);
  $('prediction-test-summary').textContent = `${rows.length} saved prediction tests shown. Each test keeps its own model description and observations.`;
  $('prediction-tests').innerHTML = rows.length ? `<table class="prediction-test-table"><thead><tr><th>Test / model</th><th>Tested settings</th><th>Live results</th><th>Saved prediction</th><th>Notes</th><th>Actions</th></tr></thead><tbody>${rows.map(test => {
    const r = test.record, prediction = test.prediction;
    const ranked = Object.entries(prediction?.probabilities || {}).sort((a, b) => b[1] - a[1]);
    return `<tr><td><strong>${escapeHTML(test.name)}</strong><small>${escapeHTML(title(test.model.category))} / ${escapeHTML(test.model.name)}</small><small>Saved: ${escapeHTML(test.created_at.replace('T', ' '))}</small>${modelDescription(test.model)}</td><td>${escapeHTML({motion:'Moving gesture', gesture:'Perform once', pose:'Held pose'}[r.mode] || 'Not recorded')}<small>Stillness: ${recordValue(r.settle_ms)} ms</small><small>Hands missing: ${recordValue(r.missing_ms)} ms</small><small>Frame interval: ${recordValue(r.frame_interval_ms)} ms</small></td><td>${recordPercent(r.live_trials && r.live_correct != null ? r.live_correct / r.live_trials : null)}<small>${recordValue(r.live_correct)} / ${recordValue(r.live_trials)} correct</small><small>Median ${test.automated ? 'camera response' : 'recognition'}: ${recordValue(r.response_ms)} ms</small>${test.automated ? automatedMeasurements(test, false) : ''}<small>Tested: ${recordValue(r.tested_on)}</small></td><td>${ranked.length ? `${escapeHTML(ranked[0][0])} · ${percent(ranked[0][1])}<small>${prediction.source === 'live' ? 'Live camera' : 'Sample file'} · ${escapeHTML(prediction.captured_at.replace('T', ' '))}</small>${prediction.sample ? `<small>${escapeHTML(prediction.sample)}</small>` : ''}` : 'No prediction attached'}</td><td><div class="record-notes">${escapeHTML(r.notes || 'No notes yet.')}</div><small>${escapeHTML(r.device || '')}</small></td><td><div class="record-actions"><button class="button secondary" type="button" data-edit-test="${escapeHTML(test.id)}" ${test.id === automatedTest ? 'disabled' : ''}>Edit test</button><button class="button secondary" type="button" data-download-test="${escapeHTML(test.id)}">Download test</button>${test.automated ? automatedDownloads(test.id) : ''}<button class="button secondary" type="button" data-retry-test="${escapeHTML(test.id)}" ${state.runs.some(m => m.id === test.run && m.checkpoint) ? '' : 'disabled'}>Try these settings</button></div></td></tr>`;
  }).join('')}</tbody></table>` : empty('No saved prediction tests match these filters. Try a model, then choose Save as test run.');
  renderTestControls();
}
function renderTestControls() {
  const run = $('predict-run').value;
  $('live-save-test').disabled = recordSaving || !token || !run;
  $('sample-save-test').disabled = recordSaving || !token || latestPredictions.sample?.run !== run;
  $('prediction-test-next').textContent = run ? `Next saved run: Test ${nextTestNumber(run)}. Add trial counts, recognition time, and notes when saving.` : '';
  $('record-test-name').required = editingTest !== null;
  $('model-record-close').disabled = recordSaving;
  $('model-record-copy').disabled = recordSaving;
  renderAutomatedControls();
}
function openTestEditor(id = '', source = 'live') {
  if (recordSaving) return;
  const saved = state.prediction_tests?.[id];
  const run = saved?.run || $('predict-run').value;
  const model = saved?.model || state.runs.find(r => r.id === run);
  if (!model) return;
  editingRecord = run;
  editingTest = saved?.id || '';
  const latest = latestPredictions[source];
  const attached = latest?.run === run ? latest : null;
  editingPrediction = saved ? saved.prediction : attached?.prediction || null;
  const record = saved ? saved.record : {...modelRecord(run), decision: 'Testing',
    live_trials: null, live_correct: null, response_ms: null,
    tested_on: new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10),
    ...(source === 'live' ? (attached?.settings || {...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value})
      : {settle_ms: null, missing_ms: null, frame_interval_ms: null, mode: null})};
  for (const name of recordFields) $('record-' + name).value = record[name] ?? (name === 'decision' ? 'Testing' : '');
  $('record-test-name').value = saved?.name || `Test ${nextTestNumber(run)}`;
  suggestedTestName = $('record-test-name').value;
  $('record-test-name-field').hidden = false;
  $('record-decision-field').hidden = true;
  $('model-record-title').textContent = `${$('record-test-name').value} · ${title(model.category)} / ${model.name}`;
  $('model-record-description').textContent = 'Save a separate test run with this model’s description, settings, and observations.';
  $('model-record-hint').textContent = 'Trial counts and recognition time are manual measurements. Leave unmeasured fields blank. Saving again updates this test; Save as test run starts a new numbered test.';
  for (const key of ['live_trials', 'live_correct', 'response_ms']) $('record-' + key).readOnly = Boolean(saved?.automated);
  if (saved?.automated) $('model-record-hint').textContent = 'Counts and response time are calculated from the recorded attempts. Edit the name, notes, or device description here. Download metrics and trials from Prediction test runs.';
  $('model-record-save').textContent = 'Save test run';
  $('model-record-error').hidden = true;
  $('model-record-editor').hidden = false;
  location.hash = 'runs';
  route();
  renderTestControls();
  $('model-record-editor').scrollIntoView({behavior: 'smooth', block: 'start'});
}
async function downloadRecord(path) {
  try {
    const response = await fetch(path, {cache: 'no-store'});
    if (!response.ok) throw new Error((await response.json()).error || 'Could not download the record.');
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] || 'model-record.json';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { notice(error.message, true); }
}
function automatedDownloads(id) {
  return ['metrics', 'trials'].map(format => `<button class="button secondary" type="button" data-test-csv="${escapeHTML(id)}" data-test-format="${format}">Download ${format} CSV</button>`).join('');
}
function automatedMeasurements(test, detailed = true) {
  const m = test.measurements;
  if (!m) return '';
  const timing = m.timing;
  const milliseconds = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';
  const summary = `<p>Accuracy: ${recordPercent(m.accuracy)} · Macro precision: ${recordPercent(m.macro_precision)} · Macro recall: ${recordPercent(m.macro_recall)} · Macro F1: ${recordPercent(m.macro_f1)} · Lowest gesture recall: ${recordPercent(m.lowest_sign_recall)}</p>
    <p>${fmt(m.correct)} / ${fmt(m.trials)} correct · ${fmt(m.outcomes.timeout)} timeouts · ${fmt(m.outcomes.incomplete)} incomplete · ${fmt(m.outcomes.stopped)} stopped · ${fmt(m.outcomes.error)} errors</p>
    <p>Camera response median: ${milliseconds(timing.median_ms)} ms · p95: ${milliseconds(timing.p95_ms)} ms · ${fmt(timing.samples)} timed predictions</p>
    <p class="hint">${m.coverage_complete ? 'All model gestures attempted.' : `Remaining gestures: ${escapeHTML(m.missing_signs.join(', '))}. Macro metrics require an attempt for every gesture.`}</p>`;
  if (!detailed) return `<details><summary>Automatic live metrics</summary>${summary}</details>`;
  return `<h3>${escapeHTML(test.name)}</h3>${summary}<div class="table-wrap" tabindex="0" aria-label="Per-gesture live test metrics"><table><thead><tr><th>Gesture</th><th>Attempts</th><th>Correct</th><th>Precision</th><th>Recall</th><th>F1</th><th>Unrecognized</th></tr></thead><tbody>${Object.entries(m.per_class).map(([label, row]) => `<tr><td>${escapeHTML(label)}</td><td>${fmt(row.support)}</td><td>${fmt(row.correct)}</td><td>${recordPercent(row.precision)}</td><td>${row.support ? recordPercent(row.recall) : '—'}</td><td>${row.support ? recordPercent(row.f1) : '—'}</td><td>${fmt(row.unrecognized)}</td></tr>`).join('')}</tbody></table></div><div class="job-actions" style="justify-content:flex-start;flex-wrap:wrap">${automatedDownloads(test.id)}<button class="button secondary" type="button" data-download-test="${escapeHTML(test.id)}">Download full JSON</button></div>`;
}
function renderAutomatedControls() {
  const test = state.prediction_tests?.[automatedTest || automatedViewedTest];
  const model = state.runs.find(r => r.id === $('predict-run').value);
  const labels = automatedTest && test ? test.model.class_labels : model?.class_labels || [];
  const choices = JSON.stringify(labels);
  if ($('auto-test-expected').dataset.choices !== choices) {
    populate($('auto-test-expected'), labels.map(label => [label, label]), 'Select a trained model');
    $('auto-test-expected').dataset.choices = choices;
  }
  const busy = automatedStarting || automatedSaving || Boolean(automatedTrial) || Boolean(automatedPending);
  $('auto-test-start').disabled = Boolean(automatedTest) || busy || !liveSession || !token || !labels.length || recordSaving;
  $('auto-test-next').disabled = !automatedTest || busy || !liveSession || !token;
  $('auto-test-finish').disabled = !automatedTest || automatedStarting || automatedSaving || Boolean(automatedPending);
  $('auto-test-retry').hidden = !automatedPending;
  $('auto-test-retry').disabled = automatedSaving;
  $('auto-test-name').disabled = Boolean(automatedTest) || automatedStarting;
  $('auto-test-device').disabled = Boolean(automatedTest) || automatedStarting;
  $('auto-test-expected').disabled = busy;
  for (const id of ['live-settle-ms', 'live-missing-ms', 'live-frame-ms']) $(id).disabled = Boolean(automatedTrial) || automatedStarting;
  const results = test ? automatedMeasurements(test) : '';
  if ($('auto-test-results').innerHTML !== results) $('auto-test-results').innerHTML = results;
}
function acceptAutomatedTest(result) {
  state.prediction_tests = {...state.prediction_tests, [result.test.id]: result.test};
  ++predictionTestsVersion;
  renderTestHistory();
}
async function startAutomatedTest() {
  if (automatedTest || automatedStarting || automatedPending || !liveSession || recordSaving) return;
  const run = $('predict-run').value, version = liveVersion;
  if (!state.runs.find(r => r.id === run)?.class_labels?.includes($('auto-test-expected').value)) return;
  automatedStarting = true; controls();
  $('auto-test-status').textContent = 'Creating a saved test session…';
  try {
    const result = await api('/api/prediction-test', {run, automated: true,
      ...($('auto-test-name').value.trim() ? {name: $('auto-test-name').value.trim()} : {}),
      ...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value,
      device: $('auto-test-device').value,
      device_details: {browser: navigator.userAgent || '', platform: navigator.platform || '',
        camera: liveStream?.getVideoTracks?.()[0]?.getSettings?.() || {}}});
    automatedTest = result.test.id; automatedViewedTest = automatedTest;
    acceptAutomatedTest(result);
    automatedStarting = false;
    if (version === liveVersion && liveSession) await beginAutomatedTrial();
    else $('auto-test-status').textContent = 'Session saved. Start the camera and select Next trial to record an attempt.';
  } catch (error) { $('auto-test-status').textContent = error.message; notice(error.message, true); }
  finally { automatedStarting = false; controls(); }
}
async function beginAutomatedTrial() {
  if (!automatedTest || automatedStarting || automatedTrial || automatedPending || automatedSaving || !liveSession) return;
  const expected = $('auto-test-expected').value;
  if (!state.prediction_tests?.[automatedTest]?.model.class_labels.includes(expected)) return;
  const version = liveVersion;
  automatedStarting = true; controls();
  try {
    if ($('live-mode').value !== 'pose') await nextGesture();
    if (version !== liveVersion || !liveSession) return;
    automatedTrial = {id: crypto.randomUUID().replaceAll('-', ''), expected,
      captured_at: new Date().toISOString(), armedAt: performance.now(), startedAt: null, handStartedAt: null,
      settings: {...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value}};
    $('auto-test-status').textContent = `Recording ${expected}. Perform the gesture once; the result will save automatically.`;
  } catch (error) { $('auto-test-status').textContent = error.message; notice(error.message, true); }
  finally { automatedStarting = false; controls(); }
}
function startAutomatedFrame(trial, startedAt) {
  if (!trial || trial !== automatedTrial || trial.startedAt !== null) return;
  trial.startedAt = startedAt;
  automatedTimer = setTimeout(() => timeoutAutomatedTrial(trial), 15000);
}
function timeoutAutomatedTrial(trial) {
  if (trial !== automatedTrial) return;
  const remaining = 15000 - (performance.now() - trial.startedAt);
  if (remaining > 0) automatedTimer = setTimeout(() => timeoutAutomatedTrial(trial), Math.ceil(remaining));
  else finishAutomatedTrial('timeout');
}
async function observeAutomatedFrame(trial, result, frameStartedAt, receivedAt = performance.now()) {
  if (!trial || trial !== automatedTrial) return;
  if (result.hand_detected && trial.handStartedAt === null) trial.handStartedAt = frameStartedAt;
  if (result.status === 'prediction') {
    trial.handStartedAt ??= frameStartedAt;
    await finishAutomatedTrial('prediction', result, receivedAt);
  } else if (result.status === 'incomplete') await finishAutomatedTrial('incomplete', {}, receivedAt);
}
async function finishAutomatedTrial(outcome, result = {}, now = performance.now()) {
  const trial = automatedTrial;
  if (!trial) return;
  automatedTrial = null;
  clearTimeout(automatedTimer); automatedTimer = null;
  automatedPending = {id: automatedTest, trial: {id: trial.id, expected: trial.expected,
    outcome, captured_at: trial.captured_at, settings: trial.settings,
    elapsed_ms: now - (trial.startedAt ?? trial.armedAt),
    response_ms: outcome === 'prediction' ? now - trial.handStartedAt : null,
    probabilities: outcome === 'prediction' ? result.probabilities : {},
    frames: result.frames ?? null, duration_ms: result.duration_ms ?? null}};
  await saveAutomatedTrial();
}
async function saveAutomatedTrial() {
  if (!automatedPending || automatedSaving) return;
  automatedSaving = true; controls();
  $('auto-test-status').textContent = 'Saving the recorded attempt…';
  try {
    const result = await api('/api/prediction-test/trial', automatedPending, {keepalive: true});
    automatedPending = null;
    acceptAutomatedTest(result);
    const trial = result.test.trials.at(-1);
    $('auto-test-status').textContent = `Attempt saved: expected ${trial.expected}; ${trial.predicted ? `recognized ${trial.predicted}` : trial.outcome}. Select the expected gesture and Next trial when ready.`;
  } catch (error) {
    $('auto-test-status').textContent = `Attempt has not been saved: ${error.message} Keep this page open and select Retry saving attempt.`;
    notice('The recorded attempt is waiting to be saved. Retry before closing this page.', true);
  } finally { automatedSaving = false; controls(); }
}
async function finishAutomatedTest() {
  if (!automatedTest || automatedStarting || automatedSaving || automatedPending) return;
  await finishAutomatedTrial('stopped');
  if (automatedPending) return;
  automatedTest = '';
  $('auto-test-status').textContent = 'Session finished. Every recorded attempt is saved. Download the metrics and trial records below or under Model runs.';
  renderTestHistory(); controls();
}
function openModelExport(id) {
  const run = state.runs.find(r => r.id === id);
  if (!run || !(run.exports?.hudyat ?? run.checkpoint)) return;
  exportingRun = id;
  $('export-name').value = `${run.category}-${run.name}`.replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 100);
  $('model-export-title').textContent = `Export for HUDYAT: ${title(run.category)} / ${run.name}`;
  $('model-export-error').hidden = true;
  $('model-export-editor').hidden = false;
  controls();
  $('model-export-editor').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function renderDatasets() {
  const search = $('category-search').value.toLowerCase();
  const rows = state.categories.filter(c => c.name.toLowerCase().includes(search));
  $('dataset-list').innerHTML = rows.length ? rows.map(c => `<article class="dataset-item"><div class="dataset-item-heading"><div><span class="category-name">${escapeHTML(title(c.name))}</span><div class="category-meta">${fmt(c.samples)} sources · ${fmt(c.classes.length)} gesture folders · ${fmt(c.datasets?.length)} prepared datasets</div></div><span class="tag ${c.prepared ? 'ready' : ''}">${c.prepared ? 'Prepared' : 'Not prepared'}</span></div><div class="chips">${c.classes.map(g => `<span class="chip">${escapeHTML(g.name)} <small>${fmt(g.samples)}</small></span>`).join('') || '<span class="hint">No gesture folders yet.</span>'}</div>${(c.datasets || []).map(d => `<div class="dataset-version"><div><strong>${escapeHTML(d.name)}</strong><p class="hint">${escapeHTML(d.path)}${d.summary.total_output_sequences ? ` · ${fmt(d.summary.total_output_sequences)} prepared sequences` : ''}</p></div><button class="button secondary submit-job" type="button" data-inspect="${escapeHTML(c.name)}" data-dataset="${escapeHTML(d.path)}">Validate dataset ↗</button><button class="button danger submit-job" type="button" data-delete-kind="dataset" data-delete-path="${escapeHTML(d.path)}">Delete dataset</button></div>`).join('')}<div class="dataset-actions"><button data-prepare="${escapeHTML(c.name)}">Prepare another dataset ↗</button></div></article>`).join('') : empty('No matching categories. Check your source and dataset folders.');
  controls();
  $('dataset-list').querySelectorAll('[data-inspect]').forEach(button => {
    button.insertAdjacentHTML('afterend', paperReportActions(button.dataset.dataset, 'dataset'));
  });
}
function paperReportActions(path, kind) {
  const url = path.split('/').map(encodeURIComponent).join('/');
  return `<button class="button secondary submit-job" type="button" data-paper-report-${kind}="${escapeHTML(path)}">Build tables &amp; charts</button> <a class="button secondary" href="/paper-report/${url}/paper_report/index.html" target="_blank" rel="noopener">Open tables &amp; charts ↗</a>`;
}
async function scan() {
  const version = ++scanVersion;
  const recordsVersion = modelRecordsVersion;
  const testsVersion = predictionTestsVersion;
  loading = true; controls();
  try {
    const next = await api(`/api/state?data=${encodeURIComponent($('data-root').value)}&datasets=${encodeURIComponent($('dataset-root').value)}`);
    if (version !== scanVersion) return;
    if (recordsVersion !== modelRecordsVersion) next.model_records = state.model_records;
    if (testsVersion !== predictionTestsVersion) next.prediction_tests = state.prediction_tests;
    state = next; token = next.token; render();
  } catch (error) { notice(error.message, true); }
  finally { if (version === scanVersion) { loading = false; controls(); } }
}
function controls() {
  const active = jobs.some(j => j.status === 'running');
  document.querySelectorAll('.submit-job').forEach(b => { b.disabled = loading || submitting || active || liveActive || !token; });
  document.querySelectorAll('.export-model-action, .benchmark-model-action').forEach(b => {
    b.disabled = loading || submitting || active || liveActive || !token || b.dataset.available !== 'true';
  });
  $('model-export-start').disabled ||= !state.runs.some(r => r.id === exportingRun && (r.exports?.hudyat ?? r.checkpoint));
  $('train-form').querySelector('[type="submit"]').disabled ||= !preparedChoices().some(d => d.path === $('train-dataset').value);
  datasetPaths();
  const loso = $('split-strategy').value === 'loso';
  $('signer-holdouts').hidden = !loso;
  $('test-signer').disabled = !loso;
  $('validation-signer').disabled = !loso;
  $('test-signer').required = loso;
  for (const key of ['validation', 'test']) {
    const input = $('prepare-form').elements[key];
    if (input) input.disabled = loso;
  }
  $('live-start').disabled = loading || submitting || active || liveActive || !token || !$('predict-run').value;
  $('live-stop').disabled = !liveActive;
  $('predict-run').disabled = liveActive || Boolean(automatedTest) || automatedStarting;
  $('live-mode').disabled = liveActive;
  $('live-next').hidden = !['gesture', 'motion'].includes($('live-mode').value);
  $('live-next').disabled = !liveActive || !liveSession || liveResetting || Boolean(automatedTrial) || automatedStarting;
  $('model-record-save').disabled = recordSaving || !token;
  $('live-record').disabled = recordSaving || !token || !$('predict-run').value;
  renderTestControls();
  const category = state.categories.find(c => c.name === $('prepare-form').elements.category.value);
  const selection = sourceSelection();
  const filtered = selection.source_folders !== null || selection.videos_only;
  const classes = selectedClasses(category);
  const included = classes.filter(c => c.samples > 0), omitted = classes.filter(c => !c.samples);
  $('source-folder-options').hidden = selection.source_folders === null;
  $('source-selection-preview').textContent = category
    ? `${fmt(included.reduce((sum, c) => sum + c.samples, 0))} matching ${selection.videos_only ? 'videos' : 'source files'}. ${included.map(c => `${c.name}: ${fmt(c.samples)}`).join(' · ')}${filtered && omitted.length ? `. Omitted gestures: ${omitted.map(c => c.name).join(', ')}.` : ''}`
    : 'Select a category to preview source counts.';
  const issue = !category ? 'Select a category with source samples to prepare a dataset.'
    : filtered ? (selection.source_folders?.length === 0 ? 'Choose at least one source folder.'
      : included.length < 2 ? 'Selected sources need at least two populated gesture classes.'
      : included.some(c => c.samples < 3) ? 'Each included gesture needs at least three independent source files.' : '')
    : category.preparation_issue || '';
  $('prepare-readiness').textContent = loading ? 'Checking project folders…' : active
    ? 'Another job is running. Wait for it to finish or stop it in Activity.'
    : issue ? `${issue} If you have added files since the last scan, click Prepare dataset to check again.`
    : 'Source counts meet the basic requirements. Source folders are checked again when you click Prepare dataset.';
  $('refresh').disabled = loading;
  $('scan-folders').disabled = loading || active;
  $('training-dot').hidden = !active;
}
async function deleteSaved(kind, path) {
  if (loading || submitting || liveActive || jobs.some(j => j.status === 'running')) return;
  const known = kind === 'model' ? state.runs.some(r => r.id === path)
    : state.categories.some(c => c.datasets?.some(d => d.path === path));
  if (!known || !window.confirm(`Move ${path} to local trash?\n\nSource recordings and other saved datasets/models stay available. Files can be recovered from .trash in the project folder.`)) return;
  submitting = true; controls();
  try {
    const result = await api('/api/delete', {kind, path, data: $('data-root').value, datasets: $('dataset-root').value});
    if (kind === 'model') {
      if (selectedRun === path) { selectedRun = ''; ++runVersion; $('run-detail').hidden = true; }
      if (editingRecord === path) { editingRecord = ''; $('model-record-editor').hidden = true; }
      if (exportingRun === path) { exportingRun = ''; $('model-export-editor').hidden = true; }
    }
    await scan();
    notice(`Deleted from workspace. Recovery folder: ${result.trash}`);
  } catch (error) { notice(error.message, true); }
  finally { submitting = false; controls(); }
}
async function startJob(data) {
  if (submitting) return;
  submitting = true; controls(); notice('');
  if (data.action === 'prepare') $('prepare-error').hidden = true;
  if (data.action === 'export') $('model-export-error').hidden = true;
  try {
    await api('/api/jobs', {data: $('data-root').value, datasets: $('dataset-root').value, ...data,
      ...(data.action === 'prepare' ? sourceSelection() : {})});
    $('job-select').value = '';
    await pollJobs();
    $('jobs-panel').scrollIntoView({behavior: 'smooth', block: 'nearest'});
  } catch (error) {
    notice(error.message, true);
    if (data.action === 'prepare') {
      await scan();
      $('prepare-error').textContent = error.message;
      $('prepare-error').hidden = false;
    }
    if (data.action === 'export') {
      $('model-export-error').textContent = error.message;
      $('model-export-error').hidden = false;
    }
  }
  finally { submitting = false; controls(); }
}
function renderJob() {
  const job = jobs.find(j => j.id === $('job-select').value) || jobs[0];
  if (!job) return;
  const log = $('job-log'), nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 65;
  $('job-state').textContent = title(job.status);
  $('job-state').className = `tag ${job.status}`;
  $('job-description').textContent = `${title(job.action)} · ${job.category}${job.dataset ? ` · ${job.dataset}` : ''} · Started ${job.started.replace('T', ' ')}${job.exit_code !== null ? ` · Exit ${job.exit_code}` : ''}`;
  const content = `${job.command}\n\n${job.logs.join('\n') || 'Starting worker…'}`;
  if (log.textContent !== content) { log.textContent = content; if (nearBottom) log.scrollTop = log.scrollHeight; }
  $('cancel-job').hidden = job.status !== 'running';
  $('cancel-job').dataset.id = job.id;
  $('download-export').hidden = job.action !== 'export' || job.status !== 'completed';
  $('download-export').dataset.id = job.id;
  $('open-hudyat-timing').hidden = job.action !== 'hudyat-timing' || job.status !== 'completed';
  $('open-hudyat-timing').href = `/hudyat-timing/${encodeURIComponent(job.id)}/`;
  const prediction = jobs.find(j => j.action === 'predict' && j.status === 'completed');
  if (prediction) {
    const output = prediction.logs.join('\n');
    try {
      const start = output.startsWith('{') ? 0 : output.lastIndexOf('\n{') + 1;
      const result = JSON.parse(output.slice(start));
      const ranked = Object.entries(result).filter(([, v]) => typeof v === 'number').sort((a, b) => b[1] - a[1]);
      if (ranked.length) {
        const [label, confidence] = ranked[0];
        latestPredictions.sample = {run: prediction.run, prediction: {source: 'sample',
          sample: prediction.sample || '', captured_at: prediction.started, probabilities: Object.fromEntries(ranked)}};
        $('prediction-result').className = '';
        $('prediction-result').innerHTML = `<div class="prediction-main">${escapeHTML(label)}</div><div class="prediction-confidence">${percent(confidence)} model confidence · ${escapeHTML(prediction.started.replace('T', ' '))}</div>${ranked.slice(0, 5).map(([name, score]) => `<div class="prediction-row"><span>${escapeHTML(name)}</span><meter min="0" max="1" value="${score}" aria-label="${escapeHTML(name)} confidence"></meter><span>${percent(score)}</span></div>`).join('')}<p class="hint">Model scores can be wrong. Verify the sign before relying on a prediction.</p>`;
      }
    } catch { /* Non-JSON worker output is still available in the activity log. */ }
  }
}
function showLiveResult(result, settings = null) {
  $('live-status').textContent = result.message || 'Live prediction running.';
  if (result.status !== 'prediction') {
    if (!result.keep_result) $('live-result').replaceChildren();
    return;
  }
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  if (!ranked.length) return;
  const [label, confidence] = ranked[0];
  latestPredictions.live = {run: $('predict-run').value,
    settings: settings || {...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value},
    prediction: {source: 'live', sample: '', captured_at: new Date().toISOString(), probabilities: result.probabilities}};
  renderTestControls();
  $('live-result').innerHTML = `<div class="prediction-main">${escapeHTML(label)}</div><div class="prediction-confidence">${percent(confidence)} model confidence${confidence < .6 ? ' · Uncertain' : ''}</div>${ranked.slice(0, 5).map(([name, score]) => `<div class="prediction-row"><span>${escapeHTML(name)}</span><meter min="0" max="1" value="${score}"></meter><span>${percent(score)}</span></div>`).join('')}`;
}
function liveTiming() {
  return {settle_ms: Number($('live-settle-ms').value || 600),
    missing_ms: Number($('live-missing-ms').value || 400)};
}
function liveFrameInterval() {
  return Number($('live-frame-ms').value || 50);
}
function renderLiveTiming() {
  const timing = liveTiming();
  $('live-settle-value').textContent = `${timing.settle_ms} ms`;
  $('live-missing-value').textContent = `${timing.missing_ms} ms`;
  $('live-frame-value').textContent = `${liveFrameInterval()} ms`;
}
async function liveFrame(version) {
  if (version !== liveVersion || !liveSession) return;
  if (liveResetting) {
    liveTimer = setTimeout(() => liveFrame(version), 200);
    return;
  }
  const captureVersion = liveCaptureVersion;
  const frameStartedAt = performance.now();
  try {
    const video = $('live-video'), canvas = $('live-canvas');
    if (!video.videoWidth || video.readyState < 2) {
      liveTimer = setTimeout(() => liveFrame(version), 200);
      return;
    }
    const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const trial = automatedTrial;
    startAutomatedFrame(trial, frameStartedAt);
    const settings = {...liveTiming(), frame_interval_ms: liveFrameInterval(), mode: $('live-mode').value};
    const result = await api('/api/live/frame', {session: liveSession,
      image: canvas.toDataURL('image/jpeg', .8), timestamp: performance.now(),
      timing: {settle_ms: settings.settle_ms, missing_ms: settings.missing_ms}});
    const receivedAt = performance.now();
    if (version !== liveVersion) return;
    if (captureVersion === liveCaptureVersion) {
      showLiveResult(result, settings);
      await observeAutomatedFrame(trial, result, frameStartedAt, receivedAt);
    }
    if (version !== liveVersion) return;
    // Keep requests serial and count processing time toward the frame interval.
    const delay = Math.max(0, liveFrameInterval() - (performance.now() - frameStartedAt));
    liveTimer = setTimeout(() => liveFrame(version), delay);
  } catch (error) {
    if (version !== liveVersion) return;
    await finishAutomatedTrial('error');
    await stopLive();
    $('live-status').textContent = error.message;
  }
}
async function startLive() {
  if (liveActive) return;
  const version = ++liveVersion;
  latestPredictions.live = null;
  liveActive = true; controls();
  $('live-result').replaceChildren();
  $('live-status').textContent = 'Requesting camera access…';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS and a supported browser.');
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
    if (version !== liveVersion) { stream.getTracks().forEach(track => track.stop()); return; }
    liveStream = stream;
    $('live-video').srcObject = stream;
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => { if (version === liveVersion) stopLive(); }));
    await $('live-video').play();
    if (version !== liveVersion) return;
    $('live-status').textContent = 'Loading model and hand detector. The first start can take a moment…';
    const ready = await api('/api/live/start', {run: $('predict-run').value, mode: $('live-mode').value,
      hand_model: $('predict-form').elements.hand_model.value});
    if (version !== liveVersion) {
      await api('/api/live/stop', {session: ready.session});
      return;
    }
    liveSession = ready.session;
    controls();
    $('live-status').textContent = 'Camera ready. Show your sign.';
    await liveFrame(version);
  } catch (error) {
    if (version !== liveVersion) return;
    await stopLive();
    $('live-status').textContent = error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access, then try again.' : error.message;
  }
}
async function stopLive() {
  ++liveVersion;
  ++liveCaptureVersion;
  liveResetting = false;
  clearTimeout(liveTimer);
  const session = liveSession;
  liveSession = ''; liveActive = false;
  if (liveStream) liveStream.getTracks().forEach(track => track.stop());
  liveStream = null;
  $('live-video').srcObject = null;
  $('live-result').replaceChildren();
  $('live-status').textContent = 'Camera stopped.';
  controls();
  await finishAutomatedTrial('stopped');
  if (session) {
    try { await api('/api/live/stop', {session}); }
    catch (error) { $('live-status').textContent = `Camera stopped. ${error.message} The idle worker will expire automatically.`; }
  }
}
async function nextGesture() {
  if (!liveSession || liveResetting) return;
  const version = liveVersion;
  latestPredictions.live = null;
  ++liveCaptureVersion;
  liveResetting = true; controls();
  $('live-result').replaceChildren();
  $('live-status').textContent = 'Preparing for the next gesture…';
  try {
    const result = await api('/api/live/reset', {session: liveSession});
    if (version === liveVersion) showLiveResult(result);
  } catch (error) {
    if (version === liveVersion) {
      await stopLive();
      $('live-status').textContent = error.message;
    }
  } finally {
    if (version === liveVersion) { liveResetting = false; controls(); }
  }
}
async function pollJobs() {
  if (polling) return;
  polling = true;
  try {
    jobs = await api('/api/jobs');
    populate($('job-select'), jobs.map(j => [j.id, `${title(j.action)} · ${j.category} · ${j.status}`]), 'No jobs');
    $('job-select').hidden = !jobs.length;
    renderJob(); controls();
    const status = jobs[0] ? `${jobs[0].id}:${jobs[0].status}` : '';
    if (status !== previousJobStatus && jobs[0] && ['completed', 'failed', 'cancelled'].includes(jobs[0].status)) {
      if (jobs[0].status === 'failed') notice('The job failed. Check the activity log for the worker’s error and dependency details.', true);
      if (jobs[0].status === 'completed') notice(jobs[0].action === 'export'
        ? 'HUDYAT bundle ready. Use Download HUDYAT bundle in Activity, then follow its import instructions.'
        : `${title(jobs[0].action)} completed. Project data has been refreshed.`);
      await scan();
    }
    previousJobStatus = status;
  } catch (error) { notice(`Cannot reach the dashboard server. ${error.message}`, true); }
  finally { polling = false; }
}
function chart(history) {
  const train = history.accuracy || [], validation = history.val_accuracy || [];
  if (!train.length && !validation.length) return empty('No completed learning history was saved for this run.');
  const width = 760, height = 215, left = 42, top = 15, right = 18, bottom = 31;
  const count = Math.max(train.length, validation.length), plotWidth = width - left - right, plotHeight = height - top - bottom;
  const x = i => left + i / Math.max(count - 1, 1) * plotWidth;
  const y = v => top + (1 - Math.min(1, Math.max(0, v))) * plotHeight;
  const lines = [0, .25, .5, .75, 1].map(v => `<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="#edf0ee"/><text x="${left-10}" y="${y(v)+3}" text-anchor="end" fill="#9da79f" font-size="9">${Math.round(v*100)}%</text>`).join('');
  const series = (values, color, name) => values.length ? `<polyline points="${values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>${values.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${color}" opacity="0.01"><title>${name}: epoch ${i+1}, ${percent(v)}</title></circle>`).join('')}` : '';
  return `<svg class="learning-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and validation accuracy over ${count} epochs"><title>Training and validation accuracy over ${count} epochs</title>${lines}${series(train, '#739c84', 'Training')}${series(validation, '#de9771', 'Validation')}<text x="${left}" y="${height-8}" fill="#9da79f" font-size="9">Epoch 1</text><text x="${width-right}" y="${height-8}" text-anchor="end" fill="#9da79f" font-size="9">Epoch ${count}</text></svg>`;
}
async function openRun(id) {
  const version = ++runVersion;
  selectedRun = id;
  location.hash = 'runs';
  $('run-detail').hidden = false;
  $('run-detail-title').textContent = 'Loading run…';
  $('run-detail-path').textContent = id;
  $('run-metrics').replaceChildren(); $('learning-chart').replaceChildren(); $('run-classes').replaceChildren();
  $('use-run').disabled = true;
  try {
    const detail = await api(`/api/run?id=${encodeURIComponent(id)}`);
    if (version !== runVersion) return;
    const run = state.runs.find(r => r.id === id);
    $('run-detail-title').textContent = run ? `${title(run.category)} / ${run.name}` : id;
    $('run-metrics').innerHTML = stat('Test accuracy', percent(detail.metrics.accuracy), 'Held-out samples', '↗') + stat('Macro precision', percent(detail.benchmark?.macro_precision), 'Equal weight per sign', '◎') + stat('Macro recall', percent(detail.benchmark?.macro_recall), 'Equal weight per sign', '◎') + stat('Macro F1', percent(detail.metrics.macro_f1), 'Equal weight per class', '◎') + stat('Lowest sign recall', percent(detail.benchmark?.lowest_sign_recall), escapeHTML((detail.benchmark?.lowest_recall_signs || []).join(', ')), '◎') + stat('Epochs recorded', fmt((detail.history.accuracy || []).length), `${fmt(detail.classes.length)} gesture classes`, '⌁');
    $('learning-chart').innerHTML = chart(detail.history);
    $('run-classes').innerHTML = detail.classes.map(c => `<span class="chip">${escapeHTML(c)}</span>`).join('');
    $('use-run').disabled = !run?.checkpoint;
    $('run-detail').scrollIntoView({behavior: 'smooth', block: 'nearest'});
  } catch (error) { notice(error.message, true); }
}
for (const [id, action] of [['prepare-form', 'prepare'], ['train-form', 'train'], ['predict-form', 'predict']]) {
  $(id).addEventListener('submit', event => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.currentTarget));
    event.currentTarget.querySelectorAll('[type="checkbox"][name]').forEach(input => { data[input.name] = input.checked; });
    startJob({action, ...data});
  });
}
document.addEventListener('click', event => {
  const deletion = event.target.closest('[data-delete-kind]');
  if (deletion) { deleteSaved(deletion.dataset.deleteKind, deletion.dataset.deletePath); return; }
  const allReports = event.target.closest('[data-paper-report-all]');
  if (allReports) { startJob({action: 'paper-report'}); return; }
  const runReport = event.target.closest('[data-paper-report-run]');
  if (runReport) { startJob({action: 'paper-report', run: runReport.dataset.paperReportRun}); return; }
  const datasetReport = event.target.closest('[data-paper-report-dataset]');
  if (datasetReport) { startJob({action: 'paper-report', dataset: datasetReport.dataset.paperReportDataset}); return; }
  const cameraTest = event.target.closest('[data-hudyat-timing-run]');
  if (cameraTest) { startJob({action: 'hudyat-timing', run: cameraTest.dataset.hudyatTimingRun, hudyat_project: $('hudyat-project').value}); return; }
  const benchmark = event.target.closest('[data-benchmark-run]');
  if (benchmark) { startJob({action: 'benchmark', run: benchmark.dataset.benchmarkRun}); return; }
  const downloadModelZip = event.target.closest('[data-download-model-zip]');
  if (downloadModelZip) downloadRecord(`/api/model/download?run=${encodeURIComponent(downloadModelZip.dataset.downloadModelZip)}`);
  const exportModel = event.target.closest('[data-export-model]');
  if (exportModel) openModelExport(exportModel.dataset.exportModel);
  const history = event.target.closest('[data-history-run]');
  if (history) {
    $('test-model-filter').value = history.dataset.historyRun;
    renderTestHistory();
    $('prediction-tests-section').scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  const downloadModel = event.target.closest('[data-download-model]');
  if (downloadModel) downloadRecord(`/api/model-record/export?run=${encodeURIComponent(downloadModel.dataset.downloadModel)}`);
  const downloadTest = event.target.closest('[data-download-test]');
  if (downloadTest) downloadRecord(`/api/prediction-test/export?id=${encodeURIComponent(downloadTest.dataset.downloadTest)}`);
  const testCSV = event.target.closest('[data-test-csv]');
  if (testCSV) downloadRecord(`/api/prediction-test/export?id=${encodeURIComponent(testCSV.dataset.testCsv)}&format=${encodeURIComponent(testCSV.dataset.testFormat)}`);
  const editTest = event.target.closest('[data-edit-test]');
  if (editTest) openTestEditor(editTest.dataset.editTest);
  const retryTest = event.target.closest('[data-retry-test]');
  if (retryTest) {
    const test = state.prediction_tests?.[retryTest.dataset.retryTest];
    if (test) useRecordedRun(test.run, test.record);
  }
  const record = event.target.closest('[data-record-run]');
  if (record) editModelRecord(record.dataset.recordRun);
  const tryRun = event.target.closest('[data-test-run]');
  if (tryRun) useRecordedRun(tryRun.dataset.testRun);
  const run = event.target.closest('[data-run]');
  if (run) openRun(run.dataset.run);
  const prepare = event.target.closest('[data-prepare]');
  if (prepare) { $('prepare-form').elements.category.value = prepare.dataset.prepare; renderSourceFolders(); controls(); $('prepare-form').scrollIntoView({behavior: 'smooth', block: 'center'}); }
  const inspect = event.target.closest('[data-inspect]');
  if (inspect) startJob({action: 'inspect', category: inspect.dataset.inspect, dataset: inspect.dataset.dataset});
});
$('refresh').addEventListener('click', () => { notice(''); scan(); pollJobs(); });
$('scan-folders').addEventListener('click', () => { notice(''); scan(); });
$('category-search').addEventListener('input', renderDatasets);
$('prepare-form').elements.category.addEventListener('change', () => { $('prepare-error').hidden = true; renderSourceFolders(); controls(); });
$('dataset-name').addEventListener('input', () => { $('prepare-error').hidden = true; datasetPaths(); });
$('dataset-root').addEventListener('input', datasetPaths);
$('train-form').elements.category.addEventListener('change', renderTrainingDatasets);
$('train-dataset').addEventListener('change', controls);
$('split-strategy').addEventListener('change', controls);
$('source-folder-mode').addEventListener('change', controls);
$('source-folder-options').addEventListener('change', controls);
$('source-videos-only').addEventListener('change', controls);
$('download-benchmarks').addEventListener('click', () => downloadRecord(`/api/benchmarks/export?category=${encodeURIComponent($('run-filter').value)}&preferred=${$('run-preferred-only').checked ? '1' : '0'}`));
$('run-filter').addEventListener('change', renderRuns);
$('run-preferred-only').addEventListener('change', renderRuns);
$('model-export-form').addEventListener('submit', event => {
  event.preventDefault();
  if (exportingRun) startJob({action:'export', run:exportingRun, export_name:$('export-name').value.trim(), hand_model:$('export-hand-model').value});
});
$('model-export-close').addEventListener('click', () => { $('model-export-editor').hidden = true; });
$('download-export').addEventListener('click', () => downloadRecord(`/api/model/export/download?id=${encodeURIComponent($('download-export').dataset.id)}`));
$('test-model-filter').addEventListener('change', renderTestHistory);
$('live-save-test').addEventListener('click', () => openTestEditor('', 'live'));
$('sample-save-test').addEventListener('click', () => openTestEditor('', 'sample'));
$('model-record-form').addEventListener('submit', event => { event.preventDefault(); saveModelRecord(); });
$('model-record-close').addEventListener('click', () => { $('model-record-editor').hidden = true; });
$('model-record-copy').addEventListener('click', copyCameraSettings);
$('live-record').addEventListener('click', recordCameraSettings);
$('auto-test-start').addEventListener('click', startAutomatedTest);
$('auto-test-next').addEventListener('click', beginAutomatedTrial);
$('auto-test-retry').addEventListener('click', saveAutomatedTrial);
$('auto-test-finish').addEventListener('click', finishAutomatedTest);
$('job-select').addEventListener('change', renderJob);
$('cancel-job').addEventListener('click', async () => {
  $('cancel-job').disabled = true;
  try { await api('/api/cancel', {id: $('cancel-job').dataset.id}); await pollJobs(); }
  catch (error) { notice(error.message, true); }
  finally { $('cancel-job').disabled = false; }
});
$('use-run').addEventListener('click', () => useRecordedRun(selectedRun));
$('predict-run').addEventListener('change', controls);
$('live-start').addEventListener('click', startLive);
$('live-stop').addEventListener('click', stopLive);
$('live-mode').addEventListener('change', controls);
$('live-next').addEventListener('click', nextGesture);
for (const id of ['live-settle-ms', 'live-missing-ms', 'live-frame-ms']) {
  $(id).addEventListener('input', renderLiveTiming);
}
renderLiveTiming();
window.addEventListener('pagehide', () => {
  const session = liveSession;
  stopLive();
  if (session) fetch('/api/live/stop', {method: 'POST', keepalive: true,
    headers: {'Content-Type': 'application/json', 'X-Dashboard-Token': token}, body: JSON.stringify({session})}).catch(() => {});
});
document.addEventListener('visibilitychange', () => { if (document.hidden && liveActive) stopLive(); });
window.addEventListener('hashchange', route);
$('run-name').value = `run-${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15).replace('T', '-')}`;
$('dataset-name').value = `dataset-${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15).replace('T', '-')}`;
route();
scan().then(pollJobs);
setInterval(() => { if (!document.hidden) pollJobs(); }, 2000);
