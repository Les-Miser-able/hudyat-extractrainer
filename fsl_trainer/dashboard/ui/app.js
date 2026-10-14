'use strict';
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => Number(value || 0).toLocaleString();
const percent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const title = value => value.charAt(0).toUpperCase() + value.slice(1).replaceAll('_', ' ');
let state = {categories: [], runs: []}, jobs = [], token = '', selectedRun = '', polling = false, loading = false, submitting = false;
let scanVersion = 0, runVersion = 0, previousJobStatus = '';
let folderCategory = '';
let liveStream = null, liveSession = '', liveActive = false, liveVersion = 0, liveTimer = null;
let liveCaptureVersion = 0, liveResetting = false;

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = !message;
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Dashboard-Token': token}, body: JSON.stringify(body)
  });
  const result = await response.json();
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
  populate($('predict-run'), state.runs.filter(r => r.checkpoint).map(r => [r.id, `${title(r.category)} / ${r.name}`]), 'No trained checkpoints found');
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
    ? `Dataset: ${selected.path} · Output: runs/${$('train-form').elements.category.value}/<run name>`
    : 'Prepare a dataset, then select it here to train a model.';
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
function renderRuns() {
  const filter = $('run-filter').value;
  $('all-runs').innerHTML = runTable(state.runs.filter(r => !filter || r.category === filter));
}
function renderDatasets() {
  const search = $('category-search').value.toLowerCase();
  const rows = state.categories.filter(c => c.name.toLowerCase().includes(search));
  $('dataset-list').innerHTML = rows.length ? rows.map(c => `<article class="dataset-item"><div class="dataset-item-heading"><div><span class="category-name">${escapeHTML(title(c.name))}</span><div class="category-meta">${fmt(c.samples)} sources · ${fmt(c.classes.length)} gesture folders · ${fmt(c.datasets?.length)} prepared datasets</div></div><span class="tag ${c.prepared ? 'ready' : ''}">${c.prepared ? 'Prepared' : 'Not prepared'}</span></div><div class="chips">${c.classes.map(g => `<span class="chip">${escapeHTML(g.name)} <small>${fmt(g.samples)}</small></span>`).join('') || '<span class="hint">No gesture folders yet.</span>'}</div>${(c.datasets || []).map(d => `<div class="dataset-version"><div><strong>${escapeHTML(d.name)}</strong><p class="hint">${escapeHTML(d.path)}${d.summary.total_output_sequences ? ` · ${fmt(d.summary.total_output_sequences)} prepared sequences` : ''}</p></div><button class="button secondary submit-job" type="button" data-inspect="${escapeHTML(c.name)}" data-dataset="${escapeHTML(d.path)}">Validate dataset ↗</button></div>`).join('')}<div class="dataset-actions"><button data-prepare="${escapeHTML(c.name)}">Prepare another dataset ↗</button></div></article>`).join('') : empty('No matching categories. Check your source and dataset folders.');
  controls();
}
async function scan() {
  const version = ++scanVersion;
  loading = true; controls();
  try {
    const next = await api(`/api/state?data=${encodeURIComponent($('data-root').value)}&datasets=${encodeURIComponent($('dataset-root').value)}`);
    if (version !== scanVersion) return;
    state = next; token = next.token; render();
  } catch (error) { notice(error.message, true); }
  finally { if (version === scanVersion) { loading = false; controls(); } }
}
function controls() {
  const active = jobs.some(j => j.status === 'running');
  document.querySelectorAll('.submit-job').forEach(b => { b.disabled = loading || submitting || active || liveActive || !token; });
  $('train-form').querySelector('[type="submit"]').disabled ||= !preparedChoices().some(d => d.path === $('train-dataset').value);
  datasetPaths();
  $('live-start').disabled = loading || submitting || active || liveActive || !token || !$('predict-run').value;
  $('live-stop').disabled = !liveActive;
  $('predict-run').disabled = liveActive;
  $('live-mode').disabled = liveActive;
  $('live-next').hidden = $('live-mode').value !== 'gesture';
  $('live-next').disabled = !liveActive || !liveSession || liveResetting;
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
async function startJob(data) {
  if (submitting) return;
  submitting = true; controls(); notice('');
  if (data.action === 'prepare') $('prepare-error').hidden = true;
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
  const prediction = jobs.find(j => j.action === 'predict' && j.status === 'completed');
  if (prediction) {
    const output = prediction.logs.join('\n');
    try {
      const start = output.startsWith('{') ? 0 : output.lastIndexOf('\n{') + 1;
      const result = JSON.parse(output.slice(start));
      const ranked = Object.entries(result).filter(([, v]) => typeof v === 'number').sort((a, b) => b[1] - a[1]);
      if (ranked.length) {
        const [label, confidence] = ranked[0];
        $('prediction-result').className = '';
        $('prediction-result').innerHTML = `<div class="prediction-main">${escapeHTML(label)}</div><div class="prediction-confidence">${percent(confidence)} model confidence · ${escapeHTML(prediction.started.replace('T', ' '))}</div>${ranked.slice(0, 5).map(([name, score]) => `<div class="prediction-row"><span>${escapeHTML(name)}</span><meter min="0" max="1" value="${score}" aria-label="${escapeHTML(name)} confidence"></meter><span>${percent(score)}</span></div>`).join('')}<p class="hint">Model scores can be wrong. Verify the sign before relying on a prediction.</p>`;
      }
    } catch { /* Non-JSON worker output is still available in the activity log. */ }
  }
}
function showLiveResult(result) {
  $('live-status').textContent = result.message || 'Live prediction running.';
  if (result.status !== 'prediction') {
    if (!result.keep_result) $('live-result').replaceChildren();
    return;
  }
  const ranked = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  if (!ranked.length) return;
  const [label, confidence] = ranked[0];
  $('live-result').innerHTML = `<div class="prediction-main">${escapeHTML(label)}</div><div class="prediction-confidence">${percent(confidence)} model confidence${confidence < .6 ? ' · Uncertain' : ''}</div>${ranked.slice(0, 5).map(([name, score]) => `<div class="prediction-row"><span>${escapeHTML(name)}</span><meter min="0" max="1" value="${score}"></meter><span>${percent(score)}</span></div>`).join('')}`;
}
async function liveFrame(version) {
  if (version !== liveVersion || !liveSession) return;
  if (liveResetting) {
    liveTimer = setTimeout(() => liveFrame(version), 200);
    return;
  }
  const captureVersion = liveCaptureVersion;
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
    const result = await api('/api/live/frame', {session: liveSession,
      image: canvas.toDataURL('image/jpeg', .8), timestamp: performance.now()});
    if (version !== liveVersion) return;
    if (captureVersion === liveCaptureVersion) showLiveResult(result);
    liveTimer = setTimeout(() => liveFrame(version), 200);
  } catch (error) {
    if (version !== liveVersion) return;
    await stopLive();
    $('live-status').textContent = error.message;
  }
}
async function startLive() {
  if (liveActive) return;
  const version = ++liveVersion;
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
  if (session) {
    try { await api('/api/live/stop', {session}); }
    catch (error) { $('live-status').textContent = `Camera stopped. ${error.message} The idle worker will expire automatically.`; }
  }
}
async function nextGesture() {
  if (!liveSession || liveResetting) return;
  const version = liveVersion;
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
      if (jobs[0].status === 'completed') notice(`${title(jobs[0].action)} completed. Project data has been refreshed.`);
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
    $('run-metrics').innerHTML = stat('Test accuracy', percent(detail.metrics.accuracy), 'Held-out samples', '↗') + stat('Macro F1', percent(detail.metrics.macro_f1), 'Equal weight per class', '◎') + stat('Epochs recorded', fmt((detail.history.accuracy || []).length), `${fmt(detail.classes.length)} gesture classes`, '⌁');
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
$('source-folder-mode').addEventListener('change', controls);
$('source-folder-options').addEventListener('change', controls);
$('source-videos-only').addEventListener('change', controls);
$('run-filter').addEventListener('change', renderRuns);
$('job-select').addEventListener('change', renderJob);
$('cancel-job').addEventListener('click', async () => {
  $('cancel-job').disabled = true;
  try { await api('/api/cancel', {id: $('cancel-job').dataset.id}); await pollJobs(); }
  catch (error) { notice(error.message, true); }
  finally { $('cancel-job').disabled = false; }
});
$('use-run').addEventListener('click', () => { $('predict-run').value = selectedRun; location.hash = 'predict'; });
$('predict-run').addEventListener('change', controls);
$('live-start').addEventListener('click', startLive);
$('live-stop').addEventListener('click', stopLive);
$('live-mode').addEventListener('change', controls);
$('live-next').addEventListener('click', nextGesture);
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
