// Exercise the dashboard control logic without a browser or ML dependencies.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');

function dashboard() {
  const nodes = new Map();
  const documentHandlers = {};
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', textContent: '', hidden: false, disabled: false, dataset: {},
      addEventListener(name, handler) { (this.handlers ??= {})[name] = handler; }, classList: {toggle() {}},
      querySelector: () => node(`${id}-submit`),
      replaceChildren() { this.innerHTML = ''; }, scrollIntoView() {},
      querySelectorAll: () => [], elements: {category: {value: 'alphabet',
        addEventListener(name, handler) { (this.handlers ??= {})[name] = handler; }}, hand_model: {value: ''}}
    });
    return nodes.get(id);
  }
  const submits = [...['prepare-form', 'train-form', 'predict-form'].map(id => node(`${id}-submit`)), node('model-export-start')];
  const context = vm.createContext({
    document: {getElementById: node, querySelectorAll: selector => selector === '.submit-job' ? submits : [],
      addEventListener(name, handler) { documentHandlers[name] = handler; }},
    window: {addEventListener() {}}, location: {hash: '#datasets'},
    // Leave the initial network scan pending; tests supply state explicitly.
    fetch: () => new Promise(() => {}), setInterval() {}, setTimeout() {}, clearTimeout() {},
    performance: {now: () => 100}, crypto, console
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../fsl_trainer/dashboard/ui/app.js'), 'utf8'), context);
  vm.runInContext("loading = false; token = 'test-token';", context);
  return {node, run: code => vm.runInContext(code, context), context, documentHandlers};
}

test('a stale missing-data scan allows a fresh preparation check', () => {
  const app = dashboard();
  app.run("state.categories = [{name: 'alphabet', preparation_issue: 'No gesture class folders found.'}]; controls();");
  assert.equal(app.node('prepare-form-submit').disabled, false);
  assert.match(app.node('prepare-readiness').textContent, /click Prepare dataset to check again/);
});

test('controls recover after a failed job and still block concurrent jobs', () => {
  const app = dashboard();
  app.run("jobs = [{status: 'running'}]; controls();");
  assert.equal(app.node('prepare-form-submit').disabled, true);
  assert.match(app.node('prepare-readiness').textContent, /Another job is running/);
  app.run("jobs = [{status: 'failed'}]; controls();");
  assert.equal(app.node('prepare-form-submit').disabled, false);
  app.run("submitting = true; controls();");
  assert.equal(app.node('prepare-form-submit').disabled, true);
  app.run("submitting = false; controls();");
  assert.equal(app.node('prepare-form-submit').disabled, false);
});

test('a rejected preparation request displays the error beside its form', async () => {
  const app = dashboard();
  const error = 'No gesture class folders found in data/alphabet.';
  const requests = [];
  app.context.fetch = async (url, options) => {
    requests.push([url, options]);
    return options?.method === 'POST'
      ? {ok: false, json: async () => ({error})}
      : {ok: false, json: async () => ({error: 'Scan unavailable'})};
  };
  await app.run("startJob({action: 'prepare', category: 'alphabet'})");
  assert.equal(app.node('prepare-error').hidden, false);
  assert.equal(app.node('prepare-error').textContent, error);
  assert.equal(app.node('prepare-form-submit').disabled, false);
  assert.equal(requests.filter(([, options]) => options?.method === 'POST').length, 1);
});


test('preparation submits video controls with the existing form fields', async () => {
  const app = dashboard();
  const requests = [];
  app.context.FormData = class {
    *[Symbol.iterator]() {
      yield ['category', 'alphabet'];
      yield ['dataset_name', 'custom preparation'];
      yield ['video_extra_samples', '3'];
      yield ['video_time_warp', '0.05'];
      yield ['seed', '42'];
      yield ['normalization', 'hand-relative-sequence-v1'];
    }
  };
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok: false, json: async () => ({error: 'Fixture rejection'})};
  };
  const form = app.node('prepare-form');
  form.querySelectorAll = selector => selector === '[type="checkbox"][name]'
    ? [{name: 'video_mirror_hands', checked: true}] : [];
  form.handlers.submit({preventDefault() {}, currentTarget: form});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].action, 'prepare');
  assert.equal(requests[0].dataset_name, 'custom preparation');
  assert.equal(requests[0].video_extra_samples, '3');
  assert.equal(requests[0].video_time_warp, '0.05');
  assert.equal(requests[0].video_mirror_hands, true);
  assert.equal(requests[0].seed, '42');
  assert.equal(requests[0].normalization, 'hand-relative-sequence-v1');
});

test('editing the dataset name updates the preparation output preview', () => {
  const app = dashboard();
  app.node('dataset-root').value = 'datasets_augmented';
  app.node('dataset-name').value = 'filtered_v2';
  app.node('dataset-name').handlers.input();
  assert.equal(app.node('prepare-output').textContent, 'Output: datasets_augmented/alphabet/filtered_v2');
  assert.equal(app.node('prepare-error').hidden, true);
});

test('dataset name input uses a valid browser pattern that accepts names and rejects paths', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../fsl_trainer/dashboard/ui/index.html'), 'utf8');
  const pattern = html.match(/id="dataset-name"[^>]*pattern="([^"]*)"/)[1]
    .replace(/&(?:quot|lt|gt);/g, entity => ({'&quot;': '"', '&lt;': '<', '&gt;': '>'})[entity]);
  const expression = new RegExp(`^(?:${pattern})$`, 'v');
  assert.equal(expression.test('custom dataset v2'), true);
  for (const value of ['', '../outside', 'a/b', 'a\\b', 'a|b', 'a:b']) {
    assert.equal(expression.test(value), false, value);
  }
});

test('training selection tracks its category and survives refreshes', () => {
  const app = dashboard();
  app.run(`state.categories = [
    {name: 'alphabet', datasets: [{name: 'v1', path: 'datasets_augmented/alphabet/v1'},
                                {name: 'v2', path: 'datasets_augmented/alphabet/v2'}]},
    {name: 'colors', datasets: [{name: 'Original dataset', path: 'datasets_augmented/colors'}]}
  ]; renderTrainingDatasets();`);
  assert.equal(app.node('train-dataset').value, 'datasets_augmented/alphabet/v1');
  app.node('train-dataset').value = 'datasets_augmented/alphabet/v2';
  app.run('renderTrainingDatasets()');
  assert.equal(app.node('train-dataset').value, 'datasets_augmented/alphabet/v2');
  assert.match(app.node('training-source').textContent, /datasets_augmented\/alphabet\/v2/);
  assert.equal(app.node('train-form-submit').disabled, false);
  const category = app.node('train-form').elements.category;
  category.value = 'colors';
  category.handlers.change();
  assert.equal(app.node('train-dataset').value, 'datasets_augmented/colors');
  assert.doesNotMatch(app.node('train-dataset').innerHTML, /alphabet/);
  category.value = 'missing';
  category.handlers.change();
  assert.equal(app.node('train-dataset').value, '');
  assert.equal(app.node('train-form-submit').disabled, true);
});

test('training submits the selected prepared dataset', async () => {
  const app = dashboard(), requests = [];
  app.context.FormData = class {
    *[Symbol.iterator]() {
      yield ['category', 'alphabet'];
      yield ['dataset', 'datasets_augmented/alphabet/filtered_v2'];
      yield ['run', 'model_v2'];
    }
  };
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok: false, json: async () => ({error: 'Fixture rejection'})};
  };
  const form = app.node('train-form');
  form.handlers.submit({preventDefault() {}, currentTarget: form});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests[0].action, 'train');
  assert.equal(requests[0].dataset, 'datasets_augmented/alphabet/filtered_v2');
});

test('validation buttons submit their own dataset path', async () => {
  const app = dashboard(), requests = [];
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok: false, json: async () => ({error: 'Fixture rejection'})};
  };
  app.documentHandlers.click({target: {closest: selector => selector === '[data-inspect]'
    ? {dataset: {inspect: 'alphabet', dataset: 'datasets_augmented/alphabet/v2'}} : null}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests[0].action, 'inspect');
  assert.equal(requests[0].category, 'alphabet');
  assert.equal(requests[0].dataset, 'datasets_augmented/alphabet/v2');
});

test('paper report controls submit dataset and model paths without starting training', async () => {
  for (const [selector, dataset, expected] of [
    ['[data-paper-report-dataset]', {paperReportDataset: 'datasets_augmented/alphabet/v2'}, {dataset: 'datasets_augmented/alphabet/v2'}],
    ['[data-paper-report-run]', {paperReportRun: 'runs/alphabet/v2'}, {run: 'runs/alphabet/v2'}],
    ['[data-paper-report-all]', {}, {}]
  ]) {
    const app = dashboard();
    const requests = [];
    app.context.fetch = async (url, options) => {
      if (options?.method === 'POST') requests.push(JSON.parse(options.body));
      return {ok: false, json: async () => ({error: 'Fixture rejection'})};
    };
    app.documentHandlers.click({target: {closest: value => value === selector ? {dataset} : null}});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests[0].action, 'paper-report');
    for (const [key, value] of Object.entries(expected)) assert.equal(requests[0][key], value);
  }
});

test('folder choices update preview and are submitted without PUB', async () => {
  const app = dashboard();
  app.node('source-folder-mode').value = 'selected';
  app.node('source-videos-only').checked = true;
  app.node('source-folder-options').querySelectorAll = () => [
    {value: 'S1', checked: true}, {value: 'S4', checked: true}
  ];
  app.run(`state.categories = [{name: 'alphabet', classes: [
    {name: 'blue', folders: {S1: {samples: 4, videos: 3}, S4: {samples: 3, videos: 3}, PUB: {samples: 20, videos: 20}}},
    {name: 'red', folders: {S1: {samples: 3, videos: 3}, S4: {samples: 3, videos: 3}}},
    {name: 'violet', folders: {PUB: {samples: 20, videos: 20}}}
  ]}]; controls();`);
  assert.match(app.node('source-selection-preview').textContent, /12 matching videos/);
  assert.match(app.node('source-selection-preview').textContent, /blue: 6 · red: 6/);
  assert.match(app.node('source-selection-preview').textContent, /Omitted gestures: violet/);
  const requests = [];
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok: false, json: async () => ({error: 'Fixture rejection'})};
  };
  await app.run("startJob({action: 'prepare', category: 'alphabet'})");
  assert.deepEqual(requests[0].source_folders, ['S1', 'S4']);
  assert.equal(requests[0].videos_only, true);
  app.node('source-folder-options').querySelectorAll = () => [];
  app.run('controls()');
  assert.match(app.node('prepare-readiness').textContent, /Choose at least one source folder/);
});

function camera(app) {
  let stopped = 0;
  const track = {stop() { stopped++; }, addEventListener() {}};
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  app.context.navigator = {mediaDevices: {getUserMedia: async () => stream}};
  const video = app.node('live-video');
  Object.assign(video, {play: async () => {}, videoWidth: 640, videoHeight: 480, readyState: 2});
  Object.assign(app.node('live-canvas'), {getContext: () => ({drawImage() {}}), toDataURL: () => 'data:image/jpeg;base64,fixture'});
  app.node('predict-run').value = 'runs/colors/first';
  app.node('live-mode').value = 'motion';
  return {stream, stopped: () => stopped};
}

test('camera inference sends serial frames and stop releases tracks and job controls', async () => {
  const app = dashboard();
  const cam = camera(app), requests = [], timers = [];
  app.context.setTimeout = fn => { timers.push(fn); return timers.length; };
  app.context.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    return {ok: true, json: async () => url.endsWith('/start') ? {session: 'live-session'}
      : url.endsWith('/frame') ? {status: 'prediction', probabilities: {blue: .8, red: .2}} : {ok: true}};
  };
  await app.run('startLive()');
  assert.equal(app.node('prepare-form-submit').disabled, true);
  assert.equal(app.node('predict-run').disabled, true);
  assert.match(app.node('live-result').innerHTML, /blue/);
  assert.equal(requests.filter(([url]) => url.endsWith('/frame')).length, 1);
  assert.equal(timers.length, 1);
  await app.run('stopLive()');
  assert.equal(cam.stopped(), 1);
  assert.equal(app.node('live-video').srcObject, null);
  assert.equal(app.node('live-result').innerHTML, '');
  assert.equal(app.node('prepare-form-submit').disabled, false);
  assert.equal(requests.at(-1)[0], '/api/live/stop');
});

test('denied camera permission displays an error without starting a worker', async () => {
  const app = dashboard();
  camera(app);
  app.context.navigator.mediaDevices.getUserMedia = async () => { throw Object.assign(new Error('Denied'), {name: 'NotAllowedError'}); };
  let requested = false;
  app.context.fetch = async () => { requested = true; };
  await app.run('startLive()');
  assert.equal(requested, false);
  assert.match(app.node('live-status').textContent, /permission was denied/);
  assert.equal(app.node('live-stop').disabled, true);
});

test('an older server missing live routes explains the restart and releases the camera', async () => {
  const app = dashboard();
  const cam = camera(app);
  app.context.fetch = async () => ({ok: false, status: 404, json: async () => ({error: 'Not found.'})});
  await app.run('startLive()');
  assert.match(app.node('live-status').textContent, /restart.*dashboard\.py/i);
  assert.equal(cam.stopped(), 1);
  assert.equal(app.node('live-stop').disabled, true);
});

test('stopping while permission is pending closes late camera tracks', async () => {
  const app = dashboard();
  const cam = camera(app);
  let allow;
  app.context.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { allow = resolve; });
  const start = app.run('startLive()');
  await app.run('stopLive()');
  allow(cam.stream);
  await start;
  assert.equal(cam.stopped(), 1);
  assert.equal(app.node('live-video').srcObject, null);
});

test('a stopped camera ignores a prediction already in flight', async () => {
  const app = dashboard();
  camera(app);
  let prediction;
  app.context.fetch = async url => url.endsWith('/frame')
    ? new Promise(resolve => { prediction = resolve; })
    : {ok: true, json: async () => ({session: 'live-session'})};
  const start = app.run('startLive()');
  await new Promise(resolve => setImmediate(resolve));
  await app.run('stopLive()');
  prediction({ok: true, json: async () => ({status: 'prediction', probabilities: {blue: 1}})});
  await start;
  assert.equal(app.node('live-result').innerHTML, '');
  assert.equal(app.node('live-status').textContent, 'Camera stopped.');
});

test('one gesture result stays visible until Next gesture resets capture', async () => {
  const app = dashboard();
  app.node('live-mode').value = 'gesture';
  app.run("liveActive = true; liveSession = 'fixture'; showLiveResult({status: 'prediction', probabilities: {evening: .9}}); controls();");
  assert.equal(app.node('live-next').hidden, false);
  assert.equal(app.node('live-next').disabled, false);
  const result = app.node('live-result').innerHTML;
  app.run("showLiveResult({status: 'held', keep_result: true, message: 'Result held.'})");
  assert.equal(app.node('live-result').innerHTML, result);
  const requests = [];
  app.context.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    return {ok: true, json: async () => ({status: 'waiting', message: 'Ready.'})};
  };
  await app.run('nextGesture()');
  assert.equal(app.node('live-result').innerHTML, '');
  assert.deepEqual(requests, [['/api/live/reset', {session: 'fixture'}]]);
  assert.equal(app.run('liveSession'), 'fixture');
});

test('Next gesture ignores a completed prediction from before the reset', async () => {
  const app = dashboard();
  camera(app);
  app.node('live-mode').value = 'gesture';
  let prediction;
  app.context.fetch = async url => url.endsWith('/frame')
    ? new Promise(resolve => { prediction = resolve; })
    : {ok: true, json: async () => url.endsWith('/start') ? {session: 'fixture'} : {status: 'waiting', message: 'Ready.'}};
  const start = app.run('startLive()');
  await new Promise(resolve => setImmediate(resolve));
  await app.run('nextGesture()');
  prediction({ok: true, json: async () => ({status: 'prediction', probabilities: {evening: .9}})});
  await start;
  assert.equal(app.node('live-result').innerHTML, '');
  await app.run('stopLive()');
});


test('continuous movement preserves a completed result when hands disappear and clears it for new capture', async () => {
  const app = dashboard();
  app.node('live-mode').value = 'motion';
  app.run("liveActive = true; liveSession = 'fixture'; controls(); showLiveResult({status: 'prediction', probabilities: {afternoon: .9}});");
  assert.equal(app.node('live-next').hidden, false);
  const result = app.node('live-result').innerHTML;
  app.run("showLiveResult({status: 'waiting', keep_result: true, message: 'Result held. Perform another gesture when ready.'})");
  assert.equal(app.node('live-result').innerHTML, result);
  app.run("showLiveResult({status: 'capturing', message: 'Capturing the full gesture…'})");
  assert.equal(app.node('live-result').innerHTML, '');
  app.context.fetch = async () => ({ok: true, json: async () => ({status: 'waiting'})});
  await app.run('nextGesture()');
  assert.equal(app.run('liveSession'), 'fixture');
});


test('camera cadence counts processing time and never adds a fixed pause after a slow frame', async () => {
  for (const processingMs of [20, 120]) {
    const app = dashboard();
    camera(app);
    let now = 0;
    const timers = [];
    app.context.performance.now = () => now;
    app.context.setTimeout = (fn, delay) => { timers.push({fn, delay}); return timers.length; };
    app.context.fetch = async url => {
      if (url.endsWith('/frame')) now = processingMs;
      return {ok: true, json: async () => url.endsWith('/start') ? {session: 'fixture'}
        : url.endsWith('/frame') ? {status: 'capturing'} : {ok: true}};
    };
    await app.run('startLive()');
    assert.equal(timers.length, 1);
    assert.equal(timers[0].delay, Math.max(0, 50 - processingMs));
    await app.run('stopLive()');
  }
});


test('timing sliders change labels and subsequent camera requests without resetting capture', async () => {
  const app = dashboard();
  camera(app);
  const requests = [], timers = [];
  app.context.setTimeout = (fn, delay) => { timers.push({fn, delay}); return timers.length; };
  app.context.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    return {ok: true, json: async () => url.endsWith('/start') ? {session: 'fixture'}
      : url.endsWith('/frame') ? {status: 'capturing'} : {ok: true}};
  };
  await app.run('startLive()');
  const first = requests.find(([url]) => url.endsWith('/frame'))[1];
  assert.deepEqual(first.timing, {settle_ms: 600, missing_ms: 400});
  for (const [id, value, output] of [
    ['live-settle-ms', '250', 'live-settle-value'],
    ['live-missing-ms', '900', 'live-missing-value'],
    ['live-frame-ms', '120', 'live-frame-value']
  ]) {
    app.node(id).value = value;
    app.node(id).handlers.input();
    assert.equal(app.node(output).textContent, value + ' ms');
    assert.equal(app.node(id).disabled, false);
  }
  await timers.shift().fn();
  const frames = requests.filter(([url]) => url.endsWith('/frame'));
  assert.equal(frames.length, 2);
  assert.deepEqual(frames[1][1].timing, {settle_ms: 250, missing_ms: 900});
  assert.equal(timers[0].delay, 120);
  assert.equal(requests.filter(([url]) => url.endsWith('/start')).length, 1);
  assert.equal(requests.filter(([url]) => url.endsWith('/reset')).length, 0);
  assert.equal(app.run('liveSession'), 'fixture');
  await app.run('stopLive()');
});


test('model comparison shows provenance and escapes user notes, with manual preferred filtering', () => {
  const app = dashboard();
  app.run("state.runs = [{id:'runs/greetings/a',name:'a',category:'greetings',classes:2,class_labels:['afternoon','evening'],accuracy:.9,checkpoint:true,metadata:{dataset:'dataset-v1',split_counts:{train:18,test:4},training:{batch_size:32,learning_rate:.001},preparation:{video_extra_samples:8}}},{id:'runs/colors/b',name:'b',category:'colors',classes:4}]; state.model_records = {'runs/greetings/a':{decision:'Preferred',notes:'<img src=x onerror=bad()>',settle_ms:700,missing_ms:450,frame_interval_ms:50}}; renderRuns();");
  assert.match(app.node('all-runs').innerHTML, /dataset-v1/);
  assert.match(app.node('all-runs').innerHTML, /700 ms/);
  assert.match(app.node('all-runs').innerHTML, /&lt;img/);
  assert.doesNotMatch(app.node('all-runs').innerHTML, /<img/);
  app.node('run-preferred-only').checked = true;
  app.run('renderRuns()');
  assert.ok(app.node('all-runs').innerHTML.includes('runs/greetings/a'));
  assert.ok(!app.node('all-runs').innerHTML.includes('runs/colors/b'));
});

test('saving a model record keeps blank timings unknown and records zero correct trials', async () => {
  const app = dashboard();
  app.run("state.runs = [{id:'runs/greetings/a',name:'a',category:'greetings',checkpoint:true}]; editModelRecord('runs/greetings/a');");
  app.node('record-decision').value = 'Preferred';
  app.node('record-live_trials').value = '10';
  app.node('record-live_correct').value = '0';
  app.node('record-notes').value = 'Confuses evening';
  const requests = [];
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push([url,payload]);
    return {ok:true,json:async()=>({model_records:{[payload.run]:payload}})};
  };
  await app.run('saveModelRecord()');
  assert.equal(requests[0][0], '/api/model-record');
  assert.equal(requests[0][1].live_correct, 0);
  assert.equal(requests[0][1].settle_ms, null);
  assert.equal(app.run("modelRecord('runs/greetings/a').decision"), 'Preferred');
  assert.match(app.node('predict-run').innerHTML, /Preferred/);
});

test('a failed record save preserves the draft and displays the server error', async () => {
  const app = dashboard();
  app.run("state.runs = [{id:'runs/greetings/a',name:'a',category:'greetings'}]; editModelRecord('runs/greetings/a');");
  app.node('record-notes').value = 'Keep this draft';
  app.context.fetch = async () => ({ok:false,json:async()=>({error:'Could not save record'})});
  await app.run('saveModelRecord()');
  assert.equal(app.node('record-notes').value, 'Keep this draft');
  assert.equal(app.node('model-record-error').hidden, false);
  assert.equal(app.node('model-record-error').textContent, 'Could not save record');
});

test('recording current camera settings preserves the active capture session', async () => {
  const app = dashboard(); camera(app);
  app.run("state.runs = [{id:'runs/colors/first',name:'first',category:'colors',checkpoint:true}]; liveActive=true; liveSession='fixture';");
  app.node('live-settle-ms').value = '700'; app.node('live-missing-ms').value = '450';
  app.node('live-frame-ms').value = '80';
  const requests=[];
  app.context.fetch=async(url,options)=>{const payload=JSON.parse(options.body);requests.push([url,payload]);return{ok:true,json:async()=>({model_records:{[payload.run]:payload}})};};
  await app.run('recordCameraSettings()');
  assert.equal(app.run('liveSession'), 'fixture');
  assert.equal(app.run('liveActive'), true);
  assert.equal(requests.length, 1);
  assert.equal(requests[0][1].frame_interval_ms, 80);
  assert.equal(requests[0][1].missing_ms, 450);
});

test('trying a model restores its recorded timing and mode', async () => {
  const app = dashboard();
  app.run("state.runs=[{id:'runs/greetings/a',checkpoint:true}]; state.model_records={'runs/greetings/a':{mode:'motion',settle_ms:700,missing_ms:450,frame_interval_ms:80}};");
  await app.run("useRecordedRun('runs/greetings/a')");
  assert.equal(app.node('predict-run').value, 'runs/greetings/a');
  assert.equal(app.node('live-mode').value, 'motion');
  assert.equal(Number(app.node('live-missing-ms').value), 450);
  assert.equal(app.node('live-frame-value').textContent, '80 ms');
  assert.equal(app.context.location.hash, 'predict');
});


test('a stale project refresh cannot replace a model record saved while it was pending', async () => {
  const app=dashboard();
  app.run("state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings',checkpoint:true}];");
  let finishScan;
  app.context.fetch=async(url,options)=>url.startsWith('/api/state')
    ? new Promise(resolve=>{finishScan=resolve;})
    : {ok:true,json:async()=>({model_records:{'runs/greetings/a':{decision:'Preferred'}}})};
  const pending=app.run('scan()');
  await app.run("persistModelRecord({run:'runs/greetings/a',decision:'Preferred'})");
  finishScan({ok:true,json:async()=>({categories:[],runs:[{id:'runs/greetings/a',name:'a',category:'greetings',checkpoint:true}],model_records:{},token:'test-token',python:'test'})});
  await pending;
  assert.equal(app.run("modelRecord('runs/greetings/a').decision"),'Preferred');
  assert.match(app.node('all-runs').innerHTML,/Preferred/);
});

test('saving a live prediction creates a separate numbered test and preserves the model record', async () => {
  const app = dashboard(), requests = [];
  app.run("state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings',checkpoint:true}]; state.model_records={'runs/greetings/a':{decision:'Preferred',notes:'Model description',live_trials:50,live_correct:45,response_ms:900}};");
  app.node('predict-run').value = 'runs/greetings/a';
  app.node('live-mode').value = 'gesture';
  app.node('live-settle-ms').value = '700';
  app.node('live-missing-ms').value = '450';
  app.node('live-frame-ms').value = '80';
  app.run("showLiveResult({status:'prediction', probabilities:{afternoon:.8, evening:.2}}); openTestEditor('', 'live');");
  assert.equal(app.node('record-test-name').value, 'Test 1');
  assert.equal(app.node('record-notes').value, 'Model description');
  assert.equal(app.node('record-live_trials').value, '');
  assert.equal(app.node('record-live_correct').value, '');
  assert.equal(app.node('record-response_ms').value, '');
  app.node('record-live_trials').value = '10';
  app.node('record-live_correct').value = '8';
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push([url, payload]);
    const saved = {id:'test-1',run:payload.run,name:'Test 1',number:1,created_at:'2026-10-15T12:00:00',
      model:{id:payload.run,name:'a',category:'greetings'},record:payload,prediction:payload.prediction};
    return {ok:true,json:async()=>({prediction_tests:{'test-1':saved},test:saved})};
  };
  await app.run('saveModelRecord()');
  assert.equal(requests[0][0], '/api/prediction-test');
  assert.equal(requests[0][1].name, undefined, 'automatic name is allocated by the server');
  assert.equal(requests[0][1].settle_ms, 700);
  assert.equal(requests[0][1].live_correct, 8);
  assert.deepEqual(requests[0][1].prediction.probabilities, {afternoon:.8,evening:.2});
  assert.equal(app.run("modelRecord('runs/greetings/a').live_trials"), 50);
  assert.equal(app.node('test-model-filter').value, 'runs/greetings/a');
  assert.equal(app.node('run-preferred-only').checked, false);
  await app.run('saveModelRecord()');
  assert.equal(requests[1][1].id, 'test-1', 'saving again edits the existing test');
  app.run("openTestEditor('', 'live')");
  assert.equal(app.node('record-test-name').value, 'Test 2');
  assert.equal(app.node('record-live_trials').value, '');
});

test('sample test attaches the model and sample used by the completed job', () => {
  const app = dashboard();
  app.run(`state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings',checkpoint:true},
    {id:'runs/colors/b',name:'b',category:'colors',checkpoint:true}];
    jobs=[{id:'job-1',action:'predict',run:'runs/greetings/a',sample:'data/greetings/one.jpg',
      category:'prediction',status:'completed',started:'2026-10-15T12:00:00',command:'predict',exit_code:0,
      logs:['{"afternoon":0.9,"evening":0.1}']}]; renderJob();`);
  app.node('predict-run').value = 'runs/colors/b';
  app.run('controls()');
  assert.equal(app.node('sample-save-test').disabled, true);
  app.node('predict-run').value = 'runs/greetings/a';
  app.run("controls(); openTestEditor('', 'sample');");
  assert.equal(app.node('sample-save-test').disabled, false);
  assert.equal(app.run('editingPrediction.sample'), 'data/greetings/one.jpg');
  assert.equal(app.node('record-settle_ms').value, '');
  assert.equal(app.node('record-mode').value, '');
});

test('saved test history shows frozen descriptions, escapes notes, and filters by model', () => {
  const app = dashboard();
  app.run(`state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings'}];
    state.prediction_tests={one:{id:'one',run:'runs/greetings/a',name:'Test <1>',number:1,
      created_at:'2026-10-15T12:00:00',model:{name:'a',category:'greetings',metadata:{dataset:'original-dataset'}},
      record:{notes:'<img src=x>',live_trials:10,live_correct:0,settle_ms:700}},
    two:{id:'two',run:'runs/colors/b',name:'Other session',number:1,created_at:'2026-10-15T12:00:00',
      model:{name:'b',category:'colors'},record:{}}}; renderRuns();`);
  assert.match(app.node('prediction-tests').innerHTML, /original-dataset/);
  assert.match(app.node('prediction-tests').innerHTML, /Test &lt;1&gt;/);
  assert.match(app.node('prediction-tests').innerHTML, /0\.00%/);
  assert.doesNotMatch(app.node('prediction-tests').innerHTML, /<img/);
  app.node('test-model-filter').value = 'runs/greetings/a';
  app.node('test-model-filter').handlers.change();
  assert.doesNotMatch(app.node('prediction-tests').innerHTML, /Other session/);
  app.run("openTestEditor('one')");
  assert.equal(app.node('record-live_correct').value, 0);
  assert.equal(app.node('record-notes').value, '<img src=x>');
  assert.equal(app.node('record-test-name').value, 'Test <1>');
});

test('retrying a test restores its own settings instead of current model settings', async () => {
  const app = dashboard();
  app.run("state.runs=[{id:'runs/greetings/a',checkpoint:true}]; state.model_records={'runs/greetings/a':{settle_ms:1200,mode:'pose'}};");
  await app.run("useRecordedRun('runs/greetings/a',{mode:'motion',settle_ms:350,missing_ms:250,frame_interval_ms:80})");
  assert.equal(app.node('live-mode').value, 'motion');
  assert.equal(Number(app.node('live-settle-ms').value), 350);
  assert.equal(Number(app.node('live-missing-ms').value), 250);
});

test('failed test saves preserve the draft and a stale scan cannot replace a successful save', async () => {
  const app = dashboard();
  app.run("state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings'}];");
  app.node('predict-run').value = 'runs/greetings/a';
  app.run("openTestEditor('', 'sample')");
  app.node('record-notes').value = 'Keep this test draft';
  app.context.fetch = async () => ({ok:false,json:async()=>({error:'Disk is full'})});
  await app.run('saveModelRecord()');
  assert.equal(app.node('record-notes').value, 'Keep this test draft');
  assert.equal(app.node('model-record-error').textContent, 'Disk is full');
  assert.equal(app.run('editingTest'), '');
  let finishScan;
  const saved = {id:'saved',run:'runs/greetings/a',name:'Test 1',number:1,created_at:'2026-10-15T12:00:00',
    model:{name:'a',category:'greetings'},record:{notes:'Keep this test draft'}};
  app.context.fetch = async url => url.startsWith('/api/state') ? new Promise(resolve=>{finishScan=resolve;})
    : {ok:true,json:async()=>({prediction_tests:{saved},test:saved})};
  const pending = app.run('scan()');
  await app.run('saveModelRecord()');
  finishScan({ok:true,json:async()=>({categories:[],runs:[{id:'runs/greetings/a',name:'a',category:'greetings'}],
    prediction_tests:{},token:'test-token',python:'test'})});
  await pending;
  assert.equal(app.run('state.prediction_tests.saved.record.notes'), 'Keep this test draft');
});

test('record downloads use the server filename and display download failures', async () => {
  const app = dashboard(), clicked = [];
  app.context.URL = {createObjectURL:()=> 'blob:test', revokeObjectURL() {}};
  app.context.document.body = {append() {}};
  app.context.document.createElement = () => ({click(){clicked.push([this.href,this.download]);},remove(){}});
  app.context.fetch = async () => ({ok:true,blob:async()=>({}),
    headers:{get:()=> 'attachment; filename="greetings-a-record.json"'}});
  await app.run("downloadRecord('/api/model-record/export?run=runs%2Fgreetings%2Fa')");
  assert.deepEqual(clicked, [['blob:test','greetings-a-record.json']]);
  app.context.fetch = async () => ({ok:false,json:async()=>({error:'Run not found'})});
  await app.run("downloadRecord('/api/model-record/export?run=missing')");
  assert.equal(app.node('notice').textContent, 'Run not found');
});

test('opening the test form reveals Model runs before scrolling to the editor', () => {
  const app = dashboard();
  const views = [{id:'view-predict',hidden:false},{id:'view-runs',hidden:true}];
  app.context.location.hash = '#predict';
  app.context.document.querySelectorAll = selector => selector === '.view' ? views : [];
  app.run("state.runs=[{id:'runs/greetings/a',name:'a',category:'greetings'}];");
  app.node('predict-run').value = 'runs/greetings/a';
  app.node('model-record-editor').scrollIntoView = () => {
    assert.equal(views[1].hidden, false, 'the target view must be visible before scrolling');
  };
  // Real Location objects normalize assignment to include the hash prefix.
  let hash = '#predict';
  Object.defineProperty(app.context.location, 'hash', {get:()=>hash,set:value=>{hash=value.startsWith('#')?value:'#'+value;}});
  app.run("openTestEditor('', 'live')");
  assert.equal(views[0].hidden, true);
  assert.equal(views[1].hidden, false);
});

test('model actions include trained ZIP download and HUDYAT export eligibility', () => {
  const app = dashboard();
  app.run(`state.runs=[{id:'runs/alphabet/a',name:'a',category:'alphabet',checkpoint:true,
    exports:{download:true,hudyat:true}}, {id:'runs/colors/arrays',name:'arrays',category:'colors',checkpoint:true,
    exports:{download:true,hudyat:false,reason:'HUDYAT needs extraction metadata'}}]; renderRuns();`);
  assert.match(app.node('all-runs').innerHTML, /Download model ZIP/);
  assert.match(app.node('all-runs').innerHTML, /Export for HUDYAT/);
  assert.match(app.node('all-runs').innerHTML, /HUDYAT needs extraction metadata/);
  assert.match(app.node('all-runs').innerHTML, /data-export-model="runs\/colors\/arrays"[^>]*disabled/);
  app.run("openModelExport('runs/alphabet/a')");
  assert.equal(app.node('export-name').value, 'alphabet-a');
  assert.equal(app.node('model-export-editor').hidden, false);
  assert.match(app.node('model-export-title').textContent, /Alphabet \/ a/);
  assert.equal(app.node('model-export-start').disabled, false);
  app.run("jobs=[{status:'running'}]; controls();");
  assert.equal(app.node('model-export-start').disabled, true);
});

test('HUDYAT export submits the selected model and custom detector through the job worker', async () => {
  const app = dashboard(), requests = [];
  app.run("state.runs=[{id:'runs/alphabet/test model',name:'test model',category:'alphabet',checkpoint:true,exports:{hudyat:true}}]; openModelExport('runs/alphabet/test model');");
  app.node('export-hand-model').value = 'assets/custom detector.task';
  app.node('export-name').value = ' Alphabet v2 ';
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push([url, JSON.parse(options.body)]);
    return {ok:true,json:async()=>options?.method === 'POST' ? {id:'export-1'} : []};
  };
  app.node('model-export-form').handlers.submit({preventDefault(){}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 1);
  assert.equal(requests[0][0], '/api/jobs');
  assert.equal(requests[0][1].action, 'export');
  assert.equal(requests[0][1].run, 'runs/alphabet/test model');
  assert.equal(requests[0][1].hand_model, 'assets/custom detector.task');
  assert.equal(requests[0][1].export_name, 'Alphabet v2');
});

test('an export rejection keeps the form and explains the detector mismatch', async () => {
  const app = dashboard();
  app.run("state.runs=[{id:'runs/alphabet/a',name:'a',category:'alphabet',checkpoint:true}]; openModelExport('runs/alphabet/a');");
  app.context.fetch = async () => ({ok:false,json:async()=>({error:'The hand detector must match training.'})});
  await app.run("startJob({action:'export',run:'runs/alphabet/a'})");
  assert.equal(app.node('model-export-error').hidden, false);
  assert.match(app.node('model-export-error').textContent, /must match training/);
  assert.equal(app.node('model-export-editor').hidden, false);
});

test('Activity offers the bundle download only after a successful export', () => {
  const app = dashboard();
  app.run("jobs=[{id:'export-1',action:'export',category:'alphabet',run:'runs/alphabet/a',status:'running',started:'2026-10-15T12:00:00',exit_code:null,logs:[]}]; renderJob();");
  assert.equal(app.node('download-export').hidden, true);
  app.run("jobs[0].status='completed'; jobs[0].exit_code=0; renderJob();");
  assert.equal(app.node('download-export').hidden, false);
  assert.equal(app.node('download-export').dataset.id, 'export-1');
  app.run("jobs[0].status='failed'; renderJob();");
  assert.equal(app.node('download-export').hidden, true);
});

test('model ZIP and HUDYAT download actions encode identifiers correctly', async () => {
  const app = dashboard();
  app.context.downloaded = [];
  app.run('downloadRecord = async path => downloaded.push(path)');
  app.documentHandlers.click({target:{closest(selector){return selector === '[data-download-model-zip]'
    ? {dataset:{downloadModelZip:'runs/alphabet/test model'}} : null;}}});
  app.node('download-export').dataset.id = 'export-1';
  app.node('download-export').handlers.click();
  assert.deepEqual(app.context.downloaded, ['/api/model/download?run=runs%2Falphabet%2Ftest%20model', '/api/model/export/download?id=export-1']);
});


test('benchmark table shows all metrics, timing scope, and missing evidence', () => {
  const app = dashboard();
  app.run(`state.runs = [{id:'runs/colors/v1', name:'v1', category:'colors', classes:6, checkpoint:true, complete:true,
    benchmark:{signs:6, required_signs:6, accuracy:.9, macro_precision:.91, macro_recall:.92,
      macro_f1:.93, lowest_sign_recall:.7, lowest_recall_signs:['<blue>'], status:'Incomplete',
      response_time_ms:null, checks:{response_time_ms:null}}}]; renderRuns();`);
  const html = app.node('benchmark-table').innerHTML;
  for (const expected of ['90.00%', '91.00%', '92.00%', '93.00%', '70.00%', 'Not measured',
    'Incomplete', 'response time ms: missing', '&lt;blue&gt;', 'data-benchmark-run="runs/colors/v1"']) {
    assert.ok(html.includes(expected), expected);
  }
  assert.ok(!html.includes('<blue>'));
  app.run(`state.runs[0].benchmark.response_time_ms = 24.5;
    state.runs[0].benchmark.timing = {p95_ms:30, samples:30}; renderRuns();`);
  assert.match(app.node('benchmark-table').innerHTML, /24.50 ms/);
  assert.match(app.node('benchmark-table').innerHTML, /HUDYAT camera-to-result · median/);
  app.node('run-filter').value = 'alphabet';
  app.run('renderRuns()');
  assert.match(app.node('benchmark-table').innerHTML, /No matching models/);
});

test('benchmark action submits the saved run without starting training', async () => {
  const app = dashboard(), requests = [];
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok:false, json:async () => ({error:'Fixture rejection'})};
  };
  app.documentHandlers.click({target:{closest:selector => selector === '[data-benchmark-run]'
    ? {dataset:{benchmarkRun:'runs/colors/v1'}} : null}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests[0].action, 'benchmark');
  assert.equal(requests[0].run, 'runs/colors/v1');
});


test('HUDYAT camera test prepares the selected run and Activity opens only completed pages', async () => {
  const app=dashboard(), requests=[];
  app.node('hudyat-project').value='../HUDYAT';
  app.context.fetch=async(url,options)=>{if(options?.method==='POST')requests.push(JSON.parse(options.body));return {ok:false,json:async()=>({error:'Fixture rejection'})};};
  app.documentHandlers.click({target:{closest:selector=>selector==='[data-hudyat-timing-run]'?{dataset:{hudyatTimingRun:'runs/colors/v1'}}:null}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(requests[0].action,'hudyat-timing');
  assert.equal(requests[0].hudyat_project,'../HUDYAT');
  app.run(`jobs=[{id:'0011223344556677',action:'hudyat-timing',category:'colors',status:'running',started:'2026-10-15T12:00:00',exit_code:null,command:'test',logs:[]}];renderJob();`);
  assert.equal(app.node('open-hudyat-timing').hidden,true);
  app.run("jobs[0].status='completed';renderJob();");
  assert.equal(app.node('open-hudyat-timing').hidden,false);
  assert.equal(app.node('open-hudyat-timing').href,'/hudyat-timing/0011223344556677/');
});

function automaticCamera(app, mode = 'pose') {
  camera(app);
  const saved = {id: 'auto-1', run: 'runs/colors/first', name: 'Test 1', number: 1,
    created_at: '2026-10-17T10:00:00', automated: true, record: {}, trials: [],
    model: {id: 'runs/colors/first', name: 'first', category: 'colors', class_labels: ['blue', 'red']}};
  app.context.automaticFixture = saved;
  app.run(`state.runs=[{...automaticFixture.model,checkpoint:true}];
    state.prediction_tests={'auto-1':automaticFixture};
    liveActive=true; liveSession='fixture'; liveVersion=7; liveCaptureVersion=2;
    automatedTest='auto-1'; automatedViewedTest='auto-1';`);
  app.node('live-mode').value = mode;
  app.node('live-settle-ms').value = '600';
  app.node('live-missing-ms').value = '400';
  app.node('live-frame-ms').value = '50';
  app.run('controls()');
  return saved;
}

function savedAutomaticReply(saved, payload) {
  const probabilities = payload.trial.probabilities;
  const predicted = Object.keys(probabilities).sort((a, b) => probabilities[b] - probabilities[a])[0] || null;
  const test = {...saved, trials: [{...payload.trial, predicted, correct: predicted === payload.trial.expected}]};
  return {ok: true, json: async () => ({test, prediction_tests: {[test.id]: test}})};
}

test('starting an automated session snapshots the model settings and resets movement capture', async () => {
  const app = dashboard(), saved = automaticCamera(app, 'gesture'), requests = [];
  app.run("automatedTest=''; automatedViewedTest=''; state.prediction_tests={};");
  app.node('auto-test-name').value = 'Camera comparison';
  app.node('auto-test-device').value = 'Laptop / signer 01';
  app.context.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    return {ok: true, json: async () => url === '/api/prediction-test'
      ? {test: saved, prediction_tests: {[saved.id]: saved}} : {status: 'waiting'}};
  };
  await app.run('startAutomatedTest()');
  assert.equal(requests[0][0], '/api/prediction-test');
  assert.equal(requests[0][1].automated, true);
  assert.equal(requests[0][1].name, 'Camera comparison');
  assert.equal(requests[0][1].device, 'Laptop / signer 01');
  assert.equal(requests[0][1].mode, 'gesture');
  assert.equal(requests[0][1].settle_ms, 600);
  assert.equal(requests[1][0], '/api/live/reset');
  assert.equal(app.run('automatedTrial.expected'), 'blue');
  assert.equal(app.run('automatedTrial.startedAt'), null, 'the timeout starts with the first submitted frame');
  assert.equal(app.node('auto-test-expected').disabled, true);
  assert.equal(app.node('live-settle-ms').disabled, true);
  assert.equal(app.node('live-next').disabled, true);
});

test('a held pose automatically records one wrong attempt rather than counting every camera frame', async () => {
  const app = dashboard(), saved = automaticCamera(app), requests = [];
  let now = 100;
  app.context.performance.now = () => now;
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push([url, payload, options]);
    if (url === '/api/prediction-test/trial') return savedAutomaticReply(saved, payload);
    now = 340;
    return {ok: true, json: async () => ({status: 'prediction', hand_detected: true,
      probabilities: {blue: .2, red: .8}, frames: 1})};
  };
  await app.run('beginAutomatedTrial();');
  await app.run('liveFrame(liveVersion)');
  await app.run('liveFrame(liveVersion)');
  const saves = requests.filter(([url]) => url === '/api/prediction-test/trial');
  assert.equal(saves.length, 1);
  assert.equal(saves[0][1].trial.expected, 'blue');
  assert.equal(saves[0][1].trial.response_ms, 240);
  assert.equal(saves[0][1].trial.elapsed_ms, 240);
  assert.deepEqual(saves[0][1].trial.probabilities, {blue: .2, red: .8});
  assert.equal(saves[0][2].keepalive, true);
  assert.equal(app.run("state.prediction_tests['auto-1'].trials[0].correct"), false);
  assert.equal(app.run('automatedTrial'), null);
  assert.equal(app.node('auto-test-next').disabled, false);
  assert.equal(app.node('auto-test-expected').disabled, false);
});

test('camera response starts at the first hand frame and a timeout cannot be overwritten by a late prediction', async () => {
  const app = dashboard(), saved = automaticCamera(app), requests = [];
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push(payload);
    return savedAutomaticReply(saved, payload);
  };
  await app.run('beginAutomatedTrial()');
  app.run('const timedTrial=automatedTrial; startAutomatedFrame(timedTrial, 100);');
  await app.run("observeAutomatedFrame(timedTrial,{status:'waiting',hand_detected:false},100,300)");
  await app.run("observeAutomatedFrame(timedTrial,{status:'capturing',hand_detected:true},500,650)");
  await app.run("observeAutomatedFrame(timedTrial,{status:'prediction',probabilities:{blue:.8,red:.2},frames:12,duration_ms:1000},1500,1800)");
  assert.equal(requests[0].trial.elapsed_ms, 1700);
  assert.equal(requests[0].trial.response_ms, 1300);
  assert.equal(requests[0].trial.duration_ms, 1000);
  await app.run('beginAutomatedTrial()');
  app.run('const expiredTrial=automatedTrial; startAutomatedFrame(expiredTrial,100);');
  app.context.performance.now = () => 15100;
  await app.run("finishAutomatedTrial('timeout')");
  await app.run("observeAutomatedFrame(expiredTrial,{status:'prediction',probabilities:{blue:.8,red:.2}},100,16000)");
  assert.equal(requests.length, 2);
  assert.equal(requests[1].trial.outcome, 'timeout');
  assert.equal(requests[1].trial.elapsed_ms, 15000);
  assert.equal(requests[1].trial.response_ms, null);
  assert.deepEqual(requests[1].trial.probabilities, {});
});

test('a failed automatic save retains its trial ID for retry and blocks further recording', async () => {
  const app = dashboard(), saved = automaticCamera(app), requests = [];
  let fail = true;
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push(payload);
    return fail ? {ok: false, json: async () => ({error: 'Disk is full'})} : savedAutomaticReply(saved, payload);
  };
  await app.run('beginAutomatedTrial()');
  app.run('startAutomatedFrame(automatedTrial, 100)');
  await app.run("finishAutomatedTrial('incomplete')");
  assert.match(app.node('auto-test-status').textContent, /Disk is full/);
  assert.equal(app.node('auto-test-retry').hidden, false);
  assert.equal(app.node('auto-test-next').disabled, true);
  await app.run('beginAutomatedTrial(); finishAutomatedTest();');
  assert.equal(requests.length, 1);
  assert.equal(app.run('automatedTest'), 'auto-1');
  fail = false;
  await app.run('saveAutomatedTrial()');
  assert.deepEqual(requests[0], requests[1]);
  assert.equal(app.run('automatedPending'), null);
  assert.equal(app.node('auto-test-retry').hidden, true);
  assert.equal(app.node('auto-test-next').disabled, false);
});

test('stopping the camera records an unfinished attempt and ignores the old in-flight frame', async () => {
  const app = dashboard(), saved = automaticCamera(app), requests = [];
  let frame;
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push([url, payload]);
    if (url === '/api/live/frame') return new Promise(resolve => { frame = resolve; });
    if (url === '/api/prediction-test/trial') return savedAutomaticReply(saved, payload);
    return {ok: true, json: async () => ({ok: true})};
  };
  await app.run('beginAutomatedTrial()');
  const pending = app.run('liveFrame(liveVersion)');
  await app.run('stopLive()');
  frame({ok: true, json: async () => ({status: 'prediction', probabilities: {blue: .8, red: .2}})});
  await pending;
  const saves = requests.filter(([url]) => url === '/api/prediction-test/trial');
  assert.equal(saves.length, 1);
  assert.equal(saves[0][1].trial.outcome, 'stopped');
  assert.equal(saves[0][1].trial.response_ms, null);
  assert.equal(app.run('liveActive'), false);
});

test('an old frame submitted before a trial is armed cannot count toward that trial', async () => {
  const app = dashboard(), saved = automaticCamera(app), requests = [];
  let frame;
  app.context.fetch = async (url, options) => {
    requests.push(url);
    if (url === '/api/live/frame') return new Promise(resolve => { frame = resolve; });
    return savedAutomaticReply(saved, JSON.parse(options.body));
  };
  const pending = app.run('liveFrame(liveVersion)');
  await app.run('beginAutomatedTrial()');
  frame({ok: true, json: async () => ({status: 'prediction', probabilities: {blue: .8, red: .2}})});
  await pending;
  assert.equal(requests.length, 1);
  assert.notEqual(app.run('automatedTrial'), null);
  assert.equal(app.run('automatedTrial.startedAt'), null);
});

test('automated test CSV actions use the correct format and editable observations retain measured counts', async () => {
  const app = dashboard(), saved = automaticCamera(app), downloads = [];
  app.run("automatedTest=''; openTestEditor('auto-1');");
  assert.equal(app.node('record-live_trials').readOnly, true);
  assert.equal(app.node('record-response_ms').readOnly, true);
  app.run("editModelRecord('runs/colors/first')");
  assert.equal(app.node('record-live_trials').readOnly, false);
  app.context.URL = {createObjectURL: () => 'blob:test', revokeObjectURL() {}};
  app.context.document.body = {append() {}};
  app.context.document.createElement = () => ({click() {}, remove() {}});
  app.context.fetch = async url => {
    downloads.push(url);
    return {ok: true, blob: async () => ({}), headers: {get: () => 'attachment; filename="trials.csv"'}};
  };
  app.documentHandlers.click({target: {closest: selector => selector === '[data-test-csv]'
    ? {dataset: {testCsv: 'auto 1', testFormat: 'trials'}} : null}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(downloads[0], '/api/prediction-test/export?id=auto%201&format=trials');
  app.run('renderTestHistory()');
  assert.match(app.node('prediction-tests').innerHTML, /Download metrics CSV/);
  assert.match(app.node('prediction-tests').innerHTML, /Download trials CSV/);
});

test('an early or stale timeout callback cannot fail an attempt before its deadline', async () => {
  const app = dashboard(), saved = automaticCamera(app), timers = [], requests = [];
  let now = 100;
  app.context.performance.now = () => now;
  app.context.setTimeout = (callback, delay) => { timers.push([callback, delay]); return timers.length; };
  app.context.fetch = async (url, options) => {
    const payload = JSON.parse(options.body); requests.push(payload);
    return savedAutomaticReply(saved, payload);
  };
  await app.run('beginAutomatedTrial()');
  app.run('startAutomatedFrame(automatedTrial,100)');
  now = 15099.5;
  timers[0][0]();
  assert.equal(requests.length, 0);
  assert.equal(timers[1][1], 1);
  now = 15100;
  timers[1][0]();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].trial.elapsed_ms, 15000);
  await app.run('beginAutomatedTrial()');
  timers[0][0]();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 1, 'the previous timer must not timeout the new trial');
  assert.notEqual(app.run('automatedTrial'), null);
});


test('LOSO controls choose signer holdouts and show the saved training protocol', () => {
  const app = dashboard();
  app.node('prepare-form').elements.validation = {disabled:false};
  app.node('prepare-form').elements.test = {disabled:false};
  app.node('split-strategy').value = 'loso';
  app.node('split-strategy').handlers.change();
  assert.equal(app.node('signer-holdouts').hidden, false);
  assert.equal(app.node('test-signer').disabled, false);
  assert.equal(app.node('test-signer').required, true);
  assert.equal(app.node('prepare-form').elements.test.disabled, true);
  app.run(`state.categories = [{name:'alphabet',datasets:[{name:'LOSO S1',path:'datasets_augmented/alphabet/loso-S1',evaluation_protocol:{strategy:'leave_one_signer_out',test_signer:'S1',validation_signer:'S2',train_signers:['S3','S4']}}]}]; renderTrainingDatasets();`);
  assert.match(app.node('training-source').textContent, /LOSO test: S1; validation: S2; train: S3, S4/);
  app.node('split-strategy').value = 'grouped';
  app.run('controls()');
  assert.equal(app.node('test-signer').disabled, true);
  assert.equal(app.node('prepare-form').elements.test.disabled, false);
});

test('delete actions name the exact artifact, honor cancellation, and use authenticated requests', async () => {
  const app = dashboard();
  const requests = [];
  app.node('data-root').value = 'data';
  app.node('dataset-root').value = 'datasets_augmented';
  app.run(`state.runs = [{id:'runs/colors/first',name:'first',category:'colors',class_labels:['blue'],checkpoint:true}]; selectedRun='runs/colors/first'; scan=async()=>{state.runs=[];};`);
  app.context.window.confirm = () => false;
  app.context.fetch = async (url, opts) => {
    requests.push([url, JSON.parse(opts.body), opts.headers]);
    return {ok:true, json:async()=>({ok:true,trash:'.trash/fixture'})};
  };
  await app.run("deleteSaved('model','runs/colors/first')");
  assert.equal(requests.length, 0);
  let confirmed = '';
  app.context.window.confirm = message => { confirmed = message; return true; };
  await app.run("deleteSaved('model','runs/colors/first')");
  assert.match(confirmed, /runs\/colors\/first/);
  assert.equal(requests[0][0], '/api/delete');
  assert.equal(requests[0][1].path, 'runs/colors/first');
  assert.equal(requests[0][1].kind, 'model');
  assert.equal(requests[0][2]['X-Dashboard-Token'], 'test-token');
  assert.equal(app.node('run-detail').hidden, true);
  assert.match(app.node('notice').textContent, /Recovery folder: .trash\/fixture/);
  assert.equal(app.run('selectedRun'), '');
});

test('delete blocks during jobs and camera use and preserves a server rejection', async () => {
  const app = dashboard();
  app.run("state.categories=[{name:'alphabet',datasets:[{name:'v1',path:'datasets_augmented/alphabet/v1'}]}]");
  let calls = 0;
  app.context.window.confirm = () => { calls++; return true; };
  app.context.fetch = async () => ({ok:false,json:async()=>({error:'Stop the running job'})});
  app.run("jobs=[{status:'running'}]");
  await app.run("deleteSaved('dataset','datasets_augmented/alphabet/v1')");
  assert.equal(calls, 0);
  app.run("jobs=[];liveActive=true");
  await app.run("deleteSaved('dataset','datasets_augmented/alphabet/v1')");
  assert.equal(calls, 0);
  app.run('liveActive=false');
  await app.run("deleteSaved('dataset','datasets_augmented/alphabet/v1')");
  assert.equal(calls, 1);
  assert.equal(app.node('notice').textContent, 'Stop the running job');
  assert.equal(app.run('submitting'), false);
  assert.equal(app.run('state.categories[0].datasets.length'), 1);
});
