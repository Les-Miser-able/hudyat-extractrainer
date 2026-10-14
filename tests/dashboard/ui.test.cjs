// Exercise the dashboard control logic without a browser or ML dependencies.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
  const nodes = new Map();
  const documentHandlers = {};
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', textContent: '', hidden: false, disabled: false,
      addEventListener(name, handler) { (this.handlers ??= {})[name] = handler; }, classList: {toggle() {}},
      querySelector: () => node(`${id}-submit`),
      replaceChildren() { this.innerHTML = ''; }, scrollIntoView() {},
      querySelectorAll: () => [], elements: {category: {value: 'alphabet',
        addEventListener(name, handler) { (this.handlers ??= {})[name] = handler; }}, hand_model: {value: ''}}
    });
    return nodes.get(id);
  }
  const submits = ['prepare-form', 'train-form', 'predict-form'].map(id => node(`${id}-submit`));
  const context = vm.createContext({
    document: {getElementById: node, querySelectorAll: selector => selector === '.submit-job' ? submits : [],
      addEventListener(name, handler) { documentHandlers[name] = handler; }},
    window: {addEventListener() {}}, location: {hash: '#datasets'},
    // Leave the initial network scan pending; tests supply state explicitly.
    fetch: () => new Promise(() => {}), setInterval() {}, setTimeout() {}, clearTimeout() {},
    performance: {now: () => 100}, console
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
    }
  };
  app.context.fetch = async (url, options) => {
    if (options?.method === 'POST') requests.push(JSON.parse(options.body));
    return {ok: false, json: async () => ({error: 'Fixture rejection'})};
  };
  const form = app.node('prepare-form');
  form.handlers.submit({preventDefault() {}, currentTarget: form});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].action, 'prepare');
  assert.equal(requests[0].dataset_name, 'custom preparation');
  assert.equal(requests[0].video_extra_samples, '3');
  assert.equal(requests[0].video_time_warp, '0.05');
  assert.equal(requests[0].seed, '42');
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
