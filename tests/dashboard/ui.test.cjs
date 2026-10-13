// Exercise the dashboard control logic without a browser or ML dependencies.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function dashboard() {
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', textContent: '', hidden: false, disabled: false,
      addEventListener(name, handler) { (this.handlers ??= {})[name] = handler; }, classList: {toggle() {}},
      querySelector: () => node(`${id}-submit`),
      replaceChildren() { this.innerHTML = ''; },
      querySelectorAll: () => [], elements: {category: {value: 'alphabet', addEventListener() {}}, hand_model: {value: ''}}
    });
    return nodes.get(id);
  }
  const submits = ['prepare-form', 'train-form', 'predict-form'].map(id => node(`${id}-submit`));
  const context = vm.createContext({
    document: {getElementById: node, querySelectorAll: selector => selector === '.submit-job' ? submits : [],
      addEventListener() {}},
    window: {addEventListener() {}}, location: {hash: '#datasets'},
    // Leave the initial network scan pending; tests supply state explicitly.
    fetch: () => new Promise(() => {}), setInterval() {}, setTimeout() {}, clearTimeout() {},
    performance: {now: () => 100}, console
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../../fsl_trainer/dashboard/ui/app.js'), 'utf8'), context);
  vm.runInContext("loading = false; token = 'test-token';", context);
  return {node, run: code => vm.runInContext(code, context), context};
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
  assert.equal(requests[0].video_extra_samples, '3');
  assert.equal(requests[0].video_time_warp, '0.05');
  assert.equal(requests[0].seed, '42');
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
