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
      addEventListener() {}, classList: {toggle() {}},
      querySelector: () => node(`${id}-submit`),
      querySelectorAll: () => [], elements: {category: {value: 'alphabet', addEventListener() {}}}
    });
    return nodes.get(id);
  }
  const submits = ['prepare-form', 'train-form', 'predict-form'].map(id => node(`${id}-submit`));
  const context = vm.createContext({
    document: {getElementById: node, querySelectorAll: selector => selector === '.submit-job' ? submits : [],
      addEventListener() {}},
    window: {addEventListener() {}}, location: {hash: '#datasets'},
    // Leave the initial network scan pending; tests supply state explicitly.
    fetch: () => new Promise(() => {}), setInterval() {}, console
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
