// Check an extracted bundle against HUDYAT's real TF.js loader, without the API or database.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {verifyCameraRuntime} from './runtime-probe.mjs';

const [bundle, project] = process.argv.slice(2);
if (!bundle || !project) throw Error('Usage: node tools/verify_hudyat_export.mjs EXTRACTED_BUNDLE HUDYAT_FOLDER');
const load = file => import(pathToFileURL(file).href);
const module = await load(path.join(project, 'client/node_modules/@tensorflow/tfjs/dist/tf.node.js'));
const tf = module.default || module;
const {buildModel, loadWeights, resample} = await load(path.join(project, 'client/src/features/fsl-recognition/category-core.js'));
const {preprocessSequence} = await import('./model-preprocessing.js');
const site = path.resolve(bundle, 'model-assets');
const catalog = JSON.parse(await readFile(path.join(site, 'catalog.json'), 'utf8'));
for (const entry of catalog.categories.filter(e => e.available)) {
  const artifacts = {};
  for (const [field, hash] of [['modelUrl','modelSha256'],['weightsUrl','weightsSha256'],['handUrl','handSha256']]) {
    const filename = path.resolve(site, entry[field]);
    if (!filename.startsWith(site + path.sep)) throw Error('Artifact path escapes the bundle.');
    artifacts[field] = await readFile(filename);
    assert.equal(createHash('sha256').update(artifacts[field]).digest('hex'), entry[hash]);
  }
  const metadata = JSON.parse(artifacts.modelUrl);
  const folder = path.dirname(path.resolve(site, entry.modelUrl));
  const probe = JSON.parse(await readFile(path.join(folder, 'parity.json'), 'utf8'));
  const camera = JSON.parse(await readFile(path.join(folder, 'camera-preprocessing-parity.json'), 'utf8'));
  let preprocessingError = 0;
  for (const sample of camera.cases) {
    const input = resample(preprocessSequence(sample.frames.map(f=>new Float32Array(f)), sample.width, sample.height, camera.preprocessing));
    const expected = sample.expected.flat();
    input.forEach((value,i) => {preprocessingError = Math.max(preprocessingError, Math.abs(value-expected[i]));});
  }
  assert.ok(preprocessingError < 2e-5, 'Camera preprocessing differs from Python: ' + preprocessingError);
  await tf.setBackend('cpu');
  await tf.ready();
  const model = buildModel(tf, metadata.classes.length);
  const bytes = artifacts.weightsUrl;
  loadWeights(tf, model, metadata, bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const input = tf.tensor3d(probe.inputs);
  const output = model.predict(input);
  const actual = await output.array();
  let predictionError = 0;
  actual.forEach((row,i) => row.forEach((value,j) => {predictionError=Math.max(predictionError,Math.abs(value-probe.expected[i][j]));}));
  input.dispose(); output.dispose(); model.dispose();
  assert.ok(predictionError < .0002, 'Model output differs from Python: ' + predictionError);
  const runtime=await verifyCameraRuntime(project,entry,artifacts,camera,tf);
  const enhanced=await verifyCameraRuntime(project,entry,artifacts,camera,tf,'enhanced');
  console.log(JSON.stringify({category:entry.id,version:entry.version,normalization:metadata.preprocessing.normalization||'image',
    preprocessingError,predictionError,runtime,enhanced,hashes:'passed'}));
}
