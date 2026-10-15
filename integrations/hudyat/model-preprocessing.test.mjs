import assert from 'node:assert/strict';
import {test} from 'node:test';
import {normalizeFrames, normalizationFor, preprocessSequence} from './model-preprocessing.js';

function frame(x = .2, y = .3) {
  const values = new Float32Array(128);
  for (let j = 0; j < 21; j++) values.set([x + j * .01, y + j * .02, j * .003], j * 3);
  values[126] = 1;
  return values;
}

test('legacy image-coordinate models retain their original inputs', () => {
  const frames = [frame()];
  assert.equal(normalizationFor({}), 'image');
  assert.equal(preprocessSequence(frames, 0, 0, {}), frames);
  assert.equal(preprocessSequence(frames, 640, 480, {normalization:'image'}), frames);
  assert.throws(() => normalizationFor({normalization:'unknown'}), /Unsupported/);
});

test('hand-relative preprocessing preserves wrist movement and zeroes missing hands', () => {
  const frames = [frame(), frame(.3, .4)];
  const result = preprocessSequence(frames, 640, 480, {normalization:'hand-relative-sequence-v1'});
  assert.equal(result[0][0], 0);
  assert.ok(result[1][0] > 0);
  assert.ok(result[1][1] > 0);
  assert.equal(result[1][126], 1);
  assert.equal(result[1][127], 0);
  assert.ok(Array.from(result[1].slice(63,126)).every(v => v === 0));
  assert.ok(Math.abs(result[0][27] - result[1][27]) < 1e-6);
});

test('hand-relative poses are invariant to image translation and uniform scale', () => {
  const first = frame(), shifted = new Float32Array(first);
  for (let j = 0; j < 21; j++) {
    shifted[j*3] = first[j*3]*.5+.1;
    shifted[j*3+1] = first[j*3+1]*.5+.2;
    shifted[j*3+2] = first[j*3+2]*.5;
  }
  const a = normalizeFrames([first], 640, 480)[0];
  const b = normalizeFrames([shifted], 640, 480)[0];
  a.forEach((value,i) => assert.ok(Math.abs(value-b[i]) < 1e-5));
  assert.deepEqual(Array.from(first), Array.from(frame()), 'source landmarks remain available for overlays');
});

test('invalid dimensions, nonfinite coordinates, and degenerate hands are handled', () => {
  assert.throws(() => normalizeFrames([frame()], 0, 480), /dimensions/);
  const invalid = frame(); invalid[5] = NaN;
  assert.throws(() => normalizeFrames([invalid], 640, 480), /Invalid/);
  const degenerate = new Float32Array(128); degenerate[126] = 1;
  assert.ok(Array.from(normalizeFrames([degenerate], 640, 480)[0]).every(v => v === 0));
});
