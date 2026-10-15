import test from 'node:test';
import assert from 'node:assert/strict';
import {createTimingTrial} from '../../integrations/hudyat/camera-timing.js';

test('camera clock begins before extraction for the first hand frame and includes confirmation/paint',()=>{
  const trial=createTimingTrial('blue'), absent=new Float32Array(128), present=new Float32Array(128);present[126]=1;
  assert.equal(trial.capture(100,absent),null);
  assert.equal(trial.capture(200,present),200);
  trial.capture(400,present);
  trial.prediction({prediction:'blue'});
  assert.deepEqual(trial.complete(true,2400),{target:'blue',predicted:'blue',correct:true,response_ms:2200,elapsed_ms:2200,timeout:false});
});
test('timeouts retain a missing response and an elapsed duration instead of an invented fast result',()=>{
  const trial=createTimingTrial('blue'), frame=new Float32Array(128);frame[127]=1;
  trial.capture(100,frame);trial.prediction({prediction:'red'});
  const result=trial.complete(false,15150,true);
  assert.equal(result.response_ms,null);assert.equal(result.predicted,null);assert.equal(result.timeout,true);assert.equal(result.elapsed_ms,15050);
  assert.throws(()=>createTimingTrial('blue').complete(true,100),/No valid camera start/);
});

test('no hand detection is retained as a timeout instead of waiting indefinitely',()=>{
  const trial=createTimingTrial('blue');trial.capture(100,new Float32Array(128));
  assert.equal(trial.started,null);
  const result=trial.complete(false,15100,true);
  assert.equal(result.response_ms,null);assert.equal(result.elapsed_ms,15000);assert.equal(result.timeout,true);
});
