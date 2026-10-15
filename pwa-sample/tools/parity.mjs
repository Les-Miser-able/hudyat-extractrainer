import * as tf from '@tensorflow/tfjs';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {buildModel,loadWeights,preprocessFrames} from '../dist/core.js';
const dist=path.resolve(import.meta.dirname,'../dist');
await tf.setBackend('cpu');
const catalog=JSON.parse(await readFile(path.join(dist,'catalog.json'),'utf8'));
for(const entry of catalog.categories.filter(c=>c.available)){
  const folder=path.dirname(path.join(dist,entry.modelUrl));
  const meta=JSON.parse(await readFile(path.join(folder,'model.json'),'utf8'));
  const bin=await readFile(path.join(folder,'weights.bin'));
  const probe=JSON.parse(await readFile(path.join(folder,'parity.json'),'utf8'));
  let camera;
  try{camera=JSON.parse(await readFile(path.join(folder,'camera-preprocessing-parity.json'),'utf8'));}
  catch(error){if(error.code!=='ENOENT')throw error;}
  if(camera)for(const c of camera.cases){
    const actual=preprocessFrames(c.frames,c.width,c.height,camera.preprocessing);
    const expected=c.expected.flat();
    const difference=Math.max(...actual.map((value,i)=>Math.abs(value-expected[i])));
    if(difference>0.00002)throw new Error(entry.id+' camera preprocessing parity failed: '+difference);
  }
  const model=buildModel(tf,meta.classes.length);
  loadWeights(tf,model,meta,bin.buffer.slice(bin.byteOffset,bin.byteOffset+bin.byteLength));
  const x=tf.tensor(probe.inputs);
  const y=model.predict(x);
  const actual=await y.array();
  let maxError=0;
  for(let i=0;i<actual.length;i++)for(let j=0;j<actual[i].length;j++)
    maxError=Math.max(maxError,Math.abs(actual[i][j]-probe.expected[i][j]));
  if(maxError>0.0002)throw new Error(entry.id+' parity failed: '+maxError);
  console.log(entry.id+': Python / TensorFlow.js maximum probability difference '+maxError);
  x.dispose();y.dispose();model.dispose();
}
