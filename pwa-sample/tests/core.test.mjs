import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resample,packResult,rank,preprocessFrames,normalizeFrames,normalizationFor} from '../dist/core.js';
const dist=new URL('../dist/',import.meta.url);
test('old model metadata keeps image features and unknown normalization is rejected',()=>{
 const f=new Float32Array(128);f[0]=.3;f[126]=1;
 assert.deepEqual(preprocessFrames([f],0,0,{}),resample([f]));
 assert.equal(normalizationFor({}),'image');
 assert.throws(()=>normalizationFor({normalization:'unknown'}));
 assert.throws(()=>normalizationFor({normalization:null}));
});
test('hand-relative camera features preserve shape across distance and encode movement separately',()=>{
 const frame=shift=>{
  const f=new Float32Array(128);
  for(let j=0;j<21;j++){f[j*3]=.4+j*.005+shift;f[j*3+1]=.8-j*.02;f[j*3+2]=-j*.001;}
  f[126]=1;return f;
 };
 const a=frame(0),b=frame(.15);
 const normalized=normalizeFrames([a,b],640,480);
 assert.equal(normalized[0][0],0);
 assert.ok(normalized[1][0]>.9);
 for(let i=3;i<63;i++)assert.ok(Math.abs(normalized[0][i]-normalized[1][i])<1e-6);
 const small=new Float32Array(a);
 for(let j=0;j<21;j++)for(let c=0;c<3;c++)small[j*3+c]=a[c]+.5*(a[j*3+c]-a[c]);
 const expected=normalizeFrames([a],640,480)[0],actual=normalizeFrames([small],640,480)[0];
 for(let i=0;i<128;i++)assert.ok(Math.abs(expected[i]-actual[i])<1e-6);
 assert.deepEqual(Array.from(actual.slice(63,126)),Array(63).fill(0));
 const degenerate=new Float32Array(128);degenerate.fill(.3,0,63);degenerate[126]=1;
 assert.ok(normalizeFrames([degenerate],640,480)[0].every(v=>v===0));
 assert.throws(()=>normalizeFrames([a],0,480));
});
test('held pose repeats coordinates and binary presence without jitter',()=>{
 const frame=new Float32Array(128);frame[0]=.25;frame[126]=1;
 const out=resample([frame]);assert.equal(out.length,4096);
 for(let i=0;i<32;i++)assert.deepEqual(out.slice(i*128,(i+1)*128),frame);
});
test('motion is linearly interpolated to 32 frames',()=>{
 for(const count of [2,17,80]){
  const frames=Array.from({length:count},(_,i)=>{const f=new Float32Array(128);f[0]=i/(count-1);f[126]=1;return f;});
  const out=resample(frames);
  for(let i=0;i<32;i++)assert.ok(Math.abs(out[i*128]-i/31)<1e-6);
 }
});
test('absent hands stay zero and presence stays binary',()=>{
 const a=new Float32Array(128).fill(1),b=new Float32Array(128);
 const out=resample([a,b]);
 for(let i=0;i<32;i++)for(let h=0;h<2;h++){
  assert.ok([0,1].includes(out[i*128+126+h]));
  if(!out[i*128+126+h])assert.ok(out.slice(i*128+h*63,i*128+(h+1)*63).every(v=>v===0));
 }
 assert.throws(()=>resample([]));assert.throws(()=>resample([[1]]));
});
test('hand order, duplicate slots and swap follow handedness',()=>{
 const points=value=>Array.from({length:21},()=>({x:value,y:value,z:value}));
 const result={landmarks:[points(.2),points(.8),points(.5)],handedness:[
 [{categoryName:'Right',score:.9}],[{categoryName:'Left',score:.9}],[{categoryName:'Left',score:.6}]]};
 const f=packResult(result);assert.ok(Math.abs(f[0]-.8)<1e-6);assert.ok(Math.abs(f[63]-.2)<1e-6);
 assert.deepEqual([...f.slice(126)],[1,1]);
 assert.equal(packResult(result,true)[0],f[63]);
 assert.ok(packResult({landmarks:[],handedness:[]}).every(v=>v===0));
});
test('ranking retains the correct class labels',()=>{
 assert.deepEqual(rank([.1,.7,.2],['A','B','C']).map(v=>v.label),['B','C','A']);
});
test('offline shell exists and does not prefetch category models',async()=>{
 const assets=JSON.parse(await readFile(new URL('shell-assets.json',dist),'utf8'));
 for(const asset of assets){assert.ok(!asset.includes('/models/')&&!asset.includes('/shared/'));await stat(new URL(asset,dist));}
 const manifest=JSON.parse(await readFile(new URL('manifest.webmanifest',dist),'utf8'));
 for(const icon of manifest.icons)await stat(new URL(icon.src,dist));
});
test('every available category has intact independent weights and a shared detector',async()=>{
 const catalog=JSON.parse(await readFile(new URL('catalog.json',dist),'utf8'));
 for(const entry of catalog.categories.filter(e=>e.available)){
  for(const [url,hash] of [[entry.modelUrl,entry.modelSha256],[entry.weightsUrl,entry.weightsSha256],[entry.handUrl,entry.handSha256]]){
   assert.equal(createHash('sha256').update(await readFile(new URL(url,dist))).digest('hex'),hash);
  }
  const meta=JSON.parse(await readFile(new URL(entry.modelUrl,dist),'utf8'));
  assert.equal(meta.category,entry.id);assert.equal(meta.classes.length,entry.classes);
  assert.deepEqual(meta.inputShape,[32,128]);
 }
});
