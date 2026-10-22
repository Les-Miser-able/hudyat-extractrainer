// Verify HUDYAT's actual TypeScript camera runtime in Node. Only the camera,
// asset transport and browser APIs are simulated; packing, preprocessing,
// resampling, model construction and inference come from the selected checkout.
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';

export async function verifyCameraRuntime(project,entry,artifacts,camera,tf,profile='baseline'){
 const client=path.resolve(project,'client');
 const require=createRequire(path.join(client,'package.json'));
 const ts=require('typescript');
 let latestInput,detectorOptions;
 const tensorRuntime={...tf,tensor3d(values,shape){
  latestInput=Array.from(values.flat?.(Infinity)??values);
  return tf.tensor3d(values,shape);
 }};
 class Video {videoWidth=640;videoHeight=480;result;}
 const payloads=new Map(['modelUrl','weightsUrl','handUrl'].map(key=>[entry[key],artifacts[key]]));
 const globals={
  window:{isSecureContext:true},HTMLVideoElement:Video,
  caches:{async open(){return {async match(url){
   const bytes=payloads.get(url);return bytes?new Response(bytes):undefined;
  }};}},
 };
 globals.window.caches=globals.caches;
 const saved=new Map(Object.keys(globals).map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 for(const [key,value] of Object.entries(globals))Object.defineProperty(globalThis,key,{value,configurable:true,writable:true});
 const modules=new Map();
 const resolve=file=>[file,file+'.ts',file+'.tsx',file+'.js'].find(existsSync);
 const load=file=>{
  file=resolve(file);
  if(!file)throw Error('Cannot resolve a HUDYAT runtime module.');
  if(modules.has(file))return modules.get(file).exports;
  const module={exports:{}};modules.set(file,module);
  const source=readFileSync(file,'utf8').replaceAll('import.meta.env.BASE_URL',"''");
  const compiled=ts.transpileModule(source,{fileName:file,compilerOptions:{
   module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,
  }}).outputText;
  const imports=specifier=>{
   if(specifier==='@tensorflow/tfjs')return tensorRuntime;
   if(specifier==='@mediapipe/tasks-vision')return {
    FilesetResolver:{async forVisionTasks(){return {};}},
    HandLandmarker:{async createFromOptions(vision,options){
     detectorOptions=options;
     return {detect:video=>video.result,detectForVideo:video=>video.result,close(){}};
    }},
   };
   if(specifier==='../../api/model-api')return {
    async fetchModelRelease(){return entry;},async fetchCandidateRelease(){return entry;},modelAssetUrl:url=>url,
   };
   return specifier.startsWith('.')?load(path.resolve(path.dirname(file),specifier)):require(specifier);
  };
  new Function('require','module','exports',compiled)(imports,module,module.exports);
  return module.exports;
 };
 let runtime;
 try{
  const folder=path.join(client,'src/features/fsl-recognition');
  const {createCategoryRuntime}=load(path.join(folder,'category-runtime.ts'));
  let options;
  if(existsSync(path.join(folder,'experiment.ts'))){
   const settings=load(path.join(folder,'experiment.ts'));
   options=settings.runtimeOptions(profile==='enhanced'?settings.ENHANCED:settings.BASELINE);
  }
  runtime=await createCategoryRuntime(entry.id,new AbortController().signal,()=>{},options);
  const metadata=JSON.parse(new TextDecoder().decode(artifacts.modelUrl));
  const expectedMode=metadata.preprocessing.normalization??'image';
  if('sequenceRelative' in runtime){
   assert.equal(runtime.sequenceRelative,expectedMode==='hand-relative-sequence-v1');
   assert.equal(runtime.handRelative,expectedMode!=='image');
  }
  assert.equal(detectorOptions.numHands,entry.id==='alphabet'?(options?.alphabetHands??2):2);
  assert.equal(detectorOptions.runningMode,entry.id==='alphabet'&&!options?.alphabetTracking?'IMAGE':'VIDEO');
  const video=new Video();
  const unpack=frame=>{
   const landmarks=[],handedness=[];
   for(let slot=0;slot<2;slot++)if(frame[126+slot]){
    const points=Array.from({length:21},(_,i)=>({x:frame[slot*63+i*3],y:frame[slot*63+i*3+1],z:frame[slot*63+i*3+2]}));
    const hand=slot^Number(Boolean(metadata.preprocessing.extraction.swap_hands));
    landmarks.push(points);handedness.push([{categoryName:hand?'Right':'Left',score:1}]);
   }
   return {landmarks,handedness};
  };
  let maxInputError=0,maxPredictionError=0,signChecks=0,predictionCases=0;
  for(const sample of camera.cases){
   video.videoWidth=sample.width;video.videoHeight=sample.height;
   const frames=sample.frames.map(frame=>{video.result=unpack(frame);return runtime.capture(video);});
   const result=await runtime.predict(frames);
   const expected=sample.expected.flat();
   assert.equal(latestInput.length,expected.length);
   let error=0;
   expected.forEach((value,i)=>{error=Math.max(error,Math.abs(value-latestInput[i]));});
   maxInputError=Math.max(maxInputError,error);
   assert.ok(error<2e-5,`HUDYAT camera runtime input differs from training: ${error}`);
   if(sample.probabilities){
    predictionCases++;
    const ranked=sample.probabilities.map((value,i)=>({value,i})).sort((a,b)=>b.value-a.value);
    assert.equal(result.prediction,metadata.classes[ranked[0].i]);
    maxPredictionError=Math.max(maxPredictionError,Math.abs(result.score-ranked[0].value),
     Math.abs(result.margin-(ranked[0].value-ranked[1].value)));
   }
   if(runtime.sequenceRelative){
    // These fixture clips are already cut at the hand window, so the sign path
    // must produce the same features without mirroring left hands or losing wrists.
    const prepared=runtime.prepareSign(frames.map(raw=>({raw})),30);
    const signInput=prepared.flatMap(row=>Array.from(row));
    assert.equal(signInput.length,expected.length);
    let signError=0;
    expected.forEach((value,i)=>{signError=Math.max(signError,Math.abs(value-signInput[i]));});
    assert.ok(signError<2e-5,`HUDYAT sign input differs from training: ${signError}`);
    await runtime.predictInput(prepared);signChecks++;
   }
  }
  assert.ok(maxPredictionError<2e-4,'HUDYAT camera prediction differs from Python: '+maxPredictionError);
  // The export must use the app's gates, never lower them to make a model pass.
  if(existsSync(path.join(folder,'gesture-lock.ts'))){
   const {DEFAULT_GATES,DEFAULT_LOCK_RULES,advanceLock,emptyLock}=load(path.join(folder,'gesture-lock.ts'));
   let lock=emptyLock();
   for(let now=0;now<=2000;now+=200)lock=advanceLock(lock,{now,present:true,label:'test',score:.7,margin:.2,steady:true});
   assert.equal(lock.locked,undefined,'A 70% prediction must not pass HUDYAT baseline gates.');
   assert.equal(DEFAULT_GATES.minScore,.8);assert.equal(DEFAULT_GATES.minMargin,.15);
   assert.equal(DEFAULT_LOCK_RULES.holdMs,1200);
  }
  return {profile,runtimeInputError:maxInputError,runtimePredictionError:predictionCases?maxPredictionError:null,
   predictionCases,signChecks,appSettings:'preserved'};
 }finally{
  runtime?.dispose();
  for(const [key,descriptor] of saved){
   if(descriptor)Object.defineProperty(globalThis,key,descriptor);else delete globalThis[key];
  }
 }
}
