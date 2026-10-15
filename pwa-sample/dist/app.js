import {HAND_EDGES,describeFrame,webcamReport} from './diagnostics.js';
import {HandLandmarker,FilesetResolver} from './vendor/vision_bundle.mjs';
import {buildModel,loadWeights,packResult,preprocessFrames,normalizationFor,sha256,rank} from './core.js';
const $=id=>document.getElementById(id);
const tf=globalThis.tf;
const MODEL_CACHE='fsl-pocket-models-v1';
const SHARED_CACHE='fsl-pocket-shared-v1';
let catalog=[],selected=null,model=null,metadata=null,stream=null,detector=null,busy=false,ready=false;
let operation=0,recording=false,installPrompt=null,swReady=null;
let live=false,lastTrial=null,trials=[];
const absolute=url=>new URL(url,document.baseURI).href;
const message=(text,error=false)=>{$('status').textContent=text;$('status').classList.toggle('error',error);};
function controls(){
  $('category').disabled=busy;
  $('download').disabled=busy || !selected?.available;
  $('remove').disabled=busy || !ready;
  $('camera').disabled=busy || !model || !!stream;
  $('pose').disabled=busy || !model || !stream;
  $('gesture').disabled=busy || !model || !stream;
  $('stop').disabled=!stream;
  $('live').disabled=busy || !model || !stream || selected?.id!=='alphabet';
  $('stop-live').disabled=!live;
  $('save-trial').disabled=!lastTrial || selected?.id!=='alphabet';
  $('camera-state').textContent=stream?'Camera on':'Camera off';
}
async function action(fn){
  if(busy)return;
  busy=true;controls();
  try{await fn();}catch(error){
    message(error.name==='NotAllowedError'?'Camera access was denied. Allow it in your browser settings to try a sign.':error.message,true);
  }finally{busy=false;$('recording').hidden=true;$('download-progress').hidden=true;controls();}
}
function clearResult(){
  $('prediction').textContent='Ready when you are.';
  $('result-note').textContent='Your result will appear here.';
  $('alternatives').replaceChildren();
}
function stopCamera(){
  operation++;recording=false;live=false;lastTrial=null;
  const overlay=$('landmarks');overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
  stream?.getTracks().forEach(track=>track.stop());stream=null;
  $('video').srcObject=null;
  $('camera-placeholder').style.display='';
  $('recording').hidden=true;controls();
}
function disposeModel(){
  stopCamera();
  detector?.close();detector=null;
  model?.dispose();model=null;metadata=null;ready=false;
}
async function availableLocally(entry){
  if(!entry?.available)return false;
  const models=await caches.open(MODEL_CACHE),shared=await caches.open(SHARED_CACHE);
  return !!(await models.match(absolute(entry.modelUrl)) && await models.match(absolute(entry.weightsUrl))
    && await shared.match(absolute(entry.handUrl)));
}
async function loadLocal(){
  const cache=await caches.open(MODEL_CACHE);
  const metaResponse=await cache.match(absolute(selected.modelUrl));
  const weightResponse=await cache.match(absolute(selected.weightsUrl));
  if(!metaResponse || !weightResponse)throw new Error('Model download is incomplete. Download it again.');
  const metaBuffer=await metaResponse.arrayBuffer(),buffer=await weightResponse.arrayBuffer();
  if(await sha256(metaBuffer)!==selected.modelSha256 || await sha256(buffer)!==selected.weightsSha256)
    throw new Error('The saved model failed its integrity check. Remove it and download again.');
  metadata=JSON.parse(new TextDecoder().decode(metaBuffer));
  if(metadata.category!==selected.id)throw new Error('The model category does not match.');
  normalizationFor(metadata.preprocessing);
  await tf.ready();
  $('expected').replaceChildren(...metadata.classes.map(label=>{const o=document.createElement('option');o.value=label;o.textContent=label;return o;}));
  const candidate=buildModel(tf,metadata.classes.length);
  try {loadWeights(tf,candidate,metadata,buffer);} catch(error){candidate.dispose();throw error;}
  model=candidate;ready=true;
  $('model-state').textContent='Saved on device';
  $('download').textContent='Download again ↓';
  message('Model ready. Turn on your camera when you’re ready.');
}
async function chooseCategory(id){
  const entry=catalog.find(c=>c.id===id);
  if(!entry)throw new Error('Unknown category.');
  disposeModel();clearResult();selected=entry;trials=[];
  $('trial-summary').textContent='No webcam trials saved in this session.';$('category').value=id;
  $('category-name').textContent=entry.name;
  $('model-state').textContent=entry.available?'Not downloaded':'Not exported yet';
  $('model-detail').textContent=entry.available?
    entry.classes+' signs · saved locally after download.':'A trained model has not been added for this category yet.';
  $('model-size').textContent=entry.available?
    (entry.bytes/1024/1024).toFixed(1)+' MB model + '+(entry.handBytes/1024/1024).toFixed(1)+' MB shared hand detector on first download.':'';
  $('download').textContent='Download model ↓';
  if(entry.available && await availableLocally(entry))await loadLocal();
  else message(entry.available?'Download this category to get started.':'Choose an available category, or export this model from your trainer.');
  controls();
}
async function fetchAndSave(url,expectedHash,cacheName){
  const cache=await caches.open(cacheName);
  const saved=await cache.match(absolute(url));
  if(saved){
    const bytes=await saved.arrayBuffer();
    if(await sha256(bytes)===expectedHash)return;
    await cache.delete(absolute(url));
  }
  const response=await fetch(url,{cache:'no-store'});
  if(!response.ok)throw new Error('Could not download '+selected.name+'. Check your connection and try again.');
  const total=Number(response.headers.get('Content-Length')||0);
  let buffer;
  if(response.body){
    const reader=response.body.getReader(),chunks=[];let loaded=0;
    $('download-progress').hidden=false;
    while(true){
      const {done,value}=await reader.read();if(done)break;
      chunks.push(value);loaded+=value.byteLength;
      if(total){$('download-progress').max=total;$('download-progress').value=loaded;}
      else $('download-progress').removeAttribute('value');
    }
    const bytes=new Uint8Array(loaded);let offset=0;
    for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
    buffer=bytes.buffer;
  }else buffer=await response.arrayBuffer();
  if(await sha256(buffer)!==expectedHash)throw new Error('Download integrity check failed. Try again.');
  await cache.put(absolute(url),new Response(buffer,{headers:{'Content-Type':response.headers.get('Content-Type')||'application/octet-stream'}}));
}
async function download(){
  if(!selected?.available)throw new Error('This category has no exported model.');
  if(!swReady)throw new Error('Offline storage requires HTTPS or localhost in a supported browser.');
  message('Saving the app for offline use…');
  await swReady;
  message('Downloading '+selected.name+' model…');
  await fetchAndSave(selected.modelUrl,selected.modelSha256,MODEL_CACHE);
  await fetchAndSave(selected.weightsUrl,selected.weightsSha256,MODEL_CACHE);
  message('Saving the shared hand detector…');
  await fetchAndSave(selected.handUrl,selected.handSha256,SHARED_CACHE);
  if(navigator.storage?.persist)await navigator.storage.persist().catch(()=>false);
  model?.dispose();model=null;
  await loadLocal();
}
async function remove(){
  const entry=selected;
  disposeModel();
  const cache=await caches.open(MODEL_CACHE);
  // Remove all saved versions of this category, leaving other categories intact.
  const prefix=absolute('./models/'+entry.id+'/');
  for(const request of await cache.keys())if(request.url.startsWith(prefix))await cache.delete(request);
  await chooseCategory(entry.id);
  message(entry.name+' model removed. Shared recognition tools remain cached.');
}
async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access needs HTTPS or localhost.');
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
  $('video').srcObject=stream;
  try{await $('video').play();}catch(error){stopCamera();throw error;}
  $('camera-placeholder').style.display='none';
  clearResult();message('Camera ready. Preview is mirrored; the model sees the original camera frame.');
}
async function createDetector(mode){
  detector?.close();detector=null;
  const shared=await caches.open(SHARED_CACHE);
  const response=await shared.match(absolute(selected.handUrl));
  if(!response)throw new Error('Hand detector is missing. Download this category again.');
  const buffer=await response.arrayBuffer();
  if(await sha256(buffer)!==selected.handSha256)throw new Error('Hand detector integrity check failed.');
  const vision=await FilesetResolver.forVisionTasks(absolute('./vendor/wasm'));
  detector=await HandLandmarker.createFromOptions(vision,{
    baseOptions:{modelAssetBuffer:new Uint8Array(buffer),delegate:'CPU'},
    runningMode:mode,numHands:2,
    minHandDetectionConfidence:metadata.preprocessing.extraction.detection_confidence,
    minHandPresenceConfidence:metadata.preprocessing.extraction.detection_confidence,
    minTrackingConfidence:0.5
  });
}
async function infer(frames){
  lastTrial=null;
  drawFrame(frames[frames.length-1]);
  if(!frames.some(f=>f[126]||f[127])){
    clearResult();$('prediction').textContent='No hand detected.';
    $('result-note').textContent='Move your hand into view and try again.';return;
  }
  message('Recognizing on this device…');
  const input=preprocessFrames(frames,$('video').videoWidth,$('video').videoHeight,metadata.preprocessing);
  if(!Array.from({length:32},(_,t)=>input[t*128+126]||input[t*128+127]).some(Boolean)){
    clearResult();$('prediction').textContent='No usable hand landmarks detected.';return;
  }
  const tensor=tf.tensor3d(input,[1,32,128]);
  let prediction;
  try{
    prediction=model.predict(tensor);
    const probabilities=await prediction.data();
    const ranked=rank(probabilities,metadata.classes);
    lastTrial={predicted:ranked[0].label,confidence:ranked[0].probability,probabilities:Array.from(probabilities),
      capturedAt:Date.now(),frames:frames.map(f=>Array.from(f)),mode:live?'live-held-pose':(frames.length===1?'held-pose':'gesture')};
    $('prediction').textContent=ranked[0].probability>=0.6?ranked[0].label:'Not sure yet.';
    $('result-note').textContent=selected.name+' · model score '+Math.round(ranked[0].probability*100)+'%'+
      (ranked[0].probability<0.6?' · try the sign again.':' · check that the sign is correct.');
    $('alternatives').replaceChildren(...ranked.slice(0,3).map(item=>{
      const li=document.createElement('li'),label=document.createElement('span'),meter=document.createElement('meter'),score=document.createElement('span');
      label.textContent=item.label;meter.min=0;meter.max=1;meter.value=item.probability;meter.setAttribute('aria-label',item.label+' model score');
      score.textContent=Math.round(item.probability*100)+'%';li.append(label,meter,score);return li;
    }));
    message('Recognition finished locally. Try another sign whenever you’re ready.');
  }finally{tensor.dispose();prediction?.dispose();}
}
async function pose(){
  const id=operation;
  message('Preparing held-pose recognition…');await createDetector('IMAGE');
  if(id!==operation || !stream)return;
  const frame=packResult(detector.detect($('video')),metadata.preprocessing.extraction.swap_hands);
  await infer([frame]);
}
async function gesture(){
  const id=operation;message('Preparing gesture recording…');await createDetector('VIDEO');
  if(id!==operation || !stream)return;
  const frames=[];let previous=-1,start=null;
  recording=true;$('recording').hidden=false;clearResult();message('Make the complete gesture now.');
  while(recording && id===operation){
    const now=await new Promise(requestAnimationFrame);
    if(!stream || document.hidden)break;
    if(start===null)start=now;
    const video=$('video');
    if(video.readyState>=2 && video.currentTime!==previous){
      previous=video.currentTime;
      frames.push(packResult(detector.detectForVideo(video,now),metadata.preprocessing.extraction.swap_hands));
    }
    if(now-start>=2000)break;
  }
  recording=false;$('recording').hidden=true;
  if(id!==operation || !stream)return;
  if(frames.length<2)throw new Error('Too few video frames were captured. Please try again.');
  await infer(frames);
}

function drawFrame(frame){
  const video=$('video'),canvas=$('landmarks');
  canvas.width=video.videoWidth||640;canvas.height=video.videoHeight||480;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  for(let h=0;h<2;h++)if(frame[126+h]){
    const point=j=>[frame[h*63+j*3]*canvas.width,frame[h*63+j*3+1]*canvas.height];
    ctx.strokeStyle=h?'#ffc857':'#55efc4';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;
    for(const [a,b] of HAND_EDGES){ctx.beginPath();ctx.moveTo(...point(a));ctx.lineTo(...point(b));ctx.stroke();}
    for(let j=0;j<21;j++){ctx.beginPath();ctx.arc(...point(j),3,0,Math.PI*2);ctx.fill();}
  }
  const hands=describeFrame(frame);
  $('hand-detail').textContent=hands.length?hands.map(h=>h.slot+' hand · '+Math.round(h.width*100)+'% frame width'+(h.clipped?' · near edge: move fully into view':'')).join(' / '):'No hand detected. Move your hand into view.';
}
async function livePoses(){
  const id=operation;
  await createDetector('IMAGE');
  if(id!==operation||!stream)return;
  live=true;controls();
  let previous=-1;
  try{
    while(live && stream && id===operation && !document.hidden){
      const began=performance.now();
      if($('video').readyState>=2 && $('video').currentTime!==previous){
        previous=$('video').currentTime;
        const frame=packResult(detector.detect($('video')),metadata.preprocessing.extraction.swap_hands);
        await infer([frame]);controls();
      }
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,200-(performance.now()-began))));
    }
  }finally{live=false;controls();}
}
$('live').addEventListener('click',()=>action(livePoses));
$('stop-live').addEventListener('click',()=>{live=false;});
$('save-trial').addEventListener('click',()=>{
  if(!lastTrial || Date.now()-lastTrial.capturedAt>5000){message('Recognize your sign again before saving a trial.',true);return;}
  trials.push({...lastTrial,actual:$('expected').value});lastTrial=null;controls();
  const report=webcamReport(trials,metadata.classes);
  $('trial-summary').textContent=report.samples+' trials · '+Math.round(report.accuracy*100)+'% raw top-1 accuracy · '+report.uncertain+' uncertain. Change pose between trials; test every letter.';
});
$('export-trials').addEventListener('click',()=>{
  if(!trials.length){message('Save at least one labeled trial first.',true);return;}
  const report={...webcamReport(trials,metadata.classes),category:selected.id,modelVersion:selected.version,preprocessing:metadata.preprocessing};
  const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='webcam-test-'+selected.id+'-'+Date.now()+'.json';a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});

$('category').addEventListener('change',()=>action(()=>chooseCategory($('category').value)));
$('download').addEventListener('click',()=>action(download));
$('remove').addEventListener('click',()=>action(remove));
$('camera').addEventListener('click',()=>action(startCamera));
$('stop').addEventListener('click',stopCamera);
$('pose').addEventListener('click',()=>action(pose));
$('gesture').addEventListener('click',()=>action(gesture));
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopCamera();});
window.addEventListener('pagehide',stopCamera);
function connection(){$('connection').textContent=navigator.onLine?'On-device recognition':'Offline · saved models available';}
window.addEventListener('online',connection);window.addEventListener('offline',connection);connection();
window.addEventListener('beforeinstallprompt',event=>{event.preventDefault();installPrompt=event;$('install').hidden=false;});
$('install').addEventListener('click',async()=>{if(installPrompt){await installPrompt.prompt();installPrompt=null;$('install').hidden=true;}});
window.addEventListener('appinstalled',()=>{$('install').hidden=true;});
async function init(){
  if(!('caches' in window) || !window.isSecureContext)throw new Error('Open this app on HTTPS or localhost for downloads and camera access.');
  if('serviceWorker' in navigator){
    // Refresh already-open pages when a corrected app version takes control.
    const hadController=!!navigator.serviceWorker.controller;
    let refreshing=false;
    navigator.serviceWorker.addEventListener('controllerchange',()=>{
      if(hadController && !refreshing){refreshing=true;location.reload();}
    });
    swReady=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('Offline setup timed out. Check your connection and reload the app.')),60000);
      navigator.serviceWorker.register('./sw.js').then(()=>navigator.serviceWorker.ready)
        .then(value=>{clearTimeout(timer);resolve(value);},error=>{clearTimeout(timer);reject(error);});
    });
    swReady.catch(error=>message('Offline setup failed: '+error.message,true));
  }
  try{await tf.setBackend('webgl');}catch{await tf.setBackend('cpu');}
  const response=await fetch('./catalog.json');
  if(!response.ok)throw new Error('Model catalog not found. Export a model first.');
  catalog=(await response.json()).categories;
  $('category').replaceChildren(...catalog.map(entry=>{
    const option=document.createElement('option');option.value=entry.id;
    option.textContent=entry.name+(entry.available?'':' — not exported');return option;
  }));
  if(!catalog.length)throw new Error('No categories are configured.');
  await chooseCategory((catalog.find(c=>c.available)||catalog[0]).id);
  const registry=document.modelContext;
  if(registry?.registerTool){
    registry.registerTool({name:'list_fsl_categories',description:'List model categories and the current selection.',
      inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},
      execute:()=>({categories:catalog.map(c=>({id:c.id,name:c.name,available:c.available})),selected:selected?.id})});
    registry.registerTool({name:'select_fsl_category',description:'Select a category without downloading or starting the camera.',
      inputSchema:{type:'object',properties:{category:{type:'string'}},required:['category'],additionalProperties:false},
      execute:async input=>{if(busy)throw new Error('An operation is in progress.');if(typeof input?.category!=='string')throw new Error('Category is required.');
        busy=true;controls();try{await chooseCategory(input.category);return {selected:selected.id,saved:ready};}finally{busy=false;controls();}}});
  }
}
action(init);
