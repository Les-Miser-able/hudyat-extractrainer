import * as tf from '@tensorflow/tfjs';
import { HandLandmarker,FilesetResolver } from '@mediapipe/tasks-vision';
import {fetchModelRelease,modelAssetUrl} from '../../api/model-api';
import {buildModel,loadWeights,packResult,resample,sha256} from './category-core.js';
export type Update={message:string;prediction?:string;score?:number;version?:string;ready?:boolean};
const cacheName='hudyat-model-files-v1';
async function artifact(url:string,hash:string,signal:AbortSignal,emit:(s:Update)=>void,label:string){
 const cache=await caches.open(cacheName),key=modelAssetUrl(url);
 const saved=await cache.match(key);
 if(saved){const bytes=await saved.arrayBuffer();if(await sha256(bytes)===hash)return bytes;await cache.delete(key);}
 emit({message:'Downloading '+label+'…'});
 const response=await fetch(key,{signal});if(!response.ok)throw Error('Model file download failed.');
 let bytes:ArrayBuffer;
 if(response.body){
  const reader=response.body.getReader(),chunks:Uint8Array[]=[];let received=0;
  const total=Number(response.headers.get('content-length'));
  while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.byteLength;
   emit({message:'Downloading '+label+'… '+(total?Math.min(100,Math.round(received/total*100))+'%':(received/1048576).toFixed(1)+' MB')});}
  const joined=new Uint8Array(received);let offset=0;for(const chunk of chunks){joined.set(chunk,offset);offset+=chunk.length;}bytes=joined.buffer;
 }else bytes=await response.arrayBuffer();
 if(await sha256(bytes)!==hash)throw Error('Model file integrity check failed.');
 await cache.put(key,new Response(bytes));return bytes;
}
export async function downloadCategory(category:string,signal:AbortSignal,emit:(s:Update)=>void){
 if(!window.isSecureContext || !('caches' in window))throw Error('Recognition requires HTTPS or localhost.');
 emit({message:'Checking category model…'});
 const release=await fetchModelRelease(category,signal);
 const metadata=JSON.parse(new TextDecoder().decode(await artifact(release.modelUrl,release.modelSha256,signal,emit,'model details')));
 if(metadata.category!==release.id || metadata.format!=='fsl-conv1d-bilstm-v1'||metadata.inputShape?.join(',')!=='32,128')throw Error('Unsupported category model.');
 const weights=await artifact(release.weightsUrl,release.weightsSha256,signal,emit,'recognition model');
 const hand=await artifact(release.handUrl,release.handSha256,signal,emit,'hand detector');
 return {release,metadata,weights,hand};
}
export async function createCategoryRuntime(category:string,signal:AbortSignal,emit:(s:Update)=>void){
 const {release,metadata,weights,hand}=await downloadCategory(category,signal,emit);
 if(signal.aborted)throw Error('Cancelled');
 emit({message:'Starting recognition engine…',version:release.version});
 await tf.ready();
 const model=buildModel(tf,metadata.classes.length);
 let detector:HandLandmarker|undefined;
 try {
  loadWeights(tf,model,metadata,weights);
  emit({message:'Starting hand tracking…',version:release.version});
  const vision=await FilesetResolver.forVisionTasks(import.meta.env.BASE_URL+'mediapipe/wasm');
  detector=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetBuffer:new Uint8Array(hand),delegate:'CPU'},runningMode:release.id==='alphabet'?'IMAGE':'VIDEO',numHands:2,
   minHandDetectionConfidence:metadata.preprocessing.extraction.detection_confidence,minHandPresenceConfidence:metadata.preprocessing.extraction.detection_confidence,minTrackingConfidence:.5});
  if(signal.aborted)throw Error('Cancelled');
  let disposed=false;
  return {version:release.version,category:release.id,classes:metadata.classes as string[],
   capture(video:HTMLVideoElement){if(disposed)throw Error('Model released');return packResult(release.id==='alphabet'?detector!.detect(video):detector!.detectForVideo(video,performance.now()),metadata.preprocessing.extraction.swap_hands);},
   async predict(frames:Float32Array[]){
    if(disposed)throw Error('Model released');
    if(!frames.some(f=>f[126]||f[127]))return {message:'No hand detected.'};
    const input=tf.tensor3d(resample(frames),[1,32,128]);let output:tf.Tensor|undefined;
    try{output=model.predict(input) as tf.Tensor;const scores=await output.data();const index=scores.indexOf(Math.max(...scores));
     return {message:scores[index]>=.6?'Model estimate — compare with the reference.':'Not sure yet. Try the sign again.',prediction:metadata.classes[index] as string,score:scores[index],margin:scores[index]-Math.max(...Array.from(scores).filter((_,i)=>i!==index))};
    }finally{input.dispose();output?.dispose();}
   },
   dispose(){if(!disposed){disposed=true;detector?.close();model.dispose();}}
  };
 }catch(error){detector?.close();model.dispose();throw error;}
}
export type CategoryRuntime=Awaited<ReturnType<typeof createCategoryRuntime>>;
