import {useEffect,useEffectEvent,useRef,useState} from 'react';
import type {RefObject} from 'react';
import type {CategoryRuntime} from './category-runtime';
import type {BackgroundProcessor} from './background';
import {BASELINE,runtimeOptions,type RecognitionSettings} from './experiment';
import {advanceLock,emptyLock,frameIssue,holdProgress,isSteady,lockResetCause,matchesTarget,motionLimit,createMatchCompletion,type FrameIssue} from './gesture-lock';
import {takeRuntime} from './preload';
import {recognitionMetrics,type FrameTiming} from './metrics';
import {createSlidingWindow} from './sliding-window';
import {recommendedInput} from './sign-capture.js';
import {createTimedSegmenter} from './timed-segmenter';
import {createEarlyCheck} from './early-check';
import {longestGap,signLog,type SignDecision} from './sign-log';

type CapturedFrame={raw:Float32Array;norm:Float32Array;at:number};
/** Frames per second the camera loop is actually running at, from the last 25 frames. */
function createRateMeter(){
 const times:number[]=[];
 return {push(at:number){times.push(at);if(times.length>25)times.shift();},
  fps(){const span=times.length>4?times[times.length-1]!-times[0]!:0;return span>0?(times.length-1)*1000/span:20;}};
}

export type RecognitionStatus={phase:'loading'|'ready'|'running'|'error';message:string;version?:string};
/** EXPERIMENT: how close the current sign is to being accepted, plus a coaching hint. */
export type HoldFeedback={progress:number;hint?:string};

type HintCause=FrameIssue|'small hand';
const HINTS:Partial<Record<HintCause,string>>={'no hand':'Keep your hand in view','unsteady':'Hold still','low score':'Make the handshape clearer','low margin':'Make the handshape clearer','small hand':'Move your hand closer'};
/** Wrist-to-middle-knuckle length (frame coordinates) of the first present hand; small = far from the camera. */
function handSize(frame:Float32Array){
 for(let h=0;h<2;h++)if(frame[126+h]){const o=h*63;return Math.hypot(frame[o+27]-frame[o],frame[o+28]-frame[o+1]);}
 return 0;
}

/** Headless recognition: no predictions or overlays are rendered. */
export function CategoryRecognition({category,targetLabel,video,active,onCorrect,onWrong,onStatus,retry,settings=BASELINE,onHold}:{category:string;targetLabel?:string;video:RefObject<HTMLVideoElement|null>;active:boolean;onCorrect:()=>void;onWrong:()=>void;onStatus:(status:RecognitionStatus)=>void;retry:number;settings?:RecognitionSettings;onHold?:(hold:HoldFeedback|null)=>void}){
 const [ready,setReady]=useState(false);
 const runtime=useRef<CategoryRuntime|null>(null),inFlight=useRef(false);
 // EXPERIMENT: optional background segmentation before hand detection.
 const background=useRef<BackgroundProcessor|null>(null);
 const [backgroundReady,setBackgroundReady]=useState(false);
 const backgroundMode=settings.detectionBackground;
 const notifyCorrect=useEffectEvent(onCorrect);
 const notifyWrong=useEffectEvent(onWrong);
 const notifyStatus=useEffectEvent(onStatus);
 const notifyHold=useEffectEvent((hold:HoldFeedback|null)=>onHold?.(hold));
 // Only these settings need a different runtime; the rest apply to the running loop.
 const {alphabetTracking,delegate,alphabetHands,alphabetModel}=settings;
 useEffect(()=>{
  const abort=new AbortController();
  const taken=takeRuntime(category,runtimeOptions({...BASELINE,alphabetTracking,delegate,alphabetHands,alphabetModel}),s=>{if(!abort.signal.aborted)notifyStatus({phase:'loading',message:s.message,version:s.version});});
  if(taken.preloaded)notifyStatus({phase:'loading',message:'Starting recognition engine…'});
  void taken.promise
   .then(value=>{if(abort.signal.aborted)return;runtime.current=value;recognitionMetrics.runtime({delegate:value.delegate,preloaded:taken.preloaded,model:value.version});setReady(true);notifyStatus({phase:'ready',message:'Model ready',version:value.version});})
   .catch(error=>{if(!abort.signal.aborted)notifyStatus({phase:'error',message:error instanceof Error?error.message:'Could not load the model.'});});
  // A preloading page keeps the runtime for the next camera step; otherwise release disposes it.
  return ()=>{abort.abort();runtime.current=null;setReady(false);const release=()=>{if(inFlight.current)setTimeout(release,20);else taken.release();};release();};
 },[category,retry,alphabetTracking,delegate,alphabetHands,alphabetModel]);
 // EXPERIMENT metrics: time the camera was live before recognition could start.
 const waitingSince=useRef<number|undefined>(undefined);
 useEffect(()=>{
  if(!active){waitingSince.current=undefined;return;}
  if(!ready){waitingSince.current??=performance.now();return;}
  recognitionMetrics.modelWait(waitingSince.current===undefined?0:performance.now()-waitingSince.current);
  waitingSince.current=undefined;
 },[active,ready]);
 useEffect(()=>{
  if(backgroundMode==='off')return;
  let cancelled=false,loaded:BackgroundProcessor|undefined;
  void import('./background').then(m=>m.createBackgroundProcessor())
   .then(value=>{loaded=value;if(cancelled){value.close();return;}background.current=value;setBackgroundReady(true);})
   .catch(error=>{if(!cancelled)notifyStatus({phase:'error',message:error instanceof Error?error.message:'Could not start background processing.'});});
  return ()=>{cancelled=true;background.current=null;setBackgroundReady(false);const release=()=>{if(inFlight.current)setTimeout(release,20);else loaded?.close();};release();};
 },[backgroundMode,retry]);
 useEffect(()=>{
  if(!active||!ready||!targetLabel||(backgroundMode!=='off'&&!backgroundReady))return;
  let cancelled=false,timer:ReturnType<typeof setTimeout>;let frames:Float32Array[]=[],start=0,lastTime=-1;
  let lock=emptyLock(),previous:Float32Array|undefined;let previousProcessingMs=0;
  let hint:{text:string;at:number}|undefined,lastProgress=0;
  const feedback=(progress:number,cause:HintCause|undefined,now:number)=>{
   if(!settings.holdFeedback)return;
   const text=cause&&HINTS[cause];
   if(text)hint={text,at:now};
   // A hint fades after 1.5s, or once the sign starts progressing again.
   else if(hint&&(now-hint.at>1500||progress>lastProgress))hint=undefined;
   lastProgress=progress;
   notifyHold({progress,hint:hint?.text});
  };
  const issueFor=(frame:Float32Array,present:boolean,issue?:FrameIssue):HintCause|undefined=>
   !present?'no hand':issue??(handSize(frame)<.08?'small hand':undefined);
  const newWindow=()=>settings.dynamicCapture==='sliding'?createSlidingWindow(settings.window):undefined;
  let sliding=newWindow();
  // DABEST sign capture: start and end of a sign in real time; the rate meter is only a fallback frame rate.
  const rate=createRateMeter();
  const newSegmenter=()=>createTimedSegmenter<CapturedFrame>({endAbsentS:settings.endAbsentMs/1000});
  let segmenter=newSegmenter(),signStart=0;
  const newEarly=()=>createEarlyCheck<CapturedFrame>(undefined,settings.earlyMinHandsMs);
  let early=newEarly(),agreed=0;
  recognitionMetrics.prompt(targetLabel,performance.now());
  const completion=createMatchCompletion(targetLabel,
   ()=>{recognitionMetrics.correct(targetLabel,performance.now());notifyCorrect();},
   label=>{recognitionMetrics.wrong(targetLabel,label);notifyWrong();});
  const run=async()=>{
   const current=runtime.current,element=video.current;
   let ownsInference=false,timing:FrameTiming|undefined;
   try{
    if(document.hidden){lock=emptyLock();previous=undefined;frames=[];start=0;segmenter=newSegmenter();early=newEarly();agreed=0;}
    else if(!cancelled&&current&&element&&element.readyState>=2&&!inFlight.current&&element.currentTime!==lastTime){
     lastTime=element.currentTime;inFlight.current=true;ownsInference=true;
     if(!current.classes.some(label=>matchesTarget(label,targetLabel))){notifyStatus({phase:'error',message:'This sign is not in the downloaded model.',version:current.version});return;}
     const frameStarted=performance.now();
     const processed=backgroundMode!=='off'&&background.current?background.current.process(element,backgroundMode):undefined;
     const captureStarted=performance.now();
     const frame=current.capture(processed?.canvas??element),present=Boolean(frame[126]||frame[127]);
     const frameTiming:FrameTiming={at:frameStarted,segmentMs:processed?.segmentMs??0,captureMs:performance.now()-captureStarted};timing=frameTiming;
     completion.observePresence(present,performance.now());
     if(present)recognitionMetrics.handSeen(targetLabel,frameStarted);
     const timed=async<T,>(work:()=>Promise<T>)=>{const t=performance.now();const result=await work();frameTiming.predictMs=performance.now()-t;return result;};
     const predict=(input:Float32Array[])=>timed(()=>current.predict(input));
     if(current.category==='alphabet'){
      const result=present?await predict([frame]):undefined;
      if(cancelled)return;
      const lockInput={now:performance.now(),present,label:result?.prediction,score:result?.score,margin:result?.margin,steady:isSteady(previous,frame,motionLimit(settings.lock)),maxGapMs:Math.max(600,previousProcessingMs+(performance.now()-frameStarted)+350)};
      const before=lock;lock=advanceLock(lock,lockInput,settings.lock);
      const resetCause=lockResetCause(before,lock,lockInput,settings.lock);if(resetCause)recognitionMetrics.lockReset(resetCause);
      feedback(holdProgress(lock,settings.lock,lockInput.now),issueFor(frame,present,frameIssue(lockInput,settings.lock)),lockInput.now);
      previous=frame;previousProcessingMs=performance.now()-frameStarted;
      notifyStatus({phase:'running',message:present?'Recognition running · hold your sign steady':'Recognition running · no hand detected',version:current.version});
      if(lock.locked){completion.accept(lock.locked);lock=emptyLock();}
     }else{
      notifyStatus({phase:'running',message:present?'Recognition running · checking gesture':'Recognition running · no hand detected',version:current.version});
      if(settings.dynamicCapture==='sign'&&current.handRelative){
      // The sign ends when no hand has been seen for 0.5 s (or after 3 s); then it is cut and squeezed like training.
      const now=performance.now();rate.push(now);const fps=rate.fps();
      const wasSigning=segmenter.signing;
      const item={raw:frame,norm:current.normalize(frame),at:now};
      const ended=segmenter.push(item,present,now);
      if(segmenter.signing&&!wasSigning)signStart=now;
      feedback(segmenter.signing?Math.min(1,(now-signStart)/3000):0,issueFor(frame,present),now);
      const check=async(items:CapturedFrame[])=>{
       const span=(items[items.length-1]!.at-items[0]!.at)/1000;
       const {width,height}=current.frameSize(frame);
       const {input}=recommendedInput(items,items.length>1&&span>0?(items.length-1)/span:fps,width,height,'auto');
       const result=await timed(()=>current.predictInput(input));
       const label=result.prediction && (result.score??0)>=.8 && (result.margin??0)>=.15?result.prediction:undefined;
       // Diagnostics for the Recognition lab: why and when this answer came.
       const hands=items.map(it=>Boolean(it.raw[126]||it.raw[127]));
       const firstHand=items[Math.max(0,hands.indexOf(true))]!.at;
       const log=(reason:SignDecision['reason'],accepted:boolean)=>signLog.add({at:now,reason,target:targetLabel,
        signMs:now-firstHand,longestGapMs:longestGap(items,hands),prediction:result.prediction,score:result.score,
        margin:result.margin,accepted});
       return {label,log};
      };
      if(ended){
       early=newEarly();agreed=0;
       const {label,log}=await check(ended.items);
       if(cancelled)return;
       log(ended.reason,Boolean(label));
       if(label)completion.accept(label);
      }else{
       // A sign held steady: check it so far; only a confident match with the target is accepted before it ends.
       const sofar=settings.earlyCheck?early.push(item,present,segmenter.signing):null;
       if(!segmenter.signing)agreed=0;
       if(sofar){
        const {label,log}=await check(sofar);
        if(cancelled)return;
        // Accepted only after the same confident answer on settings.earlyAgree checks in a row.
        agreed=label&&matchesTarget(label,targetLabel)?agreed+1:0;
        const accept=Boolean(label&&agreed>=settings.earlyAgree);
        if(label&&matchesTarget(label,targetLabel))log('early',accept);
        if(accept)completion.accept(label!);
       }
      }
      }else if(sliding){
      const clip=sliding.push(frame,performance.now());
      feedback(sliding.progress(performance.now()),issueFor(frame,present),performance.now());
      if(clip){
       const result=await predict(clip);
       if(cancelled)return;
       const accepted=sliding.vote(result);
       if(accepted)completion.accept(accepted);
      }
      }else{
      // Start a complete two-second clip when a hand enters view; retry automatically.
      feedback(start?Math.min(1,(performance.now()-start)/2000):0,issueFor(frame,present),performance.now());
      if(!start && !present)return;
      if(!start)start=performance.now();frames.push(frame);
      if(performance.now()-start>=2000){
       const result=frames.length>=8?await predict(frames):undefined;frames=[];start=0;
       if(cancelled)return;
       if(result?.prediction && (result.score??0)>=.8 && (result.margin??0)>=.15)completion.accept(result.prediction);
      }
      }
     }
    }
   }catch(error){if(!cancelled){lock=emptyLock();frames=[];start=0;sliding=newWindow();segmenter=newSegmenter();early=newEarly();agreed=0;notifyStatus({phase:'error',message:error instanceof Error?error.message:'Recognition failed.'});}}
   finally{if(timing&&!cancelled)recognitionMetrics.frame(timing);if(ownsInference)inFlight.current=false;if(!cancelled&&!completion.completed())timer=setTimeout(run,category==='alphabet'?settings.alphabetPollMs:40);}
  };
  void run();return()=>{cancelled=true;clearTimeout(timer);notifyHold(null);};
 },[active,ready,category,targetLabel,video,retry,settings,backgroundMode,backgroundReady]);
 return null;
}
