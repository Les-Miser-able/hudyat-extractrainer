import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import * as tf from '@tensorflow/tfjs';
import {CategoryRecognition} from 'hudyat-recognition';
import {createTimingTrial} from './camera-timing.js';

function App({config}) {
  const video = useRef(null), stream = useRef(null), tracker = useRef(null), timer = useRef(null), visibilityVersion = useRef(0);
  const [camera, setCamera] = useState(false), [ready, setReady] = useState(false), [active, setActive] = useState(false);
  const [target, setTarget] = useState(config.classes[0]), [trials, setTrials] = useState([]);
  const [status, setStatus] = useState('Start the camera to load the model.'), [result, setResult] = useState('');
  const [saving, setSaving] = useState(false), [arming, setArming] = useState(false), [error, setError] = useState('');
  const [device, setDevice] = useState(null);
  useEffect(() => {
    window.__hudyatTiming = {
      capture(started, frame) {
        const current=tracker.current;
        if(!current)return;
        const wasStarted=current.started!==null, hadFrame=current.firstFrame!==null;
        current.capture(started,frame);
        if(!hadFrame || (!wasStarted && current.started!==null)) {
          clearTimeout(timer.current);timer.current=setTimeout(()=>finish(false,true),15000);
        }
      },
      prediction(value) { tracker.current?.prediction(value); }
    };
    const hidden=()=>{if(document.hidden)visibilityVersion.current++;if(document.hidden && tracker.current){clearTimeout(timer.current);tracker.current=null;setActive(false);setError('Trial discarded because this tab was hidden. Keep it visible while measuring.');}};
    document.addEventListener('visibilitychange',hidden);
    return ()=>{delete window.__hudyatTiming;clearTimeout(timer.current);document.removeEventListener('visibilitychange',hidden);stream.current?.getTracks().forEach(t=>t.stop());};
  }, []);
  async function startCamera() {
    setError('');
    try {
      stream.current=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});
      video.current.srcObject=stream.current;await video.current.play();
      const track=stream.current.getVideoTracks()[0], settings=track.getSettings();
      setDevice({user_agent:navigator.userAgent,platform:navigator.platform,camera:track.label,
        width:settings.width || video.current.videoWidth,height:settings.height || video.current.videoHeight,
        frame_rate:settings.frameRate || null});
      setCamera(true);
    } catch(error) {setError(error.message);stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;}
  }
  async function begin() {
    if(!ready || !camera || active || arming || document.hidden)return;
    const currentStream=stream.current;setArming(true);
    try {const jobs=await (await fetch('/api/jobs',{cache:'no-store'})).json();
      if(jobs.some(job=>job.status==='running')){setError('Stop the trainer job before measuring camera timing.');return;}
    }catch(error){setError(error.message);return;}finally{setArming(false);}
    if(stream.current!==currentStream || document.hidden)return;
    clearTimeout(timer.current);tracker.current=createTimingTrial(target);setResult('');setError('');setActive(true);
  }
  function finish(correct, timeout=false) {
    const current=tracker.current;
    if(!current)return;
    tracker.current=null;clearTimeout(timer.current);setActive(false);
    setResult(timeout?'No confirmed result within 15 seconds.':correct?'Correct sign confirmed.':'Different sign confirmed.');
    const version=visibilityVersion.current;
    // Let React render the confirmation and allow a paint before stopping the clock.
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      if(document.hidden || version!==visibilityVersion.current){setError('Trial discarded because the tab was hidden before the result painted.');return;}
      try {const trial=current.complete(correct,performance.now(),timeout);setTrials(rows=>[...rows,trial]);}
      catch(error){setError(error.message);}
    }));
  }
  function stop() {
    tracker.current=null;clearTimeout(timer.current);setActive(false);setCamera(false);setReady(false);
    stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;video.current.srcObject=null;
  }
  async function save() {
    if(active || !trials.length)return;
    setSaving(true);setError('');
    try {
      const state=await (await fetch('/api/state',{cache:'no-store'})).json();
      const response=await fetch('/api/hudyat-timing',{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':state.token},
        body:JSON.stringify({session:config.session,trials,device:{...device,backend:tf.getBackend(),tfjs_version:tf.version.tfjs}})});
      const value=await response.json();if(!response.ok)throw Error(value.error);
      setStatus('Saved to this model. Benchmark status: '+value.benchmark.status+'. Refresh Model runs to view it.');
    }catch(error){setError(error.message);}finally{setSaving(false);}
  }
  const tested=new Set(trials.filter(t=>!t.timeout).map(t=>t.target));
  return <>
    <h1>HUDYAT camera-to-result timing</h1><p>{config.category} / {config.name}</p>
    <p className="hint">This page runs the recognition and confirmation code from your HUDYAT checkout with this trainer model. Start: the first camera frame with a detected hand, timed before hand extraction. End: the painted confirmed result. Recording, extraction, preprocessing, inference, confidence checks, and confirmation are included. Model downloads/loading are excluded. Keep this tab visible.</p>
    <p className="hint">Move hands out of view, choose a sign, then select Begin trial and perform it. Test every sign. Wrong confirmations and 15-second timeouts (including no hand detected) are retained. No camera images or videos are saved.</p>
    <video ref={video} muted playsInline autoPlay />
    <p role="status">{status}</p><p className="result" role="status">{result}</p>{error && <p className="error" role="alert">{error}</p>}
    <button onClick={startCamera} disabled={camera}>Start camera</button><button onClick={stop} disabled={!camera}>Stop camera</button>
    <label>Expected sign <select value={target} onChange={e=>setTarget(e.target.value)} disabled={active}>{config.classes.map(label=><option key={label}>{label}</option>)}</select></label>
    <button onClick={begin} disabled={!camera || !ready || active || arming}>Begin trial</button>
    <button onClick={()=>{tracker.current=null;clearTimeout(timer.current);setActive(false);setResult('Trial cancelled.');}} disabled={!active}>Cancel trial</button>
    <button onClick={save} disabled={active || !trials.length || saving}>Save timing results</button>
    <p>{tested.size} / {config.classes.length} signs measured · {trials.length} trials. Use the camera/device intended for evaluation. Results belong to this session/device.</p>
    <table><thead><tr><th>Expected</th><th>Confirmed</th><th>Camera-to-result</th><th>Outcome</th></tr></thead><tbody>{trials.map((row,i)=><tr key={i}><td>{row.target}</td><td>{row.predicted || '—'}</td><td>{row.response_ms===null?'No response':row.response_ms.toFixed(2)+' ms'}</td><td>{row.timeout?'Timeout':row.correct?'Correct':'Wrong sign'}</td></tr>)}</tbody></table>
    {camera && <CategoryRecognition category={config.category} targetLabel={target} video={video} active={active}
      onCorrect={()=>finish(true)} onWrong={()=>finish(false)} retry={0}
      onStatus={s=>{if(s.phase==='ready')setReady(true);if(s.phase==='error'){setError(s.message);tracker.current=null;setActive(false);clearTimeout(timer.current);}else setStatus(s.message);}} />}
  </>;
}
const config=await (await fetch('./session.json',{cache:'no-store'})).json();
createRoot(document.getElementById('root')).render(<App config={config}/>);
