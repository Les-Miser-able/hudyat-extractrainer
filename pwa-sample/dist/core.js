export function buildModel(tf, classes) {
  const l = tf.layers;
  const model = tf.sequential();
  model.add(l.conv1d({inputShape:[32,128], filters:64, kernelSize:3, padding:'same', activation:'relu'}));
  model.add(l.batchNormalization({momentum:0.99, epsilon:0.0001}));
  model.add(l.maxPooling1d({poolSize:2}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.conv1d({filters:128, kernelSize:3, padding:'same', activation:'relu'}));
  model.add(l.batchNormalization({momentum:0.99, epsilon:0.0001}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.bidirectional({layer:l.lstm({units:128, returnSequences:true,
    activation:'tanh', recurrentActivation:'sigmoid', recurrentDropout:0, dropout:0, kernelInitializer:'zeros', recurrentInitializer:'zeros'}), mergeMode:'concat'}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.bidirectional({layer:l.lstm({units:64, returnSequences:false,
    activation:'tanh', recurrentActivation:'sigmoid', recurrentDropout:0, dropout:0, kernelInitializer:'zeros', recurrentInitializer:'zeros'}), mergeMode:'concat'}));
  model.add(l.dropout({rate:0.3}));
  model.add(l.dense({units:64, activation:'relu'}));
  model.add(l.dense({units:classes, activation:'softmax'}));
  return model;
}

export function loadWeights(tf, model, metadata, buffer) {
  if (metadata.format !== 'fsl-conv1d-bilstm-v1' || metadata.inputShape.join(',') !== '32,128')
    throw new Error('This model format is unsupported.');
  const tensors = [];
  try {
    for (const spec of metadata.weights) {
      if (spec.dtype !== 'float32' || spec.offset % 4 || spec.offset + spec.bytes > buffer.byteLength)
        throw new Error('Invalid model weight data.');
      const count = spec.shape.reduce((a,b)=>a*b,1);
      if (count * 4 !== spec.bytes) throw new Error('Invalid weight shape.');
      tensors.push(tf.tensor(new Float32Array(buffer.slice(spec.offset,spec.offset+spec.bytes)),spec.shape));
    }
    // Keras exports weights per layer; TF.js model-level ordering groups trainable weights first.
    let offset=0;
    for(const layer of model.layers){
      const count=layer.weights.length;
      layer.setWeights(tensors.slice(offset,offset+count));offset+=count;
    }
    if(offset!==tensors.length)throw new Error('Unexpected model weight count.');
  } finally { tensors.forEach(t=>t.dispose()); }
}

export function packResult(result, swap=false) {
  const out = new Float32Array(128);
  const scores = [-1,-1];
  const hands = result.landmarks || [];
  const labels = result.handedness || result.handednesses || [];
  hands.forEach((points,i)=>{
    const category = labels[i]?.[0];
    const label = (category?.categoryName || category?.displayName || '').toLowerCase();
    if (points.length !== 21 || !['left','right'].includes(label)) return;
    const slot = (label === 'left' ? 0 : 1) ^ Number(swap);
    const score = category.score;
    if (score <= scores[slot] || !points.every(p=>[p.x,p.y,p.z].every(Number.isFinite))) return;
    points.forEach((p,j)=>out.set([p.x,p.y,p.z],slot*63+j*3));
    out[126+slot] = 1;
    scores[slot] = score;
  });
  return out;
}

export function resample(frames) {
  if (!frames.length || frames.some(f=>f.length!==128)) throw new Error('No valid frames captured.');
  const out = new Float32Array(32*128);
  for (let t=0;t<32;t++) {
    const position=t*(frames.length-1)/31;
    const left=Math.floor(position), right=Math.min(left+1,frames.length-1);
    const ratio=position-left, nearest=Math.floor(position+0.5);
    for(let c=0;c<126;c++) out[t*128+c]=frames[left][c]*(1-ratio)+frames[right][c]*ratio;
    for(let h=0;h<2;h++) {
      out[t*128+126+h]=frames[nearest][126+h];
      if(!out[t*128+126+h]) out.fill(0,t*128+h*63,t*128+(h+1)*63);
    }
  }
  return out;
}

export function normalizationFor(preprocessing={}) {
  const mode=Object.hasOwn(preprocessing,'normalization')?preprocessing.normalization:'image';
  if(!['image','hand-relative-sequence-v1'].includes(mode))
    throw new Error('Unsupported model preprocessing: '+mode);
  return mode;
}

export function normalizeFrames(frames,width,height) {
  if(!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)
    throw new Error('Hand-relative features require valid image dimensions.');
  if(!frames.length||frames.some(f=>f.length!==128||!Array.from(f).every(Number.isFinite)
    ||![0,1].includes(f[126])||![0,1].includes(f[127])))
    throw new Error('Invalid landmark frames.');
  const out=frames.map(()=>new Float32Array(128));
  for(let h=0;h<2;h++) {
    const offset=h*63,valid=[],palms=[];
    for(let t=0;t<frames.length;t++) {
      const f=frames[t];
      const palm=Math.hypot((f[offset+27]-f[offset])*width,(f[offset+28]-f[offset+1])*height);
      if(f[126+h]===1&&palm>=1e-6){valid.push(t);palms.push(palm);}
    }
    if(!valid.length)continue;
    palms.sort((a,b)=>a-b);
    const mid=Math.floor(palms.length/2);
    const scale=palms.length%2?palms[mid]:(palms[mid-1]+palms[mid])/2;
    const first=frames[valid[0]],origin=[first[offset]*width,first[offset+1]*height,first[offset+2]*width];
    for(const t of valid) {
      const f=frames[t],palm=Math.hypot((f[offset+27]-f[offset])*width,(f[offset+28]-f[offset+1])*height);
      for(let j=0;j<21;j++)for(let c=0;c<3;c++) {
        const unit=c===1?height:width;
        const value=j===0?(f[offset+c]*unit-origin[c])/scale
          :(f[offset+j*3+c]-f[offset+c])*unit/palm;
        out[t][offset+j*3+c]=value;
        if(!Number.isFinite(out[t][offset+j*3+c]))throw new Error('Hand-relative coordinates exceed float32 range.');
      }
      out[t][126+h]=1;
    }
  }
  return out;
}

export function preprocessFrames(frames,width,height,preprocessing={}) {
  return resample(normalizationFor(preprocessing)==='image'?frames:normalizeFrames(frames,width,height));
}

export async function sha256(buffer) {
  const hash=await crypto.subtle.digest('SHA-256',buffer);
  return [...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join('');
}

export function rank(probabilities, classes) {
  return classes.map((label,i)=>({label,probability:probabilities[i]}))
    .sort((a,b)=>b.probability-a.probability);
}
