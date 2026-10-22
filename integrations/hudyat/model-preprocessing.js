// Camera features shared with the trainer export convention.
export function normalizationFor(preprocessing={}) {
  const mode=Object.hasOwn(preprocessing,'normalization')?preprocessing.normalization:'image';
  if(!['image','hand-relative-v1','hand-relative-sequence-v1'].includes(mode))
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

export function preprocessSequence(frames, width, height, preprocessing = {}) {
  const mode=normalizationFor(preprocessing);
  if(mode==='image')return frames;
  if(mode==='hand-relative-v1')return frames.map(frame=>normalizeFrames([frame],width,height)[0]);
  return normalizeFrames(frames, width, height);
}
