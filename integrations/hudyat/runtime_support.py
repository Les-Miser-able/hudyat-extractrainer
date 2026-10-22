"""Prepare the small HUDYAT camera compatibility update before writing files."""
import json
import os
from pathlib import Path
import re
import tempfile

SUPPORT = Path(__file__).resolve().parent


def inside(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Import paths must stay inside the selected HUDYAT folder.')
    return path


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.hudyat-import-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('HUDYAT camera code has changed. Review category-runtime.ts before applying this update.')
    return source.replace(old, new, 1)


def current_runtime(source):
    """Adapt model inputs while keeping HUDYAT's detector options and UI rules."""
    marker = '// FSL trainer sequence preprocessing adapter v2'
    if marker in source:
        if all(value in source for value in ('const modelFrames=prepareFrames(frames);',
                                             'prepareSign(items:', 'sequenceRelative',
                                             "from './model-preprocessing.js'")):
            return source
        raise ValueError('The HUDYAT trainer preprocessing integration has changed. Review it before importing.')
    source = replace_once(source,
        "import {HAND_RELATIVE,normalizeFrame} from './normalize';",
        "import {HAND_RELATIVE,normalizeFrame} from './normalize';\n"
        "import {normalizationFor,preprocessSequence} from './model-preprocessing.js';\n"
        "import {signWindow} from './sign-capture.js';")
    source = replace_once(source,
        'const handRelative=metadata.preprocessing?.normalization===HAND_RELATIVE;',
        marker + "\n const normalization=normalizationFor(metadata.preprocessing);\n"
        " const sequenceRelative=normalization==='hand-relative-sequence-v1';\n"
        " const handRelative=sequenceRelative||normalization===HAND_RELATIVE;")
    source = replace_once(source,
        'const frameSizes=new WeakMap<Float32Array,{width:number;height:number}>();',
        '''const frameSizes=new WeakMap<Float32Array,{width:number;height:number}>();
 const prepareFrames=(frames:Float32Array[])=>{
  if(sequenceRelative){
   const size=frameSizes.get(frames[0]);
   if(!size||frames.some(frame=>{const s=frameSizes.get(frame);return !s||s.width!==size.width||s.height!==size.height;}))
    throw Error('Capture a complete gesture at one camera resolution.');
   return preprocessSequence(frames,size.width,size.height,metadata.preprocessing);
  }
  return handRelative?frames.map(frame=>{const size=frameSizes.get(frame)??{width:0,height:0};return normalizeFrame(frame,size.width,size.height);}):frames;
 };''')
    source = replace_once(source,
        'const modelFrames=handRelative?frames.map(frame=>{const size=frameSizes.get(frame)??{width:0,height:0};return normalizeFrame(frame,size.width,size.height);}):frames;',
        'const modelFrames=prepareFrames(frames);')
    source = replace_once(source,
        'classes:metadata.classes as string[],delegate,handRelative,',
        'classes:metadata.classes as string[],delegate,handRelative,sequenceRelative,')
    source = replace_once(source, 'if(!handRelative)return frame;',
                          'if(!handRelative||sequenceRelative)return frame;')
    source = replace_once(source, 'async predictInput(rows:ArrayLike<number>[]){', '''prepareSign(items:{raw:Float32Array}[],fps:number){
    if(!items.length)throw Error('No gesture frames captured.');
    // Keep HUDYAT's hand-window trimming; sequence-trained models retain anatomical
    // slots and wrist trajectories instead of the pose model's automatic mirroring.
    const [start,end]=signWindow(items.map(item=>Boolean(item.raw[126]||item.raw[127])),fps);
    const frames=items.slice(start,end+1).map(item=>item.raw);
    const flat=resample(prepareFrames(frames));
    return Array.from({length:32},(_,i)=>flat.subarray(i*128,(i+1)*128));
   },
   async predictInput(rows:ArrayLike<number>[]){''')
    return source


def current_recognition(source):
    marker = '// FSL trainer: prepare sequence features using the model contract.'
    if marker in source and 'current.prepareSign(items,captureFps)' in source:
        return source
    return replace_once(source,
        "const {input}=recommendedInput(items,items.length>1&&span>0?(items.length-1)/span:fps,width,height,'auto');",
        marker + '\n       const captureFps=items.length>1&&span>0?(items.length-1)/span:fps;\n'
        '       const input=current.sequenceRelative?current.prepareSign(items,captureFps)\n'
        "        :recommendedInput(items,captureFps,width,height,'auto').input;")


def runtime_changes(project):
    runtime = inside(project, 'client/src/features/fsl-recognition/category-runtime.ts')
    source = runtime.read_text(encoding='utf-8')
    package_path = inside(project, 'client/package.json')
    package = json.loads(package_path.read_text(encoding='utf-8'))
    changes = {}
    if ('const frameSizes=new WeakMap' in source
            and 'async predictInput(rows:ArrayLike<number>[]){' in source):
        source = current_runtime(source)
        recognition = inside(project, 'client/src/features/fsl-recognition/category-recognition.tsx')
        changes[recognition] = current_recognition(recognition.read_text(encoding='utf-8')).encode('utf-8')
    elif "from './model-preprocessing.js'" not in source:
        replacements = [
            ("import {buildModel,loadWeights,packResult,resample,sha256} from './category-core.js';",
             "import {buildModel,loadWeights,packResult,resample,sha256} from './category-core.js';\n"
             "import {normalizationFor,preprocessSequence} from './model-preprocessing.js';"),
            ('const model=buildModel(tf,metadata.classes.length);',
             'normalizationFor(metadata.preprocessing);\n const model=buildModel(tf,metadata.classes.length);'),
            ('let disposed=false;', 'let disposed=false;\n  let frameWidth=0,frameHeight=0;'),
            ('return packResult(release.id',
             'frameWidth=video.videoWidth;frameHeight=video.videoHeight;return packResult(release.id'),
            ('resample(frames),[1,32,128]',
             'resample(preprocessSequence(frames,frameWidth,frameHeight,metadata.preprocessing)),[1,32,128]'),
        ]
        for old, new in replacements:
            source = replace_once(source, old, new)
    elif 'resample(preprocessSequence(frames,frameWidth,frameHeight,metadata.preprocessing))' not in source:
        raise ValueError('HUDYAT has a different preprocessing integration. Review it before importing.')
    folder = 'client/src/features/fsl-recognition/'
    for name in ('model-preprocessing.js', 'model-preprocessing.d.ts'):
        changes[inside(project, folder + name)] = (SUPPORT / name).read_bytes()
    test = (SUPPORT / 'model-preprocessing.test.mjs').read_text(encoding='utf-8').replace(
        "from './model-preprocessing.js'", "from '../src/features/fsl-recognition/model-preprocessing.js'")
    changes[inside(project, 'client/tests/model-preprocessing.test.mjs')] = test.encode('utf-8')
    test_path = 'tests/model-preprocessing.test.mjs'
    current = package.get('scripts', {}).get('test', '')
    if test_path not in current:
        if not re.match(r'^node(?:\s+--[\w=-]+)*\s+--test(?:\s|$)', current):
            raise ValueError('HUDYAT uses a different test command. Review client/package.json before importing.')
        package['scripts']['test'] = current + ' ' + test_path
        changes[package_path] = (json.dumps(package, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    changes[runtime] = source.encode('utf-8')
    return changes


def apply_changes(changes):
    for path, payload in changes.items():
        if not path.is_file() or path.read_bytes() != payload:
            atomic_write(path, payload)
            print('Updated ' + str(path))
