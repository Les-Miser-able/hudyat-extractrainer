// Build an isolated timing page from HUDYAT's real recognition sources. HUDYAT is read only.
import {readFile, writeFile, mkdir, cp} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const [project, output, session] = process.argv.slice(2);
if (!project || !output || !/^[a-f0-9]{16}$/.test(session || '')) throw Error('Usage: node build_hudyat_timing.mjs HUDYAT OUTPUT SESSION');
const client = path.resolve(project, 'client'), dest = path.resolve(output);
const feature = path.join(client, 'src/features/fsl-recognition');
const recognition = path.join(feature, 'category-recognition.tsx').replaceAll('\\','/');
const runtime = path.join(feature, 'category-runtime.ts').replaceAll('\\','/');
const sources = {};
for (const name of ['category-recognition.tsx', 'category-runtime.ts', 'category-core.js', 'gesture-lock.ts', 'model-preprocessing.js']) {
  sources[name] = createHash('sha256').update(await readFile(path.join(feature, name))).digest('hex');
}
const {build} = await import(pathToFileURL(path.join(client, 'node_modules/vite/dist/node/index.js')).href);
const originalRuntime = JSON.stringify(runtime);
const transport = '\0timing-model-transport', instrument = '\0timing-runtime';
const base = '/hudyat-timing/' + session + '/';
await mkdir(dest, {recursive:true});
await build({root:client, configFile:false, publicDir:false, cacheDir:path.join(dest, '.vite'),
  resolve:{alias:Object.fromEntries(['react', 'react-dom', '@tensorflow/tfjs', '@mediapipe/tasks-vision'].map(name=>[name,path.join(client,'node_modules',name)]))},
  define:{'import.meta.env.BASE_URL':JSON.stringify(base), 'import.meta.env.VITE_API_URL':'""', 'process.env.NODE_ENV':'"production"'},
  plugins:[{name:'trainer-timing',enforce:'pre',
    resolveId(source, importer) {
      if(source==='hudyat-recognition') return recognition;
      if(source==='./category-runtime' && importer?.replaceAll('\\','/')===recognition) return instrument;
      if(source==='../../api/model-api' && importer?.replaceAll('\\','/')===runtime) return transport;
    },
    load(id) {
      if(id===transport) return 'export const modelAssetUrl=url=>new URL(url,location.origin).href; export async function fetchModelRelease(category,signal){ const catalog=await (await fetch("'+base+'model-assets/catalog.json",{signal,cache:"no-store"})).json(); const entry=catalog.categories.find(e=>e.id===category&&e.available); if(!entry)throw Error("Model release unavailable"); return {...entry,...Object.fromEntries(["modelUrl","weightsUrl","handUrl"].map(key=>[key,new URL(entry[key],location.origin+"'+base+'model-assets/").href]))}; }';
      if(id===instrument) return 'import {createCategoryRuntime as original} from '+originalRuntime+'; export async function createCategoryRuntime(...args){ const runtime=await original(...args); return {...runtime,capture(video){const started=performance.now();const frame=runtime.capture(video);window.__hudyatTiming?.capture(started,frame);return frame;},async predict(frames){const result=await runtime.predict(frames);window.__hudyatTiming?.prediction(result);return result;}}; }';
    }}],
  build:{outDir:dest,emptyOutDir:false,minify:false,lib:{entry:path.resolve(import.meta.dirname,'../integrations/hudyat/timing-app.jsx'),formats:['es'],fileName:()=> 'timing.js'},
    rollupOptions:{output:{inlineDynamicImports:true}}}
});
const bundleText = await readFile(path.join(dest,'timing.js'),'utf8');
if (!bundleText.includes(base+'model-assets/catalog.json') || !bundleText.includes('window.__hudyatTiming?.capture') || !bundleText.includes('window.__hudyatTiming?.prediction')) {
  throw Error('HUDYAT runtime could not be instrumented. No camera session is ready. Check its recognition imports.');
}
await cp(path.join(client,'public/mediapipe/wasm'),path.join(dest,'mediapipe/wasm'),{recursive:true});
const config=JSON.parse(await readFile(path.join(dest,'session.json'),'utf8'));
config.runtime_sources=sources;
config.dependencies={};
for(const name of ['react','react-dom','@tensorflow/tfjs','@mediapipe/tasks-vision','vite']){
  config.dependencies[name]=JSON.parse(await readFile(path.join(client,'node_modules',name,'package.json'),'utf8')).version;
}
await writeFile(path.join(dest,'session.json'),JSON.stringify(config,null,2));
await writeFile(path.join(dest,'index.html'),'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HUDYAT camera timing</title><style>body{font:16px system-ui;background:#f6f7f9;color:#26302e;margin:0}main{max-width:960px;margin:auto;padding:28px}video{display:block;width:min(100%,640px);background:#17251f;border-radius:12px}button,select{font:inherit;padding:10px 14px;margin:6px;border:1px solid #bdc9c2;border-radius:6px}button{cursor:pointer}button:disabled{opacity:.5}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}.result{font-size:24px;min-height:36px}.hint{color:#626d67;line-height:1.6}.error{color:#a52d24}</style></head><body><main id="root"></main><script type="module" src="./timing.js"></script></body></html>');
console.log('Camera timing page ready: '+base);
