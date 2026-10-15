const SHELL='fsl-pocket-shell-v5';
const MODELS='fsl-pocket-models-v1';
const SHARED='fsl-pocket-shared-v1';
self.addEventListener('install',event=>event.waitUntil((async()=>{
  const assets=await (await fetch('./shell-assets.json',{cache:'no-store'})).json();
  const cache=await caches.open(SHELL);
  await cache.addAll([...assets,'./catalog.json','./shell-assets.json'].map(url=>new Request(url,{cache:'reload'})));
  await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const name of await caches.keys())if(name.startsWith('fsl-pocket-shell-') && name!==SHELL)await caches.delete(name);
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET' || new URL(event.request.url).origin!==self.location.origin)return;
  event.respondWith((async()=>{
    const url=new URL(event.request.url);
    if(url.pathname.endsWith('/catalog.json')){
      try{const response=await fetch(event.request);if(response.ok){const cache=await caches.open(SHELL);await cache.put(event.request,response.clone());}return response;}
      catch{return await (await caches.open(SHELL)).match(event.request)||Response.error();}
    }
    for(const name of [MODELS,SHARED,SHELL]){
      const match=await (await caches.open(name)).match(event.request);
      if(match)return match;
    }
    try{return await fetch(event.request);}
    catch{
      if(event.request.mode==='navigate')return await (await caches.open(SHELL)).match('./index.html')||Response.error();
      return Response.error();
    }
  })());
});
