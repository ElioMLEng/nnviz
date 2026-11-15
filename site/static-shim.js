"use strict";
// Static build only (copied into site/ by export_static.py): answers the page's /api/* requests from pre-exported
// data files, so nnviz runs with no Python and no server — even opened straight from disk (file://).
(function(){
  const cache = {}, waiting = {};
  window.__nnvizData = (key, obj) => { cache[key] = obj; (waiting[key] || []).forEach(r => r(obj)); delete waiting[key]; };
  function load(key){
    if (cache[key]) return Promise.resolve(cache[key]);
    return new Promise((res, rej) => {
      (waiting[key] = waiting[key] || []).push(res);
      if (waiting[key].length > 1) return;
      const s = document.createElement("script"); s.src = `data/${key}.js`; s.async = true;
      s.onerror = () => { delete waiting[key]; rej(new Error(`missing data file data/${key}.js`)); };
      s.onload = () => s.remove();
      document.head.appendChild(s);
    });
  }
  const ok = obj => new Response(JSON.stringify(obj), {headers: {"Content-Type": "application/json"}});
  let current = null, manifest = null;
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = new URL(typeof input === "string" ? input : input.url, "http://x/");
    if (!url.pathname.startsWith("/api/")) return realFetch(input, init);
    const q = url.searchParams;
    try {
      switch (url.pathname){
        case "/api/models": { const m = await load("models"); if (!current) current = m.current; return ok({...m, current}); }
        case "/api/load": current = q.get("model"); manifest = null; return ok({ok: true});
        case "/api/graph": { const g = await load(`${current}/graph`); manifest = g.static; return ok(g); }
        case "/api/activations": {
          if (!manifest) manifest = (await load(`${current}/graph`)).static;
          const cls = Math.min(+q.get("cls") || 0, manifest.classes - 1), seed = (+q.get("seed") || 0) % manifest.seeds;
          const g = await load(`${current}/graph`);
          const idx = (g.controls || []).map((c, i) => {   // nearest recorded slider value
            const v = +q.get("p_" + c.id), vals = manifest.grids[i];
            return isNaN(v) ? vals.indexOf(c.default) : vals.reduce((b, x, j) => Math.abs(x - v) < Math.abs(vals[b] - v) ? j : b, 0);
          });
          return ok(await load(`${current}/a_${cls}_${seed}` + idx.map(k => "_" + k).join("")));
        }
        case "/api/upload": return ok({error: "Importing your own model needs the Python version of nnviz (see README)."});
      }
      return ok({error: "unknown endpoint " + url.pathname});
    } catch (e) { return ok({error: String(e.message || e)}); }
  };
  // hide what the static build can't do (live pretrained download, model import)
  const st = document.createElement("style");
  st.textContent = `label.chip:has(#pre), #upbtn, #ctl details{display:none !important}`;
  document.head.appendChild(st);
  window.NNVIZ_STATIC = true;
})();
