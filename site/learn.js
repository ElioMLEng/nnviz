"use strict";
// "Explain NNs": a short course opened from the panel header (button, L key, or ?learn=1 / ?learn=<chapter number>).
// Add chapters by appending to LEARN; `part` groups them in the sidebar.
(function(){
const $ = id => document.getElementById(id);
const C = {acc:"#5cdcff", pink:"#ff4fa3", green:"#66bb6a", amber:"#ffca28", violet:"#ab47bc", mut:"#8a8a96", line:"#3a3a48", fg:"#e4e4ea", box:"#14141b"};
const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c || C.fg}" font-size="${o.s || 12}" text-anchor="${o.a || "middle"}" font-family="system-ui,Segoe UI,sans-serif"${o.w ? ` font-weight="${o.w}"` : ""}>${s}</text>`;
const R = (x, y, w, h, o = {}) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${o.r ?? 8}" fill="${o.f || C.box}" stroke="${o.s || C.line}" stroke-width="${o.sw || 1.2}"/>`;
const ARROW = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.mut}"/></marker>
  <marker id="aha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.pink}"/></marker></defs>`;
const L = (x1, y1, x2, y2, o = {}) => `<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${o.c || C.mut}" stroke-width="${o.w || 1.4}" fill="none"${o.d ? ` stroke-dasharray="${o.d}"` : ""} marker-end="url(#${o.m || "ah"})"/>`;
const fig = (svg, cap, vb) => `<figure><svg viewBox="${vb}" role="img" aria-label="${cap}">${ARROW}${svg}</svg><figcaption>${cap}</figcaption></figure>`;

// ---------- diagrams ----------
const figML = fig(
  R(10, 40, 110, 56) + T(65, 64, "Data", {w: 600}) + T(65, 82, "inputs x, targets y", {c: C.mut, s: 10.5}) +
  R(160, 40, 120, 56, {s: C.acc}) + T(220, 64, "Model", {w: 600, c: C.acc}) + T(220, 82, "f(x; θ)", {c: C.mut, s: 11}) +
  R(320, 40, 110, 56) + T(375, 64, "Prediction", {w: 600}) + T(375, 82, "ŷ = f(x; θ)", {c: C.mut, s: 11}) +
  R(470, 40, 120, 56, {s: C.pink}) + T(530, 64, "Loss", {w: 600, c: C.pink}) + T(530, 82, "how wrong: L(ŷ, y)", {c: C.mut, s: 10.5}) +
  L(120, 68, 158, 68) + L(280, 68, 318, 68) + L(430, 68, 468, 68) +
  `<path d="M530 98 C530 138 220 138 220 100" stroke="${C.pink}" stroke-width="1.4" fill="none" stroke-dasharray="5 4" marker-end="url(#aha)"/>` +
  T(375, 140, "adjust θ to lower the loss (gradient descent), repeat", {c: C.pink, s: 11}),
  "Training loop: predict, measure the error, nudge the parameters, repeat.", "0 0 600 150");

const figDL = fig((() => {
  const st = [["Pixels", "raw numbers", C.mut], ["Edges", "layer 1–2", C.acc], ["Textures", "layers 3–5", C.acc], ["Parts", "deeper layers", C.violet], ["“Cat”", "output", C.green]];
  return st.map(([a, b, c], i) => R(8 + i*120, 26, 100, 52, {s: c}) + T(58 + i*120, 50, a, {w: 600, c}) + T(58 + i*120, 67, b, {c: C.mut, s: 10.5}) +
    (i < st.length - 1 ? L(108 + i*120, 52, 126 + i*120, 52) : "")).join("") +
    T(300, 104, "each layer re-describes the previous one, more abstractly", {c: C.mut, s: 11});
})(), "Deep learning: a stack of layers that learns its own features, from simple to abstract.", "0 0 600 112");

const figNeuron = fig(
  [0, 1, 2].map(i => `<circle cx="40" cy="${30 + i*45}" r="15" fill="${C.box}" stroke="${C.acc}" stroke-width="1.3"/>` + T(40, 34 + i*45, `x${"₁₂₃"[i]}`) +
    L(56, 30 + i*45, 222, 75 + (i - 1)*10, {c: [C.acc, C.pink, C.acc][i], w: [2.2, 1.4, 1][i]}) + T(130, 38 + i*36, `w${"₁₂₃"[i]}`, {c: [C.acc, C.pink, C.acc][i], s: 11})).join("") +
  `<circle cx="250" cy="75" r="28" fill="${C.box}" stroke="${C.fg}" stroke-width="1.4"/>` + T(250, 80, "Σ + b", {s: 13, w: 600}) +
  L(278, 75, 328, 75) + R(330, 55, 74, 40, {s: C.amber}) + T(367, 80, "σ( · )", {c: C.amber, w: 600}) + L(404, 75, 456, 75) +
  `<circle cx="474" cy="75" r="16" fill="${C.box}" stroke="${C.green}" stroke-width="1.4"/>` + T(474, 79, "y", {c: C.green, w: 600}) +
  T(250, 132, "weighted sum", {c: C.mut, s: 10.5}) + T(367, 132, "non-linearity", {c: C.mut, s: 10.5}) + T(474, 132, "output", {c: C.mut, s: 10.5}),
  "One artificial neuron: y = σ(w₁x₁ + w₂x₂ + w₃x₃ + b). Line thickness = weight size, pink = negative weight.", "0 0 520 145");

const figTensor = fig((() => {
  const cell = (x, y, f) => `<rect x="${x}" y="${y}" width="16" height="16" rx="2" fill="${f}" stroke="#0a0a0e"/>`;
  const shade = (i) => ["#1d4a5c", "#2b6a86", "#3d8fb0", "#5cdcff"][i % 4];
  let s = cell(40, 60, shade(3)) + T(48, 110, "Scalar", {w: 600}) + T(48, 126, "rank 0 · []", {c: C.mut, s: 10.5});
  for (let i = 0; i < 4; i++) s += cell(120 + i*17, 60, shade(i));
  s += T(154, 110, "Vector", {w: 600}) + T(154, 126, "rank 1 · [4]", {c: C.mut, s: 10.5});
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) s += cell(240 + c*17, 43 + r*17, shade(r + c));
  s += T(274, 110, "Matrix", {w: 600}) + T(274, 126, "rank 2 · [3, 4]", {c: C.mut, s: 10.5});
  for (let d = 2; d >= 0; d--) for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) s += cell(372 + c*17 + d*9, 43 + r*17 - d*9, shade(r + c + d));
  s += T(412, 110, "3-D tensor", {w: 600}) + T(412, 126, "rank 3 · [3, 3, 4]", {c: C.mut, s: 10.5});
  s += T(530, 62, "Image batch", {w: 600, s: 11}) + T(530, 78, "[N, C, H, W]", {c: C.acc, s: 11}) + T(530, 94, "rank 4", {c: C.mut, s: 10.5});
  return s;
})(), "A tensor is an n-dimensional array. Its shape lists the size of each axis.", "0 0 600 136");

const figGPU = fig((() => {
  let s = R(10, 20, 250, 130) + T(135, 14, "CPU · a few powerful cores", {c: C.fg, s: 11.5, w: 600});
  for (let i = 0; i < 4; i++) s += R(22 + (i % 2)*60, 32 + Math.floor(i/2)*55, 52, 47, {r: 5, s: C.amber}) + T(48 + (i % 2)*60, 60 + Math.floor(i/2)*55, "core", {c: C.amber, s: 10.5});
  s += R(150, 32, 98, 102, {r: 5, s: C.line, f: "#1a1a22"}) + T(199, 80, "large cache", {c: C.mut, s: 10.5}) + T(199, 95, "+ control", {c: C.mut, s: 10.5});
  s += R(300, 20, 290, 130) + T(445, 14, "GPU · thousands of simple cores", {c: C.fg, s: 11.5, w: 600});
  for (let r = 0; r < 8; r++) for (let c = 0; c < 22; c++) s += `<rect x="${310 + c*12.4}" y="${32 + r*14.5}" width="9.5" height="11" rx="1.5" fill="${(r + c) % 7 === 0 ? C.green : "#1d5a44"}"/>`;
  return s;
})(), "CPUs are built for fast sequential work with branching; GPUs run the same operation on huge amounts of data in parallel.", "0 0 600 158");

const figMatmul = fig((() => {
  const grid = (x, y, rows, cols, hl, col) => { let s = "";
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
      s += `<rect x="${x + c*15}" y="${y + r*15}" width="13" height="13" rx="2" fill="${hl(r, c) ? col : "#1c1c26"}"/>`; return s; };
  return grid(20, 30, 4, 6, r => r === 1, C.acc) + T(65, 108, "A · [M=4, K=6]", {c: C.mut, s: 11}) + T(130, 64, "×", {s: 18}) +
    grid(150, 15, 6, 3, (r, c) => c === 2, C.pink) + T(172, 124, "B · [K=6, N=3]", {c: C.mut, s: 11}) + T(225, 64, "=", {s: 18}) +
    grid(245, 30, 4, 3, (r, c) => r === 1 && c === 2, C.green) + T(267, 108, "C · [M=4, N=3]", {c: C.mut, s: 11}) +
    T(330, 50, "one output = row · column", {a: "start", s: 12}) + T(330, 68, "= K multiply-adds = 2K FLOPs", {a: "start", c: C.mut, s: 11.5}) +
    T(330, 92, "all of C = 2 · M · K · N FLOPs", {a: "start", c: C.green, s: 12, w: 600});
})(), "Matrix multiplication: the operation behind linear layers, convolutions and attention.", "0 0 560 132");

// ---------- building blocks for the layer chapters ----------
const glance = (what, analogy, where) => `<div class="glance"><div><b>Purpose</b>${what}</div><div><b>Analogy</b>${analogy}</div><div><b>Found in</b>${where}</div></div>`;
const procon = (pros, cons) => `<div class="pc"><div class="pro"><b>Strengths</b><ul>${pros.map(p => `<li>${p}</li>`).join("")}</ul></div>` +
  `<div class="con"><b>Limits</b><ul>${cons.map(p => `<li>${p}</li>`).join("")}</ul></div></div>`;
const uses = items => `<div class="uses">${items.map(([i, t, d]) => `<div><i aria-hidden="true">${i}</i><span><b>${t}.</b> ${d}</span></div>`).join("")}</div>`;
const card = (k, name, sub, what, ana, where, fx) => `<div class="card" style="--k:${k}"><h5><span></span>${name}</h5>` +
  `<div style="color:var(--mut);font-size:11px;margin-top:-2px">${sub}</div><p>${what}</p>` +
  (fx ? `<div class="fx sm">${fx.map(l => `<div>${l}</div>`).join("")}</div>` : "") +
  `<p class="ana">${ana}</p><div class="where"><b>Where:</b> ${where}</div></div>`;
// formula block: equation lines, then a key explaining every symbol
const formula = (lines, key, note) => `<h4>Formula</h4><div class="fx">${lines.map(l => `<div>${l}</div>`).join("")}</div>` +
  `<table class="sym">${key.map(([s, d]) => `<tr><td>${s}</td><td>${d}</td></tr>`).join("")}</table>` + (note ? `<p class="fnote">${note}</p>` : "");

const figFC = fig((() => {
  const ins = [0, 1, 2, 3, 4], outs = [0, 1, 2], X0 = 150, X1 = 400, yi = i => 22 + i*28, yo = i => 36 + i*35;
  const w = [[.9, -.3, .5, .1, -.7], [-.2, .8, -.6, .4, .3], [.5, .2, .1, -.9, .6]];
  let s = "";
  outs.forEach(o => ins.forEach(i => { const v = w[o][i];
    s += `<line x1="${X0}" y1="${yi(i)}" x2="${X1}" y2="${yo(o)}" stroke="${v > 0 ? "#e4e4ea" : C.pink}" stroke-opacity="${.25 + Math.abs(v)*.6}" stroke-width="${.6 + Math.abs(v)*2.2}"/>`; }));
  ins.forEach(i => s += `<circle cx="${X0}" cy="${yi(i)}" r="9" fill="${C.box}" stroke="${C.acc}" stroke-width="1.3"/>`);
  outs.forEach(o => s += `<circle cx="${X1}" cy="${yo(o)}" r="11" fill="${C.box}" stroke="${C.green}" stroke-width="1.4"/>`);
  return s + T(X0, 160, "inputs x · [5]", {c: C.acc, s: 11}) + T(X1, 160, "outputs y · [3]", {c: C.green, s: 11}) +
    T(275, 160, "W · [3, 5] = 15 weights", {c: C.mut, s: 11}) + T(40, 85, "every", {c: C.mut, s: 11}) + T(40, 100, "input", {c: C.mut, s: 11}) +
    T(510, 85, "→ every", {c: C.mut, s: 11}) + T(510, 100, "output", {c: C.mut, s: 11});
})(), "Fully connected: 5 inputs × 3 outputs = 15 connections, each with its own learned weight (white = positive, pink = negative).", "0 0 560 170");

const figConv = fig((() => {
  const img = [[0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,1,1,1],[0,0,0,1,1,1,1]];
  const ker = [[-1,0,1],[-1,0,1],[-1,0,1]];
  let s = "";
  img.forEach((row, r) => row.forEach((v, c) => s += `<rect x="${20 + c*18}" y="${20 + r*18}" width="17" height="17" rx="2" fill="${v ? "#cfd8dc" : "#263238"}"/>`));
  s += `<rect x="${20 + 1*18 - 1.5}" y="${20 + 2*18 - 1.5}" width="${3*18 + 2}" height="${3*18 + 2}" rx="3" fill="none" stroke="${C.acc}" stroke-width="2.5"/>`;
  s += T(83, 160, "input image (vertical edge)", {c: C.mut, s: 11});
  ker.forEach((row, r) => row.forEach((v, c) => s += `<rect x="${215 + c*22}" y="${45 + r*22}" width="21" height="21" rx="3" fill="${v > 0 ? "#2b6a86" : v < 0 ? "#6a2b4c" : "#1c1c26"}"/>` +
    T(225.5 + c*22, 60 + r*22, v > 0 ? "+1" : v < 0 ? "−1" : "0", {s: 10.5})));
  s += T(248, 128, "3×3 filter", {c: C.acc, s: 11, w: 600}) + T(248, 143, "“dark → light?”", {c: C.mut, s: 10.5});
  s += T(190, 82, "×", {s: 16}) + T(300, 82, "Σ", {s: 16}) + L(313, 78, 345, 78);
  const out = [[0,3,3,0,0],[0,3,3,0,0],[0,3,3,0,0],[0,3,3,0,0],[0,3,3,0,0]];
  out.forEach((row, r) => row.forEach((v, c) => s += `<rect x="${355 + c*20}" y="${30 + r*20}" width="19" height="19" rx="2" fill="${v ? C.green : "#1c1c26"}" fill-opacity="${v ? .85 : 1}"` +
    `${r === 2 && c === 1 ? ` stroke="${C.acc}" stroke-width="2.5"` : ""}/>`));
  s += T(405, 160, "feature map: edge found here", {c: C.mut, s: 11});
  return s + T(500, 70, "same filter,", {c: C.mut, s: 11, a: "start"}) + T(500, 85, "every position", {c: C.mut, s: 11, a: "start"});
})(), "A 3×3 edge filter slides over the image; the output lights up exactly where dark meets light.", "0 0 600 170");

const figAttn = fig((() => {
  const toks = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"], w = [.03, .62, .05, .01, .02, .12, .03, 0, .04, .08];
  const xs = []; let x = 18;
  toks.forEach(t => { const wd = 12 + t.length*7.2; xs.push(x + wd/2); x += wd + 8; });
  const q = 7; let s = "";
  toks.forEach((t, i) => { if (i === q) return;
    const x1 = xs[q], x2 = xs[i], h = 20 + Math.abs(x1 - x2)*0.22;
    s += `<path d="M${x1} 92 C${x1} ${92 - h - 10} ${x2} ${92 - h - 10} ${x2} 92" fill="none" stroke="${C.acc}" stroke-opacity="${.2 + w[i]*1.3}" stroke-width="${.8 + w[i]*9}"/>`;
    if (w[i] >= .1) s += T((x1 + x2)/2, 92 - .75*(h + 10) - 6, `${Math.round(w[i]*100)}%`, {c: C.acc, s: 10.5, w: 600}); });
  toks.forEach((t, i) => { const wd = 12 + t.length*7.2;
    s += R(xs[i] - wd/2, 94, wd, 24, {r: 6, s: i === q ? C.pink : C.line, f: i === q ? "#2a1220" : C.box}) + T(xs[i], 110.5, t, {c: i === q ? C.pink : C.fg, s: 12}); });
  return s + T(xs[q], 136, "query: “it”", {c: C.pink, s: 11, w: 600}) + T(xs[1], 136, "best-matching key", {c: C.acc, s: 11});
})(), "Attention from the word “it”: most of its weight goes to “cat”, so its new representation absorbs what “it” refers to.", "0 -22 600 167");

const figAct = fig((() => {
  const W = 150, H = 90, plot = (ox, name, f, col) => {
    let d = ""; for (let i = 0; i <= 60; i++){ const xv = -3 + i*0.1, yv = f(xv); d += `${i ? "L" : "M"}${ox + (xv + 3)/6*W} ${10 + H/2 - yv*H/6}`; }
    return `<rect x="${ox}" y="10" width="${W}" height="${H}" rx="6" fill="#0d0d13" stroke="${C.line}"/>` +
      `<path d="M${ox} ${10 + H/2}H${ox + W}M${ox + W/2} 10V${10 + H}" stroke="#2a2a35"/>` + `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.2"/>` +
      T(ox + W/2, H + 28, name, {c: col, s: 12, w: 600}); };
  const erf = x => { const t = 1/(1 + .3275911*Math.abs(x)), y = 1 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t - .284496736)*t + .254829592)*t*Math.exp(-x*x); return x >= 0 ? y : -y; };
  return plot(10, "ReLU", x => Math.max(0, x), "#ff9800") + plot(180, "GELU", x => .5*x*(1 + erf(x/Math.SQRT2)), C.amber) +
         plot(350, "Sigmoid", x => 3/(1 + Math.exp(-x)), C.violet);
})(), "Three common activation functions (x from −3 to 3). Sigmoid is drawn ×3 vertically for readability.", "0 0 510 125");

// ---------- chapters ----------
const LEARN = [
 {part: "Foundations", title: "Machine learning", html: `
  <p class="lede">Instead of writing the rules by hand, you show the computer many examples and let it <b>find the rules itself</b>.</p>
  <p>Classic programming: <i>rules + data → answers</i>. Machine learning turns it around: <i>data + answers → rules</i>.
  The learned "rules" are a <b>model</b>, a function <code>f(x; θ)</code> that maps an input <code>x</code> (an image, a sentence, a sound)
  to an output, controlled by numbers <code>θ</code> called <b>parameters</b> (or weights).</p>
  ${figML}
  <h4>How it learns</h4>
  <ul><li><b>Loss</b>: one number that says how wrong the predictions are (e.g. cross-entropy for classification).</li>
  <li><b>Optimisation</b>: change θ a little in the direction that lowers the loss, over and over, across millions of examples.</li>
  <li><b>Generalisation</b>: the real goal is to do well on data the model has <i>never seen</i>. That's why data is split into
  training, validation and test sets. A model that memorises its training set without generalising is <b>overfitting</b>.</li></ul>
  <h4>Main flavours</h4>
  <table class="kv">
   <tr><td>Supervised</td><td>Learn from labelled pairs (image → "cat"). Classification, regression.</td></tr>
   <tr><td>Self-supervised</td><td>Labels come from the data itself, e.g. predict the next word. This is how large language models are pre-trained.</td></tr>
   <tr><td>Unsupervised</td><td>Find structure without labels: clusters, compressed codes (autoencoders), generative models.</td></tr>
   <tr><td>Reinforcement</td><td>Learn actions from rewards by trial and error (games, robotics, fine-tuning chat models).</td></tr>
  </table>
  <div class="try"><b>In nnviz.</b> Every model in the list is an <code>f(x; θ)</code>. The number next to each name is how many parameters θ it has.</div>`},

 {part: "Foundations", title: "Deep learning", html: `
  <p class="lede">Deep learning is machine learning with models made of <b>many stacked layers</b>, so the model learns its own features
  instead of relying on hand-designed ones.</p>
  <p>A deep model is a composition <code>f = f<sub>L</sub> ∘ … ∘ f<sub>2</sub> ∘ f<sub>1</sub></code>. Each layer turns its input into a new
  <b>representation</b>. Nobody tells the network what to detect: the useful intermediate features emerge from training.</p>
  ${figDL}
  <h4>Why it took off around 2012</h4>
  <ul><li><b>Data</b>: large labelled datasets (ImageNet: 1.2 M images) and, later, web-scale text.</li>
  <li><b>Compute</b>: GPUs made training tens of times faster (AlexNet, 2012, trained on two gaming GPUs).</li>
  <li><b>Better recipes</b>: ReLU activations, dropout, batch normalisation, residual connections (ResNet, 2015), the Adam optimiser,
  attention and the Transformer (2017).</li></ul>
  <h4>Scale</h4>
  <p>Model size grew from about 60 M parameters (AlexNet) to 100 M+ (BERT, 2018), 175 B (GPT-3, 2020) and beyond. Empirically,
  performance improves predictably with more parameters, data and compute: the <b>scaling laws</b>.</p>
  <div class="try"><b>In nnviz.</b> Compare <code>alexnet</code> and <code>resnet50</code>. Turn on <i>Show depth</i> (View: Line) to see the
  feature maps shrink in space while growing in channels: the network trades "where" for "what".</div>`},

 {part: "Foundations", title: "Neural networks", html: `
  <p class="lede">A neural network is a big function built from simple units, <b>neurons</b>, arranged in layers and tuned by gradient descent.</p>
  ${figNeuron}
  <h4>From neuron to layer</h4>
  <p>A neuron computes a weighted sum of its inputs, adds a bias, then applies a <b>non-linearity</b> σ (ReLU, GELU, sigmoid…).
  A layer runs many neurons on the same input at once, which is exactly a matrix product:</p>
  <code class="eq">y = σ(W·x + b)     W: [out, in] weights, b: [out] biases</code>
  <p>The non-linearity is essential: without it, any stack of layers collapses into a single linear map, however deep.</p>
  <h4>Training: backpropagation</h4>
  <ol style="padding-left:18px;margin:6px 0"><li><b>Forward pass</b>: run a batch of inputs through the layers and compute the loss.</li>
  <li><b>Backward pass</b>: apply the chain rule from the loss back to the input to get ∂L/∂θ, the gradient, for every parameter.</li>
  <li><b>Update</b>: <code>θ ← θ − η · ∇L</code>, where η is the learning rate (Adam and friends adapt it per parameter).</li></ol>
  <h4>Common architectures</h4>
  <table class="kv">
   <tr><td>MLP</td><td>Fully connected layers. The basic building block.</td></tr>
   <tr><td>CNN</td><td>Small filters slide over images and share their weights across positions (ResNet, VGG, EfficientNet).</td></tr>
   <tr><td>Transformer</td><td>Attention lets every token look at every other token (GPT, Llama, ViT).</td></tr>
   <tr><td>U-Net / VAE / GAN / diffusion</td><td>Generate or reconstruct data: encoder → bottleneck → decoder, or denoise step by step.</td></tr>
  </table>
  <div class="try"><b>In nnviz.</b> Click a <i>Linear</i> layer to see its weight matrix W, or a <i>Convolution</i> to see its filters.
  Colour by <i>Mean |activation|</i> to see which neurons fire for the current input.</div>`},

 {part: "Foundations", title: "Tensors", html: `
  <p class="lede">Everything a neural network reads, stores and produces is a <b>tensor</b>: an n-dimensional array of numbers with a
  <b>shape</b> and a <b>data type</b>.</p>
  ${figTensor}
  <h4>Shapes you'll meet</h4>
  <table class="kv">
   <tr><td>Image batch</td><td><code>[N, C, H, W]</code>, e.g. <code>[32, 3, 224, 224]</code>: 32 RGB images of 224×224 pixels.</td></tr>
   <tr><td>Text tokens</td><td><code>[N, T]</code> integer ids → embeddings <code>[N, T, d]</code>, e.g. d = 768 in GPT-2 small.</td></tr>
   <tr><td>Conv weights</td><td><code>[C_out, C_in, k, k]</code>: one k×k filter per input/output channel pair.</td></tr>
   <tr><td>Attention scores</td><td><code>[N, heads, T, T]</code>: how much each token attends to each other token.</td></tr>
  </table>
  <h4>Data types and memory</h4>
  <p>Memory = number of elements × bytes per element. <code>float32</code> = 4 bytes, <code>bfloat16</code>/<code>float16</code> = 2,
  <code>int8</code>/<code>fp8</code> = 1. Example: a 7-billion-parameter model in bfloat16 takes 7·10⁹ × 2 B ≈ <b>14 GB</b> just for its weights.</p>
  <h4>What you do with them</h4>
  <p>Element-wise ops (add, ReLU), reductions (sum, mean, max), reshapes and transposes (free: they only change how memory is read),
  <b>broadcasting</b> (a <code>[d]</code> bias is added to every row of an <code>[N, d]</code> tensor), and above all <b>matrix multiplication</b>.</p>
  <div class="try"><b>In nnviz.</b> Every box is a tensor, drawn to its shape (labels show e.g. <code>[4×8×8]</code>). The <i>Size envelope</i>
  and the profile strip at the bottom follow how many values flow through each layer.</div>`},

 {part: "Foundations", title: "GPUs", html: `
  <p class="lede">Neural networks spend most of their time on huge, regular, independent arithmetic. That is exactly what
  <b>GPUs</b> are built for.</p>
  ${figGPU}
  <h4>Why GPUs win</h4>
  <ul><li><b>Massive parallelism</b>: thousands of cores apply the same instruction to different data. Every output element of a matrix
  product can be computed independently.</li>
  <li><b>Tensor cores</b>: dedicated units that multiply small matrix tiles in one step, at low precision (FP16, BF16, FP8).</li>
  <li><b>High-bandwidth memory (HBM)</b>: terabytes per second, because feeding the cores is often the real bottleneck.</li></ul>
  <h4>Orders of magnitude</h4>
  <table class="kv">
   <tr><td>Laptop CPU</td><td>≈ 1 TFLOP/s (10¹² operations per second)</td></tr>
   <tr><td>Gaming GPU</td><td>≈ 100+ TFLOP/s at low precision</td></tr>
   <tr><td>Data-centre GPU (H100 class)</td><td>≈ 1,000 TFLOP/s dense BF16 · ≈ 3 TB/s memory · 80 GB</td></tr>
  </table>
  <h4>Software stack</h4>
  <p>PyTorch → optimised kernels (cuBLAS, cuDNN, FlashAttention) → CUDA / ROCm → GPU. Large models are split across many GPUs:
  <b>data parallel</b> (each GPU gets different examples), <b>tensor parallel</b> (each layer is split) and <b>pipeline parallel</b> (layers are spread across GPUs).</p>
  <div class="try"><b>In nnviz.</b> The weights you see are the numbers a GPU streams from memory on every forward pass. Large models
  (<code>vit_b_16</code>, <code>gpt2-small</code>) show why memory bandwidth matters.</div>`},

 {part: "Foundations", title: "Computation", html: `
  <p class="lede">The cost of deep learning is measured in <b>FLOPs</b> (floating-point operations) and <b>bytes moved</b>. A few rules of
  thumb let you estimate both.</p>
  ${figMatmul}
  <h4>Counting FLOPs</h4>
  <table class="kv">
   <tr><td>Multiply-add</td><td>2 FLOPs</td></tr>
   <tr><td>Linear layer</td><td><code>2 · in · out</code> per input vector</td></tr>
   <tr><td>Convolution</td><td><code>2 · k² · C_in · C_out · H_out · W_out</code></td></tr>
   <tr><td>Self-attention</td><td>grows with <code>T² · d</code>: doubling the context length quadruples the cost</td></tr>
  </table>
  <h4>Training vs inference</h4>
  <p>The backward pass costs about twice the forward pass, so <b>training ≈ 3× forward</b>. For Transformers with N parameters:</p>
  <code class="eq">inference ≈ 2·N FLOPs per token      training ≈ 6·N·D FLOPs for D tokens</code>
  <p>Example: a 7 B model on 1 T tokens needs 6 × 7·10⁹ × 10¹² ≈ 4·10²² FLOPs. At an effective 400 TFLOP/s per GPU, that's about
  <b>29,000 GPU-hours</b>.</p>
  <h4>Memory during training</h4>
  <p>Weights + gradients + optimiser state (Adam keeps 2 extra values per parameter) ≈ <b>16 bytes per parameter</b> in mixed precision,
  plus the <b>activations</b> kept for the backward pass, which grow with batch size and sequence length.</p>
  <h4>Compute-bound vs memory-bound</h4>
  <p><b>Arithmetic intensity</b> = FLOPs per byte moved. Big matrix products reuse each number many times and keep the cores busy.
  Element-wise ops and normalisation barely compute and wait on memory, so libraries <b>fuse</b> them into one kernel.</p>
  <div class="try"><b>In nnviz.</b> Set <i>Colour layers by</i> to <i>FLOPs</i> to find where the compute goes. Click a layer to see its
  approximate FLOPs and parameters in the inspector.</div>`},

 // ================= Part 2: layers =================
 {part: "Layers", title: "Fully connected", html: `
  <p class="lede">The fully connected (or <b>linear</b>, <b>dense</b>) layer is the original neural layer: <b>every output listens to every input</b>.</p>
  ${glance("Mixes all input features into new ones: “which combinations of what I see matter?”",
           "A jury where every member hears every witness, and each member weighs the testimonies differently.",
           "Classifier heads, MLP blocks of every Transformer, tabular-data models, recommender systems.")}
  ${figFC}
  <h4>How it works</h4>
  <ol class="steps"><li>Take the input as a vector of numbers <code>x</code> (e.g. 512 features).</li>
  <li>Each output neuron has its own weight for <b>every</b> input. Together they form the matrix <code>W [out, in]</code>.</li>
  <li>Compute <code>y = W·x + b</code>: one matrix product gives all outputs at once.</li>
  <li>Usually follow it with a non-linearity (ReLU, GELU) so that stacked layers can bend the decision boundary.</li></ol>
  ${formula([`y = W·x + b`, `y<sub>i</sub> = Σ<sub>j</sub> W<sub>ij</sub> · x<sub>j</sub> + b<sub>i</sub>`,
             `<span class="c">parameters = n<sub>in</sub> · n<sub>out</sub> + n<sub>out</sub>      FLOPs ≈ 2 · n<sub>in</sub> · n<sub>out</sub></span>`],
    [["x ∈ ℝ<sup>n<sub>in</sub></sup>", "input vector"], ["W ∈ ℝ<sup>n<sub>out</sub> × n<sub>in</sub></sup>", "weight matrix: row i holds the weights of output neuron i"],
     ["b ∈ ℝ<sup>n<sub>out</sub></sup>", "bias: a learned offset per output"], ["y<sub>i</sub>", "output neuron i: a weighted sum of all inputs"]],
    "For a batch, X has shape [N, n<sub>in</sub>] and the layer computes Y = X·Wᵀ + b in one matrix product.")}
  ${procon(["Universal: with enough neurons, two layers can approximate any continuous function.", "One fast matrix multiply, ideal for GPUs."],
           ["Parameter-hungry: in × out weights. A 224×224 RGB image flattened into 1,000 neurons is 150 M weights.", "Ignores structure: it doesn't know that neighbouring pixels or consecutive words are related."])}
  <h4>In real life</h4>
  ${uses([["🏷️", "Final decision", "The last layer of an image classifier turns 2,048 features into 1,000 class scores."],
          ["💬", "Inside every LLM", "About two thirds of GPT-style parameters sit in the MLP (two linear layers) of each block."],
          ["🛒", "Recommendations", "Dense layers combine user and product embeddings to rank what you see next."],
          ["📈", "Tabular predictions", "Credit scoring, demand forecasting, sensor readings: small MLPs on feature vectors."]])}
  <div class="try"><b>In nnviz.</b> Load <code>toy-mixed</code> and click <code>fc1</code>: the inspector shows W as a heat map. Weight lines are
  white for positive and pink for negative weights.</div>`},

 {part: "Layers", title: "Convolution", html: `
  <p class="lede">A convolution slides a <b>small learned filter</b> across an image and asks the same question at every position:
  <i>“is my pattern here?”</i></p>
  ${glance("Detects local patterns (edges, textures, shapes) wherever they appear in the image.",
           "A magnifying glass with a stencil, swept over a photo line by line, scoring how well each spot matches.",
           "Vision models (ResNet, EfficientNet), U-Nets for segmentation and diffusion, audio and video models.")}
  ${figConv}
  <h4>How it works</h4>
  <ol class="steps"><li>A filter (kernel) is a tiny block of weights, typically 3×3 across all input channels.</li>
  <li>Place it on a patch of the input, multiply element by element, and sum: one number that says how strongly the patch matches.</li>
  <li>Slide it one step (the <b>stride</b>) and repeat. The grid of scores is a <b>feature map</b>.</li>
  <li>A layer learns many filters (64, 128, 512…). Each produces one output channel: one for vertical edges, one for fur texture, and so on.</li>
  <li>Stacking layers widens the <b>receptive field</b>: deep neurons “see” large regions, so they detect whole objects.</li></ol>
  ${formula([`y[o, i, j] = b[o] + Σ<sub>c</sub> Σ<sub>u=0</sub><sup>k−1</sup> Σ<sub>v=0</sub><sup>k−1</sup> W[o, c, u, v] · x[c, s·i + u, s·j + v]`,
             `<span class="c">H<sub>out</sub> = ⌊(H + 2p − k) / s⌋ + 1      parameters = C<sub>out</sub> · C<sub>in</sub> · k² + C<sub>out</sub></span>`,
             `<span class="c">FLOPs ≈ 2 · k² · C<sub>in</sub> · C<sub>out</sub> · H<sub>out</sub> · W<sub>out</sub></span>`],
    [["x[c, h, w]", "input: C<sub>in</sub> channels of size H × W (padded with p zeros on each side)"], ["W[o, c, u, v]", "filter weights: C<sub>out</sub> filters, each C<sub>in</sub> × k × k"],
     ["y[o, i, j]", "output channel o at position (i, j)"], ["k, s, p", "kernel size, stride (step), padding"], ["b[o]", "one bias per output channel"]],
    "Strictly this is a cross-correlation (the filter isn't flipped); deep-learning libraries call it convolution anyway.")}
  ${procon(["Weight sharing: the same 3×3 filter is reused everywhere, so there are very few parameters.", "Translation equivariance: a cat is recognised in any corner of the image.", "Built-in prior that nearby pixels belong together."],
           ["Local view: relating distant parts of an image needs many layers.", "Fixed grid: less natural for irregular data such as graphs or sets."])}
  <h4>In real life</h4>
  ${uses([["🩻", "Medical imaging", "Spotting tumours on scans, segmenting organs with U-Nets."],
          ["🚗", "Driver assistance", "Lane, pedestrian and sign detection from car cameras in real time."],
          ["📱", "Your phone camera", "Face unlock, portrait blur, night-mode denoising."],
          ["🎨", "Image generation", "The U-Net inside diffusion models (Stable Diffusion) is built from convolutions."]])}
  <div class="try"><b>In nnviz.</b> With <i>Kernel animation</i> on, the translucent cuboid sliding over each convolution is the filter.
  Click <code>conv1</code> to see the learned filters, and compare them with deeper layers in <code>resnet18</code> (tick <i>Pretrained weights</i>).</div>`},

 {part: "Layers", title: "Attention", html: `
  <p class="lede">Attention lets every element of a sequence <b>look at all the others</b> and decide which ones are relevant to it.
  It's the core of Transformers and of every modern large language model.</p>
  ${glance("Routes information between positions based on content: each token gathers what it needs from the rest.",
           "A meeting where each person asks a question (query), everyone shows a name tag (key), and you listen most to the best matches (values).",
           "ChatGPT, Claude, Llama, BERT; Vision Transformers; speech recognition (Whisper); AlphaFold.")}
  ${figAttn}
  <h4>How it works</h4>
  <ol class="steps"><li>Each token's vector is projected three ways: a <b>query</b> Q (“what am I looking for?”), a <b>key</b> K (“what do I contain?”)
  and a <b>value</b> V (“what do I pass on?”).</li>
  <li>Compare every query with every key using a dot product: a high score means “relevant to me”.</li>
  <li><b>Softmax</b> turns each row of scores into weights that add up to 100%.</li>
  <li>Each token's output is the weighted average of all values: <code>softmax(Q·Kᵀ / √d) · V</code>.</li>
  <li><b>Multi-head</b>: several attention patterns run in parallel (one tracks grammar, another references, another position) and are then merged.</li></ol>
  ${formula([`Q = X·W<sub>Q</sub>     K = X·W<sub>K</sub>     V = X·W<sub>V</sub>`,
             `Attention(Q, K, V) = softmax( Q·Kᵀ / √d<sub>k</sub> + M ) · V`,
             `softmax(z)<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>z<sub>j</sub></sup>`,
             `MultiHead(X) = Concat(head<sub>1</sub>, …, head<sub>h</sub>) · W<sub>O</sub>`,
             `<span class="c">parameters ≈ 4 · d² (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub>)      score matrix: T × T per head</span>`],
    [["X ∈ ℝ<sup>T × d</sup>", "the sequence: T tokens, each a d-dimensional vector"], ["W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>", "learned projections into queries, keys and values (d<sub>k</sub> = d / h per head)"],
     ["Q·Kᵀ", "T × T table of similarity scores: row = query token, column = key token"], ["√d<sub>k</sub>", "scaling that keeps scores in a range where softmax still learns"],
     ["M", "mask: −∞ above the diagonal in GPT-style models, so a token can't see the future (0 otherwise)"], ["W<sub>O</sub>", "output projection that merges the h heads"]])}
  ${procon(["Any token can reach any other in a single step, so long-range dependencies come easily.", "Dynamic: the connections change for every input, unlike fixed convolution filters.", "Highly parallel on GPUs, which made scaling to trillions of tokens possible."],
           ["Cost grows with the square of the length: 2× the context means 4× the compute and memory for the scores.", "Needs positional encodings, because on its own attention ignores word order."])}
  <h4>In real life</h4>
  ${uses([["🤖", "Chat assistants", "Each new word is produced by attending over the whole conversation so far."],
          ["🌍", "Translation", "Attention was invented (2014) to align words between languages; it still powers translation."],
          ["🧬", "Biology", "AlphaFold attends between amino acids to predict 3-D protein structures."],
          ["🖼️", "Vision", "Vision Transformers cut images into 16×16 patches and attend across them."]])}
  <div class="try"><b>In nnviz.</b> Open <code>gpt2-tiny</code> or <code>vit_tiny_16</code> and click an attention layer: the inspector
  shows one attention map per head (rows = queries, columns = keys). <code>gpt-oss-style</code> adds grouped-query and sliding-window attention.</div>`},

 {part: "Layers", title: "Other essential layers", html: `
  <p class="lede">Around the big three, a handful of smaller layers make deep networks <b>trainable, stable and efficient</b>.</p>
  ${figAct}
  <div class="cards">
  ${card("#ff9800", "Activation functions", "ReLU, GELU, sigmoid, SiLU",
    "Bend the signal after each linear step. ReLU keeps positives and zeroes negatives; without a non-linearity, depth would be useless.",
    "A dimmer switch that only lets light through above a threshold.", "after almost every linear or conv layer", ["ReLU(x) = max(0, x)", "GELU(x) = x · Φ(x)", "σ(x) = 1 / (1 + e<sup>−x</sup>)", "SiLU(x) = x · σ(x)"])}
  ${card("#ffca28", "Normalisation", "BatchNorm, LayerNorm, RMSNorm",
    "Re-centres and re-scales activations so every layer sees numbers in a healthy range. Training becomes faster and far more stable.",
    "Auto-levels on a photo before passing it to the next editor.", "ResNets (BatchNorm), every Transformer block (LayerNorm / RMSNorm)", ["y = γ ⊙ (x − μ) / √(σ² + ε) + β", "RMSNorm: y = γ ⊙ x / √(mean(x²) + ε)", "<span class=\"c\">μ, σ²: mean and variance over features (LayerNorm) or over the batch (BatchNorm); γ, β learned</span>"])}
  ${card("#bcaaa4", "Pooling", "max pool, average pool",
    "Shrinks a feature map by summarising each small window (its maximum or average). Cuts compute and adds tolerance to small shifts.",
    "Reading a map at a coarser zoom: you keep the landmarks, not every street.", "CNNs between stages; global pooling before the classifier", ["max: y[i, j] = max<sub>u,v &lt; k</sub> x[s·i + u, s·j + v]", "avg: y[i, j] = (1/k²) Σ<sub>u,v</sub> x[s·i + u, s·j + v]", "<span class=\"c\">no parameters · H<sub>out</sub> = ⌊(H − k) / s⌋ + 1</span>"])}
  ${card("#4db6ac", "Embedding", "token / patch / position embeddings",
    "A lookup table that turns discrete ids (words, tokens, user ids) into learned vectors. Similar meanings end up close together.",
    "A dictionary whose definitions are coordinates in a “meaning space”.", "the first layer of every language model; recommenders", ["e = E[id]  (row lookup)", "= one_hot(id) · E,   E ∈ ℝ<sup>V × d</sup>", "<span class=\"c\">V: vocabulary size · d: vector size · parameters = V · d</span>"])}
  ${card("#ffffff", "Residual connection", "skip connection, x + F(x)",
    "Adds a layer's input to its output, so each block only learns a correction. Gradients flow straight through, so networks with 100+ layers train.",
    "Editing a draft with tracked changes instead of rewriting it from scratch.", "ResNet, every Transformer, U-Net (as concatenated skips)", ["y = x + F(x)", "∂y/∂x = I + ∂F/∂x", "<span class=\"c\">the identity term I lets gradients pass even when ∂F/∂x is tiny</span>"])}
  ${card("#d4e157", "Mixture of experts", "MoE router + experts",
    "A small router sends each token to a few specialist sub-networks out of many. Huge total capacity, but only a fraction runs per token.",
    "A hospital triage desk sending each patient to two specialists out of fifty.", "Mixtral, DeepSeek, GPT-OSS-style models", ["g = softmax( TopK(W<sub>r</sub> · x) )", "y = Σ<sub>i ∈ TopK</sub> g<sub>i</sub> · E<sub>i</sub>(x)", "<span class=\"c\">W<sub>r</sub>: router weights · E<sub>i</sub>: expert i (an MLP) · only k experts run</span>"])}
  ${card("#5c6bc0", "Up-sampling", "transposed conv, interpolation",
    "Enlarges feature maps again, to rebuild a full-resolution image from a compact code.",
    "Zooming a sketch back up and re-drawing the details.", "U-Net decoders, VAE / GAN generators, super-resolution", ["nearest: y[i, j] = x[⌊i/s⌋, ⌊j/s⌋]", "transposed conv: H<sub>out</sub> = (H − 1)·s − 2p + k", "<span class=\"c\">s: scale factor (stride) · k: kernel size · p: padding</span>"])}
  ${card("#8a8a96", "Dropout", "regularisation (training only)",
    "Randomly switches off a fraction of neurons during training so the network can't rely on any single one. Reduces overfitting.",
    "Training a team where random members are absent each day, so everyone learns every role.", "classifier heads, older Transformers, small datasets", ["y = m ⊙ x / (1 − p),   m<sub>i</sub> ~ Bernoulli(1 − p)", "<span class=\"c\">training only; at inference y = x · p: drop rate (e.g. 0.1)</span>"])}
  </div>
  <div class="try"><b>In nnviz.</b> These layers have their own colours in the legend (bottom-left). Colour by <i>Residual change</i> to see how much
  each block actually modifies its input, and open <code>gpt-oss-style</code> to see an MoE router choose experts per token.</div>`},
];

let idx = 0, ret = null;
const nav = $("lnav");
let lastPart = null;
LEARN.forEach((ch, i) => {
  if (ch.part !== lastPart){ const p = document.createElement("div"); p.className = "part"; p.textContent = ch.part; nav.appendChild(p); lastPart = ch.part; }
  const b = document.createElement("button"); b.innerHTML = `<em>${i + 1}</em><span>${ch.title}</span>`; b.onclick = () => go(i); nav.appendChild(b);
});
function go(i){
  idx = Math.max(0, Math.min(LEARN.length - 1, i));
  const ch = LEARN[idx];
  $("lstep").textContent = `${ch.part} · Chapter ${idx + 1} of ${LEARN.length}`;
  $("ltitle").textContent = ch.title; $("lbody").innerHTML = ch.html; $("lbody").scrollTop = 0;
  nav.querySelectorAll("button").forEach((b, j) => { b.classList.toggle("on", j === idx); if (j === idx) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current"); });
  $("lprev").disabled = idx === 0;
  $("lnext").textContent = idx === LEARN.length - 1 ? "Finish" : `Next: ${LEARN[idx + 1].title}`;
  $("lprog").style.width = ((idx + 1) / LEARN.length * 100) + "%";
  try { localStorage.setItem("nnviz.learn", idx); } catch (e) {}
}
function open(i){
  if (i == null){ try { i = +localStorage.getItem("nnviz.learn") || 0; } catch (e) { i = 0; } }
  ret = document.activeElement; $("learn").hidden = false; go(i); $("lnext").focus();
}
function close(){ $("learn").hidden = true; if (ret && ret.focus) ret.focus(); }
$("learnbtn").onclick = () => open();
$("lclose").onclick = close;
$("lprev").onclick = () => go(idx - 1);
$("lnext").onclick = () => idx === LEARN.length - 1 ? close() : go(idx + 1);
$("learn").addEventListener("mousedown", e => { if (e.target === $("learn")) close(); });
addEventListener("keydown", e => {
  if ($("learn").hidden){
    if ((e.key === "l" || e.key === "L") && $("help").hidden && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)){ open(); e.preventDefault(); }
    return;
  }
  if (e.key === "Escape") close();
  else if (e.key === "ArrowRight") go(idx + 1);
  else if (e.key === "ArrowLeft") go(idx - 1);
  else if (e.key === "Tab"){
    const f = [...$("learn").querySelectorAll("button:not(:disabled)")], a = f[0], z = f[f.length - 1];
    if (e.shiftKey && document.activeElement === a){ z.focus(); e.preventDefault(); }
    else if (!e.shiftKey && document.activeElement === z){ a.focus(); e.preventDefault(); }
  } else if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "PageDown" || e.key === "PageUp" || e.key === " ") return;   // let the page scroll
  e.stopImmediatePropagation();
}, true);
window.learnOpen = open;
})();
