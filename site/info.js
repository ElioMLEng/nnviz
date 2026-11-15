"use strict";
// Plain-language descriptions of layer types: what the layer does, how to interpret it, what to look at in the 3D view.
const LAYER_INFO = {
  Conv2d: {
    title: "2-D convolution",
    does: "Slides a small bank of learned filters (kernels) over the image or feature map. Every output channel is one filter's response at every position; all positions share the same weights.",
    meaning: "Early filters behave like edge, colour and texture detectors; deeper ones respond to parts and objects. Weight sharing makes the layer translation-equivariant and far cheaper than a dense layer.",
    view: "The translucent cuboid is one k×k×C_in kernel sliding over the input volume; the highlighted cell is the output it produces. One output slab = one filter.",
  },
  ConvTranspose2d: {
    title: "Transposed convolution (learned up-sampling)",
    does: "Spreads each input value over a k×k patch of a larger output, so spatial size grows (stride 2 doubles it). It is the gradient of a convolution used as a forward layer.",
    meaning: "Used in decoders, generators and segmentation heads to turn compact features back into spatial maps. Can leave checkerboard artifacts when kernel size is not divisible by stride.",
    view: "Output slabs are larger than the input slabs; channels usually shrink while resolution grows.",
  },
  Linear: {
    title: "Linear / fully-connected layer",
    does: "y = W·x + b. Every output neuron is a weighted sum of all input features.",
    meaning: "Mixes information across all features. In a Transformer MLP it expands then contracts the width (typically ×4); at the end of a network it is the classifier or vocabulary head that turns features into logits.",
    view: "Each sphere is a neuron; white/magenta lines are positive/negative weights (strongest shown).",
  },
  Embedding: {
    title: "Embedding lookup",
    does: "A table with one learned vector per token id (or position). The layer simply indexes the row.",
    meaning: "Places discrete symbols in a continuous space where similar tokens end up close; positional embeddings do the same for positions so the model knows word order.",
    view: "Output is a tokens × features grid: one row per token.",
  },
  LayerNorm: {
    title: "Layer normalization",
    does: "For each token, subtracts the mean and divides by the standard deviation across its feature vector, then applies a learned scale and shift.",
    meaning: "Keeps activations on a stable scale so very deep stacks train reliably; independent of batch size, which is why Transformers use it instead of BatchNorm.",
    view: "The yellow double cage: same shape in and out, only the scale of the values changes.",
  },
  RMSNorm: {
    title: "RMS normalization",
    does: "Divides each token's feature vector by its root-mean-square (no mean subtraction), then applies a learned scale.",
    meaning: "A cheaper LayerNorm that works as well in practice; used by Llama-family and most recent LLMs.",
    view: "Same shape in and out; values rescaled per token.",
  },
  BatchNorm2d: {
    title: "Batch normalization",
    does: "Normalizes each channel using the mean and variance over the batch and all spatial positions, then applies a learned scale and shift. At inference it uses running statistics.",
    meaning: "Smooths the loss landscape, allows higher learning rates and acts as a mild regularizer. It is why ResNet-style CNNs train easily.",
    view: "The yellow cage: per-channel rescaling, shape unchanged.",
  },
  GroupNorm: {
    title: "Group normalization",
    does: "Splits channels into groups and normalizes within each group per sample, independent of the batch.",
    meaning: "BatchNorm's batch-size-independent cousin; standard in diffusion U-Nets and detection models where batches are small.",
    view: "Per-sample, per-group rescaling; shape unchanged.",
  },
  ReLU: {
    title: "ReLU activation",
    does: "max(0, x): passes positives, zeroes negatives.",
    meaning: "The non-linearity that lets stacked linear layers represent complex functions. Creates sparse activations; cheap and avoids vanishing gradients for positive inputs.",
    view: "The orange plate shows the function curve (read left→right looking along the data flow). Cells that go to zero turn dark.",
  },
  GELU: {
    title: "GELU activation",
    does: "x·Φ(x): a smooth gate that softly suppresses negative values instead of cutting them.",
    meaning: "The standard activation in Transformers (BERT, GPT-2, ViT). Smoothness gives slightly better optimization than ReLU.",
    view: "Plate shows the smooth knee near zero.",
  },
  SiLU: {
    title: "SiLU / Swish activation",
    does: "x·sigmoid(x): smooth, non-monotonic near zero.",
    meaning: "Used in EfficientNet, diffusion U-Nets and, inside SwiGLU, in Llama-style LLMs.",
    view: "Plate shows the small dip below zero.",
  },
  Hardswish: {
    title: "Hard-swish activation",
    does: "A piecewise-linear approximation of x·sigmoid(x) built from ReLU6.",
    meaning: "Gets most of Swish's accuracy at a fraction of the cost: designed for mobile CPUs (MobileNetV3).",
    view: "Plate shows a swish-like curve.",
  },
  Sigmoid: {
    title: "Sigmoid activation",
    does: "1 / (1 + e^−x): squashes values into (0, 1).",
    meaning: "Interpreted as a probability or a gate (e.g. 'real vs fake' in a GAN discriminator, squeeze-and-excitation channel gates, VAE pixel intensities).",
    view: "Plate shows the S-curve.",
  },
  Tanh: {
    title: "Tanh activation",
    does: "Squashes values into (−1, 1), zero-centred.",
    meaning: "Output layer of image generators (pixels in [−1, 1]); older recurrent nets used it everywhere.",
    view: "Plate shows the S-curve centred at zero.",
  },
  LeakyReLU: {
    title: "Leaky ReLU",
    does: "x if x > 0 else αx (small α, e.g. 0.2).",
    meaning: "Keeps a gradient alive for negative inputs; common in GAN discriminators so the generator always gets a training signal.",
    view: "Plate shows the slightly tilted left half.",
  },
  Softmax: {
    title: "Softmax",
    does: "Exponentiates and normalizes so the outputs are positive and sum to 1.",
    meaning: "Turns scores into a probability distribution over classes or tokens.",
    view: "Output cells are probabilities.",
  },
  MaxPool2d: {
    title: "Max pooling",
    does: "Keeps the maximum value in each k×k window, usually halving the spatial size.",
    meaning: "Down-samples while keeping the strongest response: adds small-shift invariance and enlarges the receptive field of later layers. Has no parameters.",
    view: "The grey window slides over one channel; output slabs have half the resolution.",
  },
  AvgPool2d: {
    title: "Average pooling",
    does: "Averages each k×k window.",
    meaning: "Smoother down-sampling, used in transition layers (DenseNet) and as the final global pool.",
    view: "Same sliding window as max-pool, values averaged.",
  },
  AdaptiveAvgPool2d: {
    title: "Adaptive average pooling",
    does: "Averages whatever spatial extent it receives down to a fixed output size (usually 1×1).",
    meaning: "Global average pooling: collapses a feature map to one value per channel so the classifier does not depend on image size.",
    view: "Spatial dimensions collapse; channels are preserved.",
  },
  Upsample: {
    title: "Up-sampling (interpolation)",
    does: "Enlarges the spatial size by nearest-neighbour or bilinear interpolation. No parameters.",
    meaning: "Cheap, artifact-free alternative to transposed convolutions; usually followed by a regular conv (diffusion U-Nets).",
    view: "Larger slabs, same number of channels.",
  },
  MultiheadAttention: {
    title: "Multi-head self-attention",
    does: "Each token produces a query, key and value. Attention weights = softmax(Q·Kᵀ/√d) say how much every token reads from every other; the output is the weighted sum of values, per head, then re-projected by W_o.",
    meaning: "Lets each token gather context from anywhere in the sequence in a single layer. Different heads specialise (positions, syntax, copying, …).",
    view: "X → Q, K, V lanes (real projection weights) → one T×T attention map per head → A·V → W_o. Columns are boxed per head.",
  },
  moe: {
    title: "Mixture-of-Experts feed-forward",
    does: "A small router scores every token against E expert MLPs and sends it to only the top-k; the expert outputs are summed with the router's gate weights.",
    meaning: "Grows the parameter count (knowledge capacity) without growing per-token compute: only k of E experts run for each token.",
    view: "Router plane (tokens × experts): bright cells are the experts each token was sent to.",
  },
  input: {
    title: "Model input",
    does: "The tensor fed to the network: an image (C×H×W), token ids, a latent vector or a timestep.",
    meaning: "Everything downstream is a function of this. Change the input class or noise in the top bar and watch activations change.",
    view: "Slabs are channels (e.g. RGB); each cell is one pixel/value.",
  },
  output: {
    title: "Model output",
    does: "The final tensor: class logits, next-token logits, a reconstructed image, a predicted noise map…",
    meaning: "Interpretation depends on the task; logits go through a softmax to become probabilities.",
    view: "One cell per output unit (sampled when there are thousands).",
  },
  // tensor ops (by function / method name)
  add: { title: "Element-wise add (residual connection)", does: "Adds two tensors of the same shape.",
         meaning: "The residual connection y = x + F(x): the block learns a correction to its input, giving gradients a direct path through depth. Also adds position to token embeddings and timestep conditioning.",
         view: "Two incoming links (one is often the dashed skip arc)." },
  mul: { title: "Element-wise multiply (gating)", does: "Multiplies two tensors element by element.",
         meaning: "Acts as a learned gate: one branch decides how much of the other passes (SwiGLU, squeeze-and-excitation, VAE σ·ε).",
         view: "Two incoming links combined." },
  cat: { title: "Concatenation", does: "Stacks tensors along an axis (usually channels).",
         meaning: "Skip connections that *keep* earlier features side by side (U-Net, DenseNet) instead of summing them; channel count grows.",
         view: "Output has the sum of the input channel counts." },
  flatten: { title: "Flatten", does: "Collapses trailing dimensions into one.", meaning: "Bridges a spatial feature map to a vector or a token sequence; no computation.", view: "Same values, new layout." },
  transpose: { title: "Transpose / permute", does: "Swaps two dimensions.", meaning: "Re-orders axes, e.g. channels↔tokens, so the next layer reads the right axis; no computation.", view: "Same values, axes swapped." },
  permute: { title: "Permute", does: "Re-orders dimensions.", meaning: "Layout change only (e.g. image patches → sequence of tokens).", view: "Same values, new layout." },
  reshape: { title: "Reshape", does: "Changes how elements are grouped into dimensions.", meaning: "Pure bookkeeping: splits/merges heads, patches or spatial axes.", view: "Same values, new shape." },
  view: { title: "Reshape (view)", does: "Re-interprets the same memory with a different shape.", meaning: "Bookkeeping: no computation.", view: "Same values, new shape." },
  mean: { title: "Mean reduction", does: "Averages over one or more dimensions.", meaning: "Pooling over tokens or positions to get a single vector (e.g. sequence mean-pooling before a classifier).", view: "A dimension disappears." },
  getitem: { title: "Indexing / slicing", does: "Selects part of a tensor.", meaning: "e.g. taking the [CLS] token, or one half of a split.", view: "Smaller tensor." },
  randn_like: { title: "Random noise ε ~ N(0, I)", does: "Samples Gaussian noise with the same shape as its input.", meaning: "The stochastic part of the VAE reparameterization trick: z = μ + σ·ε. Re-sampled on every run, so the output changes.", view: "Cells show the sampled noise." },
  exp: { title: "Exponential", does: "e^x element-wise.", meaning: "Turns log-variance into a standard deviation (σ = e^{½·logvar}).", view: "Same shape." },
  sin: { title: "Sine", does: "sin(x) element-wise.", meaning: "Half of the sinusoidal timestep/position embedding.", view: "Same shape." },
  cos: { title: "Cosine", does: "cos(x) element-wise.", meaning: "Other half of the sinusoidal embedding.", view: "Same shape." },
  softmax: { title: "Softmax", does: "Normalizes to a probability distribution.", meaning: "Turns scores into weights that sum to 1.", view: "Same shape." },
  matmul: { title: "Matrix multiply", does: "Batched matrix product of two tensors.", meaning: "The workhorse of attention (Q·Kᵀ, A·V).", view: "Two incoming links." },
  sigmoid: { title: "Sigmoid", does: "1/(1+e^−x).", meaning: "Gate in (0, 1).", view: "Same shape." },
};

// class-name aliases → entry
const LAYER_ALIAS = {
  BatchNorm1d: "BatchNorm2d", LayerNorm2d: "LayerNorm", ReLU6: "ReLU", Hardsigmoid: "Sigmoid", ELU: "ReLU", Dropout: "ReLU",
  AdaptiveMaxPool2d: "AdaptiveAvgPool2d", UpsamplingNearest2d: "Upsample", UpsamplingBilinear2d: "Upsample",
  CausalSelfAttention: "MultiheadAttention", SelfAttention: "MultiheadAttention", GQAttention: "MultiheadAttention",
  BiAttention: "MultiheadAttention", MoE: "moe", unsqueeze: "reshape", contiguous: "reshape", squeeze: "reshape",
  expand: "reshape", float: "reshape", repeat_interleave: "reshape", mean_: "mean", sum: "mean", Flatten: "flatten", Unflatten: "reshape",
  x: "input", t: "input", ids: "input", z: "input",
};

function layerEntry(n){
  const key = n.label;
  let e = LAYER_INFO[key] || LAYER_INFO[LAYER_ALIAS[key]];
  if (!e && n.kind === "attention") e = LAYER_INFO.MultiheadAttention;
  if (!e && n.kind === "moe") e = LAYER_INFO.moe;
  if (!e && n.kind === "input") e = LAYER_INFO.input;
  if (!e && n.kind === "output") e = LAYER_INFO.output;
  if (!e && n.kind === "norm") e = LAYER_INFO.LayerNorm;
  if (!e && n.kind === "conv") e = LAYER_INFO.Conv2d;
  if (!e && n.kind === "linear") e = LAYER_INFO.Linear;
  if (!e && n.kind === "activation") e = LAYER_INFO.ReLU;
  if (!e && n.kind === "pool") e = LAYER_INFO.MaxPool2d;
  if (!e && n.kind === "upsample") e = LAYER_INFO.Upsample;
  return e || null;
}

// facts that depend on this particular layer's hyper-parameters
function layerFacts(n){
  const a = n.attrs || {}, out = [];
  const arr = v => Array.isArray(v) ? v : [v, v];
  if (n.kind === "conv" && a.kernel_size){
    const [kh, kw] = arr(a.kernel_size), [sh] = arr(a.stride ?? 1);
    const tr = n.label === "ConvTranspose2d";
    out.push(`${a.in_channels} → ${a.out_channels} channels, ${kh}×${kw} kernel, stride ${sh}${a.groups > 1 ? `, groups ${a.groups}` : ""}.`);
    if (kh === 1 && kw === 1) out.push("A 1×1 convolution is a dense layer applied independently at every pixel: it only mixes channels.");
    if (a.groups > 1 && a.groups === a.in_channels) out.push("Depthwise: each channel is filtered on its own (cheap); a following 1×1 conv mixes channels.");
    if (!tr && sh >= 2 && kh === sh && kh >= 8) out.push(`Kernel = stride = ${kh}: this is a patch embedding (non-overlapping ${kh}×${kh} patches → one vector each).`);
    else if (!tr && sh >= 2) out.push(`Stride ${sh} halves the spatial size (learned down-sampling).`);
    if (tr) out.push(`Stride ${sh} multiplies the spatial size by ${sh}.`);
    if (!tr && kh >= 5) out.push(`Large ${kh}×${kw} kernel: a wide receptive field in a single layer.`);
  }
  if (n.kind === "linear" && a.in_features){
    out.push(`${a.in_features} → ${a.out_features} features${a.out_features === 4 * a.in_features ? " (×4 expansion: the Transformer MLP up-projection)" : a.in_features === 4 * a.out_features ? " (×4 contraction: the MLP down-projection)" : ""}.`);
    if (a.out_features >= 20000) out.push("Output width equals the vocabulary size: this is the language-model head producing next-token logits.");
    else if (a.out_features === 1000) out.push("1000 outputs: the ImageNet classifier head.");
  }
  if (n.kind === "norm" && a.num_features) out.push(`${a.num_features} channels each get their own scale and shift.`);
  if (n.kind === "norm" && a.normalized_shape) out.push(`Normalizes over the last dimension (${[].concat(a.normalized_shape).join("×")}).`);
  if (n.kind === "attention"){
    const H = a.heads || a.num_heads;
    if (H) out.push(`${H} heads${a.kv_heads && a.kv_heads !== H ? `, ${a.kv_heads} shared key/value heads (grouped-query attention: each K/V head serves ${H / a.kv_heads} query heads)` : ""}.`);
    if (a.window) out.push(`Sliding-window attention: each token only sees the previous ${a.window} tokens (banded attention map).`);
    if (/Causal|GQ/.test(n.label)) out.push("Causal: a token cannot attend to later tokens, so attention maps are lower-triangular.");
    if (n.sampled) out.push("Display is sampled: a few heads, tokens and feature columns are shown.");
  }
  if (n.kind === "moe" && a.num_experts) out.push(`${a.num_experts} experts, top-${a.top_k} routed per token → only ${a.top_k}/${a.num_experts} of the expert parameters are active for any token.`);
  if (n.kind === "pool" && a.kernel_size){ const [k] = arr(a.kernel_size), [s] = arr(a.stride ?? k); out.push(`${k}×${k} window, stride ${s}.`); }
  if (n.kind === "upsample" && a.scale_factor) out.push(`Scale factor ${a.scale_factor}.`);
  if (n.kind === "embedding" && a.num_embeddings) out.push(`${a.num_embeddings} entries × ${a.embedding_dim} dimensions.`);
  if (n.params) out.push(`${n.params.toLocaleString()} learned parameters.`); else if (n.kind !== "input" && n.kind !== "output") out.push("No learned parameters.");
  return out;
}

function describeLayer(n){
  const e = layerEntry(n);
  const facts = layerFacts(n);
  if (!e && !facts.length) return "";
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return `<div class="desc">` +
    (e ? `<div class="dt">${esc(e.title)}</div><p><b>What it does.</b> ${esc(e.does)}</p>
          <p><b>How to interpret it.</b> ${esc(e.meaning)}</p>
          <p><b>In this view.</b> ${esc(e.view)}</p>` : "") +
    (facts.length ? `<p><b>This layer.</b> ${facts.map(esc).join(" ")}</p>` : "") + `</div>`;
}
