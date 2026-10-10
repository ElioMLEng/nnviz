"""Model cards shown in the viewer: facts per model (release, what it is known for, what it introduced)."""

INFO = {
    "toy-mixed": dict(
        year="2026", org="this project", paper="—",
        tagline="A deliberately tiny network that contains one of everything.",
        known_for=["Teaching toy: small enough to read every neuron and every weight."],
        innovations=["Convolution → pooling → LayerNorm → self-attention (with residual) → MLP in a single model of only 835 parameters."],
        technical=["Trained for ~300 steps on 8×8 images of a horizontal, vertical or diagonal line plus noise.",
                   "2 attention heads over 16 tokens (the 4×4 feature map flattened), so the whole attention matrix is visible."],
        look_for=["The kernel cuboid sliding over the input in conv1/conv2.", "The residual arc over the attention block.",
                  "How the attention maps change between the three input classes."]),

    "resnet18": dict(
        year="Dec 2015 (CVPR 2016)", org="Microsoft Research (He, Zhang, Ren, Sun)", paper="Deep Residual Learning for Image Recognition",
        tagline="The network that made very deep networks trainable.",
        known_for=["Won ILSVRC 2015 classification (3.57 % top-5 error with an ensemble) and the CVPR 2016 best paper award.",
                   "Its skip connection is now inside almost every deep architecture, Transformers included."],
        innovations=["Residual block: y = x + F(x). The layers only learn a correction to the identity.",
                     "Solved the degradation problem: plain 56-layer nets trained worse than 20-layer ones; residual nets kept improving to 152 layers (and 1000+ in follow-ups)."],
        technical=["Basic block = two 3×3 convs + BatchNorm + ReLU, with an identity (or 1×1 strided conv) shortcut.",
                   "Global average pooling instead of big fully-connected layers; stem = 7×7 stride-2 conv + max-pool.",
                   "Gradients flow unattenuated along the shortcuts, which is why depth stops hurting."],
        look_for=["The dashed white arcs: those are the shortcuts.", "Stride-2 blocks with a 1×1 projection shortcut (a pair of parallel branches).",
                  "Channels double (64→128→256→512) each time the spatial size halves."]),

    "resnet50": dict(
        year="Dec 2015 (CVPR 2016)", org="Microsoft Research (He, Zhang, Ren, Sun)", paper="Deep Residual Learning for Image Recognition",
        tagline="The workhorse backbone of computer vision for the last decade.",
        known_for=["Default ImageNet-pretrained backbone for detection, segmentation and transfer learning (Faster/Mask R-CNN, etc.).",
                   "Still the baseline that new vision architectures are compared against."],
        innovations=["Bottleneck block: 1×1 (reduce) → 3×3 → 1×1 (expand) keeps the expensive 3×3 conv on a narrow tensor, allowing much deeper nets at similar cost.",
                     "Residual connections, as in ResNet-18, plus BatchNorm in every block."],
        technical=["16 bottleneck blocks in four stages (3, 4, 6, 3); 2048-channel final feature map.",
                   "76 % top-1 on ImageNet with the original recipe, ~80 % with modern training recipes."],
        look_for=["Many 1×1 convs: they are per-pixel channel mixers.", "Tensors get narrower then wider inside each bottleneck."]),

    "vgg16": dict(
        year="Sep 2014 (ICLR 2015)", org="Oxford Visual Geometry Group (Simonyan, Zisserman)", paper="Very Deep Convolutional Networks for Large-Scale Image Recognition",
        tagline="Proof that depth with tiny filters matters, in the simplest possible design.",
        known_for=["ILSVRC 2014: 1st in localization, 2nd in classification.", "Its features are still used for style transfer and perceptual (LPIPS) losses."],
        innovations=["Only 3×3 convolutions and 2×2 max-pools: two stacked 3×3 convs see a 5×5 area with fewer parameters and more non-linearity than one 5×5."],
        technical=["13 conv + 3 fully-connected layers, no normalization, no skip connections.",
                   "~90 % of the parameters sit in the first fully-connected layer (25088 → 4096); the conv part is only 15 M."],
        look_for=["The huge dense layer near the output (display is down-sampled).", "A plain, regular stack: channels double after each pool."]),

    "alexnet": dict(
        year="2012 (NeurIPS 2012)", org="University of Toronto (Krizhevsky, Sutskever, Hinton)", paper="ImageNet Classification with Deep Convolutional Neural Networks",
        tagline="The model that started the deep learning era in vision.",
        known_for=["Won ILSVRC 2012 with 15.3 % top-5 error against 26.2 % for the runner-up: a gap the field had never seen."],
        innovations=["ReLU instead of tanh (much faster training), dropout in the dense layers, heavy data augmentation, GPU training.",
                     "Overlapping max-pooling and local response normalization (later dropped by the field)."],
        technical=["5 conv + 3 FC layers, 11×11 stride-4 first filter.", "Trained ~6 days on two GTX 580 GPUs. torchvision ships the single-GPU variant."],
        look_for=["The very large first kernel (11×11) versus the 3×3 kernels of later models."]),

    "mobilenet_v3_small": dict(
        year="May 2019 (ICCV 2019)", org="Google (Howard et al.)", paper="Searching for MobileNetV3",
        tagline="A CNN designed (partly by search) to run fast on a phone CPU.",
        known_for=["Standard lightweight backbone for on-device vision."],
        innovations=["Depthwise-separable convs (V1, 2017) + inverted residuals with linear bottlenecks (V2, 2018).",
                     "V3 adds squeeze-and-excitation, the cheap h-swish activation, and an architecture found by NAS (MnasNet) refined with NetAdapt."],
        technical=["Depthwise 3×3/5×5 conv followed by 1×1 pointwise conv costs ~8–9× fewer FLOPs than a regular conv.",
                   "Narrow → wide → narrow blocks (inverted bottleneck), with the shortcut on the narrow ends."],
        look_for=["Depthwise convs: each output channel reads only its own input channel (diagonal weights).",
                  "Squeeze-and-excitation: global pool → tiny MLP → per-channel gate."]),

    "efficientnet_b0": dict(
        year="May 2019 (ICML 2019)", org="Google Brain (Tan, Le)", paper="EfficientNet: Rethinking Model Scaling for CNNs",
        tagline="Scale depth, width and resolution together, with a fixed recipe.",
        known_for=["State-of-the-art ImageNet accuracy at 5–10× fewer parameters/FLOPs than previous nets; the B0–B7 family."],
        innovations=["Compound scaling: depth α^φ, width β^φ, resolution γ^φ with one coefficient φ, instead of scaling one axis ad hoc.",
                     "B0 baseline found by neural architecture search, built from MBConv (MobileNetV2) blocks with squeeze-and-excitation and SiLU."],
        technical=["77.1 % top-1 with 5.3 M parameters and 0.39 GFLOPs.", "Stochastic depth and dropout during training."],
        look_for=["Repeated MBConv blocks with residual arcs only where input and output shapes match."]),

    "densenet121": dict(
        year="Aug 2016 (CVPR 2017 best paper)", org="Cornell / Tsinghua / Facebook (Huang et al.)", paper="Densely Connected Convolutional Networks",
        tagline="Every layer sees the outputs of all previous layers.",
        known_for=["CVPR 2017 best paper; strong accuracy with very few parameters."],
        innovations=["Dense connectivity: each layer's input is the concatenation of all earlier feature maps in the block (not a sum, as in ResNet).",
                     "Feature reuse: layers can be narrow (growth rate k = 32) because they need not re-learn earlier features."],
        technical=["4 dense blocks (6, 12, 24, 16 layers) joined by 1×1 conv + average-pool 'transition' layers.",
                   "Concatenation makes the channel count grow linearly inside a block."],
        look_for=["Many 'cat' ops: the tensor grows by 32 channels each layer.", "The dense web of arcs back to earlier layers."]),

    "vit_b_16": dict(
        year="Oct 2020 (ICLR 2021)", org="Google Research (Dosovitskiy et al.)", paper="An Image is Worth 16×16 Words",
        tagline="A plain Transformer on image patches, no convolutions.",
        known_for=["Showed that with enough pretraining data a pure Transformer matches or beats the best CNNs.",
                   "Foundation of CLIP, DINO, SAM, MAE and most modern vision-language models."],
        innovations=["Cut the image into 16×16 patches, linearly embed each into a token, add position embeddings and a [CLS] token, feed a standard Transformer encoder.",
                     "Almost no image-specific inductive bias (no locality, no translation equivariance): it must be learned from data."],
        technical=["224×224 → 14×14 = 196 patches + [CLS] = 197 tokens; 12 layers, 12 heads, width 768, MLP 3072 (pre-LayerNorm, GELU).",
                   "Pretrained on JFT-300M / ImageNet-21k, then fine-tuned. The patch embedding is a 16×16 stride-16 conv."],
        look_for=["The first conv: it is the patch embedding.", "12 identical encoder blocks, each with two residual arcs (attention and MLP).",
                  "Attention maps are 197×197 (shown sampled)."]),

    "vit_tiny_16": dict(
        year="Dec 2020 (ICML 2021)", org="Facebook AI (Touvron et al., DeiT)", paper="Training data-efficient image transformers & distillation through attention",
        tagline="ViT sized for ImageNet-1k alone: 12 layers but only 192 dimensions and 3 heads.",
        known_for=["DeiT showed ViTs can be trained competitively on ImageNet-1k only, in about 3 days on one machine, with the right recipe."],
        innovations=["Strong augmentation/regularization recipe and a distillation token that learns from a CNN teacher (not included in this plain ViT-Ti)."],
        technical=["Same architecture as ViT-B/16 but width 192, 3 heads, MLP 768. 197 tokens.", "Random weights here: the viewer shows the structure, not a trained model."],
        look_for=["Only 3 attention heads per block: each head's own columns are easy to see in Q, K and V."]),

    "gpt2-small": dict(
        year="Feb 2019", org="OpenAI (Radford et al.)", paper="Language Models are Unsupervised Multitask Learners",
        tagline="A decoder-only Transformer trained to predict the next token; it generalizes to many tasks zero-shot.",
        known_for=["Triggered the 'staged release' debate; the 1.5 B version was the largest language model of its time.",
                   "Ancestor of GPT-3/4 and of the decoder-only architecture used by essentially all current LLMs."],
        innovations=["Scale + a simple objective (next-token prediction on WebText, 40 GB): tasks like translation or summarization emerge without task-specific training.",
                     "Pre-LayerNorm placement, an extra LayerNorm at the end, residual-branch weight scaling, byte-level BPE (50 257 tokens)."],
        technical=["Small = 12 layers, 12 heads, d = 768, context 1024 (this viewer: 32 tokens, random weights).",
                   "Causal self-attention: each token only attends to itself and earlier tokens (lower-triangular attention maps).",
                   "MLP expands ×4 with GELU; learned position embeddings."],
        look_for=["Attention maps are lower-triangular: that is the causal mask.", "Two embeddings (token + position) added before block 0.",
                  "The lm_head maps 768 → 50 257 vocabulary logits (sampled for display)."]),

    "gpt2-tiny": dict(
        year="2019 design, scaled down here", org="this project", paper="GPT-2 design",
        tagline="Same design as GPT-2 at toy scale so the whole model is easy to explore.",
        known_for=["Educational: 4 layers, 4 heads, d = 128, vocabulary 2048."],
        innovations=["None: a faithful miniature of the GPT-2 block structure."],
        technical=["Causal self-attention, pre-LayerNorm, GELU MLP, learned positions, 32-token context."],
        look_for=["With 4 heads you can see all of them without sampling."]),

    "llama-style": dict(
        year="Feb 2023 (LLaMA); Jul 2023 (Llama 2); 2024 (Llama 3)", org="Meta AI (Touvron et al.)", paper="LLaMA: Open and Efficient Foundation Language Models",
        tagline="The open-weights recipe that most current open LLMs follow.",
        known_for=["LLaMA-13B outperformed GPT-3 (175 B) on most benchmarks while being ~10× smaller; trained on public data only.",
                   "Started the open-weights LLM ecosystem (Alpaca, Mistral, Qwen… share this block design)."],
        innovations=["RMSNorm (cheaper than LayerNorm: no mean-centering) before each sub-layer.",
                     "SwiGLU feed-forward: down(SiLU(gate(x)) ⊙ up(x)), hidden size ≈ 2/3·4d.",
                     "Rotary position embeddings (RoPE): positions are encoded by rotating Q and K, so attention depends on relative offsets.",
                     "No biases. Later versions add grouped-query attention (GQA) to shrink the KV cache."],
        technical=["This viewer: 8 layers, 8 heads, d = 512, plain MHA (no GQA), 32-token context, random weights.",
                   "Compute-optimal philosophy: train smaller models on more tokens (1–1.4 T)."],
        look_for=["Two parallel linear layers (gate, up) multiplied together: the 'mul' op.", "RMSNorm instead of LayerNorm.",
                  "Causal triangular attention maps, as in GPT-2."]),

    "gpt-oss-style": dict(
        year="Aug 2025 (gpt-oss-20b / gpt-oss-120b, Apache-2.0 open weights)", org="OpenAI", paper="gpt-oss model card",
        tagline="The most recent GPT-family architecture OpenAI has published: a sparse Mixture-of-Experts Transformer.",
        known_for=["OpenAI's first open-weights language models since GPT-2; 120 B runs on a single 80 GB GPU, 20 B on a 16 GB device.",
                   "Note: the internals of GPT-4 / GPT-5 are not public. gpt-oss is the closest public representation of current GPT design."],
        innovations=["Mixture-of-Experts MLP: a router picks top-4 of 32 (20b) or 128 (120b) experts per token, so only 3.6 B / 5.1 B of 21 B / 117 B parameters are active.",
                     "Grouped-query attention (64 query heads share 8 KV heads) and alternating full / sliding-window (128-token) attention layers.",
                     "Learned per-head attention sinks, RMSNorm, SwiGLU experts, RoPE extended to 131 k context, MXFP4-quantized expert weights."],
        technical=["Real gpt-oss-20b: 24 layers, d = 2880, 64 heads × 64, 8 KV heads, 32 experts top-4, ~201 k vocabulary.",
                   "THIS VIEWER is a scaled-down model with the same block design: 8 layers, d = 256, 8 heads / 2 KV heads, 8 experts top-2, 8-token window on alternate layers, random weights.",
                   "Details above are from the public model card as I know them; check the card for exact numbers."],
        look_for=["Router plane next to each MoE: tokens × experts, only the top-k light up.", "Attention maps: banded (sliding window) on every other layer.",
                  "K and V lanes repeat in groups: that is GQA sharing."]),

    "unet": dict(
        year="May 2015 (MICCAI 2015)", org="University of Freiburg (Ronneberger, Fischer, Brox)", paper="U-Net: Convolutional Networks for Biomedical Image Segmentation",
        tagline="Encoder–decoder with skip connections that copy detail across.",
        known_for=["Won the ISBI 2015 cell-tracking challenge by a large margin with very few training images.",
                   "The denoising network inside DDPM, Stable Diffusion and most image diffusion models is a U-Net."],
        innovations=["Contracting path (context) + expanding path (localization), with feature maps from the encoder concatenated into the decoder at each scale.",
                     "Heavy elastic-deformation data augmentation, enabling training from tens of annotated images."],
        technical=["Each level: two 3×3 convs + ReLU; down by 2×2 max-pool, up by 2×2 transposed conv.",
                   "This viewer: 4 levels, base 32 channels, BatchNorm, same-padding, 64×64 input, 2-class output."],
        look_for=["The 'U': cat ops joining a decoder tensor with the encoder tensor of the same resolution.", "Transposed convs doubling spatial size."]),

    "vae": dict(
        year="Dec 2013 (ICLR 2014)", org="Kingma & Welling (University of Amsterdam)", paper="Auto-Encoding Variational Bayes",
        tagline="An autoencoder whose code is a distribution, so you can sample from it.",
        known_for=["One of the founding deep generative models; the latent-space autoencoder in Stable Diffusion is a VAE."],
        innovations=["The encoder outputs a mean μ and log-variance of a Gaussian over the latent z; z = μ + σ ⊙ ε with ε ~ N(0, I) (the reparameterization trick) keeps sampling differentiable.",
                     "Trained by maximizing the ELBO: reconstruction term + KL(q(z|x) ‖ N(0, I)), which keeps the latent space smooth and sample-able."],
        technical=["This viewer: conv encoder (28×28 → 7×7), 16-dim latent, transposed-conv decoder, sigmoid output.",
                   "Each run draws new noise ε, so the decoded image changes with the same input."],
        look_for=["The fork into μ and logvar, the randn_like noise node, and the mul/add that implement z = μ + σ·ε.", "The decoder mirrors the encoder."]),

    "dcgan-generator": dict(
        year="Nov 2015 (ICLR 2016); GAN idea: Jun 2014", org="Radford, Metz & Chintala (indico / FAIR); GANs by Goodfellow et al.", paper="Unsupervised Representation Learning with Deep Convolutional GANs",
        tagline="Turns a 100-number noise vector into a 64×64 image with transposed convolutions.",
        known_for=["Made GAN training stable enough to be routine; latent-space arithmetic (man with glasses − man + woman…)."],
        innovations=["Architecture guidelines: strided (transposed) convs instead of pooling, BatchNorm, no fully-connected layers, ReLU in G, LeakyReLU in D, Tanh output."],
        technical=["z (100×1×1) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64.", "Trained adversarially against the discriminator (random weights in this viewer)."],
        look_for=["Spatial size doubles at every transposed conv while channels halve."]),

    "dcgan-discriminator": dict(
        year="Nov 2015 (ICLR 2016); GAN idea: Jun 2014", org="Radford, Metz & Chintala; GANs by Goodfellow et al.", paper="Unsupervised Representation Learning with Deep Convolutional GANs",
        tagline="A CNN classifier that says 'real' or 'fake'; the generator's training signal.",
        known_for=["The adversary in the GAN game: min over G, max over D of log D(x) + log(1 − D(G(z)))."],
        innovations=["Strided convs for learned down-sampling, LeakyReLU(0.2) to keep gradients alive for the generator."],
        technical=["3×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1 (sigmoid probability of 'real')."],
        look_for=["It is the generator's mirror image: same shapes in reverse."]),

    "ddpm-unet": dict(
        year="Jun 2020 (NeurIPS 2020)", org="UC Berkeley (Ho, Jain, Abbeel); idea: Sohl-Dickstein et al. 2015", paper="Denoising Diffusion Probabilistic Models",
        tagline="The noise-prediction network behind modern image diffusion.",
        known_for=["Made diffusion models competitive with GANs; ancestor of Stable Diffusion, DALL·E 2, Imagen (latent diffusion: Rombach et al., Dec 2021)."],
        innovations=["Train a network ε_θ(x_t, t) to predict the noise that was added to an image at step t, with a plain MSE loss.",
                     "Generate by starting from pure noise and denoising step by step (1000 steps in the paper; DDIM and others use far fewer)."],
        technical=["U-Net with GroupNorm, SiLU, residual blocks, self-attention at low resolution, and a sinusoidal timestep embedding injected into every block.",
                   "This viewer: scaled-down (base 32 channels, 3 resolutions, one block per level, attention at 8×8); the paper's CIFAR-10 model is 35.7 M parameters.",
                   "Two inputs: the noisy image x_t and the timestep t."],
        look_for=["The 't' input branch: sinusoidal embedding → MLP → added to each ResBlock.", "Skip concatenations like the U-Net.", "Attention only at the lowest resolution."]),
}

