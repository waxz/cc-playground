# LDM: High-Resolution Image Synthesis with Latent Diffusion Models

| | |
|---|---|
| Authors | Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer |
| Venue / year | CVPR 2022 |
| Link | https://arxiv.org/abs/2112.10752 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Pixel-space diffusion spends most of its capacity on imperceptible high-frequency detail and is expensive to train and sample. LDM separates the problem into *perceptual compression*, done once by an autoencoder, and *semantic generation*, done by a diffusion model in the autoencoder's latent space. This two-stage design is now universal: the WAN2.2 VAE under [DreamDojo](dreamdojo.md), [LingBot-VA](lingbot-va.md) and [Cosmos3](cosmos3.md), the SDXL VAE in [IRASim](irasim.md) and the frozen VAE in [WorldReforge](worldreforge.md) all instantiate it. LDM also introduces cross-attention conditioning, which became the default route for text (and in some world models, action) conditions.

## 2. Method

An encoder $\mathcal E$ maps an image $x\in\mathbb R^{H\times W\times3}$ to $z=\mathcal E(x)\in\mathbb R^{h\times w\times c}$ with downsampling factor $f=H/h\in\{1,2,4,8,16,32\}$, and a decoder reconstructs $\tilde x=\mathcal D(z)$. The autoencoder is trained with a perceptual (LPIPS) loss and a patch-based adversarial loss, and regularised either by a small KL penalty toward $\mathcal N(0,I)$ ("KL-reg") or by a vector-quantisation layer absorbed into the decoder ("VQ-reg"). The diffusion model is then trained in latent space with the DDPM objective
$$L_{LDM}=\mathbb E_{\mathcal E(x),y,\epsilon,t}\big\Vert\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))\big\Vert^2 ,$$
where $\tau_\theta$ encodes a condition $y$ (text, class, semantic map) that enters the U-Net through cross-attention, $\mathrm{softmax}(QK^\top/\sqrt d)V$ with queries from the image features and keys/values from $\tau_\theta(y)$. Spatially aligned conditions (semantic maps, low-resolution images) are instead concatenated to $z_t$.

## 3. Evidence

The downsampling sweep on ImageNet shows that $f=1$ and $f=2$ train slowly, $f=32$ loses too much information, and LDM-4 to LDM-16 balance efficiency and fidelity, with an FID gap of 38 between pixel-space LDM-1 and LDM-8 after 2M steps. LDMs reach competitive or state-of-the-art results on unconditional face and scene generation, class-conditional ImageNet (FID 3.60 with guidance), text-to-image (LAION), inpainting, super-resolution and semantic synthesis, at a fraction of the compute of pixel-space diffusion.

## 4. Critical analysis

The decisive design choice is that the autoencoder is trained separately and then frozen, so the generative model cannot recover information the latent discarded. For robot world models this matters in a specific way: small objects, gripper fingertips and contact regions occupy few latent cells after 8×–16× spatial (and, for video VAEs, 4× temporal) compression. WorldReforge's Interaction Alignment, which renormalises the loss over the few latent cells covering the object, is a response to this, and its masks are computed as fractional overlaps on the latent grid. A frozen tokenizer also caps PSNR-based evaluation: reconstruction error of the VAE itself bounds achievable PSNR, so gains of a fraction of a decibel are hard to interpret without the VAE's own reconstruction PSNR as reference, which robot world-model papers rarely report. LDM's cross-attention conditioning was designed for global conditions such as captions; [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md) show that per-frame action conditions need finer, frame-aligned routes.

## 5. Cross-references

**Builds on.** [DDPM](ddpm.md); VQGAN-style perceptual autoencoders.
**Extended by.** [DiT](dit.md) replaces the latent U-Net with a transformer; latent video models (Stable Video Diffusion in [Ctrl-World](ctrl-world.md), WAN in [DreamDojo](dreamdojo.md), [LingBot-VA](lingbot-va.md), [Cosmos3](cosmos3.md)) extend the latent to space–time.
**Contrasts with.** [Genie](genie.md) uses a discrete VQ tokenizer and a masked-token dynamics model instead of continuous latent diffusion.

## 6. Relevance to the study question

Question 1: LDM fixes the tokenisation stage of the modern world-model pipeline (frozen perceptual VAE, generator in latent space) and explains why contact-scale physics is under-represented in the loss, which motivates region-weighted objectives such as WorldReforge's IA.
