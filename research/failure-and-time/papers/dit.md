# DiT: Scalable Diffusion Models with Transformers

| | |
|---|---|
| Authors | William Peebles, Saining Xie |
| Venue / year | ICCV 2023 |
| Link | https://arxiv.org/abs/2212.09748 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Until DiT, diffusion models used convolutional U-Nets. DiT asks whether the U-Net inductive bias is necessary and shows that a plain Vision Transformer over latent patches, trained in an [LDM](ldm.md) latent space, matches or beats U-Nets and scales predictably with compute. DiT is the backbone family of almost every world model after 2023: [IRASim](irasim.md) is a DiT with frame-level adaptive layer norm, [DreamDojo](dreamdojo.md) builds on the Cosmos-Predict2.5 DiT, [LingBot-VA](lingbot-va.md) on the WAN DiT, and WorldReforge's Figure 3 depicts an action-conditioned DiT. Its adaptive layer-norm conditioning is also the most common route for injecting timestep and action embeddings.

## 2. Method

A latent of size $32\times32\times4$ (256×256 images through the Stable Diffusion VAE) is split into $p\times p$ patches ($p\in\{2,4,8\}$), linearly embedded with sinusoidal position embeddings, and processed by $N$ transformer blocks. The paper compares four ways to inject the timestep $t$ and class label $c$: in-context tokens, cross-attention, adaptive layer norm (adaLN), which regresses per-block scale and shift $(\gamma,\beta)$ from the sum of the $t$ and $c$ embeddings, and **adaLN-Zero**, which additionally regresses a residual gate $\alpha$ initialised to zero so that each block starts as the identity. The model is trained with the improved-DDPM objective ($\epsilon$-prediction plus learned variances) and sampled with 250 DDPM steps and classifier-free guidance. Model sizes S, B, L and XL span 0.3 to 118.6 Gflops.

## 3. Evidence

adaLN-Zero gives the lowest FID of the four conditioning schemes while being the most compute-efficient; at 400K iterations its FID is nearly half that of in-context conditioning. FID decreases consistently with Gflops, whether obtained from depth/width or from smaller patches (more tokens), and larger models are more compute-efficient in terms of FID per training flop. DiT-XL/2 reaches FID 2.27 on class-conditional ImageNet 256×256 and 3.04 at 512×512 with guidance, surpassing prior diffusion models including LDM.

## 4. Critical analysis

DiT's evidence is for class-conditional images; its relevance to world models lies in two transferable findings. First, the conditioning mechanism matters as much as model size: global adaLN with a single embedding per sample is what IRASim calls "video-level" conditioning, and IRASim shows that robot trajectories need the per-frame variant. DreamDojo's injection of chunked latent actions by adding them to the timestep embedding before adaLN, with a zero-initialised last layer, is a direct descendant of adaLN-Zero, whose zero-initialisation also explains why new action pathways can be grafted onto a pretrained video model without destroying it. Second, scaling with tokens implies quadratic attention costs for video, which is why world models factorise attention over space and time or compress temporally. The Gflops-based scaling claim measures FID on ImageNet, not physical accuracy; whether physical fidelity of world models scales the same way is unclear, and WorldReforge's comparison across Cosmos3 Edge, Nano and Super shows pretrained object error that is not monotone in scale (8.40, 9.07, 8.63 px).

## 5. Cross-references

**Builds on.** [LDM](ldm.md) (latent space), ViT, [DDPM](ddpm.md).
**Extended by.** [SD3](sd3.md) (multimodal DiT with rectified flow), [IRASim](irasim.md) (frame-level adaLN), [Cosmos3](cosmos3.md) and [LingBot-VA](lingbot-va.md) (mixture-of-transformers variants).
**Contrasts with.** U-Net backbones in [UniSim](unisim.md), [RoboDreamer](robodreamer.md) and [Video Diffusion Models](video-diffusion.md).

## 6. Relevance to the study question

Question 1: DiT defines the backbone and the conditioning interface (adaLN-Zero) through which actions are injected in current world models, and supplies the zero-initialisation trick used when post-training adds a new action adapter.
