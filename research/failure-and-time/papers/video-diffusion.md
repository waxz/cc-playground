# VDM: Video Diffusion Models

| | |
|---|---|
| Authors | Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet |
| Venue / year | NeurIPS 2022 |
| Link | https://arxiv.org/abs/2204.03458 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Video Diffusion Models (VDM) is the first demonstration that a standard Gaussian diffusion model, with minimal changes to the image architecture, generates temporally coherent video. It introduces three ingredients that later robot world models reuse: a space–time factorised network, joint training on images and videos, and a principled way to condition on earlier frames for extending videos (reconstruction guidance). [UniSim](unisim.md) uses VDM's video U-Net directly, [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md) adopt its factorised spatial/temporal attention in transformer form, and WorldReforge cites it as reference [22] for its generative objective.

## 2. Method

The model diffuses a block of frames jointly, in pixel space, with the continuous-time Gaussian process $z_t=\alpha_tx+\sigma_t\epsilon$ and a cosine schedule. The denoiser is a 3D U-Net factorised over space and time: spatial 2D convolutions and spatial attention operate within each frame, and a temporal attention layer (with relative position embeddings) mixes information across frames at each spatial location. Removing the temporal attention turns the network into an image model, which permits *joint training*: batches mix videos with independent images, and the paper finds this improves sample quality. Some models use the $v$-parameterisation $v=\alpha_t\epsilon-\sigma_tx$ of Salimans and Ho, which avoids the colour shifts of $\epsilon$-prediction at high resolution.

For conditional generation (longer videos, higher resolution), the paper compares naive "replacement" (substituting the noised known frames into the sample) with **reconstruction-guided sampling**, which adds the gradient of the reconstruction error of the conditioning frames,
$$\tilde x^b_\theta(z_t)=\hat x^b_\theta(z_t)-\frac{w_r\alpha_t}{2}\nabla_{z^b_t}\Vert x^a-\hat x^a_\theta(z_t)\Vert^2 ,$$
so that the generated block $b$ is consistent with the observed block $a$. Text conditioning and classifier-free guidance ([CFG](cfg.md)) are used for text-to-video, with cascaded spatial and temporal super-resolution.

## 3. Evidence

VDM reports state-of-the-art results at the time on unconditional UCF-101 generation and on Kinetics-600 frame prediction (FVD), and shows on a text-conditioned dataset of 10 million captioned videos that joint image–video training and classifier-free guidance both improve FVD and Inception-based metrics. Reconstruction guidance yields more coherent autoregressive extensions and spatial super-resolution than the replacement method.

## 4. Critical analysis

VDM works in pixel space and at small resolutions (16 frames at 64×64 for the base text-conditioned model), so its cost scales poorly; [latent diffusion](ldm.md) and its video successors move the same idea into a compressed latent. Its autoregressive extension conditions each new block on previously *generated* frames, which reintroduces exposure bias: errors in one block are treated as ground truth by the next. Reconstruction guidance improves consistency within the training horizon but does not address drift over long rollouts, the problem that [Diffusion Forcing](diffusion-forcing.md) later targets with per-frame noise levels. The joint image–video finding anticipates the data-mixing strategy of [UniSim](unisim.md) and [Cosmos3](cosmos3.md), where static images supply appearance diversity that video and action data lack.

## 5. Cross-references

**Builds on.** [DDPM](ddpm.md) for the objective; [CFG](cfg.md) for guidance.
**Extended by.** [UniSim](unisim.md) (action-conditioned video U-Net), [RoboDreamer](robodreamer.md) (compositional text-to-video on the same family), [IRASim](irasim.md) and [Ctrl-World](ctrl-world.md) (space–time factorised transformers).
**Contrasts with.** [DiT](dit.md)-based video models that replace the U-Net with a transformer over spatio-temporal latent patches.

## 6. Relevance to the study question

Question 1: VDM introduces the video architecture (space–time factorisation), the data recipe (joint image–video training) and the conditioning mechanism (clean or guided past frames) that action-conditioned world models adopt; its block-wise extension is the first instance of the long-horizon drift problem the theme report discusses.
