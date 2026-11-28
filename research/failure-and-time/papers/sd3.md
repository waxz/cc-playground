# SD3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

| | |
|---|---|
| Authors | Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach |
| Venue / year | ICML 2024 |
| Link | https://arxiv.org/abs/2403.03206 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) (corpus addition) |
| Source | full text read (Sections 2–5) |

## 1. Problem and position in the lineage

[Rectified flow](rectified-flow.md) and [flow matching](flow-matching.md) promised straighter, cheaper sampling but had not been shown to beat tuned diffusion formulations at scale. SD3 runs a controlled comparison of diffusion and rectified-flow formulations and timestep samplers, then scales a multimodal transformer (MM-DiT) trained with the winning recipe to 8B parameters. It was added to the corpus because it is the origin of two details that appear in the world models studied here: logit-normal timestep sampling (used by [Cosmos3](cosmos3.md) for action, image and audio) and the resolution-dependent timestep shift $\sigma=st/(1+(s-1)t)$ (Cosmos3's "rectified-flow shift", WorldReforge's inference "timestep shift 10").

## 2. Method

All formulations are written as $z_t=a_tx_0+b_t\epsilon$ trained with a weighted noise-prediction loss $\mathcal L_w=-\tfrac12\mathbb E_{t,\epsilon}\big[w_t\lambda_t'\Vert\epsilon_\Theta(z_t,t)-\epsilon\Vert^2\big]$, where $\lambda_t=\log(a_t^2/b_t^2)$ is the log signal-to-noise ratio. In this notation rectified flow ($a_t=1-t$, $b_t=t$) corresponds to $w_t^{RF}=\frac{t}{1-t}$, while $\epsilon$-prediction, $v$-prediction and EDM correspond to other weightings. Because $w_t$ and the timestep density $\pi(t)$ are interchangeable, the paper studies densities: uniform, **logit-normal** $\pi_{ln}(t;m,s)=\frac{1}{s\sqrt{2\pi}}\frac{1}{t(1-t)}\exp\!\big(-\frac{(\mathrm{logit}(t)-m)^2}{2s^2}\big)$, a heavy-tailed "mode" sampler, and cosine-mapped. For high resolutions, it argues that more pixels make a given noise level less destructive, and derives a shift that preserves the per-pixel uncertainty: $t_m=\frac{\sqrt{m/n}\,t_n}{1+(\sqrt{m/n}-1)t_n}$ for resolution ratio $m/n$. MM-DiT uses separate weights for text and image tokens joined in one attention operation.

## 3. Evidence

Training 61 formulations on ImageNet and CC12M and ranking them by CLIP score and FID (non-dominated sorting over 24 combinations of sampler settings, EMA and dataset), rectified flow with logit-normal sampling (rf/lognorm(0.00, 1.00)) is the most consistent performer and outperforms rectified flow with uniform sampling, confirming that intermediate timesteps matter most; it is also robust to reducing the number of sampling steps. MM-DiT outperforms DiT, cross-attention DiT and UViT variants. Scaling to 8B parameters gives smooth decreases in validation loss that correlate with human preference, and the final models outperform SDXL and other open models in human evaluation of prompt following and typography.

## 4. Critical analysis

SD3's key lesson for world models is that "flow matching" is not a single objective: performance depends on the joint choice of path, weighting and timestep density, and the optimum shifts with the number of tokens. Video world models have many more tokens per sample than images, which is why Cosmos3 uses shifts of 1/3/5 at 256p/480p/720p in pretraining and 3/5/10 in mid-training, and samples with shifts of 5 or 10. When a post-training paper reports only "backbone-specific $\omega(\sigma)$", as WorldReforge does, the effective objective is unspecified, and any auxiliary loss computed in another parameterisation changes it further. The paper's evidence is for text-to-image, with FID and CLIP score as metrics; no claim is made about physical or action-conditioned accuracy, and the optimal weighting for predicting small contact-driven motions could differ from that for aesthetic images, a question no paper in the corpus tests.

## 5. Cross-references

**Builds on.** [Rectified flow](rectified-flow.md), [flow matching](flow-matching.md), [DiT](dit.md), [LDM](ldm.md).
**Extended by.** [Cosmos3](cosmos3.md) adopts logit-normal sampling, mode sampling for video and the shift reparameterisation; its mixture-of-transformers routes modalities to separate weights much as MM-DiT does.
**Contrasts with.** [DDPM](ddpm.md)'s uniform timestep sampling with $\epsilon$-prediction.

## 6. Relevance to the study question

Question 1: SD3 explains the weighting $\omega(\sigma)$ and timestep shift that complete the flow-matching objective in current world models, and therefore what a phrase such as WorldReforge's "backbone-specific training weight" leaves undetermined.
