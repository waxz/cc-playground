# DDPM: Denoising Diffusion Probabilistic Models

| | |
|---|---|
| Authors | Jonathan Ho, Ajay Jain, Pieter Abbeel |
| Venue / year | NeurIPS 2020 |
| Link | https://arxiv.org/abs/2006.11239 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Diffusion probabilistic models (Sohl-Dickstein et al., 2015) define a generative model as the reversal of a fixed noising process, but before DDPM their samples were not competitive with GANs. DDPM shows that a particular parameterisation and a simplified, re-weighted variational objective produce high-quality images, and it establishes the equivalence between this objective and denoising score matching over many noise levels. Every video world model in this corpus from [Video Diffusion Models](video-diffusion.md) to [Cosmos3](cosmos3.md) and [WorldReforge](worldreforge.md) trains a network with a descendant of DDPM's loss. The TGI cookbook summarises DDPM and guidance in [m-diffusion-guidance](../../tgi-lineage/math-cookbook.md#m-diffusion-guidance); this report records the parts that matter for world-model training.

## 2. Method

The forward process adds Gaussian noise with a variance schedule $\beta_1,\dots,\beta_T$ ($T=1000$, linear from $10^{-4}$ to $0.02$), which admits the closed form
$$q(x_t\mid x_0)=\mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,(1-\bar\alpha_t)I\big),\qquad \bar\alpha_t=\textstyle\prod_{s\le t}(1-\beta_s).$$
The reverse process $p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I)$ is trained by the variational bound $\mathbb E[-\log p_\theta(x_0)]\le \mathbb E_q[L_T+\sum_{t>1}L_{t-1}+L_0]$, where each $L_{t-1}$ is a KL divergence between Gaussians. Parameterising the mean through a noise predictor, $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\big)$, turns each KL into a weighted squared error on $\epsilon$. DDPM then drops the weights:
$$L_{\text{simple}}=\mathbb E_{t,x_0,\epsilon}\big\Vert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\Vert^2 .$$
The authors interpret $L_{\text{simple}}$ as a re-weighted bound that down-weights the easy small-$t$ terms, and they show that $\epsilon$-prediction makes sampling resemble annealed Langevin dynamics with the learned score $\nabla\log p_t\approx-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$. The network is a U-Net with self-attention at low resolution and sinusoidal time embeddings.

## 3. Evidence

On unconditional CIFAR-10, DDPM reaches an Inception score of 9.46 and FID 3.17 (5.24 against the test set), better than most published GANs at the time, with 256×256 LSUN and CelebA-HQ samples of comparable quality to ProgressiveGAN. The ablation shows that predicting $\epsilon$ with $L_{\text{simple}}$ gives the best FID, whereas predicting $\tilde\mu$ works only with the true bound, and learning the reverse variances is unstable. The log-likelihood ($\le$3.75 bits/dim on CIFAR-10 for $L_{\text{simple}}$, $\le$3.70 for the true bound) is worse than likelihood-based competitors: the simplified objective trades likelihood for sample quality.

## 4. Critical analysis

DDPM's central empirical fact, that the *weighting* across noise levels rather than the bound itself determines sample quality, is the thread that runs through the whole theme. The later choices of $v$-prediction ([Video Diffusion Models](video-diffusion.md)), flow-matching velocity ([flow matching](flow-matching.md)), logit-normal timestep sampling ([SD3](sd3.md)) and the "backbone-specific" $\omega(\sigma)$ in WorldReforge are all re-weightings of the same family of denoising losses. A consequence for world models is that adding a loss in a different parameterisation, as WorldReforge does with its $x_0$-space Interaction Alignment term, implicitly changes the noise-level weighting. DDPM's 1,000 sampling steps were impractical; deterministic samplers, distillation and straighter paths ([rectified flow](rectified-flow.md)) reduced this to tens or a few steps, which is what allows real-time world models such as the distilled [DreamDojo](dreamdojo.md).

## 5. Cross-references

**Builds on.** Sohl-Dickstein et al. (2015) and denoising score matching (Song and Ermon, 2019).
**Extended by.** [Latent diffusion](ldm.md) moves the process to an autoencoder latent; [Video Diffusion Models](video-diffusion.md) to video; [classifier-free guidance](cfg.md) to conditional sampling; [Diffusion Forcing](diffusion-forcing.md) to per-token noise levels.
**Contrasts with.** [SV2P](sv2p.md) obtains stochasticity from a single variational latent; [flow matching](flow-matching.md) replaces the discrete Markov chain by a continuous velocity field.

## 6. Relevance to the study question

Question 1: DDPM supplies the objective template (noise a clean target, regress a parameterised denoiser, weight across noise levels) on which every modern action-conditioned video world model is trained, and its ELBO interpretation is used in the theme report's Mathematical background. It has no direct bearing on failure data or time embedding beyond the sinusoidal noise-level embedding.
