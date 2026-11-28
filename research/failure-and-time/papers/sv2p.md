# SV2P: Stochastic Variational Video Prediction

| | |
|---|---|
| Authors | Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, Sergey Levine |
| Venue / year | ICLR 2018 |
| Link | https://arxiv.org/abs/1710.11252 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Deterministic video predictors such as [Finn et al. 2016](finn-video-prediction.md) trained with squared error predict the expectation over possible futures and therefore blur exactly where the outcome is uncertain, for example when the unknown mass of a pushed toy determines how far it slides. SV2P introduces a latent variable so that each sample of the latent yields one sharp, plausible future. It is the first multi-frame stochastic predictor demonstrated on real videos, including action-conditioned robot pushing, and it is the ancestor of the variational latent dynamics models used in [PlaNet](../../tgi-lineage/papers/planet.md) and Dreamer.

## 2. Method

Given context frames $x_{0:c-1}$ (and optionally actions), the generative model factorises $p(x_{c:T}\mid x_{0:c-1},z)=\prod_t p_\theta(x_t\mid x_{0:t-1},z)$ with a latent $z\sim\mathcal N(0,I)$, either sampled once per video (time-invariant) or per frame (time-variant). The generator is the CDNA network of Finn et al. An inference network $q_\phi(z\mid x_{0:T})$ conditions on *all* frames, including the future, and the model is trained on the evidence lower bound
$$\mathcal L(x)=-\mathbb E_{q_\phi(z\mid x_{0:T})}\log p_\theta(x_{c:T}\mid x_{0:c-1},z)+D_{KL}(q_\phi(z\mid x_{0:T})\,\Vert\,p(z)),$$
the same reconstruction-plus-KL structure as the RSSM bound ([m-rssm-elbo](../../tgi-lineage/math-cookbook.md#m-rssm-elbo)). The authors stress that the posterior must see the future: a filtering posterior $q(z\mid x_{0:t-1})$ carries no information beyond the context and the generator learns to ignore it.

Training is staged because the latent is otherwise ignored or the KL collapses. Phase one trains the generator alone with $z$ sampled from the prior, which makes it deterministic; phase two switches on the inference network with zero KL weight, giving very low reconstruction error but a posterior far from the prior; phase three ramps the KL weight up so that sampling from $p(z)$ at test time becomes accurate.

## 3. Evidence

On a toy dataset of shapes moving in random directions, the deterministic model predicts the average of all directions while SV2P samples distinct, correct-looking motions. On the robotic pushing dataset of Finn et al. (action-conditioned), the BAIR robot pushing dataset (used action-free and action-conditioned) and Human3.6M, SV2P improves PSNR and SSIM over the same network without stochasticity and over other stochastic baselines when the best of many samples (100, matching the VPN comparison) is scored against the ground truth, and the advantage grows with the prediction horizon. Time-variant latents generalise better beyond the training horizon.

## 4. Critical analysis

SV2P demonstrates that a latent variable resolves the blur of deterministic prediction, but it also exposes two problems that remain. First, evaluation is ambiguous: scoring the best sample out of many rewards diversity and cannot detect samples that are implausible, while scoring a single sample penalises legitimate stochasticity. WorldReforge, which samples once with a fixed seed and scores against one recorded future, inherits the second problem. Second, the variational objective requires a careful KL schedule; posterior collapse and the three-phase recipe are symptoms of weak latent variable training that diffusion models later sidestep by injecting noise at every level of a fixed forward process ([DDPM](ddpm.md)). For action-conditioned robot prediction, the paper's own motivating example (unknown object mass) is a case of physical parameters that the action does not determine; a world model must represent this uncertainty rather than average it, which is relevant to failed grasps whose outcomes depend on contact details.

## 5. Cross-references

**Builds on.** [Finn et al. 2016](finn-video-prediction.md): the CDNA generator.
**Extended by.** [Visual Foresight](visual-foresight.md) uses stochastic predictors of this type for planning; [PlaNet](../../tgi-lineage/papers/planet.md) moves the stochastic latent into a recurrent state-space model.
**Contrasts with.** [DDPM](ddpm.md) and its successors obtain stochasticity from the denoising process rather than a single global latent.

## 6. Relevance to the study question

Question 1: SV2P marks the transition from deterministic regression to likelihood-based stochastic world models, and its ELBO is the conceptual parent of the diffusion ELBO. Question 2: its argument that action-conditioned outcomes are genuinely stochastic is the reason failure prediction cannot be evaluated by a single-sample pixel error without caveats.
