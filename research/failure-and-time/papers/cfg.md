# CFG: Classifier-Free Diffusion Guidance

| | |
|---|---|
| Authors | Jonathan Ho, Tim Salimans |
| Venue / year | NeurIPS 2021 Workshop on Deep Generative Models (arXiv 2022) |
| Link | https://arxiv.org/abs/2207.12598 |
| Theme | [t1-training-framework](../themes/t1-training-framework.md) (corpus addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Classifier guidance (Dhariwal and Nichol, 2021) improves sample fidelity of conditional diffusion models by adding the gradient of a separately trained noisy-image classifier, trading diversity for quality. This requires an extra classifier trained on noisy inputs and was suspected of acting like an adversarial attack on that classifier. Classifier-free guidance (CFG) obtains the same trade-off with no classifier, by mixing conditional and unconditional predictions of a single network. CFG is the standard knob for condition strength in every text- or action-conditioned video model in the corpus: [UniSim](unisim.md) applies it to *actions*, [RoboDreamer](robodreamer.md) composes several conditions with it, [Cosmos3](cosmos3.md) and [LingBot-VA](lingbot-va.md) train with 10% condition dropout and sample with guidance scales of 3–5, and [DreamDojo](dreamdojo.md) disables it. It was added to the corpus because guidance strength directly affects action-following and is a hidden variable in world-model comparisons.

## 2. Method

During training the condition $c$ is replaced by a null token $\varnothing$ with probability $p_{\text{uncond}}$ (0.1 or 0.2 in the paper), so one network learns both $\epsilon_\theta(z_\lambda,c)$ and $\epsilon_\theta(z_\lambda)$. At sampling time the two are extrapolated:
$$\tilde\epsilon_\theta(z_\lambda,c)=(1+w)\,\epsilon_\theta(z_\lambda,c)-w\,\epsilon_\theta(z_\lambda).$$
In score form, since $\epsilon\propto-\nabla\log p$, this approximates sampling from $\tilde p(z\mid c)\propto p(z\mid c)\,p(c\mid z)^w$ with an *implicit* classifier $p(c\mid z)\propto p(z\mid c)/p(z)$. The same linear combination applies to $v$- or velocity-prediction networks, which is how flow-matching models use it.

## 3. Evidence

On class-conditional ImageNet at 64×64 and 128×128, sweeping $w$ traces an FID–Inception-score trade-off similar to classifier guidance: small $w$ (around 0.1–0.3) gives the best FID, larger $w$ gives higher Inception score and less diversity. $p_{\text{uncond}}=0.1$ or $0.2$ works better than 0.5, so only a small fraction of capacity needs to go to the unconditional model. Results hold for different numbers of sampling steps.

## 4. Critical analysis

CFG is a sampling-time distortion of the learned conditional: with $w>0$ the sampler no longer draws from $p(z\mid c)$ but from a sharpened distribution that over-represents outcomes strongly associated with $c$. For a world model this has a physical meaning. Guidance on actions (UniSim's $\eta$) amplifies the action's effect, which may help a weakly controllable model follow commands but also exaggerates typical consequences and suppresses rarer ones; if failures are rare given an action in the training data, strong guidance pushes predictions toward the modal, successful outcome. Conversely, guidance on text alone can override the action signal. Different papers use incompatible conventions (WorldReforge reports guidance scale 1 for Cosmos3, 0 for DreamDojo and 5 for LingBot-VA), so cross-backbone comparisons at "default" guidance conflate model quality with sampler settings. None of the world-model papers in the corpus reports action-following or physical metrics as a function of guidance scale, which would be a cheap and informative ablation.

## 5. Cross-references

**Builds on.** [DDPM](ddpm.md); classifier guidance; see also [m-diffusion-guidance](../../tgi-lineage/math-cookbook.md#m-diffusion-guidance).
**Used by.** [Video Diffusion Models](video-diffusion.md), [LDM](ldm.md), [DiT](dit.md), [UniSim](unisim.md) (action guidance), [RoboDreamer](robodreamer.md) (compositional guidance), [Cosmos3](cosmos3.md), [LingBot-VA](lingbot-va.md).
**Contrasts with.** Reward or cost guidance in [Diffusion Forcing](diffusion-forcing.md), which steers samples with an external gradient rather than an implicit classifier.

## 6. Relevance to the study question

Question 1: CFG is the inference-time component of action conditioning; its scale changes how strongly predicted futures depend on the action and, by sharpening toward typical outcomes, plausibly how often a world model predicts failures (question 2).
