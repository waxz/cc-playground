# NPMP: Neural Probabilistic Motor Primitives for Humanoid Control

| | |
|---|---|
| Authors | Josh Merel, Leonard Hasenclever, Alexandre Galashov, Arun Ahuja, Vu Pham, Greg Wayne, Yee Whye Teh, Nicolas Heess |
| Venue / year | ICLR 2019 |
| Link | https://arxiv.org/abs/1811.11711 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2018 single-clip experts in the style of [DeepMimic](deepmimic.md) could reproduce individual mocap snippets robustly, but each was a separate, time-indexed network. NPMP asks how thousands of such experts can be compressed into one module that (i) performs one-shot imitation of a reference trajectory it has not seen and (ii) exposes a latent space that new high-level controllers can reuse. Its answer is an inverse model with a latent-variable bottleneck, trained entirely offline by supervised cloning of the experts.

NPMP is the earliest latent-skill method in this lineage and the template for the "tracker distilled into a latent space" pattern later scaled by [PULSE](pulse.md) and [MaskedMimic](maskedmimic.md). It is also the paper [SuperMimic](supermimic.md) lists first in its Table 1, with "SVG(0)" for both the mimic and downstream stages and "BC" for the latent stage. It was added here (depth 3) because it fills the gap between DeepMimic's per-clip experts and ASE's reusable embeddings, and because its use of a pathwise estimator makes it a necessary point of comparison for the analytic-gradient line.

## 2. Method

Experts are trained per snippet (2707 snippets of 2–6 s from CMU mocap) with a DeepMimic-style tracking reward, RSI and early termination, using the off-policy actor–critic SVG(0) with Retrace. SVG(0) differentiates a learned Q-function with respect to the reparameterised action ([pathwise gradient and SVG](../math-cookbook.md#m-reparam)); it does not differentiate through dynamics, so it is a value-gradient, not a model-gradient, method.

The student is an encoder $q(z_t|z_{t-1},x_t)$ over a short look-ahead of reference states $x_t=[s_t,\dots,s_{t+K}]$, $K=5$, and a decoder (the motor primitive policy) $\pi(a_t|z_t,s_t)$. The prior is an AR(1) process, $z_t=\alpha z_{t-1}+\sqrt{1-\alpha^2}\,\epsilon$, $\alpha=0.95$, which makes temporally adjacent snippets close in latent space. The objective is the ELBO

$$\sum_t \mathbb E_q\big[\log\pi(a_t|s_t,z_t) + \beta\log p_z(z_t|z_{t-1}) - \log q(z_t|z_{t-1},x_t)\big],$$

a conditional information bottleneck between future trajectory and action ([latent skills](../math-cookbook.md#m-latent-skills)). Two offline data schemes are compared ([distillation](../math-cookbook.md#m-distillation)): behavioural cloning from noisy expert rollouts that log the expert's mean action (DART-like), and linear-feedback policy cloning (LFPC), which logs the expert's action–state Jacobian $J_t=\partial\mu_E/\partial s$ along one nominal trajectory and trains on perturbed states with targets $a^\star_t+J_t\,\delta s$. For reuse, a high-level policy outputs $z_t$ and is trained by SVG(0) with Retrace on sparse task rewards.

## 3. Evidence

For single-skill transfer, LFPC from one trajectory performs on par with behavioural cloning from hundreds of trajectories under small and moderate action noise (standard deviation 0.05 and 0.1 on normalised actions), while open-loop replay fails under any noise. Compressing all 2707 experts works best with cloning from 100 noisy rollouts per expert and $\beta=0.1$; LFPC is somewhat worse at full scale; a 20-D instead of 60-D latent, weaker regularisation, or $\alpha=0$ (i.i.d. latents) all hurt. Performance is reported relative to experts on training and held-out clips (Fig. 4, no table). For held-out clips below 50% relative performance, directly optimising the latent sequence raises median relative performance from 43% to 78%, indicating that the encoder rather than the decoder is the main failure point. Reuse: high-level controllers solve go-to-target, target-speed, bumpy-terrain and a vision-based gap-jumping course with humanlike motion; more regularised modules train faster than $\beta=0.001$.

## 4. Critical analysis

NPMP shows that large expert libraries can be compressed offline into a stabilising, reusable decoder; one-shot imitation with open-loop latents succeeds only because the decoder closes the loop on state. The cost is hidden upstream: 2707 experts must each be trained by RL, so total compute exceeds that of a universal tracker trained directly, which is the route [PHC](phc.md) took. Offline cloning provides no correction on states the student visits but the experts never did, the drift that DAgger-style online distillation in [PULSE](pulse.md) removes. Quantitative evidence is mostly plotted, and downstream tasks are compared within NPMP variants rather than against alternative skill spaces. The SVG(0) label invites confusion with model-based analytic gradients: the gradient flows through a critic, which has its own bias, and not through any model of the body, so NPMP is better grouped with [DDPG](ddpg.md)/[SAC](sac.md)-style learners than with [SuperTrack](supertrack.md) or [ControlVAE](controlvae.md). LFPC is the one place NPMP uses Jacobians of a controller, an idea related to linear-feedback trees and DDP ([LQR and DDP](../math-cookbook.md#m-lqr-ddp)).

## 5. Cross-references

- Builds on: [DeepMimic](deepmimic.md) (expert training recipe); [SVG](svg.md) (policy optimiser); DART, DAgger and Mordatch et al. (2015) for cloning feedback controllers.
- Extended by: [PULSE](pulse.md) (online distillation, AR-like latent regulariser, learned prior); [MaskedMimic](maskedmimic.md); [AMP](amp.md) uses an NPMP-like latent model as its comparison baseline.
- Contrasts with: [ASE](ase.md) (adversarial pre-training instead of cloning); [ControlVAE](controlvae.md) (ELBO on states maximised through a learned world model instead of on actions from experts).

## 6. Relevance to TGI

NPMP's direct influence on TGI is small; TGI has no latent action space or expert library. Its relevance is clarifying: SuperMimic's Table 1 places NPMP's "SVG(0)" beside its own "MBRL", and the distinction between differentiating a critic (NPMP) and differentiating a learned dynamics model over a horizon with a terminal critic (SHAC, SuperMimic, TGI) is exactly the design choice that TGI's gradient-cut mechanism refines.
