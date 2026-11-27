# MAAC: Model-Augmented Actor-Critic: Backpropagating through Paths

| | |
|---|---|
| Authors | Ignasi Clavera, Violet Fu, Pieter Abbeel |
| Venue / year | ICLR 2020 |
| Link | https://arxiv.org/abs/2005.08068 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Model-based RL around 2019 ([PETS](pets.md), [MBPO](mbpo.md), STEVE) used learned models as black-box sample generators: imagined transitions augmented the data of a model-free learner or were scored by a planner. Clavera et al. argue that this wastes the model's most useful property, its differentiability. MAAC computes the policy gradient by backpropagating through H steps of a learned probabilistic ensemble and closing the rollout with a learned Q-function, an objective that interpolates between model-free actor-critic (H = 0) and pure shooting (H = ∞). It is the direct deep-RL ancestor of the "short imagined rollout + terminal value + pathwise gradient" scheme that [Dreamer](dreamer.md) uses in latent space, [SHAC](shac.md) uses with a differentiable simulator, and [PWM](pwm.md), [SuperMimic](supermimic.md) and [TGI](tgi.md) use with learned models.

## 2. Method

The actor maximises

$$J_\pi(\theta) = \mathbb{E}\Big[\sum_{t=0}^{H-1}\gamma^t r(\hat s_t) + \gamma^H Q_\psi(\hat s_H, a_H)\Big] + \beta\,\mathcal{H}(\pi_\theta),\qquad \hat s_{t+1}\sim \hat f_\phi(\hat s_t,a_t),\; a_t\sim\pi_\theta(\hat s_t),$$

with $\hat s_0$ sampled from the replay buffer, the gradient obtained by backpropagation through time using the reparameterised Gaussian policy and Gaussian model ([m-reparam](../math-cookbook.md#m-reparam), [m-short-horizon](../math-cookbook.md#m-short-horizon)). The model is a bootstrap ensemble of probabilistic networks ([m-ensembles](../math-cookbook.md#m-ensembles)), with one member drawn uniformly per rollout. Two Q-functions are trained on real and imagined transitions with STEVE-style model-based value-expansion targets, and an entropy bonus as in [SAC](sac.md) is added to the actor.

The theoretical contribution concerns *gradient* error rather than prediction error. Lemma 4.1 bounds

$$\mathbb{E}\big[\|\nabla_\theta J_\pi - \nabla_\theta \hat J_\pi\|^2\big] \le c_1(H)\,\epsilon_f + c_2(H)\,\epsilon_Q,$$

where $\epsilon_f$ is the maximum error in the model's Jacobians along the rollout and $\epsilon_Q$ the error in $\nabla Q$ at the horizon, under Lipschitz-gradient assumptions on $f,\hat f, Q, \hat Q$; $c_1$ grows with $H$ while $c_2$ is discounted by $\gamma^H$. Lemma 4.2 converts a gradient error into a bound on the total-variation distance between the policies after one gradient step, and Theorem 4.1 turns this into a Kakade–Langford-style lower bound on the return of the updated policy. The bound is used qualitatively: $H$ trades model-Jacobian error against critic-gradient error and is treated as a hyperparameter.

MAAC also states its departure from [SVG](svg.md) explicitly: SVG evaluated gradients on real trajectories, which requires inferring the model noise and, MAAC argues, re-introduces likelihood-ratio-type variance; MAAC rolls the model out instead.

## 3. Evidence

On four MuJoCo benchmark tasks (ant, half-cheetah, hopper, walker2d) MAAC is compared with [SAC](sac.md), [MBPO](mbpo.md), STEVE (re-implemented on SAC) and SVG(1). The paper reports reaching a return of 10,000 on half-cheetah in "just over 50 rollouts", about half the data required by prior model-based methods, and near-optimal ant performance within 150 rollouts (Fig. 2; curves, no tables). On a double integrator with linear dynamics and quadratic cost, where the true gradient is analytic, the L1 policy-gradient error first decreases and then increases with $H$ when the learned model is used, while with the true model it decreases monotonically as the critic's contribution is discounted (Fig. 3); this is the only direct test of Lemma 4.1. Ablations show that setting $H=0$ causes the largest performance drop, that STEVE targets matter little, and that single-sample pathwise estimates are insufficient on ant. Test-time CEM planning on top of the converged policy changes final returns modestly (ant 3.06k → 3.97k, half-cheetah 10.7k → 10.9k, with overlapping standard deviations).

## 4. Critical analysis

MAAC's objective is now standard, and the paper's emphasis on derivative accuracy, rather than one-step prediction accuracy, anticipates [PWM](pwm.md)'s finding that a less accurate but smoother model can give better policies. Its evidence, however, supports less than its framing. The gradient-error experiment uses a linear system, where none of the pathologies that later dominated the discussion (chaos, stiff contact, discontinuities; see [PIPPS](pipps.md), [Suh et al.](suh-diffsim-gradients.md)) arise; the U-shape in $H$ is therefore a statement about compounding model-Jacobian error only. The constants $c_1(H), c_2(H)$ depend on Lipschitz constants that contact dynamics do not have, so the bound is vacuous exactly where [AHAC](ahac.md) and TGI need guidance. The benchmarks are short-horizon gym tasks with dense, smooth rewards and learned models with abundant data; there is no robotics hardware, no terrain, and no examination of how gradient quality varies within a rollout. The comparison with SVG(1) uses SVG's one-step variant and therefore does not test the decoupling question that SVG(∞) raised and that [DMO](amigo-decoupled.md) later re-examined with a modernised MAAC baseline.

## 5. Cross-references

- Builds on: [SVG](svg.md) (pathwise gradients through learned models); [PETS](pets.md) (probabilistic ensembles); [SAC](sac.md) and [TD3](td3.md) (entropy bonus, twin critics); [MBPO](mbpo.md) (short branched rollouts from replay states).
- Contrasts with: [MBPO](mbpo.md), which uses the same model only as a data generator.
- Extended by: [Dreamer](dreamer.md) (the same actor objective with λ-returns in a latent RSSM); [SHAC](shac.md) (same structure through a differentiable simulator); [PWM](pwm.md) and [SuperMimic](supermimic.md) (learned models at scale); [DMO](amigo-decoupled.md) (uses a modernised MAAC as baseline and shows that decoupling forward and backward passes helps).

## 6. Relevance to TGI

TGI's policy objective is MAAC's with three substitutions: a deterministic, physics-structured world model conditioned on simulator terrain instead of a probabilistic ensemble; a state-value critic trained on λ-returns ([m-lambda-return](../math-cookbook.md#m-lambda-return)) instead of twin Q-functions; and a per-sample, event-dependent truncation of the horizon instead of a fixed $H$. Starting imagined rollouts from replay-buffer simulator states, as MAAC does, keeps compounding error bounded by the 32-step horizon. MAAC's gradient-error bound gives the formal reason for the event cut: when the model's Jacobian error $\epsilon_f$ is locally large (at a foot landing on an edge, where training data are sparse and the true dynamics are discontinuous), the bootstrapped critic is the lower-error source of gradient, so the cut moves the effective horizon to that step.
