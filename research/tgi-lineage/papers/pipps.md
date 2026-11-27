# PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos

| | |
|---|---|
| Authors | Paavo Parmas, Carl Edward Rasmussen, Jan Peters, Kenji Doya |
| Venue / year | ICML 2018 (PMLR 80); arXiv 2019 |
| Link | https://arxiv.org/abs/1902.01240 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[PILCO](pilco.md) learned cart-pole swing-up in a handful of trials by propagating Gaussian state distributions through a Gaussian-process model with analytic moment matching (MM) and differentiating the resulting expected cost. MM is exact-gradient but inflexible: it requires closed-form moments, which rules out neural-network models and most policies. The natural replacement, sampling particles and differentiating them with the reparameterisation (RP) trick, had been reported not to work, and the failure had been attributed to local minima. Parmas et al. keep everything about PILCO fixed except the prediction scheme and show that the actual cause is gradient variance: RP gradients through long chains of nonlinear maps become essentially random, a phenomenon they name the *curse of chaos*. The paper is the first careful demonstration in model-based RL that an exact pathwise gradient of a smooth expected objective can be useless, and the source of the hybrid estimators (likelihood-ratio plus pathwise) that [Suh et al.](suh-diffsim-gradients.md), [Onoda et al.](onoda-composite-gradients.md) and, in a simple fixed-weight form, [TGI](tgi.md) use.

## 2. Method

The objective is $J(\theta) = \mathbb{E}_{\tau}[\sum_t c(x_t)]$ under particle rollouts of a GP model $\hat f$ with a deterministic policy. Three gradient estimators are compared (background in [m-reparam](../math-cookbook.md#m-reparam) and [m-pg-theorem](../math-cookbook.md#m-pg-theorem)):

- **RP**: backpropagate each particle's cost through the sampled path, $\frac{dc}{dx}\frac{dx}{d\theta}$, averaged over particles.
- **Model-based LR**: since the policy is deterministic, the stochasticity lives in the model, and the score is taken with respect to the *state* distribution, $\mathbb{E}\big[\sum_t \tfrac{d}{d\theta}\log p(x_{t+1}\mid x_t;\theta)\,(G_{t+1}-b_{t+1})\big]$, where $p(x_{t+1}\mid x_t;\theta)=\hat f(x_{t+1}\mid x_t,\pi(x_t;\theta))$ is differentiable through the policy.
- **Batch importance-weighted LR (BIW-LR)**: treat the particle cloud at time $t$ as a mixture $q(x_{t+1})=\frac1P\sum_i p(x_{t+1}\mid x_{i,t})$ and importance-weight every particle against every mixture component, with leave-one-out self-normalised baselines.

**Total propagation (TP)** fuses the two in one backward pass. At each time step it computes both the RP and LR gradients with respect to the distribution parameters $\zeta_t$ (mean and covariance of each particle's predictive Gaussian), estimates their sample variances $\sigma^2_{RP},\sigma^2_{LR}$ across particles, and passes back the inverse-variance-weighted combination

$$\frac{dG}{d\zeta_t} = k_{LR}\frac{dG^{LR}}{d\zeta_t} + (1-k_{LR})\frac{dG^{RP}}{d\zeta_t},\qquad k_{LR} = \frac{1}{1+\sigma^2_{LR}/\sigma^2_{RP}} .$$

Because the combination is formed recursively, TP effectively takes the union over all RP path lengths: short pathwise segments are used where they are informative, and the LR estimator takes over where long Jacobian products have blown up. Optimisation uses an RMSprop-like normaliser that uses the batch estimate $\mathbb{E}[g]^2+\mathbb{V}[g]/P$ directly.

## 3. Evidence

The central experiment (Fig. 2) fixes the random seed on cart-pole swing-up, moves the policy parameters along a random direction, and plots the objective and the projected gradient with 1000 particles. The averaged objective is smooth, and finite differences agree with LR and TP gradients; the RP gradient, which with a fixed seed is the exact gradient of the plotted sample objective, is accurate near the trained parameters but its variance jumps by a factor of about $4\times10^5$ at $\Delta\theta=1.5$, implying about $4\times10^8$ particles for an accurate estimate. Plots of the value as a function of the initial particle position explain why: at $\Delta\theta = 1.5$ the per-particle return is a fractal-like function of the start state, so its derivative carries no information about the derivative of its expectation. Over 0–500 particles (Fig. 3), RP variance at $\Delta\theta=1.5$ lies between $10^8$ and $10^{15}$, BIW-LR is orders of magnitude lower, and TP is lowest, with a 10–50 % variance reduction over BIW-LR for ≤250 particles. In learning experiments with 300 particles, PIPPS with TP matches PILCO's data efficiency on cart-pole and unicycle; on unicycle balancing TP succeeds in 0.94 of runs versus PILCO's 0.91. Gaussian resampling of particles at every step, which mimics MM, also removes the chaos, which the authors use to explain why MM worked.

## 4. Critical analysis

The paper's lasting contribution is diagnostic. It separates two properties that are easily conflated: smoothness of the *expected* objective and smoothness of *individual sample paths*. Pathwise estimators differentiate sample paths; when the system is chaotic (positive Lyapunov exponents, see [m-bptt](../math-cookbook.md#m-bptt)) the per-sample Jacobian product grows exponentially even though the expectation is benign. [Metz et al.](metz-gradients.md) later generalised this to Brax physics, learned optimisers and RNNs; [Suh et al.](suh-diffsim-gradients.md) list chaos as one of two causes of high first-order variance; [PWM](pwm.md) measures it on the double pendulum.

The evidence has clear limits. The tasks are two low-dimensional systems with GP models and episodes of tens of steps; the dramatic variance curves come from a single slice through parameter space. TP's inverse-variance weighting presumes that sample variance reflects estimator error, which [Suh et al.](suh-diffsim-gradients.md) show fails under "empirical bias" at near-discontinuities: a biased pathwise estimator can have near-zero sample variance and would be weighted up. [Onoda et al.](onoda-composite-gradients.md) (with Parmas as co-author) later argue that on standard dflex locomotion benchmarks a per-step inverse-variance scheme is nevertheless sufficient, i.e. variance rather than bias dominates there. The LR estimator used here is with respect to model noise, not action noise; its variance grows with state dimension, and the paper does not test high-dimensional systems where that would matter.

A conceptual point the paper raises, and which later deep-RL work largely sidesteps rather than solves, is that there is no single "right" horizon for pathwise gradients: TP chooses the effective depth per step from data. [SHAC](shac.md) and its successors instead fix a short horizon and bootstrap with a critic, and [AHAC](ahac.md) adapts the horizon from contact stiffness.

## 5. Cross-references

- Builds on: [PILCO](pilco.md) (the moment-matching framework it replaces); [SVG](svg.md) (pathwise gradients through learned models); [REINFORCE](reinforce.md) (the LR estimator).
- Extended by: [Metz et al.](metz-gradients.md) (chaos as a general failure of differentiable programming); [Suh et al.](suh-diffsim-gradients.md) (α-order interpolation with a bias guard); [Onoda et al.](onoda-composite-gradients.md) (per-step inverse-variance weighting, by the same first author).
- Contrasts with: [SHAC](shac.md) and [MAAC](maac.md), which avoid long pathwise chains by truncation and a terminal critic rather than by estimator mixing.

## 6. Relevance to TGI

PIPPS supplies the theoretical reason TGI does not backpropagate through whole episodes: per-sample pathwise gradients through a nonlinear contact system become uninformative with depth, even when the expected return is smooth. Two components of TGI echo it directly. The default objective adds a likelihood-ratio term on imagined actions (`--lr_coef 0.1`, with exploration noise 0.2 and normalised λ-return advantages) to the pathwise return gradient, a fixed-weight version of the RP/LR mixture that TP weights adaptively. The event-gated cut replaces the pathwise continuation with the critic at touchdowns on poorly supported terrain, which in TP's language shortens the RP path exactly where its variance is expected to explode; TGI chooses that depth from a geometric event rather than from measured variance.
