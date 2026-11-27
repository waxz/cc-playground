# SHAC: Accelerated Policy Learning with Parallel Differentiable Simulation

| | |
|---|---|
| Authors | Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg, Miles Macklin |
| Venue / year | ICLR 2022 |
| Link | https://arxiv.org/abs/2204.07137 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Before SHAC, policy learning with differentiable simulators was mostly plain backpropagation through time (BPTT) over whole episodes, which worked on smooth, short, contact-free tasks and failed on the contact-rich locomotion benchmarks where model-free RL ([PPO](ppo.md), [SAC](sac.md)) was standard. Xu et al. attribute the failure to two properties of the long-horizon objective: an optimisation landscape that is noisy and discontinuous at the resolution of single parameters (made worse by early termination), and exploding gradient norms. They respond with a GPU-parallel differentiable simulator (dflex) and Short-Horizon Actor-Critic (SHAC), which backpropagates through windows of $h \ll H$ steps and closes each window with a learned value function. SHAC is the direct algorithmic parent of [AHAC](ahac.md), [PWM](pwm.md), [SuperMimic](supermimic.md) and [TGI](tgi.md); TGI's README names it as the source of its policy objective.

## 2. Method

**Simulator.** dflex integrates $M\ddot q = J^\top f(q,\dot q) + c(q,\dot q) + \tau(q,\dot q,a)$ with semi-implicit Euler, caching the CRBA Cholesky factorisation for the backward pass, and uses source-to-source transformation for adjoints ($\partial L/\partial s_t = \partial L/\partial s_{t+1}\cdot\partial F/\partial s_t$, likewise for $a_t$). Contact is deliberately smoothed: a penalty normal force $f_c = (-k_n + k_d\dot d)\min(d,0)\,n$ and a regularised friction $f_t = -\frac{v_t}{\|v_t\|}\min(k_t\|v_t\|,\mu\|f_c\|)$, which replaces the Coulomb step by a ramp, plus penalty joint limits (see [m-rigid-contact](../math-cookbook.md#m-rigid-contact), [m-contact-gradients](../math-cookbook.md#m-contact-gradients)).

**Landscape analysis.** Perturbing a single weight of a trained humanoid policy over $\Delta\theta_k\in[-1,1]$, the 1000-step BPTT loss is highly non-convex and discontinuous and its gradient norm exceeds $10^6$, while the SHAC objective with $h=32$ is a smooth approximation of it (Fig. 2).

**Objective.** $N$ parallel environments are rolled for $h$ steps from where the previous window ended, and the policy loss is

$$L_\theta = -\frac{1}{Nh}\sum_{i=1}^N\Big[\sum_{t=t_0}^{t_0+h-1}\gamma^{t-t_0}R(s^i_t,a^i_t) + \gamma^h V_{\phi'}(s^i_{t_0+h})\Big],$$

differentiated through the simulator with a reparameterised Gaussian policy; gradients are cut between windows, the discount is reset at early termination, and the policy takes one Adam step per window. The critic is fit by MSE to TD(λ) targets $\tilde V(s_t) = (1-\lambda)\sum_{k=1}^{h-t-1}\lambda^{k-1}G^k_t + \lambda^{h-t-1}G^{h-t}_t$, treated as constants, with a Polyak-averaged target critic $V_{\phi'}$ ([m-short-horizon](../math-cookbook.md#m-short-horizon), [m-lambda-return](../math-cookbook.md#m-lambda-return)). The authors argue three benefits: the critic absorbs the noisy long-horizon landscape and the termination discontinuities into a smooth function; short windows bound the depth of Jacobian products ([m-bptt](../math-cookbook.md#m-bptt)); and frequent updates exploit parallel simulation.

## 3. Evidence

Benchmarks are cart-pole swing-up ($H=240$), ant, humanoid and a 152-muscle-tendon-unit lower-body humanoid (Humanoid MTU), all $H=1000$, with differentiable rewards, five seeds, tuned baselines on identical hardware. SHAC is more sample-efficient than PPO and SAC on every task, with the gap widening with dimension; in wall-clock time the methods are much closer on simple tasks, and PPO is faster early on. On Humanoid MTU SHAC reaches PPO's reward level in about 35 minutes and 1.7 M simulation steps, reported as more than 17× (PPO) and 30× (SAC) less wall-clock time and 382× and 170× fewer samples, and after 1.5 h reaches twice the reward of the best RL policy. BPTT with 64/128-step windows solves cart-pole but converges to local minima on contact tasks (ant learns to lean forward and stand still); first-order PODS stalls on cart-pole. Removing the terminal critic ("No Critic") fails on all four problems. A horizon sweep on ant ($h\in\{4,\dots,128\}$ at equal samples per episode) gives the best reward at $h=16$–$32$, with training time growing linearly in $h$. The backward pass costs about 2× the forward pass.

## 4. Critical analysis

SHAC demonstrates convincingly that, on dflex's smoothed contact model, a short pathwise window plus a terminal critic beats both full-horizon BPTT and model-free RL in samples, and is competitive in wall-clock time. Three qualifications matter for the lineage. First, the result is conditional on the simulator's contact smoothing; [PWM](pwm.md) later stiffened dflex contact on hopper and reported a 48 % drop in SHAC's reward, and [AHAC](ahac.md) traces SHAC's instability to stiff-contact gradient error. SHAC itself acknowledges non-smoothness only in the simulator design, not in the algorithm. Second, the landscape figure shows one coordinate of one trained policy; it is illustrative, not a measurement of gradient quality, and the paper does not measure bias or variance of its estimator. Third, the horizon $h$ is a fixed hyperparameter; the authors flag an adaptive schedule as future work, which AHAC took up. The comparison also rewards SHAC's differentiable, dense reward design; [Song et al.](song-diffsim-quadruped.md) and [DMO](amigo-decoupled.md) note that non-differentiable terminal penalties and survival bonuses contribute nothing to first-order gradients and must be routed through the critic.

## 5. Cross-references

- Builds on: [MAAC](maac.md) and [SVG](svg.md) (pathwise gradient through a model with a terminal value); [PIPPS](pipps.md) (landscape slices); [GAE](gae.md) / TD(λ) critic targets.
- Extended by: [AHAC](ahac.md) (contact-adaptive horizon); [PWM](pwm.md) (same actor-critic through a learned world model); [DMO](amigo-decoupled.md) (DMO-SHAC: simulator forward, model backward); [SuperMimic](supermimic.md) (SHAC-style update through a learned humanoid model, with survival weights replacing hard termination and no target critic).
- Contrasts with: [Suh et al.](suh-diffsim-gradients.md), which mixes estimators instead of truncating; [DiffMimic](diffmimic.md), which reports that truncated windows hurt on motion mimicking.

## 6. Relevance to TGI

TGI's policy objective is SHAC's: a 32-step pathwise return (0.64 s at 20 ms per step, the same $h=32$ SHAC found best) bootstrapped with a critic fit on λ-returns ($\lambda = 0.95$) over the imagined window, with the critic's input gradient flowing back into the states. The differences are inherited from SuperMimic: the dynamics are a learned model rather than a differentiable simulator, rollouts start from replayed simulator states, terminations are soft survival weights, and there is no target critic. TGI's own addition, the event-gated cut, is the step SHAC lacks: SHAC truncates every window at a fixed length, whereas TGI additionally truncates individual rollouts at imagined touchdowns on poorly supported terrain, bootstrapping from the critic at that step. In SHAC's terms, the critic is used as a smooth surrogate not only at the end of the window but wherever the dynamics are least trustworthy.
