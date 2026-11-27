# Mathematics cookbook for the TGI lineage

This document collects the mathematics that the paper and group reports of this study rely on. Each entry carries a fixed anchor (listed in [CONVENTIONS](CONVENTIONS.md)), states a precise definition, derives the central result in enough steps to be checked by hand, gives a short intuition, and lists the paper reports that use it. The entries are ordered so that each depends only on earlier ones: foundations of sequential decision making, model-free gradient estimators, model-based reinforcement learning, analytic (first-order) gradients, optimal control, legged-robot mechanics and perception, imitation, and a final group of geometric and architectural tools.

The thread that ties the entries together is the design of Terrain-Grounded Imagination ([TGI](papers/tgi.md)): a legged-robot policy trained by backpropagating short-horizon returns through a learned world model, closed by a terminal critic, with terrain heights obtained by differentiable lookups, a gradient cut at touchdowns that the terrain poorly supports, and an impact penalty. Where an entry bears directly on one of these components, the connection is made explicit.

## Notation

The same symbols are used throughout the study; paper reports map their source notation onto these.

| Symbol | Meaning |
|---|---|
| $s_t \in \mathcal S$, $o_t \in \mathcal O$ | state and observation at step $t$; $o_t$ may be a noisy, partial function of $s_t$ |
| $a_t \in \mathcal A$ | action (for legged robots usually a normalised joint set-point, see [m-pd-actuator](#m-pd-actuator)) |
| $g_t$ | goal or command (velocity command, reference motion frame) |
| $\pi_\theta(a \mid s)$, $a = \pi_\theta(s)$ | stochastic or deterministic policy with parameters $\theta$ |
| $p(s' \mid s, a)$, $s' = f(s, a)$ | true transition kernel; deterministic transition function |
| $\hat f_\omega$, $\hat r_\omega$, $\hat d_\omega$ | learned dynamics, reward and termination heads (the hat marks any model counterpart of a true quantity) |
| $r_t = r(s_t, a_t)$ | reward received after acting in $s_t$ (some sources, e.g. [SuperMimic](papers/supermimic.md), write $r_{t+1}$) |
| $d_t \in \lbrace 0, 1 \rbrace$ | termination flag; $c_t = \prod_{k \le t}(1 - d_k)$ the survival weight |
| $\gamma \in [0,1)$, $\lambda \in [0,1]$ | discount and trace-decay parameters |
| $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$ | discounted return from $t$ |
| $J(\theta) = \mathbb E[G_0]$ | policy objective |
| $V^\pi$, $Q^\pi$, $A^\pi$ | value, action-value and advantage functions of $\pi$ |
| $V_\psi$, $Q_\psi$ | learned critics with parameters $\psi$ |
| $d^\pi(s) = (1-\gamma) \sum_t \gamma^t \Pr(s_t = s)$ | normalised discounted state occupancy |
| $T$, $H$ | episode length; imagination (truncation) horizon, $H \ll T$ |
| $\Phi_t = \partial s_{t+1} / \partial s_t$ | closed-loop state Jacobian along a trajectory |
| $\mathcal H(x, y)$ | terrain height at planar position $(x, y)$; $\mathsf H_{ij}$ the stored heightmap grid |
| $q, \dot q$ | generalised coordinates and velocities of the robot |
| $\mathrm{sg}(\cdot)$ | stop-gradient operator: identity in the forward pass, zero Jacobian in the backward pass |

Expectations without a subscript are taken over the trajectory distribution induced by the current policy and the relevant dynamics. Vectors are columns; for a scalar $\ell$ and a vector $x$, $\nabla_x \ell$ is a column and $\partial y / \partial x$ is the Jacobian with rows indexed by $y$.

## Contents

1. Foundations: [MDP and occupancy](#m-mdp) · [Bellman equations and TD](#m-bellman) · [λ-returns](#m-lambda-return)
2. Model-free estimators: [score-function gradient and policy gradient theorem](#m-pg-theorem) · [advantage and GAE](#m-advantage-gae) · [trust regions and TRPO](#m-trust-region) · [PPO](#m-ppo-clip) · [deterministic policy gradient](#m-dpg) · [maximum-entropy RL](#m-maxent)
3. Model-based RL: [model learning and compounding error](#m-model-learning) · [ensembles and uncertainty](#m-ensembles) · [RSSM and the ELBO](#m-rssm-elbo) · [Dyna and MBPO](#m-dyna-mbpo) · [MPC, CEM, MPPI](#m-mpc)
4. Analytic gradients: [reparameterisation](#m-reparam) · [BPTT through dynamics](#m-bptt) · [short-horizon objective with terminal critic](#m-short-horizon) · [gradients through contact](#m-contact-gradients)
5. Optimal control: [LQR, iLQR, DDP](#m-lqr-ddp)
6. Legged robotics: [rigid-body dynamics with contact](#m-rigid-contact) · [PD control and actuators](#m-pd-actuator) · [LIPM, ZMP, capture point](#m-lipm-zmp) · [terrain lookups and SDFs](#m-terrain-lookup) · [domain randomisation and teacher–student](#m-domain-rand) · [curricula](#m-curriculum)
7. Imitation: [tracking rewards](#m-tracking-reward) · [adversarial imitation](#m-adversarial-imitation) · [latent skills and VAEs](#m-latent-skills) · [behaviour cloning, DAgger, distillation](#m-distillation)
8. Geometry and architecture: [rotations](#m-rotations) · [attention encoders](#m-attention) · [symmetry](#m-symmetry)

---

## 1. Foundations

<a id="m-mdp"></a>
### MDP, POMDP, return, discount, occupancy measure

**Definition.** A Markov decision process is a tuple $\mathcal M = (\mathcal S, \mathcal A, p, r, \rho_0, \gamma)$ with state space $\mathcal S$, action space $\mathcal A$, transition kernel $p(s' \mid s, a)$, reward $r(s, a)$, initial-state distribution $\rho_0$ and discount $\gamma$. A policy $\pi(a \mid s)$ and $\mathcal M$ together induce a distribution over trajectories $\tau = (s_0, a_0, s_1, a_1, \dots)$,

$$p_\pi(\tau) = \rho_0(s_0) \prod_{t \ge 0} \pi(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t),$$

and the objective is the expected discounted return $J(\pi) = \mathbb E_{\tau \sim p_\pi}\left[\sum_{t \ge 0} \gamma^t r(s_t, a_t)\right]$. Episodic tasks with termination are embedded by an absorbing state with zero reward; equivalently the return is weighted by the survival indicator, $\sum_t \gamma^t c_t r_t$, which [SuperMimic](papers/supermimic.md) relaxes to a predicted survival probability so that it stays differentiable in imagination.

A partially observable MDP (POMDP) adds an observation space and an emission kernel $o_t \sim \mathcal O(\cdot \mid s_t)$; the policy acts on observations or histories $o_{\le t}$. Legged locomotion is a POMDP in practice: the policy sees proprioception, a command and perhaps a local heightmap, not friction, payload or the full terrain. The standard remedies are a history encoder or recurrent state ([m-rssm-elbo](#m-rssm-elbo)), privileged training ([m-domain-rand](#m-domain-rand)) and the asymmetric actor–critic, in which the critic sees $s_t$ while the actor sees $o_t$.

**Occupancy measure.** The normalised discounted state occupancy is $d^\pi(s) = (1-\gamma)\sum_{t \ge 0} \gamma^t \Pr(s_t = s \mid \pi)$ and the state–action occupancy is $\rho^\pi(s, a) = d^\pi(s)\pi(a \mid s)$. Both are probability distributions, and the objective is linear in the latter:

$$J(\pi) = \frac{1}{1-\gamma}\, \mathbb E_{(s,a) \sim \rho^\pi}[r(s, a)].$$

The proof exchanges the sum and expectation: $\mathbb E[\sum_t \gamma^t r(s_t,a_t)] = \sum_{s,a} r(s,a) \sum_t \gamma^t \Pr(s_t = s, a_t = a)$. The occupancy viewpoint underlies the policy gradient theorem ([m-pg-theorem](#m-pg-theorem)), the TRPO bound ([m-trust-region](#m-trust-region)) and adversarial imitation, which matches occupancy measures rather than actions ([m-adversarial-imitation](#m-adversarial-imitation)).

**Discount as horizon.** Since $\sum_t \gamma^t = 1/(1-\gamma)$, the discount sets an effective horizon of order $1/(1-\gamma)$ steps; $\gamma = 0.99$ at 50 Hz corresponds to roughly two seconds of control. The discount is also a variance-reduction device: it biases the objective towards near-term reward in exchange for smaller return variance, a trade-off that reappears explicitly as the truncation horizon in [m-short-horizon](#m-short-horizon).

**Intuition.** The MDP abstracts the robot and its environment as a sampler of next states; everything downstream either estimates how the objective changes with the policy by sampling (model-free) or by differentiating a model of $p$ (model-based, analytic).

**Used in:** [REINFORCE](papers/reinforce.md), [policy gradient theorem](papers/pg-theorem.md), [TRPO](papers/trpo.md), [GAIL](papers/gail.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md), and implicitly every other report.

<a id="m-bellman"></a>
### Value and Q functions, Bellman equations, TD learning, n-step returns

**Definition.** For a policy $\pi$, $V^\pi(s) = \mathbb E[G_t \mid s_t = s]$, $Q^\pi(s,a) = \mathbb E[G_t \mid s_t = s, a_t = a]$ and $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$. Splitting the return as $G_t = r_t + \gamma G_{t+1}$ and conditioning gives the Bellman expectation equations

$$V^\pi(s) = \mathbb E_{a \sim \pi,\, s' \sim p}\left[r(s,a) + \gamma V^\pi(s')\right], \qquad Q^\pi(s,a) = r(s,a) + \gamma\, \mathbb E_{s' \sim p,\, a' \sim \pi}[Q^\pi(s', a')].$$

The optimal value functions satisfy the Bellman optimality equation $Q^\ast(s,a) = r(s,a) + \gamma\,\mathbb E_{s'}[\max_{a'} Q^\ast(s', a')]$. The Bellman operator $(\mathcal T^\pi V)(s) = \mathbb E[r + \gamma V(s')]$ is a $\gamma$-contraction in the sup norm: $\lVert \mathcal T^\pi V - \mathcal T^\pi U \rVert_\infty \le \gamma \lVert V - U\rVert_\infty$, because the two images differ only through $\gamma\,\mathbb E[V(s') - U(s')]$. Banach's fixed-point theorem then gives uniqueness of $V^\pi$ and convergence of repeated application.

**TD learning.** Temporal-difference learning replaces the expectation by a sample and moves the estimate towards the bootstrapped target. With the TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, tabular TD(0) updates $V(s_t) \leftarrow V(s_t) + \alpha \delta_t$; with a parametric critic one minimises $\mathbb E[(V_\psi(s_t) - y_t)^2]$ with the target $y_t$ held fixed ($\mathrm{sg}$ on the target), which is a semi-gradient method rather than gradient descent on a fixed loss. Target networks, an exponential moving average $\bar\psi \leftarrow \alpha \bar\psi + (1-\alpha)\psi$, slow the movement of the target and are used by [DDPG](papers/ddpg.md), [SAC](papers/sac.md) and [SHAC](papers/shac.md); [SuperMimic](papers/supermimic.md) notes it omits them.

**n-step returns.** The $n$-step return truncates the sampled sum and bootstraps:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n}).$$

If $V = V^\pi$ exactly, $\mathbb E[G^{(n)}_t] = V^\pi(s_t)$ for every $n$. With an approximate critic of error $e(s) = V(s) - V^\pi(s)$, the bias of $G^{(n)}_t$ is $\gamma^n \mathbb E[e(s_{t+n})]$, shrinking geometrically in $n$, while its variance grows with the number of sampled rewards and transitions. This is the bias–variance dial that λ-returns ([m-lambda-return](#m-lambda-return)), GAE ([m-advantage-gae](#m-advantage-gae)) and the truncated analytic objective ([m-short-horizon](#m-short-horizon)) all turn.

**Intuition.** A critic converts a long-horizon credit-assignment problem into a one-step consistency condition. Its errors are discounted by $\gamma^n$ when used as a bootstrap $n$ steps ahead, which is why a terminal critic can be imperfect and still useful.

**Used in:** [Dyna](papers/dyna.md), [A3C](papers/a3c.md), [DDPG](papers/ddpg.md), [TD3](papers/td3.md), [SAC](papers/sac.md), [TD-MPC](papers/td-mpc.md), [SHAC](papers/shac.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-lambda-return"></a>
### λ-returns and TD(λ)

**Definition.** The λ-return is the exponentially weighted average of all $n$-step returns,

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)},$$

the weights $(1-\lambda)\lambda^{n-1}$ summing to one. On a finite window ending at $t + H$ with bootstrap $V(s_{t+H})$, the residual weight is assigned to the longest available return: $G_t^\lambda = (1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}G_t^{(n)} + \lambda^{H-1} G_t^{(H)}$, which is exactly the critic target of SHAC's Eq. 7 ([SHAC](papers/shac.md)).

**Recursive form.** Writing $G_t^{(n)} = r_t + \gamma G_{t+1}^{(n-1)}$ for $n \ge 2$ and $G_t^{(1)} = r_t + \gamma V(s_{t+1})$,

$$G_t^\lambda = (1-\lambda)\left(r_t + \gamma V(s_{t+1})\right) + \sum_{n \ge 2} (1-\lambda)\lambda^{n-1}\left(r_t + \gamma G_{t+1}^{(n-1)}\right) = r_t + \gamma\left[(1-\lambda)V(s_{t+1}) + \lambda G_{t+1}^\lambda\right],$$

using $\sum_{n \ge 1}(1-\lambda)\lambda^{n-1} = 1$ and reindexing the second sum. With termination this becomes $G_t^\lambda = r_t + \gamma(1 - d_{t+1})[(1-\lambda)V_{t+1} + \lambda G_{t+1}^\lambda]$, initialised at the horizon by $G_H^\lambda = V(s_H)$. This backward recursion is how Dreamer, SHAC, SuperMimic (its Eq. 3) and TGI compute critic targets over imagined windows in $O(H)$.

**TD(λ) and eligibility traces.** The backward view accumulates an eligibility trace $e_t = \gamma\lambda e_{t-1} + \nabla_\psi V_\psi(s_t)$ and updates $\psi \leftarrow \psi + \alpha \delta_t e_t$. For offline updates over a complete episode the sum of these increments equals the forward-view update towards $G^\lambda_t$; this equivalence follows from the telescoping identity $G^\lambda_t - V(s_t) = \sum_{k \ge 0}(\gamma\lambda)^k \delta_{t+k}$, which is derived in [m-advantage-gae](#m-advantage-gae).

**Intuition.** $\lambda = 0$ trusts the critic after one step (low variance, high bias); $\lambda = 1$ trusts the sampled rewards until the end of the window (high variance, low bias). In imagination, the "samples" are model predictions, so a large λ also places more trust in model accuracy deep into the rollout.

**Used in:** [Dreamer](papers/dreamer.md), [DreamerV2](papers/dreamer-v2.md), [DreamerV3](papers/dreamer-v3.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

---

## 2. Model-free gradient estimators

<a id="m-pg-theorem"></a>
### Score-function (likelihood-ratio) gradient, REINFORCE, baselines, policy gradient theorem

**Score-function estimator.** For a density $p_\theta(x)$ and a function $F$ that does not depend on $\theta$,

$$\nabla_\theta\, \mathbb E_{x \sim p_\theta}[F(x)] = \int F(x)\nabla_\theta p_\theta(x)\,dx = \int F(x) p_\theta(x)\nabla_\theta \log p_\theta(x)\,dx = \mathbb E_{x \sim p_\theta}\left[F(x)\nabla_\theta \log p_\theta(x)\right],$$

using $\nabla p = p \nabla \log p$ and assuming that differentiation and integration can be exchanged. The estimator requires only samples of $x$ and evaluations of $F$; $F$ may be discontinuous or a black box. This is its strength and the root of its variance: nothing about how $F$ varies with $x$ is used.

**REINFORCE.** Applying this to trajectories, $\log p_\pi(\tau) = \log\rho_0(s_0) + \sum_t \log\pi_\theta(a_t \mid s_t) + \sum_t \log p(s_{t+1} \mid s_t, a_t)$, and only the policy terms depend on $\theta$; the unknown dynamics drop out. Hence

$$\nabla_\theta J = \mathbb E_\tau\left[\left(\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right) \left(\sum_{t'} \gamma^{t'} r_{t'}\right)\right].$$

Williams ([REINFORCE](papers/reinforce.md)) gave this estimator together with the observation that a baseline may be subtracted.

**Causality and baselines.** A reward at $t'$ cannot depend on actions taken later, and for any function $b(s_t)$, $\mathbb E_{a_t \sim \pi}[\nabla_\theta\log\pi_\theta(a_t \mid s_t)\, b(s_t)] = b(s_t)\nabla_\theta \int \pi_\theta(a \mid s_t)\,da = 0$. Dropping the past rewards (each has zero-mean product with the future score) and subtracting a state-dependent baseline gives the lower-variance estimator

$$\nabla_\theta J = \mathbb E_\tau\left[\sum_t \gamma^t \nabla_\theta \log\pi_\theta(a_t \mid s_t)\left(G_t - b(s_t)\right)\right].$$

The variance-minimising constant baseline per parameter is $b = \mathbb E[\lVert g\rVert^2 G]/\mathbb E[\lVert g\rVert^2]$ with $g$ the score; in practice $b = V_\psi(s_t)$ is used, which turns $G_t - b$ into an advantage estimate.

**Policy gradient theorem.** Sutton et al. ([policy gradient theorem](papers/pg-theorem.md)) state the result in terms of the occupancy measure. Starting from $V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$ and differentiating,

$$\nabla V^\pi(s) = \sum_a \left[\nabla\pi(a \mid s) Q^\pi(s,a) + \pi(a \mid s)\gamma \sum_{s'} p(s' \mid s,a)\nabla V^\pi(s')\right].$$

Unrolling the recursion expresses $\nabla V^\pi(s_0)$ as $\sum_s \sum_{k \ge 0}\gamma^k \Pr(s_0 \to s, k)\sum_a \nabla\pi(a \mid s) Q^\pi(s,a)$, where $\Pr(s_0 \to s, k)$ is the probability of reaching $s$ in $k$ steps. Averaging over $\rho_0$ and normalising,

$$\nabla_\theta J(\theta) = \frac{1}{1-\gamma}\, \mathbb E_{s \sim d^{\pi},\, a \sim \pi_\theta}\left[\nabla_\theta \log\pi_\theta(a \mid s)\, Q^{\pi}(s,a)\right].$$

The theorem's content is that the gradient does not involve $\nabla_\theta d^\pi$, the derivative of the state distribution, even though changing $\theta$ changes which states are visited. Replacing $Q^\pi$ by $A^\pi$ leaves the expectation unchanged (baseline argument) and is the form used by actor–critic methods.

**Variance.** For a Gaussian policy $a = \mu_\theta(s) + \sigma\epsilon$, the score is $\nabla_\theta\mu^\top \epsilon/\sigma$; the estimator's variance therefore scales as $1/\sigma^2$ and, summed over $T$ steps with uncorrelated noise, linearly in the horizon at best. The contrast with the pathwise estimator, whose variance does not blow up as $\sigma \to 0$, is worked out in [m-reparam](#m-reparam).

**Intuition.** The score-function gradient raises the probability of actions that turned out better than the baseline. It learns from correlations between random perturbations and outcomes, and so needs many samples whenever the action space is high-dimensional or the outcome is noisy.

**Used in:** [REINFORCE](papers/reinforce.md), [policy gradient theorem](papers/pg-theorem.md), [A3C](papers/a3c.md), [TRPO](papers/trpo.md), [PPO](papers/ppo.md), [DreamerV2](papers/dreamer-v2.md), [DreamerV3](papers/dreamer-v3.md), [SuperMimic](papers/supermimic.md) (as the PPO baseline it contrasts with), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md).

<a id="m-advantage-gae"></a>
### Advantage function, generalised advantage estimation

**Definition.** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ measures how much better $a$ is than the policy's average action in $s$; $\mathbb E_{a\sim\pi}[A^\pi(s,a)] = 0$. With the TD residual $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$, if $V = V^\pi$ then $\mathbb E[\delta_t^V \mid s_t, a_t] = A^\pi(s_t, a_t)$, so $\delta_t^V$ is an unbiased but noisy one-step advantage estimate.

**k-step advantage estimates.** Summing residuals telescopes the intermediate values:

$$\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l}^V = -V(s_t) + \sum_{l=0}^{k-1}\gamma^l r_{t+l} + \gamma^k V(s_{t+k}) = G_t^{(k)} - V(s_t).$$

**GAE as an exponentially weighted sum of TD residuals.** Schulman et al. ([GAE](papers/gae.md)) define $\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{k \ge 1}\lambda^{k-1}\hat A^{(k)}_t$. Collect the coefficient of $\delta_{t+l}$: it appears in every $\hat A^{(k)}$ with $k > l$, with weight $\gamma^l$, so its total coefficient is

$$\gamma^l (1-\lambda)\sum_{k = l+1}^{\infty}\lambda^{k-1} = \gamma^l(1-\lambda)\frac{\lambda^l}{1-\lambda} = (\gamma\lambda)^l.$$

Therefore

$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l \ge 0}(\gamma\lambda)^l \delta^V_{t+l}, \qquad \hat A_t = \delta_t^V + \gamma\lambda(1 - d_{t+1})\hat A_{t+1},$$

and $\hat A_t + V(s_t) = G_t^\lambda$, the λ-return of [m-lambda-return](#m-lambda-return). The special cases are $\lambda = 0$, $\hat A_t = \delta_t$ (biased unless $V = V^\pi$, low variance), and $\lambda = 1$, $\hat A_t = G_t - V(s_t)$ (unbiased for any $V$, high variance).

**Bias.** GAE is unbiased for every λ if $V = V^\pi$. For an approximate $V$, the estimator is "γ-just" (unbiased for the discounted objective) only at $\lambda = 1$; the bias for $\lambda < 1$ is $\sum_l (\gamma\lambda)^l \mathbb E[\gamma e(s_{t+l+1}) - e(s_{t+l})]$ with $e = V - V^\pi$. GAE also treats $\gamma$ itself as a variance-reduction parameter: the discounted gradient is a biased estimate of the undiscounted one.

**Intuition.** GAE spreads credit for a TD surprise backwards with geometric decay $\gamma\lambda$; it is the advantage analogue of an eligibility trace. In the analytic methods of section 4, the same quantity is used only to build critic targets, because the actor gradient comes from differentiation rather than from advantage-weighted scores.

**Used in:** [GAE](papers/gae.md), [TRPO](papers/trpo.md), [PPO](papers/ppo.md), [legged-gym](papers/legged-gym.md), [RWM-O](papers/rwm-o.md), [SuperMimic](papers/supermimic.md) (critic targets, its Eq. 3), [TGI](papers/tgi.md).

<a id="m-trust-region"></a>
### KL divergence, Fisher information, natural gradient, TRPO's monotonic improvement bound

**KL divergence and Fisher information.** $D_{\mathrm{KL}}(p \,\Vert\, q) = \mathbb E_p[\log p - \log q] \ge 0$ with equality iff $p = q$. For a parametric family, expanding $D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\theta + \Delta})$ to second order around $\Delta = 0$: the zeroth- and first-order terms vanish (the expected score is zero), leaving

$$D_{\mathrm{KL}}(\pi_\theta \,\Vert\, \pi_{\theta+\Delta}) \approx \tfrac12 \Delta^\top F(\theta)\Delta, \qquad F(\theta) = \mathbb E_{s,\, a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a \mid s)\,\nabla_\theta\log\pi_\theta(a \mid s)^\top\right].$$

**Natural gradient.** Maximising the linearised objective $g^\top\Delta$ subject to $\tfrac12\Delta^\top F\Delta \le \delta$ gives, by a Lagrangian argument, $\Delta = \sqrt{2\delta/(g^\top F^{-1}g)}\,F^{-1}g$. The direction $F^{-1}g$ is invariant to smooth reparameterisations of the policy, because both $g$ and $F$ transform covariantly; the plain gradient is not.

**Performance difference lemma.** For any two policies $\pi$ and $\tilde\pi$,

$$J(\tilde\pi) = J(\pi) + \mathbb E_{\tau\sim\tilde\pi}\left[\sum_t\gamma^t A^\pi(s_t,a_t)\right] = J(\pi) + \frac{1}{1-\gamma}\mathbb E_{s\sim d^{\tilde\pi},\,a\sim\tilde\pi}[A^\pi(s,a)].$$

Proof: write $A^\pi(s_t,a_t) = \mathbb E[r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)]$ and sum with discount along a trajectory of $\tilde\pi$; the value terms telescope to $-V^\pi(s_0)$, leaving $\mathbb E_{\tilde\pi}[\sum_t\gamma^t r_t] - \mathbb E[V^\pi(s_0)] = J(\tilde\pi) - J(\pi)$.

**Surrogate and bound.** The lemma requires states from the new policy. TRPO ([TRPO](papers/trpo.md)), building on Kakade and Langford's conservative policy iteration, replaces $d^{\tilde\pi}$ by $d^\pi$ to obtain the local surrogate

$$L_\pi(\tilde\pi) = J(\pi) + \frac{1}{1-\gamma}\mathbb E_{s\sim d^\pi,\,a\sim\tilde\pi}[A^\pi(s,a)] = J(\pi) + \frac{1}{1-\gamma}\mathbb E_{s\sim d^\pi,\,a\sim\pi}\left[\frac{\tilde\pi(a \mid s)}{\pi(a \mid s)}A^\pi(s,a)\right],$$

which matches $J$ to first order at $\tilde\pi = \pi$. With $\alpha = \max_s D_{\mathrm{TV}}(\pi(\cdot\mid s), \tilde\pi(\cdot \mid s))$, a coupling argument shows that the two policies disagree at a given step with probability at most $\alpha$, so the state distributions at time $t$ differ in total variation by at most $1-(1-\alpha)^t$, and the expected advantage under $\tilde\pi$ in any state is at most $2\alpha\epsilon$ in magnitude with $\epsilon = \max_{s,a}\lvert A^\pi(s,a)\rvert$. Summing the discounted per-step discrepancies yields Theorem 1 of TRPO:

$$J(\tilde\pi) \ge L_\pi(\tilde\pi) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\alpha^2 \ge L_\pi(\tilde\pi) - C\, D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi), \qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2},$$

the second step by Pinsker's inequality $D_{\mathrm{TV}}^2 \le D_{\mathrm{KL}}$ (with TRPO's convention for the TV distance). (TRPO writes $\eta$ for unnormalised quantities; the constant is unchanged in this form because $L$ and $J$ are on the same scale.) Maximising the right-hand side is a minorise–maximise scheme: the bound touches $J$ at $\tilde\pi = \pi$, so each maximisation cannot decrease $J$.

**Practical TRPO.** The penalty coefficient $C$ is far too conservative, so TRPO instead solves $\max_\theta L_{\theta_{\text{old}}}(\theta)$ subject to $\mathbb E_{s\sim d^{\pi_{\text{old}}}}[D_{\mathrm{KL}}(\pi_{\text{old}}\Vert\pi_\theta)] \le \delta$, approximating the constraint quadratically, computing $F^{-1}g$ by conjugate gradients with Fisher–vector products, and finishing with a backtracking line search that checks the actual KL and surrogate improvement.

**Intuition.** A sample-based gradient is only trustworthy near the policy that generated the samples, because the surrogate ignores the change in state distribution. Trust regions formalise "near" in the geometry of action distributions rather than parameters. The same logic, with the model's validity in place of the policy's, appears in MBPO's branched-rollout bound ([m-dyna-mbpo](#m-dyna-mbpo)).

**Used in:** [TRPO](papers/trpo.md), [GAE](papers/gae.md), [PPO](papers/ppo.md), [MBPO](papers/mbpo.md), [GAIL](papers/gail.md).

<a id="m-ppo-clip"></a>
### PPO clipped surrogate, KL-penalty variant, practical PPO loop

**Clipped objective.** With the probability ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$, the TRPO surrogate is $L^{\mathrm{CPI}}(\theta) = \hat{\mathbb E}_t[\rho_t(\theta)\hat A_t]$. Unconstrained maximisation of $L^{\mathrm{CPI}}$ drives $\rho_t$ arbitrarily far from one. PPO ([PPO](papers/ppo.md)) replaces it with

$$L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb E}_t\left[\min\left(\rho_t(\theta)\hat A_t,\ \mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\right)\right].$$

**Why the minimum is pessimistic.** Consider one sample. If $\hat A_t > 0$, the objective is $\min(\rho_t, 1+\epsilon)\hat A_t$ for $\rho_t \ge 1 - \epsilon$: it increases with $\rho_t$ until $1+\epsilon$ and is flat thereafter, so the gradient vanishes once the action has been made sufficiently more likely. For $\rho_t < 1-\epsilon$ the unclipped term is the smaller and the gradient remains, so a step that wrongly reduced a good action's probability is still corrected. If $\hat A_t < 0$, the objective is $\max(\rho_t, 1-\epsilon)\hat A_t$ for $\rho_t \le 1 + \epsilon$: it rewards reducing $\rho_t$ only down to $1-\epsilon$, and for $\rho_t > 1 + \epsilon$ the unclipped term again dominates and the gradient pushes back. In every case $L^{\mathrm{CLIP}} \le L^{\mathrm{CPI}}$, with equality near $\rho_t = 1$; the clipped objective is a lower bound on the surrogate that removes the incentive to move the ratio outside $[1-\epsilon, 1+\epsilon]$ in the direction that would improve the surrogate, while retaining the incentive to undo moves that made it worse. It is a heuristic counterpart of the penalised bound in [m-trust-region](#m-trust-region), not a guarantee: the ratio can still leave the interval across several minibatch epochs, and later empirical studies attribute much of PPO's behaviour to implementation details (advantage normalisation, value clipping, learning-rate annealing, observation normalisation).

**KL-penalty variant.** The alternative objective $\hat{\mathbb E}_t[\rho_t\hat A_t - \beta D_{\mathrm{KL}}(\pi_{\text{old}}(\cdot\mid s_t)\Vert\pi_\theta(\cdot\mid s_t))]$ adapts $\beta$ after each update: double it if the measured KL exceeds $1.5\, d_{\text{targ}}$, halve it if below $d_{\text{targ}}/1.5$. Many locomotion codebases ([legged-gym](papers/legged-gym.md) and descendants) combine clipping with a KL-adaptive learning rate in the same spirit.

**Practical loop.** Collect $N$ parallel rollouts of $T$ steps with $\pi_{\theta_{\text{old}}}$; compute GAE advantages ([m-advantage-gae](#m-advantage-gae)) and value targets; for $K$ epochs over shuffled minibatches, minimise $-L^{\mathrm{CLIP}} + c_1(V_\psi(s_t) - \hat V_t^{\text{targ}})^2 - c_2\,\mathcal H[\pi_\theta(\cdot\mid s_t)]$; repeat. Massive GPU parallelism (thousands of environments, short rollouts of 24 steps) made PPO the default for legged robots ([legged-gym](papers/legged-gym.md)).

**Intuition.** PPO is first-order trust-region optimisation with a soft wall. Its estimator is still the score function, so the per-sample gradient carries the high variance of [m-pg-theorem](#m-pg-theorem); PPO compensates with very large batches. TGI and SuperMimic replace this estimator, not the outer actor–critic structure.

**Used in:** [PPO](papers/ppo.md), [legged-gym](papers/legged-gym.md), [hwangbo-anymal](papers/hwangbo-anymal.md), [RMA](papers/rma.md), [miki-wild](papers/miki-wild.md), [extreme-parkour](papers/extreme-parkour.md), [watch-your-step](papers/watch-your-step.md), [DeepMimic](papers/deepmimic.md), [AMP](papers/amp.md), [BeyondMimic](papers/beyondmimic.md), [RWM](papers/rwm.md), [SuperMimic](papers/supermimic.md).

<a id="m-dpg"></a>
### Deterministic policy gradient, DDPG/TD3

**Deterministic policy gradient.** For a deterministic policy $a = \mu_\theta(s)$, the action has no density and the score function is undefined. Silver et al. showed that

$$\nabla_\theta J(\theta) = \frac{1}{1-\gamma}\mathbb E_{s\sim d^{\mu}}\left[\nabla_\theta\mu_\theta(s)^\top\, \nabla_a Q^{\mu}(s,a)\big|_{a=\mu_\theta(s)}\right].$$

The derivation mirrors the policy gradient theorem: differentiate $V^\mu(s) = Q^\mu(s, \mu_\theta(s))$ to get $\nabla_\theta V^\mu(s) = \nabla_\theta\mu_\theta(s)^\top\nabla_a Q^\mu(s,a)|_{a=\mu(s)} + \gamma\,\mathbb E_{s'}[\nabla_\theta V^\mu(s')]$, where the second term arises because $Q^\mu(s,a) = r(s,a) + \gamma\mathbb E_{s'}[V^\mu(s')]$ depends on $\theta$ through $V^\mu$. Unrolling gives the occupancy-weighted sum. The DPG is the limit of the stochastic policy gradient as the policy variance goes to zero. It is a pathwise gradient through the critic: the critic's action-gradient stands in for the derivative of the unknown environment, which is the conceptual bridge to [m-reparam](#m-reparam) and to SVG(1).

**DDPG.** [DDPG](papers/ddpg.md) makes DPG practical with deep networks: off-policy learning from a replay buffer, target networks for actor and critic, exploration noise added to $\mu_\theta(s)$, and the critic fit to $y = r + \gamma Q_{\bar\psi}(s', \mu_{\bar\theta}(s'))$.

**TD3.** [TD3](papers/td3.md) addresses the overestimation bias created by maximising through an approximate critic. Its three changes are clipped double-Q learning, $y = r + \gamma\min_{i=1,2}Q_{\bar\psi_i}(s', \tilde a)$; target policy smoothing, $\tilde a = \mu_{\bar\theta}(s') + \mathrm{clip}(\epsilon, -c, c)$ with Gaussian $\epsilon$, which regularises the critic to be smooth in the action; and delayed actor updates, which let the critic settle before its gradient is trusted.

**Intuition.** The actor follows the critic's action-gradient. Its quality is bounded by how well $\nabla_a Q_\psi$ approximates $\nabla_a Q^\mu$, which value regression does not directly train; overestimation and spurious gradient directions follow. Analytic methods replace most of this gradient with the derivative of the dynamics over $H$ steps and use the critic only at the end ([m-short-horizon](#m-short-horizon)).

**Used in:** [DDPG](papers/ddpg.md), [TD3](papers/td3.md), [SVG](papers/svg.md), [MAAC](papers/maac.md), [TD-MPC](papers/td-mpc.md), [TD-MPC2](papers/td-mpc2.md).

<a id="m-maxent"></a>
### Maximum-entropy RL, soft Bellman, SAC

**Objective.** Maximum-entropy RL augments the reward with the policy entropy, $J_\alpha(\pi) = \mathbb E\left[\sum_t\gamma^t\left(r_t + \alpha\,\mathcal H[\pi(\cdot\mid s_t)]\right)\right]$, with temperature $\alpha > 0$.

**Soft Bellman equations.** Define $Q(s,a) = r(s,a) + \gamma\,\mathbb E_{s'}[V(s')]$ and $V(s) = \mathbb E_{a\sim\pi}[Q(s,a) - \alpha\log\pi(a\mid s)]$. For fixed $Q$, the policy that maximises $V(s)$ is found by writing $V(s) = -\alpha D_{\mathrm{KL}}\left(\pi(\cdot\mid s)\,\Vert\,\exp(Q(s,\cdot)/\alpha)/Z(s)\right) + \alpha\log Z(s)$, so the optimum is the Boltzmann policy $\pi^\ast(a\mid s) = \exp\left((Q(s,a) - V(s))/\alpha\right)$ with $V(s) = \alpha\log\int\exp(Q(s,a)/\alpha)\,da$, a soft maximum that tends to $\max_a Q$ as $\alpha\to 0$.

**SAC.** [SAC](papers/sac.md) alternates soft policy evaluation, regressing $Q_\psi$ towards $r + \gamma(\min_i Q_{\bar\psi_i}(s',a') - \alpha\log\pi_\theta(a'\mid s'))$ with $a' \sim \pi_\theta(\cdot\mid s')$, and soft policy improvement, minimising the KL to the Boltzmann policy, which with a reparameterised policy $a = f_\theta(\epsilon; s)$ ([m-reparam](#m-reparam)) is

$$J_\pi(\theta) = \mathbb E_{s\sim\mathcal D,\,\epsilon\sim\mathcal N}\left[\alpha\log\pi_\theta(f_\theta(\epsilon;s)\mid s) - Q_\psi(s, f_\theta(\epsilon;s))\right].$$

Actions are squashed by $\tanh$, which requires the change-of-variables correction $\log\pi(a\mid s) = \log\mathcal N(u) - \sum_i\log(1 - \tanh^2 u_i)$. The temperature is adapted to a target entropy $\bar{\mathcal H}$ by minimising $\mathbb E[-\alpha(\log\pi(a\mid s) + \bar{\mathcal H})]$. Soft policy iteration converges to the optimal max-entropy policy in the tabular case.

**Intuition.** Entropy keeps exploration alive and makes the improvement step a well-conditioned KL projection. SAC's actor update is a reparameterised, one-step analytic gradient through a learned critic; analytic model-based methods (SHAC, PWM, TGI) can be read as SAC's actor update extended through $H$ steps of differentiable dynamics before the critic is consulted.

**Used in:** [SAC](papers/sac.md), [MBPO](papers/mbpo.md), [MAAC](papers/maac.md), [TD-MPC2](papers/td-mpc2.md) (entropy-regularised actor), [DreamerV3](papers/dreamer-v3.md) (entropy regulariser on the actor).

---

## 3. Model-based reinforcement learning

<a id="m-model-learning"></a>
### One-step vs multi-step / autoregressive model loss, compounding error

**One-step (teacher-forced) loss.** Given transitions $(s_t, a_t, s_{t+1})$ from a replay buffer, the simplest model fit minimises $\mathcal L_1(\omega) = \mathbb E\left[\lVert\hat f_\omega(s_t, a_t) - s_{t+1}\rVert^2\right]$, or the negative log-likelihood of a probabilistic model ([m-ensembles](#m-ensembles)). The model always receives true states as input, which is called teacher forcing.

**Compounding error.** In use, the model is rolled out on its own predictions, $\hat s_{t+1} = \hat f_\omega(\hat s_t, a_t)$. Suppose the one-step error is bounded, $\lVert\hat f(s,a) - f(s,a)\rVert \le \varepsilon$, and $\hat f$ is $L$-Lipschitz in $s$. Then $e_k = \lVert\hat s_k - s_k\rVert$ satisfies

$$e_{k+1} \le \lVert\hat f(\hat s_k, a_k) - \hat f(s_k, a_k)\rVert + \lVert\hat f(s_k,a_k) - f(s_k,a_k)\rVert \le L e_k + \varepsilon, \qquad e_k \le \varepsilon\,\frac{L^k - 1}{L - 1}.$$

For $L > 1$ the error grows geometrically; legged dynamics through contact have local Lipschitz constants well above one. A second mechanism compounds the first: the model's inputs drift off the data distribution, where $\varepsilon$ itself is larger (exposure bias). In value terms, the simulation lemma bounds the return gap between true and model MDPs: if $\max_{s,a} D_{\mathrm{TV}}(p(\cdot\mid s,a), \hat p(\cdot\mid s,a)) \le \varepsilon_m$ and $\lvert r\rvert \le r_{\max}$, then $\lvert J_{\mathcal M}(\pi) - J_{\hat{\mathcal M}}(\pi)\rvert \le 2\gamma r_{\max}\varepsilon_m/(1-\gamma)^2$, the quadratic dependence on the effective horizon being the value-level expression of compounding.

**Multi-step (autoregressive) loss.** The remedy is to train as the model is used. Seed at a logged state, roll forward on logged actions for $k$ steps, and penalise all predictions:

$$\mathcal L_k(\omega) = \mathbb E\left[\sum_{j=1}^{k}\alpha^{j}\,\ell\left(\hat s_j, s_j\right)\right], \qquad \hat s_0 = s_0,\ \hat s_{j} = \hat f_\omega(\hat s_{j-1}, a_{j-1}).$$

Gradients of $\mathcal L_k$ flow through the rollout, so fitting the model is itself BPTT ([m-bptt](#m-bptt)) and trains the model to correct its own errors: a perturbation of $\hat s_j$ that later steps amplify is penalised. [RWM](papers/rwm.md) uses this form with a history of $M$ observation–action pairs, forecast horizon $N$, decay $\alpha$ and an auxiliary loss on privileged quantities such as contacts, $\mathcal L = \frac1N\sum_{k=1}^{N}\alpha^k[\mathcal L_o(o'_{t+k}, o_{t+k}) + \mathcal L_c(c'_{t+k}, c_{t+k})]$, and observes that teacher forcing is the special case $N = 1$. [SuperMimic](papers/supermimic.md) fits a deterministic model on $k = 8$ step windows with state, reward, noise-free observation and termination (binary cross-entropy) terms, and reports that the multi-step fit keeps rollouts stable well beyond $k$, including the $H = 32$ imagination horizon used for the policy.

**Values are not Jacobians.** For analytic policy gradients ([m-bptt](#m-bptt), [m-short-horizon](#m-short-horizon)) the policy update uses $\partial\hat f/\partial s$ and $\partial\hat f/\partial a$, not only $\hat f$. A model whose predictions match the data can have Jacobians that do not: a function can be uniformly close to another while its derivative oscillates. The error of the analytic gradient therefore depends on Jacobian error, which a value-matching loss controls only indirectly (through smoothness priors of the network, multi-step losses that penalise mis-propagated perturbations, and the data's coverage of nearby actions). This is why SuperMimic collects data with action noise, "to widen the range of actions on which the world model is fit", and why Sobolev-style losses that also match derivatives are a natural, if rarely used, extension. Near contact events the true dynamics are discontinuous ([m-contact-gradients](#m-contact-gradients)), and a smooth network necessarily replaces the jump by a steep ramp whose slope is set by architecture and data density rather than physics; TGI's gradient cut at poorly supported touchdowns can be read as declining to trust the model's Jacobian exactly where this is worst.

**Intuition.** One-step error is the wrong target when the model is to be iterated. Train on the rollout distribution, and for gradient-based policy learning remember that the policy sees the model through its derivatives.

**Used in:** [nagabandi-mbmf](papers/nagabandi-mbmf.md), [PETS](papers/pets.md), [PlaNet](papers/planet.md) (latent overshooting), [MBPO](papers/mbpo.md), [RWM](papers/rwm.md), [RWM-O](papers/rwm-o.md), [MAAC](papers/maac.md), [PWM](papers/pwm.md), [amigo-decoupled](papers/amigo-decoupled.md), [ControlVAE](papers/controlvae.md), [SuperTrack](papers/supertrack.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-ensembles"></a>
### Probabilistic ensembles, epistemic vs aleatoric uncertainty, uncertainty penalties (MOPO)

**Probabilistic model.** A Gaussian output model predicts $\hat p_\omega(s'\mid s,a) = \mathcal N(\mu_\omega(s,a), \Sigma_\omega(s,a))$, with diagonal $\Sigma$, and is trained by the negative log-likelihood

$$\mathcal L_{\mathrm{NLL}}(\omega) = \mathbb E\left[\tfrac12(\mu_\omega - s')^\top\Sigma_\omega^{-1}(\mu_\omega - s') + \tfrac12\log\det\Sigma_\omega\right],$$

in which the log-determinant prevents the variance from growing without bound and the Mahalanobis term prevents it from collapsing. Practical implementations bound the log-variance softly.

**Two kinds of uncertainty.** Aleatoric uncertainty is the irreducible noise of the system, captured by $\Sigma_\omega$. Epistemic uncertainty is the model's ignorance from limited data, captured by disagreement among an ensemble $\lbrace\omega_b\rbrace_{b=1}^B$ trained from different initialisations (and, in [PETS](papers/pets.md), bootstrapped data). For the mixture, the law of total variance separates the two:

$$\mathrm{Var}[s'] = \underbrace{\mathbb E_b[\Sigma_{\omega_b}]}_{\text{aleatoric}} + \underbrace{\mathrm{Var}_b[\mu_{\omega_b}]}_{\text{epistemic}}.$$

PETS propagates uncertainty by trajectory sampling: each particle either keeps one ensemble member for the whole rollout (TS∞) or resamples a member at every step (TS1).

**Uncertainty penalties.** In offline or heavily model-based settings, the policy exploits regions where the model is optimistic. [MOPO](papers/mopo.md) bounds the true return of a policy by its return in the model under a penalised reward. Let $G^\pi_{\hat{\mathcal M}}(s,a) = \mathbb E_{s'\sim\hat p}[V^\pi_{\mathcal M}(s')] - \mathbb E_{s'\sim p}[V^\pi_{\mathcal M}(s')]$; a telescoping argument analogous to the performance difference lemma gives $J_{\mathcal M}(\pi) = J_{\hat{\mathcal M}}(\pi) - \gamma\,\mathbb E_{(s,a)\sim\hat\rho^\pi}[G^\pi_{\hat{\mathcal M}}(s,a)]/(1-\gamma)$ up to normalisation. If $V^\pi_{\mathcal M}$ lies in a function class whose integral probability metric is $d_{\mathcal F}$, with scale $c$, then $\lvert G\rvert \le c\, d_{\mathcal F}(\hat p(\cdot\mid s,a), p(\cdot\mid s,a)) \le c\,u(s,a)$ for an admissible error estimator $u$. Optimising the penalised reward $\tilde r(s,a) = \hat r(s,a) - \lambda u(s,a)$, $\lambda = \gamma c$, therefore maximises a lower bound on the true return. MOPO's practical $u$ is the largest Frobenius norm of the predicted covariances across the ensemble, $u(s,a) = \max_b\lVert\Sigma_{\omega_b}(s,a)\rVert_F$, a heuristic that conflates aleatoric and epistemic terms.

**Intuition.** An ensemble's disagreement is a cheap proxy for "the model has not seen this". For analytic gradients the same quantity can gate trust in the model's Jacobians; a gradient cut triggered by low support or high disagreement is the derivative-level analogue of MOPO's reward penalty.

**Used in:** [PETS](papers/pets.md), [MBPO](papers/mbpo.md), [MOPO](papers/mopo.md), [MAAC](papers/maac.md), [RWM-O](papers/rwm-o.md), [TD-MPC2](papers/td-mpc2.md) (Q-ensembles), [PILCO](papers/pilco.md) (Gaussian-process epistemic uncertainty).

<a id="m-rssm-elbo"></a>
### Latent state-space models, RSSM, ELBO, KL balancing, symlog/two-hot

**Latent state-space model.** When observations are high-dimensional or partial, the model is defined over a latent state $z_t$: a transition prior $p(z_t\mid z_{t-1}, a_{t-1})$, an observation decoder $p(o_t\mid z_t)$, a reward head $p(r_t\mid z_t)$, and an approximate posterior (encoder) $q(z_t\mid z_{t-1}, a_{t-1}, o_t)$.

**ELBO derivation.** For a sequence, with $q(z_{1:T}\mid o_{1:T}, a_{1:T}) = \prod_t q(z_t\mid z_{t-1}, a_{t-1}, o_t)$,

$$\log p(o_{1:T}\mid a_{1:T}) = \log\mathbb E_{q}\left[\prod_t\frac{p(o_t\mid z_t)\,p(z_t\mid z_{t-1},a_{t-1})}{q(z_t\mid z_{t-1},a_{t-1},o_t)}\right] \ge \sum_{t=1}^T\mathbb E_q\left[\log p(o_t\mid z_t)\right] - \mathbb E_q\left[D_{\mathrm{KL}}\left(q(z_t\mid z_{t-1},a_{t-1},o_t)\,\Vert\,p(z_t\mid z_{t-1},a_{t-1})\right)\right],$$

by importance-weighting the marginal likelihood with $q$ and applying Jensen's inequality to the concave logarithm; the gap equals $D_{\mathrm{KL}}(q(z_{1:T})\Vert p(z_{1:T}\mid o_{1:T},a_{1:T}))$. Gradients of the reconstruction term with respect to the encoder use the reparameterisation trick ([m-reparam](#m-reparam)) for Gaussian latents or straight-through estimators for categorical ones.

**RSSM.** [PlaNet](papers/planet.md) argues that purely stochastic transitions struggle to remember information over many steps, and purely deterministic ones cannot represent multiple futures. The recurrent state-space model splits the state into a deterministic part $h_t = f_\omega(h_{t-1}, z_{t-1}, a_{t-1})$ (a GRU) and a stochastic part with prior $p(z_t\mid h_t)$ and posterior $q(z_t\mid h_t, o_t)$. The ELBO above applies with $h_t$ as a deterministic function of the past. PlaNet additionally proposes latent overshooting, a multi-step KL that trains the prior to be consistent over $d$-step predictions, the latent analogue of [m-model-learning](#m-model-learning).

**KL balancing and free bits.** The KL term trains both the posterior (towards the prior) and the prior (towards the posterior). [DreamerV2](papers/dreamer-v2.md) weights the two roles differently with stop-gradients,

$$\mathcal L_{\mathrm{KL}} = \alpha\, D_{\mathrm{KL}}(\mathrm{sg}(q)\,\Vert\,p) + (1-\alpha)\,D_{\mathrm{KL}}(q\,\Vert\,\mathrm{sg}(p)), \qquad \alpha = 0.8,$$

so that the prior learns faster than the posterior is regularised, avoiding posteriors that collapse towards a poorly trained prior. DreamerV2 also replaces Gaussian latents with 32 categorical variables of 32 classes each, trained with straight-through gradients. [DreamerV3](papers/dreamer-v3.md) adds free bits, clipping each KL term below one nat ($\max(1, D_{\mathrm{KL}})$) so that the model does not spend capacity compressing already-simple latents.

**Symlog and two-hot.** DreamerV3 handles targets of unknown scale with $\mathrm{symlog}(x) = \mathrm{sign}(x)\ln(\lvert x\rvert + 1)$ and its inverse $\mathrm{symexp}(x) = \mathrm{sign}(x)(e^{\lvert x\rvert} - 1)$; observations are regressed in symlog space. Rewards and values are predicted as a categorical distribution over exponentially spaced bins $B = \mathrm{symexp}([-20, \dots, 20])$; a scalar target $y$ between bins $b_k \le y < b_{k+1}$ is encoded as the two-hot vector with weights $(b_{k+1} - y)/(b_{k+1}-b_k)$ and $(y - b_k)/(b_{k+1} - b_k)$ on those bins, trained by cross-entropy. The cross-entropy gradient does not scale with the target magnitude, which makes the losses robust across domains.

**Actor gradients in Dreamer.** [Dreamer](papers/dreamer.md) (V1) trains the actor by backpropagating λ-returns ([m-lambda-return](#m-lambda-return)) through the learned latent dynamics, which is the closest ancestor of TGI's actor update in the world-model line. DreamerV2 found pure dynamics backpropagation inferior on discrete-action Atari and used REINFORCE there (with a mixture for continuous control), and DreamerV3 uses REINFORCE with percentile-normalised returns throughout. The swing away from analytic gradients in this line, followed by their return in SHAC, PWM and SuperMimic, is a central tension of the study.

**Intuition.** The RSSM is a learned filter: the posterior incorporates the latest observation, the prior predicts without it, and the KL term makes the prediction good enough that imagination can run on the prior alone.

**Used in:** [World Models](papers/world-models.md), [PlaNet](papers/planet.md), [Dreamer](papers/dreamer.md), [DreamerV2](papers/dreamer-v2.md), [DreamerV3](papers/dreamer-v3.md), [DayDreamer](papers/daydreamer.md), [TD-MPC2](papers/td-mpc2.md) (symlog/two-hot), [RWM](papers/rwm.md) (contrast: no stochastic latent).

<a id="m-dyna-mbpo"></a>
### Dyna-style model rollouts, branched rollouts, MBPO return bound

**Dyna.** [Dyna](papers/dyna.md) interleaves three processes: act in the real environment and store transitions, learn a model from them, and apply the same value or policy update to both real transitions and simulated transitions drawn from the model. The model is used as a generator of additional training data; the learner need not know which data are imagined.

**Branched rollouts.** Model rollouts from the initial-state distribution over the full horizon suffer compounding error ([m-model-learning](#m-model-learning)). [MBPO](papers/mbpo.md) instead starts short model rollouts of length $k$ from states sampled out of the replay buffer, which places imagined data where the real data are and keeps rollouts short. SuperMimic and TGI seed imagined rollouts the same way, from replay-buffer states, and describe this as Dyna-style.

**MBPO's return bound.** Let $\epsilon_\pi = \max_s D_{\mathrm{TV}}(\pi(\cdot\mid s)\Vert\pi_D(\cdot\mid s))$ measure the shift between the current policy and the data-collecting policy, and $\epsilon_m = \max_t\mathbb E_{s\sim\pi_{D,t}}[D_{\mathrm{TV}}(p(\cdot\mid s,a)\Vert\hat p(\cdot\mid s,a))]$ the model's generalisation error on the data distribution. For full model rollouts (Theorem 4.1 of MBPO),

$$J(\pi) \ge \hat J(\pi) - \left[\frac{2\gamma r_{\max}(\epsilon_m + 2\epsilon_\pi)}{(1-\gamma)^2} + \frac{4 r_{\max}\epsilon_\pi}{1-\gamma}\right],$$

which is vacuous at realistic error levels because model error enters with the factor $1/(1-\gamma)^2$. For $k$-step branched rollouts (Theorem 4.3),

$$J(\pi) \ge J^{\mathrm{branch}}(\pi) - 2r_{\max}\left[\frac{\gamma^{k+1}\epsilon_\pi}{(1-\gamma)^2} + \frac{\gamma^k + 2}{1-\gamma}\epsilon_\pi + \frac{k}{1-\gamma}(\epsilon_m + 2\epsilon_\pi)\right].$$

The model error now enters linearly in $k$ rather than through the full effective horizon, while the policy-shift terms decay with $\gamma^k$. With the bound written in terms of $\epsilon_m$ alone the optimal $k$ is zero; MBPO argues that the relevant error is the model's error on the current policy's distribution, $\epsilon_{m'} \approx \epsilon_m + \epsilon_\pi\,d\epsilon_{m'}/d\epsilon_\pi$, and empirically finds the slope small, which makes $k > 0$ favourable. Both proofs proceed by bounding the total-variation distance between the trajectory distributions under two policy–dynamics pairs, step by step, and summing the resulting reward discrepancies.

**Intuition.** Short rollouts branched from real states trade a little horizon for a lot of trust: the model is only asked to be right for $k$ steps near data. TGI's short imagination horizon closed by a critic is the analytic-gradient instance of this principle.

**Used in:** [Dyna](papers/dyna.md), [MBPO](papers/mbpo.md), [MOPO](papers/mopo.md), [MAAC](papers/maac.md), [RWM](papers/rwm.md) (MBPO-PPO), [RWM-O](papers/rwm-o.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-mpc"></a>
### Receding-horizon control, CEM, MPPI

**Receding horizon.** Model predictive control solves, at every control step, a finite-horizon problem $\max_{a_{t:t+H-1}}\sum_{k=0}^{H-1}\gamma^k\hat r(\hat s_{t+k}, a_{t+k}) + \gamma^H\hat V(\hat s_{t+H})$ from the current state, executes the first action, and re-plans. Feedback comes from re-planning; the terminal value $\hat V$ (absent in classic MPC, present in [TD-MPC](papers/td-mpc.md)) compensates for the truncated horizon.

**Cross-entropy method.** CEM, used by [PETS](papers/pets.md) and [PlaNet](papers/planet.md), maintains a Gaussian over action sequences. It samples $N$ sequences, evaluates them in the model, refits the mean and variance to the top $K$ elites, and iterates a few times. It is gradient-free and handles multimodal, discontinuous objectives at the cost of sample counts that grow with the dimension $H\cdot\dim\mathcal A$.

**MPPI.** Model predictive path integral control weights all samples rather than selecting elites. Consider the KL-regularised problem $\min_q\mathbb E_q[S(u)] + \lambda D_{\mathrm{KL}}(q\Vert p)$ over distributions of control sequences $u$, with cost $S$ and a sampling distribution $p$. Its solution is $q^\ast(u)\propto p(u)\exp(-S(u)/\lambda)$, obtained exactly as the Boltzmann policy in [m-maxent](#m-maxent). Projecting $q^\ast$ onto Gaussians with fixed covariance by moment matching gives the update

$$\mu \leftarrow \sum_{i=1}^N w_i u^{(i)}, \qquad w_i = \frac{\exp(-S(u^{(i)})/\lambda)}{\sum_j\exp(-S(u^{(j)})/\lambda)},$$

with temperature $\lambda$ trading greediness for averaging. [TD-MPC](papers/td-mpc.md) and [TD-MPC2](papers/td-mpc2.md) run MPPI in a learned latent space with a terminal Q-function and warm-start the sample distribution with a learned policy, whose actor is trained with a DPG-style gradient through the latent model and critic.

**Intuition.** Sampling-based MPC is a zeroth-order optimiser of the same truncated objective that analytic methods differentiate ([m-short-horizon](#m-short-horizon)); it is robust to discontinuities and expensive at run time, whereas a policy trained by differentiating the objective amortises the optimisation into a network.

**Used in:** [nagabandi-mbmf](papers/nagabandi-mbmf.md), [PETS](papers/pets.md), [PlaNet](papers/planet.md), [TD-MPC](papers/td-mpc.md), [TD-MPC2](papers/td-mpc2.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [DTC](papers/dtc.md), [ilqr-online-to](papers/ilqr-online-to.md).

---

## 4. Analytic (first-order) gradients

<a id="m-reparam"></a>
### Reparameterisation / pathwise gradient, comparison with score-function estimator, SVG

**Pathwise estimator.** Suppose $x\sim p_\theta$ can be written as a differentiable transformation of parameter-free noise, $x = g_\theta(\epsilon)$ with $\epsilon\sim p(\epsilon)$; for a Gaussian, $x = \mu_\theta + \sigma_\theta\odot\epsilon$, $\epsilon\sim\mathcal N(0, I)$. Then the expectation is over a fixed distribution and, for differentiable $F$,

$$\nabla_\theta\,\mathbb E_{x\sim p_\theta}[F(x)] = \nabla_\theta\,\mathbb E_{\epsilon}[F(g_\theta(\epsilon))] = \mathbb E_{\epsilon}\left[\left(\frac{\partial g_\theta(\epsilon)}{\partial\theta}\right)^{\top}\nabla_x F(x)\Big|_{x = g_\theta(\epsilon)}\right].$$

Unlike the score-function estimator ([m-pg-theorem](#m-pg-theorem)), this uses $\nabla_x F$: it requires $F$ to be differentiable (and, for unbiasedness, sufficiently regular, e.g. Lipschitz) but exploits how $F$ varies with $x$.

**Variance contrast.** Take $x\sim\mathcal N(\theta,\sigma^2)$ and $F(x) = cx$, so $\nabla_\theta\mathbb E[F] = c$. The pathwise estimator is $c$ exactly, with zero variance. The score-function estimator is $F(x)(x-\theta)/\sigma^2$; substituting $x = \theta+\sigma\epsilon$ gives $c\theta\epsilon/\sigma + c\epsilon^2$, whose mean is $c$ and whose variance is $c^2(\theta^2/\sigma^2 + 2)$, unbounded as $\sigma\to 0$ and growing with the offset $\theta$ (a baseline removes the $\theta$ term but not the $\epsilon^2$ term). More generally, for $L$-Lipschitz $F$ the pathwise estimator's variance is at most $\mathbb E\lVert\nabla_x F\rVert^2\lVert\partial g/\partial\theta\rVert^2 \le L^2\lVert\partial g/\partial\theta\rVert^2$ independently of the dimension of the noise, while the score-function variance grows with the dimension and with $\mathbb E[F^2]/\sigma^2$. The ranking reverses when $F$ is discontinuous or very stiff: the pathwise gradient of a step function is zero almost everywhere, while the true gradient of its Gaussian smoothing is not ([m-contact-gradients](#m-contact-gradients)).

**Stochastic value gradients.** [SVG](papers/svg.md) applies the pathwise estimator to RL. Policy noise and dynamics noise are both reparameterised, $a_t = \pi_\theta(s_t, \eta_t)$, $s_{t+1} = \hat f(s_t, a_t, \xi_t)$, so that the return becomes a deterministic function of $\theta$ and the noise, and its gradient is obtained by backpropagation along the sampled trajectory. For a real trajectory the dynamics noise $\xi_t$ is unknown; SVG infers it from the observed transition under the model (for additive Gaussian noise, $\xi_t = s_{t+1} - \hat\mu(s_t,a_t)$ up to scaling), so that the model's Jacobians are evaluated along the real trajectory and the gradient remains consistent with real outcomes even if the model's mean is biased. The family is indexed by how far the model is unrolled: SVG(∞) differentiates the whole episode through the model; SVG(1) unrolls one step and bootstraps with a learned $V$, $\nabla_\theta[r(s,a) + \gamma V(\hat f(s,a,\xi))]$; SVG(0) differentiates a learned $Q(s,a)$ directly, as in DPG and SAC. SHAC, MAAC, PWM and TGI occupy the middle of this family with $H$-step unrolls closed by a value function.

**Intuition.** The score-function estimator asks "which random perturbations happened to help?"; the pathwise estimator asks "in which direction does the outcome change?". The second is far more informative when the outcome is a smooth function of the action, and misleading when it is not.

**Used in:** [SVG](papers/svg.md), [PIPPS](papers/pipps.md), [SAC](papers/sac.md), [Dreamer](papers/dreamer.md), [MAAC](papers/maac.md), [SHAC](papers/shac.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [PWM](papers/pwm.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md), [ControlVAE](papers/controlvae.md).

<a id="m-bptt"></a>
### Backpropagation through time for dynamics, Jacobian products, exploding gradients, chaos, Lyapunov exponents

**Setting.** Consider deterministic closed-loop dynamics $s_{t+1} = f(s_t, a_t)$, $a_t = \pi_\theta(s_t)$, and the objective $J(\theta) = \sum_{t=0}^{T-1}\gamma^t r(s_t, a_t)$ from a fixed $s_0$ (stochastic policies are handled by reparameterisation, [m-reparam](#m-reparam)). Define the Jacobians along the trajectory, $F^s_t = \partial f/\partial s$, $F^a_t = \partial f/\partial a$, $P^s_t = \partial\pi_\theta/\partial s$, $P^\theta_t = \partial\pi_\theta/\partial\theta$, all evaluated at $(s_t, a_t)$, and the closed-loop Jacobian $\Phi_t = F^s_t + F^a_t P^s_t$.

**Forward (sensitivity) form.** Differentiating the recursion,

$$\frac{ds_{t+1}}{d\theta} = \Phi_t\frac{ds_t}{d\theta} + F^a_t P^\theta_t, \qquad \frac{ds_0}{d\theta} = 0 \quad\Longrightarrow\quad \frac{ds_t}{d\theta} = \sum_{k=0}^{t-1}\left(\Phi_{t-1}\Phi_{t-2}\cdots\Phi_{k+1}\right)F^a_k P^\theta_k.$$

The total derivative of the objective is

$$\nabla_\theta J = \sum_{t=0}^{T-1}\gamma^t\left[\left(P^\theta_t\right)^{\top}\nabla_a r_t + \left(\frac{ds_t}{d\theta}\right)^{\top}\left(\nabla_s r_t + (P^s_t)^\top\nabla_a r_t\right)\right],$$

so every action's effect on every later reward passes through a product of closed-loop Jacobians $\Phi_{t-1}\cdots\Phi_{k+1}$.

**Reverse (adjoint) form.** Reverse-mode differentiation computes the same quantity in $O(T)$ by propagating costates backwards. With $\bar s_T = 0$ (or $\gamma^T\nabla V(s_T)$ if a terminal value is present),

$$\bar a_t = \gamma^t\nabla_a r_t + (F^a_t)^\top\bar s_{t+1}, \qquad \bar s_t = \gamma^t\nabla_s r_t + (F^s_t)^\top\bar s_{t+1} + (P^s_t)^\top\bar a_t, \qquad \nabla_\theta J = \sum_{t}(P^\theta_t)^\top\bar a_t.$$

The costate $\bar s_t = \partial J/\partial s_t$ is the discrete Pontryagin adjoint; it reappears as $V_x$ in the DDP backward pass ([m-lqr-ddp](#m-lqr-ddp)).

**Exploding and vanishing gradients.** The magnitude of the term linking action $k$ to reward $t$ is governed by $\lVert\Phi_{t-1}\cdots\Phi_{k+1}\rVert$. For a linear time-invariant system $\Phi_t = \Phi$, this is $\lVert\Phi^{t-k-1}\rVert\approx\rho(\Phi)^{t-k-1}$ for spectral radius $\rho$; gradients explode if $\rho > 1$ and vanish if $\rho < 1$. For nonlinear systems the relevant quantity is the largest Lyapunov exponent,

$$\lambda_{\max} = \lim_{t\to\infty}\frac1t\log\left\lVert\Phi_{t-1}\Phi_{t-2}\cdots\Phi_0\, v\right\rVert$$

for almost every initial tangent vector $v$ (the limit exists and is independent of $v$ for almost every trajectory by Oseledets' theorem). Hence $\lVert\Phi_{t-1}\cdots\Phi_{k+1}\rVert\sim e^{\lambda_{\max}(t-k)}$, and $\lVert\nabla_\theta J\rVert$ grows like $e^{(\lambda_{\max} + \log\gamma)T}$ whenever $\lambda_{\max} + \log\gamma > 0$, because the dominant terms are those with the longest Jacobian chains.

**Chaos and the variance of BPTT.** In a chaotic system ($\lambda_{\max} > 0$) the objective is a highly oscillatory function of $\theta$ at long horizons: two parameter vectors $\delta$ apart produce trajectories that separate like $\delta e^{\lambda_{\max}t}$. The exact gradient of a single rollout is then a poor guide to the gradient of the expected (or smoothed) objective. If the initial state is random, the BPTT gradient remains an unbiased estimator of the gradient of the expectation where the expectation is differentiable, but its variance grows like $e^{2\lambda_{\max}T}$. Metz et al. (2021, "Gradients are not all you need") document this failure across physics, meta-learning and recurrent networks, and [suh-diffsim-gradients](papers/suh-diffsim-gradients.md) analyse it for contact-rich simulation. Legged locomotion is prone to it: contact switching, balance near unstable equilibria and the closed-loop feedback of the policy all contribute positive local expansion rates.

**Remedies.** The literature uses four. Truncation (cut gradients after $H$ steps and bootstrap with a critic, [m-short-horizon](#m-short-horizon)) bounds the chain length. Gradient clipping bounds the norm but not the direction error. Smoothing the dynamics or the objective ([m-contact-gradients](#m-contact-gradients)) reduces local expansion. Stabilising feedback in the policy itself reduces $\rho(\Phi_t)$: a policy that rejects perturbations makes its own closed-loop Jacobians contractive, so as training succeeds the gradients tend to become better conditioned. The last point also explains a failure mode early in training: an unstable initial policy produces exactly the expanding Jacobian products that make its gradients least reliable.

**Intuition.** BPTT through physics multiplies sensitivity matrices; their product grows or shrinks exponentially with the number of steps at a rate set by how fast nearby trajectories separate. Long horizons make analytic gradients exact but useless.

**Used in:** [PILCO](papers/pilco.md), [SVG](papers/svg.md), [PIPPS](papers/pipps.md), [Dreamer](papers/dreamer.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [DiffMimic](papers/diffmimic.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [PWM](papers/pwm.md), [amigo-decoupled](papers/amigo-decoupled.md), [SuperTrack](papers/supertrack.md), [ControlVAE](papers/controlvae.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-short-horizon"></a>
### Truncated-horizon objective with terminal critic (SHAC), bias–variance in horizon length

**Objective.** [SHAC](papers/shac.md) samples $N$ trajectories of short horizon $h$ from the differentiable simulator, continuing from the end states of the previous window, and minimises (its Eq. 5)

$$\mathcal L_\theta = -\frac{1}{Nh}\sum_{i=1}^N\left[\sum_{t=t_0}^{t_0+h-1}\gamma^{t-t_0}r(s^i_t, a^i_t) + \gamma^h V_{\bar\psi}(s^i_{t_0+h})\right],$$

with a reparameterised Gaussian policy, a target critic $V_{\bar\psi}$, gradients cut between windows, and the critic fit by MSE to TD(λ) targets computed over the window ([m-lambda-return](#m-lambda-return)). In this study's notation, with imagination horizon $H$, learned model $\hat f_\omega$ and survival weights, the objective used by [SuperMimic](papers/supermimic.md) (its Eq. 2) and by TGI is

$$\hat J_H(\theta) = \mathbb E_{s_0\sim\mathcal D}\left[\sum_{t=0}^{H-1}\gamma^t\hat c_t\,\hat r_t + \gamma^H\hat c_H\,V_\psi(\hat s_H)\right], \qquad \hat s_{t+1} = \hat f_\omega(\hat s_t, a_t),\ a_t = \pi_\theta(\hat o_t, g_t),$$

with $s_0$ drawn from the replay buffer $\mathcal D$ ([m-dyna-mbpo](#m-dyna-mbpo)), $\hat c_{t+1} = \hat c_t(1 - \hat d_{t+1})$ a differentiable survival weight in place of SHAC's hard reset, and $\omega$ frozen during the policy update.

**Gradient.** The adjoint recursion of [m-bptt](#m-bptt) applies with the terminal condition $\bar s_H = \gamma^H\nabla_s V_\psi(\hat s_H)$ (times $\hat c_H$). The update therefore combines two sources of gradient information: the model's Jacobians over $H$ steps and the critic's spatial gradient at the horizon. Special cases organise the family: $H\to\infty$ without critic is full BPTT; $H = 1$ is $\nabla_\theta[r(s,a) + \gamma V_\psi(\hat f(s,a))]$, i.e. SVG(1) ([m-reparam](#m-reparam)); and a critic-free finite $H$ is the reward-sum objective used by [DiffMimic](papers/diffmimic.md), [ControlVAE](papers/controlvae.md) and [SuperTrack](papers/supertrack.md).

**Bias.** If $V_\psi = V^\pi$ and the model is exact, $\hat J_H = J$ for every $H$ (by the Bellman equation unrolled $H$ times). With critic error $e = V_\psi - V^\pi$, the objective bias is $J_H - J = \gamma^H\mathbb E[c_H\, e(s_H)]$, and the gradient bias is

$$\nabla_\theta J_H - \nabla_\theta J = \gamma^H\,\mathbb E\left[c_H\left(\frac{ds_H}{d\theta}\right)^\top\nabla_s e(s_H)\right] + \gamma^H\,\mathbb E\left[e(s_H)\,\nabla_\theta c_H\right].$$

Two observations follow. The bias decays as $\gamma^H$, so short horizons rely more heavily on the critic. And what enters is the error in the critic's gradient $\nabla_s e$, which value regression does not directly control: a critic can fit values well and still have poor spatial gradients, the same issue as for model Jacobians in [m-model-learning](#m-model-learning). Model error contributes a second bias that grows with $H$ (compounding, [m-model-learning](#m-model-learning)), so with a learned model the bias is not monotone in $H$.

**Variance.** The pathwise gradient's variance across sampled start states and noise grows with the norms of the Jacobian products, roughly as $e^{2\lambda_{\max}H}$ ([m-bptt](#m-bptt)), and with the number and stiffness of contact events inside the window ([m-contact-gradients](#m-contact-gradients)). The horizon is therefore the principal bias–variance dial: small $H$ gives low variance and high critic-induced bias; large $H$ gives low critic bias but explosive variance and compounding model error. SHAC also argues a landscape effect: the terminal value replaces the rugged long-horizon return surface by a smooth surrogate, absorbing discontinuities from early termination. [AHAC](papers/ahac.md) makes the horizon adaptive, truncating windows when the dynamics become stiff, and SuperMimic reports that the critic matters most after perturbations, where recovery rewards lie beyond the imagination window.

**Learned-model variant.** Replacing the simulator by $\hat f_\omega$ ([PWM](papers/pwm.md), [SuperMimic](papers/supermimic.md), TGI) changes the character of the gradients: a neural model is smooth by construction, so the Jacobian products are typically better conditioned than those of a stiff simulator, at the cost of model bias near contact. PWM argues that this implicit smoothing is a benefit; the gradient cut and impact penalty in TGI are explicit controls on the same trade-off.

**Intuition.** Differentiate where derivatives are reliable (the near future, through dynamics) and estimate where they are not (the far future, through a critic); choose the boundary by how fast trajectories diverge and how good the model and critic are.

**Used in:** [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [PWM](papers/pwm.md), [MAAC](papers/maac.md), [Dreamer](papers/dreamer.md), [amigo-decoupled](papers/amigo-decoupled.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-contact-gradients"></a>
### Gradients through contact: discontinuities, empirical bias of first-order estimators, smoothing, gradient cutting

**Smoothed objective and two estimators.** Following [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), let $f(\theta)$ be a (possibly discontinuous) rollout cost and define its Gaussian smoothing $F_\sigma(\theta) = \mathbb E_{w\sim\mathcal N(0,\sigma^2 I)}[f(\theta + w)]$, which is what a stochastic policy effectively optimises. Two Monte Carlo estimators of $\nabla F_\sigma$ are

$$\hat\nabla^{[0]}F = \frac{1}{N}\sum_{i=1}^N\frac{1}{\sigma^2}\left(f(\theta + w_i) - f(\theta)\right)w_i, \qquad \hat\nabla^{[1]}F = \frac1N\sum_{i=1}^N\nabla f(\theta + w_i),$$

the zeroth-order (score-function, with baseline $f(\theta)$) and first-order (pathwise) batched gradients. $\hat\nabla^{[0]}$ is unbiased whenever $F_\sigma$ is differentiable, which Gaussian smoothing guarantees for bounded $f$. $\hat\nabla^{[1]}$ is unbiased only if one may exchange gradient and expectation, which requires $f$ to be (locally Lipschitz) continuous.

**Bias at discontinuities.** Take a step $f(\theta) = \mathbb 1[\theta\ge 0]$, a caricature of "the foot touches down or not". Then $F_\sigma(\theta) = \Phi_{\mathcal N}(\theta/\sigma)$ and $\nabla F_\sigma(\theta) = \varphi_{\mathcal N}(\theta/\sigma)/\sigma > 0$, but $\nabla f = 0$ almost everywhere, so $\hat\nabla^{[1]} = 0$ with probability one: the first-order estimator is completely biased because the Dirac mass of the true derivative at the jump is never sampled. Physical contact produces such jumps in the state as a function of parameters, for example whether a swing foot clears an edge.

**Empirical bias at stiff continuity.** Real simulators and learned models are continuous but stiff: replace the step by a ramp of width $\nu\ll\sigma$ and slope $1/\nu$. Now $\hat\nabla^{[1]}$ is unbiased, but each sample lands on the ramp with probability $p\approx\nu\varphi_{\mathcal N}(\theta/\sigma)/\sigma$ and contributes $1/\nu$ when it does. With $N$ samples and $Np\ll 1$, the estimate is zero most of the time and enormous occasionally; its variance is of order $p/(N\nu^2)\approx\varphi/(N\sigma\nu)$, which diverges as $\nu\to 0$. Suh et al. call the typical-sample behaviour empirical bias: the estimator is unbiased in expectation but almost every realisation is biased. They also show that chaotic Jacobian products ([m-bptt](#m-bptt)) inflate $\hat\nabla^{[1]}$'s variance in the same way, while $\hat\nabla^{[0]}$'s variance is bounded by $\mathbb E[f^2]\,d/(N\sigma^2)$ regardless of stiffness.

**Interpolation.** The α-order estimator $\hat\nabla^{[\alpha]} = \alpha\hat\nabla^{[1]} + (1-\alpha)\hat\nabla^{[0]}$ chooses $\alpha$ per step to minimise estimated variance, subject to a confidence check that the two estimators agree within a tolerance, falling back to $\alpha = 0$ when the first-order samples look inconsistent with the zeroth-order ones (a symptom of discontinuity).

**Smoothing the physics.** A complementary approach changes $f$. Soft contact models replace the complementarity condition ([m-rigid-contact](#m-rigid-contact)) by a penetration-dependent force, $\lambda_n = k\,\kappa\,\mathrm{softplus}(-\phi/\kappa)$, or by the convex, regularised formulation of [MuJoCo](papers/mujoco.md); barrier and randomised smoothing variants exist. Smoothing introduces a modelling bias (the robot "feels" ground before reaching it) in exchange for finite, informative gradients. A neural world model performs this smoothing implicitly and uncontrollably ([m-model-learning](#m-model-learning)).

**Gradient through a touchdown: the saltation matrix.** For a hybrid system with guard $g(x) = 0$ (e.g. foot height above terrain), reset map $R$ (the impact law) and vector fields $F^-$ before and $F^+$ after the event, the sensitivity of the post-event state to the pre-event state, including the effect of shifting the event time, is the saltation matrix

$$\Xi = D_xR + \frac{\left(F^+ - D_xR\,F^-\right)D_xg}{D_xg\,F^-}.$$

The denominator $D_xg\,F^-$ is the rate at which the guard is approached: the foot's normal velocity relative to the terrain surface. For a foot at position $p$ above a heightmap, $g = p_z - \mathcal H(p_x, p_y)$ and $D_xg = (-\partial_x\mathcal H, -\partial_y\mathcal H, 1)\,\partial p/\partial x$. Two situations make $\Xi$ ill-conditioned or ill-defined: grazing contact, where the approach velocity tends to zero and the second term diverges; and contact near a terrain discontinuity (a stair edge or gap border), where $\nabla\mathcal H$ is undefined or jumps between grid cells ([m-terrain-lookup](#m-terrain-lookup)), so that infinitesimally different touchdown locations have very different guard normals. These are precisely the touchdowns that the terrain poorly supports.

**Gradient cutting.** A gradient cut applies $\mathrm{sg}$ to a subset of the computation graph, so that $\Phi_t$ is replaced by $m_t\Phi_t$ with $m_t\in\lbrace 0, 1\rbrace$ chosen by a criterion. The resulting estimator drops the terms whose Jacobian chain crosses a cut step; its bias is exactly the sum of those dropped terms, and its variance is bounded by the longest uncut chain. SHAC's cut between windows is the time-triggered version; a cut at selected touchdown events is the event-triggered version. TGI cuts the gradient at touchdowns poorly supported by the terrain: rewards after such a touchdown still train the actions taken after it, and the critic's value at the horizon still trains actions within the window through uncut paths, but the policy is not pushed by the model's (unreliable) saltation-like sensitivity at that event. The criterion trades the systematic bias of the cut against the heavy-tailed variance (and empirical bias) of the uncut gradient at exactly the events where Suh et al.'s analysis predicts the first-order estimator fails.

**Impact penalties.** A penalty on touchdown severity, for instance on the pre-impact normal foot velocity $\lVert v_{n}^-\rVert^2$ or on the change in contact force, acts on the same quantity from the reward side. Physically it reduces impulsive loads. Mathematically it drives the policy towards touchdowns with small velocity jumps $F^+ - D_xR\,F^-$, which shrinks the second term of $\Xi$ and the corresponding stiffness of the learned model's Jacobians, provided the approach velocity does not become so small that contacts become grazing. An impact penalty and a gradient cut are therefore complementary: the penalty makes most touchdowns benign; the cut removes the gradient contribution of the remainder.

**Intuition.** First-order gradients are excellent where the cost is smooth and treacherous at contact events, where they are biased (discontinuities) or have heavy-tailed variance (stiffness). The options are to smooth the physics, mix in zeroth-order estimates, shorten horizons, or remove the offending terms; each buys variance with bias.

**Used in:** [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [DiffMimic](papers/diffmimic.md), [PWM](papers/pwm.md), [MuJoCo](papers/mujoco.md), [TGI](papers/tgi.md).

---

## 5. Optimal control

<a id="m-lqr-ddp"></a>
### LQR, Riccati recursion, iLQR/DDP

**Finite-horizon LQR.** For linear dynamics $x_{t+1} = Ax_t + Bu_t$ and quadratic cost $\sum_{t=0}^{T-1}\tfrac12(x_t^\top Q x_t + u_t^\top R u_t) + \tfrac12 x_T^\top Q_T x_T$ with $Q\succeq 0$, $R\succ 0$, dynamic programming proceeds backwards with a quadratic value $V_t(x) = \tfrac12 x^\top P_t x$, $P_T = Q_T$. At step $t$,

$$V_t(x) = \min_u\ \tfrac12 x^\top Qx + \tfrac12 u^\top Ru + \tfrac12(Ax + Bu)^\top P_{t+1}(Ax + Bu).$$

Setting the derivative in $u$ to zero gives $(R + B^\top P_{t+1}B)u = -B^\top P_{t+1}Ax$, so the optimal control is linear feedback $u_t = K_t x_t$ with $K_t = -(R + B^\top P_{t+1}B)^{-1}B^\top P_{t+1}A$. Substituting back yields the Riccati recursion

$$P_t = Q + A^\top P_{t+1}A - A^\top P_{t+1}B\left(R + B^\top P_{t+1}B\right)^{-1}B^\top P_{t+1}A.$$

For $T\to\infty$ under stabilisability and detectability, $P_t$ converges to the solution of the discrete algebraic Riccati equation and $K$ to a stationary stabilising gain.

**iLQR and DDP.** For nonlinear dynamics $x_{t+1} = f(x_t, u_t)$ and general stage cost $\ell(x,u)$, differential dynamic programming iterates around a nominal trajectory $(\bar x_t, \bar u_t)$. Define the local action-value of a perturbation $(\delta x, \delta u)$, $Q_t(\delta x, \delta u) = \ell(\bar x + \delta x, \bar u + \delta u) + V_{t+1}(f(\bar x + \delta x, \bar u + \delta u))$, and expand to second order with $V'$ denoting derivatives of $V_{t+1}$:

$$\begin{aligned} Q_x &= \ell_x + f_x^\top V'_x, & Q_u &= \ell_u + f_u^\top V'_x,\\ Q_{xx} &= \ell_{xx} + f_x^\top V'_{xx}f_x + V'_x\cdot f_{xx}, & Q_{uu} &= \ell_{uu} + f_u^\top V'_{xx}f_u + V'_x\cdot f_{uu},\\ Q_{ux} &= \ell_{ux} + f_u^\top V'_{xx}f_x + V'_x\cdot f_{ux}. \end{aligned}$$

Minimising the quadratic model in $\delta u$ gives $\delta u = k + K\delta x$ with $k = -Q_{uu}^{-1}Q_u$ and $K = -Q_{uu}^{-1}Q_{ux}$, and the value derivatives propagate as

$$V_x = Q_x + K^\top Q_{uu}k + K^\top Q_u + Q_{ux}^\top k, \qquad V_{xx} = Q_{xx} + K^\top Q_{uu}K + K^\top Q_{ux} + Q_{ux}^\top K.$$

A forward pass applies $u_t = \bar u_t + \alpha k_t + K_t(x_t - \bar x_t)$ with line search $\alpha$. DDP keeps the second-order dynamics tensors $V'_x\cdot f_{xx}$ etc.; iLQR drops them, a Gauss–Newton approximation that guarantees $Q_{uu}\succ 0$ when $\ell_{uu}\succ 0$ and is cheaper. [ilqr-online-to](papers/ilqr-online-to.md) (Tassa, Erez, Todorov) adds Levenberg–Marquardt regularisation of $V'_{xx}$ (adding $\mu I$, which regularises in state space and keeps the feedback gains well behaved) and control limits, and runs the optimiser online as MPC ([m-mpc](#m-mpc)).

**Equivalence with BPTT plus second-order terms.** Consider an open-loop control sequence and the total cost. Setting all second-order terms to zero, the backward recursion reduces to $V_x = Q_x = \ell_x + f_x^\top V'_x$ (with $K = 0$) and $Q_u = \ell_u + f_u^\top V'_x$: exactly the adjoint recursion $\bar s_t = \nabla_s r_t + (F^s_t)^\top\bar s_{t+1}$, $\bar a_t = \nabla_a r_t + (F^a_t)^\top\bar s_{t+1}$ of [m-bptt](#m-bptt), with costs in place of rewards. Thus $Q_u$ is the BPTT gradient with respect to $u_t$, and a gradient step is $\delta u_t = -\eta Q_u$. DDP replaces this by a stagewise Newton step, $-Q_{uu}^{-1}Q_u$, and adds the feedback term $K\delta x$, which accounts for how later controls will respond to earlier changes; iLQR is the Gauss–Newton version. When BPTT trains a feedback policy $\pi_\theta$, the policy Jacobian $P^s_t$ in $\Phi_t$ plays the role of $K_t$: the gradient already accounts for the policy's feedback, but only to first order.

**Intuition.** LQR solves the linear-quadratic problem exactly by propagating a quadratic value backwards. DDP applies this locally to nonlinear problems; its backward pass is BPTT augmented with curvature, which is why it converges in few iterations on smooth problems and struggles, like BPTT, at contact discontinuities.

**Used in:** [ilqr-online-to](papers/ilqr-online-to.md), [MuJoCo](papers/mujoco.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [kajita-zmp-preview](papers/kajita-zmp-preview.md) (LQ preview control), [DTC](papers/dtc.md), [PILCO](papers/pilco.md), [SHAC](papers/shac.md) (contrast).

---

## 6. Legged robotics

<a id="m-rigid-contact"></a>
### Rigid-body dynamics with contacts, complementarity, friction cone

**Equations of motion.** For a floating-base robot with generalised coordinates $q\in\mathbb R^{6+n}$ (base pose and $n$ joint angles) and velocities $v = \dot q$ (with the base orientation handled as in [m-rotations](#m-rotations)),

$$M(q)\dot v + h(q, v) = S^\top\tau + J_c(q)^\top\lambda,$$

where $M\succ 0$ is the mass matrix, $h = C(q,v)v + g(q)$ collects Coriolis, centrifugal and gravity terms, $S = [0_{n\times 6}\ I_n]$ selects the actuated joints (the base is unactuated, so the robot moves its body only through contact forces), $\tau$ are joint torques, $J_c$ stacks contact Jacobians mapping $v$ to contact-point velocities, and $\lambda$ stacks contact forces. The first six rows are the Newton–Euler equations of the whole robot: the rate of change of total momentum equals gravity plus contact wrenches, which is where the centroidal models of [m-lipm-zmp](#m-lipm-zmp) come from.

**Normal complementarity.** With signed gap $\phi(q)\ge 0$ between a contact point and the terrain and normal force $\lambda_n$, rigid non-penetrating contact requires $0\le\lambda_n\perp\phi(q)\ge 0$: force is non-negative, the gap is non-negative, and at least one is zero. In velocity-level time stepping (Stewart–Trinkle, Anitescu–Potra) this becomes $0\le\lambda_n\perp v_n^+\ge 0$ for active contacts, with $v_n^+$ the post-step normal velocity.

**Friction cone.** Coulomb friction constrains the tangential force, $\lVert\lambda_t\rVert\le\mu\lambda_n$, and the maximum dissipation principle selects $\lambda_t$ opposing sliding: $\lambda_t = -\mu\lambda_n v_t/\lVert v_t\rVert$ when $v_t\ne 0$. Linearising the cone by a pyramid with $m$ directions $d_j$ turns the problem, together with the discretised dynamics $M(v^+ - v) = \Delta t(S^\top\tau - h) + J_c^\top\Lambda$ with impulses $\Lambda$, into a linear complementarity problem: find $z\ge 0$ with $w = Az + b\ge 0$ and $z^\top w = 0$. LCPs may have multiple or no solutions for frictional contact (Painlevé paradoxes), and their solution map is piecewise smooth with nondifferentiable kinks at mode switches.

**Impacts.** For a perfectly inelastic impact at a newly active contact set, the impulse that zeroes the contact velocity gives $v^+ = \left(I - M^{-1}J_c^\top(J_cM^{-1}J_c^\top)^{-1}J_c\right)v^-$, a projection onto the constraint-consistent velocities in the metric of $M$. The jump is discontinuous in the touchdown event, which is the source of the saltation term in [m-contact-gradients](#m-contact-gradients).

**Soft and convex models.** [MuJoCo](papers/mujoco.md) replaces strict complementarity by a convex optimisation over contact forces with a regularisation that admits small, controlled constraint violation; the forward dynamics are then unique and smooth almost everywhere, and an inverse dynamics exists. Penalty (spring–damper) models used in many GPU simulators set $\lambda_n = \max(0, -k\phi - b\dot\phi)$. These choices determine which gradients a differentiable simulator returns and which dynamics a learned world model must reproduce.

**Intuition.** Locomotion is the art of moving an unactuated base through unilateral, friction-limited contact forces. Complementarity encodes "push but never pull"; the friction cone encodes "do not slip". Both are inequality constraints whose activation switches discretely, which is the root of every nonsmoothness discussed above.

**Used in:** [MuJoCo](papers/mujoco.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [DTC](papers/dtc.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SHAC](papers/shac.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [hwangbo-anymal](papers/hwangbo-anymal.md), [TGI](papers/tgi.md).

<a id="m-pd-actuator"></a>
### PD set-point control, actuator networks, action spaces

**PD set-point control.** Almost all learned legged controllers output joint position targets that a low-level loop at higher frequency converts to torque,

$$\tau = K_p(q^\ast - q) - K_d\dot q, \qquad q^\ast = q_0 + \sigma_a a,$$

with nominal posture $q_0$, action scale $\sigma_a$ and gains $K_p, K_d$, followed by saturation at torque limits. The policy thus chooses the equilibrium of a virtual spring–damper rather than the force directly. The effective stiffness $K_p$ sets a trade-off between tracking accuracy and compliance under impacts.

**Why position targets.** Peng and van de Panne (2017) compared torque, target-angle, target-velocity and muscle action spaces for learned locomotion and found PD targets learned fastest and most robustly. The PD loop provides immediate local feedback, so the policy's output is low-frequency and the closed-loop dynamics are better damped. For analytic gradients this also matters: the Jacobian $\partial\tau/\partial a = K_p\sigma_a$ (before saturation) is constant, and the damping term reduces the local expansion rate of $\Phi_t$ ([m-bptt](#m-bptt)). Saturation makes $\partial\tau/\partial a$ zero, a vanishing-gradient trap that clipping-aware parameterisations avoid.

**Actuator networks.** [hwangbo-anymal](papers/hwangbo-anymal.md) found that series-elastic actuators of ANYmal deviate substantially from the ideal PD model, and trained a network $\tau_t = \mathcal N(e_{t}, e_{t-1}, e_{t-2}, \dot q_t, \dot q_{t-1}, \dot q_{t-2})$, with $e = q^\ast - q$, on data from the real actuator. Inserted in simulation, it closed much of the sim-to-real gap. The actuator network is an early instance of a learned component of a world model inserted into a known simulator.

**Intuition.** The action space is part of the dynamics the policy sees. A PD interface makes that dynamics smoother, more stable and easier to learn, whether by sampling or by differentiation.

**Used in:** [hwangbo-anymal](papers/hwangbo-anymal.md), [legged-gym](papers/legged-gym.md), [RMA](papers/rma.md), [DeepMimic](papers/deepmimic.md), [peng-animals](papers/peng-animals.md), [BeyondMimic](papers/beyondmimic.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-lipm-zmp"></a>
### Linear inverted pendulum, ZMP, preview control, capture point

**LIPM.** Model the robot as a point mass $m$ at the centre of mass $c = (x, y, z)$ supported by a massless leg from a pivot $p$ on flat ground, and constrain the height to $z = z_c$. Horizontal force balance with the leg force directed from $p$ to $c$ gives, per horizontal axis,

$$\ddot x = \omega^2(x - p_x), \qquad \omega = \sqrt{g/z_c},$$

a linear system with eigenvalues $\pm\omega$: one stable and one unstable mode.

**ZMP.** The zero-moment point is the point on the ground about which the horizontal components of the moment of the contact forces vanish; for flat ground it coincides with the centre of pressure and must lie in the support polygon (convex hull of contact points) for the feet not to roll. Taking moments of the total contact force about $p$ for a body with negligible angular momentum about the CoM, $m\ddot x\,z_c = m g(x - p_x)$, so

$$p_x = x - \frac{z_c}{g}\ddot x,$$

which is the LIPM relation read the other way: the ZMP is the pivot that produces the observed CoM acceleration.

**Preview control.** [kajita-zmp-preview](papers/kajita-zmp-preview.md) treats the CoM jerk $u = \dddot x$ as input to the "cart-table" model with state $(x, \dot x, \ddot x)$ and output $p_x = x - (z_c/g)\ddot x$, and minimises $\sum_k Q_e(p_x^{\mathrm{ref}}(k) - p_x(k))^2 + R\,u(k)^2$ (with an integral-of-error augmentation). The LQ solution ([m-lqr-ddp](#m-lqr-ddp)) has the form $u(k) = -G_i\sum_{j\le k}e(j) - G_x\,\mathbf x(k) - \sum_{j=1}^{N_L}G_p(j)\,p_x^{\mathrm{ref}}(k+j)$: feedback plus a feedforward on the next $N_L$ reference ZMP values. Because the CoM must start moving before the ZMP shifts (the unstable mode), previewing the future footstep plan is necessary for walking.

**Capture point.** Write the LIPM in terms of $\xi = x + \dot x/\omega$, the divergent component of motion. Then $\dot\xi = \dot x + \ddot x/\omega = \dot x + \omega(x - p_x) = \omega(\xi - p_x)$, and the convergent component $x - \dot x/\omega$ decays. With a constant pivot, $\xi(t) - p_x = (\xi(0) - p_x)e^{\omega t}$: the robot comes to rest above $p_x$ if and only if $p_x = \xi$. [capture-point](papers/capture-point.md) names this location the (instantaneous) capture point: stepping onto it stops the robot, and the reachable set of capture points defines N-step capturability. The orbital energy $E = \tfrac12\dot x^2 - \tfrac12\omega^2(x - p_x)^2$ is conserved and equals zero exactly on the trajectories that come to rest.

**Intuition.** Balance in the sagittal plane is a scalar unstable system; the only control is where to put the pivot. The capture point is where a foot must land to cancel the divergence, which is why foothold placement, and the terrain's ability to support a foothold there, dominates balance on rough ground.

**Used in:** [kajita-zmp-preview](papers/kajita-zmp-preview.md), [capture-point](papers/capture-point.md), [raibert](papers/raibert.md) (foot placement heuristics), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [DTC](papers/dtc.md), [watch-your-step](papers/watch-your-step.md) (foothold anticipation), [TGI](papers/tgi.md).

<a id="m-terrain-lookup"></a>
### Heightmaps, bilinear interpolation, differentiable lookups, signed distance fields

**Heightmap.** A 2.5-D terrain representation stores heights $\mathsf H_{ij} = \mathcal H(x_0 + i\Delta, y_0 + j\Delta)$ on a grid of resolution $\Delta$ (typically 0.04–0.1 m), in the world frame or, as an elevation map, in a robot-centric frame. It cannot represent overhangs; it is the standard exteroceptive input for perceptive locomotion. A policy typically observes heights sampled at a fixed pattern of points around the base, or, in [watch-your-step](papers/watch-your-step.md), $k\times k$ patches around predicted footholds, located by rounding to the nearest cell.

**Bilinear interpolation.** For a query $(x, y)$ let $u = (x - x_0)/\Delta$, $v = (y - y_0)/\Delta$, $i = \lfloor u\rfloor$, $j = \lfloor v\rfloor$, $\alpha = u - i$, $\beta = v - j$. Then

$$\mathcal H(x,y) \approx (1-\alpha)(1-\beta)\mathsf H_{ij} + \alpha(1-\beta)\mathsf H_{i+1,j} + (1-\alpha)\beta\,\mathsf H_{i,j+1} + \alpha\beta\,\mathsf H_{i+1,j+1},$$

continuous across cell boundaries. Its gradient within a cell is

$$\frac{\partial\mathcal H}{\partial x} = \frac{1}{\Delta}\left[(1-\beta)(\mathsf H_{i+1,j} - \mathsf H_{ij}) + \beta(\mathsf H_{i+1,j+1} - \mathsf H_{i,j+1})\right],\qquad \frac{\partial\mathcal H}{\partial y} = \frac{1}{\Delta}\left[(1-\alpha)(\mathsf H_{i,j+1} - \mathsf H_{ij}) + \alpha(\mathsf H_{i+1,j+1} - \mathsf H_{i+1,j})\right].$$

Three properties matter for differentiable use. The gradient is discontinuous across cell boundaries (bilinear interpolation is $C^0$, not $C^1$). At a step of height $\Delta h$ (a stair riser), the gradient is $\Delta h/\Delta$ inside the one cell that straddles the edge and zero elsewhere, an artificial ramp whose slope depends on grid resolution, not on the terrain; for $\Delta h = 0.15$ m and $\Delta = 0.05$ m it is 3, a 72° slope. And nearest-neighbour lookup has zero gradient almost everywhere. Bicubic interpolation or pre-blurring the map yields $C^1$ surfaces at the price of rounding edges, i.e. bias in where the robot believes support ends.

**Differentiable terrain terms in imagination.** In TGI the world model is grounded by terrain lookups: quantities such as the gap of each foot to the ground, $\phi_{\mathrm{foot}} = p_z - \mathcal H(p_x, p_y)$, are computed analytically from the predicted foot position and the heightmap rather than predicted by the network, so the policy receives gradients through $\partial\mathcal H/\partial(p_x, p_y)$. The lookup gradient enters the guard derivative $D_xg$ of the saltation matrix ([m-contact-gradients](#m-contact-gradients)) and every terrain-dependent reward. A touchdown that straddles an edge cell, or a foot whose footprint covers cells of widely different heights, is "poorly supported" in two senses at once: physically, part of the sole has no ground beneath it; numerically, the lookup gradient there is a resolution artefact. A support measure computed from the footprint, for example the fraction of footprint samples whose height lies within a tolerance of the sole, or the spread of $\mathsf H$ under the foot, is a natural criterion for the gradient cut. (The [TGI](papers/tgi.md) report gives the exact criterion used.)

**Surface normal and slope.** The upward normal of the height surface is $n = (-\partial_x\mathcal H, -\partial_y\mathcal H, 1)/\sqrt{1 + \lVert\nabla\mathcal H\rVert^2}$, needed to decompose contact forces into normal and tangential parts for the friction cone ([m-rigid-contact](#m-rigid-contact)).

**Signed distance fields.** A signed distance field $\phi:\mathbb R^3\to\mathbb R$ gives the distance to the nearest surface, positive outside and negative inside, and satisfies the eikonal equation $\lVert\nabla\phi\rVert = 1$ almost everywhere, with $\nabla\phi$ the outward normal at the surface. The vertical gap $p_z - \mathcal H$ is not a distance: for a locally planar surface it relates to the true distance by $\phi\approx(p_z - \mathcal H)/\sqrt{1 + \lVert\nabla\mathcal H\rVert^2}$, so on steep slopes and near edges the vertical gap overstates clearance and has the wrong gradient direction. SDFs represent full 3-D geometry (overhangs, walls), give $C^1$ fields away from medial axes, and are used for collision costs in trajectory optimisation; their cost is storage and the need to compute or learn them from sensor data.

**Intuition.** A heightmap lookup is cheap and differentiable almost everywhere, but its derivatives inherit the grid: they are informative on smooth slopes and fictitious at edges. Using them for gradients requires deciding where they can be believed.

**Used in:** [lee-challenging-terrain](papers/lee-challenging-terrain.md), [miki-wild](papers/miki-wild.md), [legged-gym](papers/legged-gym.md), [anymal-parkour](papers/anymal-parkour.md), [extreme-parkour](papers/extreme-parkour.md), [humanoid-parkour](papers/humanoid-parkour.md), [beamdojo](papers/beamdojo.md), [attention-map-encoding](papers/attention-map-encoding.md), [watch-your-step](papers/watch-your-step.md), [deepgait](papers/deepgait.md), [DTC](papers/dtc.md), [TGI](papers/tgi.md).

<a id="m-domain-rand"></a>
### Domain randomisation, privileged teacher–student, DAgger, latent adaptation

**Domain randomisation.** Let $\xi$ denote simulator parameters (friction, masses, motor strength, latency, terrain). Domain randomisation optimises the expected return over a distribution, $\max_\theta\mathbb E_{\xi\sim p(\xi)}[J(\theta;\xi)]$. A policy that cannot observe $\xi$ is optimal for the POMDP in which $\xi$ is hidden state; it learns to be robust (one behaviour good for all $\xi$) or adaptive (infer $\xi$ from history), depending on its inputs.

**Privileged teacher–student.** [lee-challenging-terrain](papers/lee-challenging-terrain.md) trains a teacher $\pi^T(a\mid s, \xi)$ with privileged information (true terrain, contact states, friction) by RL, then a student $\pi^S(a\mid o_{t-K:t})$ from observation history by imitation. Because the student's state distribution differs from the teacher's, the student is trained on its own rollouts with the teacher's actions as labels, which is DAgger ([m-distillation](#m-distillation)). [miki-wild](papers/miki-wild.md) extends this to noisy elevation maps with a belief encoder that learns when to trust exteroception.

**Latent adaptation.** [RMA](papers/rma.md) factors the teacher as $a = \pi(s, z)$ with an extrinsics latent $z = \mu(\xi)$ trained jointly by RL, then trains an adaptation module $\hat z_t = \phi(s_{t-K:t}, a_{t-K-1:t-1})$ by regression $\lVert\hat z_t - z_t\rVert^2$ on the student's rollouts. At deployment, $\phi$ infers $\hat z$ online. Regressing a low-dimensional latent rather than raw parameters ignores the parameters that do not affect behaviour.

**Context-conditioned world models.** A learned world model can condition on $\xi$ directly, $\hat s_{t+1} = \hat f_\omega(\hat s_t, a_t, \xi)$, so that imagination under randomisation is deterministic given the episode context, as in [SuperMimic](papers/supermimic.md) (friction, calibration offsets, torso centre of mass). The policy's gradient then optimises the expected return over $\xi$ without needing $\xi$ as input.

**Intuition.** Randomise what you cannot model precisely, give the teacher what the robot cannot sense, and then teach the student to act as if it knew.

**Used in:** [hwangbo-anymal](papers/hwangbo-anymal.md), [lee-challenging-terrain](papers/lee-challenging-terrain.md), [RMA](papers/rma.md), [miki-wild](papers/miki-wild.md), [extreme-parkour](papers/extreme-parkour.md), [radosavovic-humanoid](papers/radosavovic-humanoid.md), [h2o](papers/h2o.md), [omnih2o](papers/omnih2o.md), [asap](papers/asap.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-curriculum"></a>
### Terrain curricula, success-guided sampling

**Game-inspired terrain curriculum.** [legged-gym](papers/legged-gym.md) arranges terrains in a grid of types × difficulty levels. Each robot is assigned a level; at reset it moves up a level if it walked farther than half the terrain tile, and down if it covered less than half the distance implied by its commanded velocity over the episode; robots that solve the hardest level are resampled uniformly. The resulting distribution concentrates training at the frontier of competence without a hand-tuned schedule.

**Success-guided sampling.** For motion libraries or task sets, sample item $i$ with probability increasing in its failure rate, for instance $p_i\propto(\bar f_i + \epsilon)^\beta$ with a running failure estimate $\bar f_i$, mixed with a uniform floor to avoid forgetting. Tracking systems such as [BeyondMimic](papers/beyondmimic.md) apply this per time bin within long motions, so that difficult segments (jumps, falls) are visited more often.

**Interaction with imagination.** In Dyna-style training ([m-dyna-mbpo](#m-dyna-mbpo)) the curriculum shapes the replay buffer and hence both the model's training distribution and the start states of imagined rollouts. Terrain that the policy rarely visits is also terrain the model rarely sees, so the curriculum indirectly controls where model gradients can be trusted.

**Intuition.** Train where success is uncertain; that is where the gradient signal, of either kind, is largest.

**Used in:** [legged-gym](papers/legged-gym.md), [lee-challenging-terrain](papers/lee-challenging-terrain.md), [extreme-parkour](papers/extreme-parkour.md), [anymal-parkour](papers/anymal-parkour.md), [humanoid-parkour](papers/humanoid-parkour.md), [beamdojo](papers/beamdojo.md), [watch-your-step](papers/watch-your-step.md), [BeyondMimic](papers/beyondmimic.md), [PHC](papers/phc.md), [TGI](papers/tgi.md).

---

## 7. Imitation

<a id="m-tracking-reward"></a>
### Motion-tracking rewards (exp kernels), phase, reference state initialisation, early termination

**Exponential-kernel rewards.** Given a reference motion with poses $\hat q_t$ (joint angles, body positions and orientations), [DeepMimic](papers/deepmimic.md) uses a weighted sum of exponentiated errors,

$$r_t = \sum_k w_k\exp\left(-\frac{\lVert e_{k,t}\rVert^2}{\sigma_k^2}\right),$$

with terms for joint rotations (quaternion differences, [m-rotations](#m-rotations)), joint velocities, end-effector positions and the centre of mass. Each term is bounded in $(0, 1]$, so no single error dominates, and $\sigma_k$ sets the error scale at which the reward is informative. Its gradient is $\nabla_e r = -(2e/\sigma^2)\exp(-\lVert e\rVert^2/\sigma^2)$, largest at $\lVert e\rVert = \sigma/\sqrt 2$ and vanishing for large errors: a tracker that has drifted far from the reference receives almost no gradient, a property that matters more for analytic gradients than for PPO, which needs only reward differences. [BeyondMimic](papers/beyondmimic.md) defines anchor-relative body position, orientation and velocity terms of this form, and SuperMimic adopts its reward without the self-collision penalty.

**Phase.** A phase variable $\phi_t\in[0,1)$ (or the reference frame index) is given to the policy so that it knows which part of the motion to produce; for cyclic motions it is encoded as $(\sin 2\pi\phi, \cos 2\pi\phi)$. Goal-conditioned trackers replace phase by the next reference frames.

**Reference state initialisation and early termination.** RSI starts episodes from states sampled along the reference, so that late parts of a motion are trained before the policy can reach them; this is the imitation analogue of seeding imagined rollouts from the replay buffer. Early termination ends an episode when, for instance, the torso touches the ground or the tracking error exceeds a threshold; it removes uninformative failed states from the data and implicitly penalises falling by forfeiting future reward.

**Intuition.** Tracking turns imitation into dense reward maximisation. The exponential kernel is a soft trust region around the reference.

**Used in:** [DeepMimic](papers/deepmimic.md), [peng-animals](papers/peng-animals.md), [DiffMimic](papers/diffmimic.md), [SuperTrack](papers/supertrack.md), [PHC](papers/phc.md), [h2o](papers/h2o.md), [exbody](papers/exbody.md), [omnih2o](papers/omnih2o.md), [asap](papers/asap.md), [BeyondMimic](papers/beyondmimic.md), [SONIC](papers/sonic.md), [SuperMimic](papers/supermimic.md).

<a id="m-adversarial-imitation"></a>
### GAN, GAIL, AMP discriminator, least-squares GAN, f-divergences

**GAN.** A generator $G$ and discriminator $D$ play $\min_G\max_D\ \mathbb E_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathbb E_{x\sim p_G}[\log(1 - D(x))]$. For fixed $G$ the pointwise maximiser is $D^\ast(x) = p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x) + p_G(x))$, and substituting gives $2\,D_{\mathrm{JS}}(p_{\mathrm{data}}\Vert p_G) - \log 4$, so the generator minimises the Jensen–Shannon divergence.

**f-divergences.** For convex $f$ with $f(1) = 0$, $D_f(P\Vert Q) = \mathbb E_Q[f(dP/dQ)]$. Its variational form $D_f(P\Vert Q)\ge\sup_T\mathbb E_P[T(x)] - \mathbb E_Q[f^\ast(T(x))]$, with convex conjugate $f^\ast$, turns any f-divergence into a discriminator objective (f-GAN). JS, KL and reverse KL are special cases.

**GAIL.** [GAIL](papers/gail.md) shows that maximum-entropy inverse RL followed by RL, with a particular regulariser on the cost, reduces to matching occupancy measures ([m-mdp](#m-mdp)). The practical objective is

$$\min_\pi\max_D\ \mathbb E_{\rho^\pi}[\log D(s,a)] + \mathbb E_{\rho^{E}}[\log(1 - D(s,a))] - \lambda\mathcal H(\pi),$$

where in GAIL's convention $D$ scores policy samples high, and the policy is trained by TRPO on the reward $-\log D(s,a)$. Matching occupancies rather than actions avoids the compounding error of behaviour cloning ([m-distillation](#m-distillation)).

**AMP.** [AMP](papers/amp.md) applies adversarial imitation to state transitions $(s, s')$ (actions of a motion capture clip are unavailable) and uses the least-squares GAN objective with targets $\pm 1$,

$$\min_D\ \mathbb E_{\mathcal M}[(D(s,s') - 1)^2] + \mathbb E_{\pi}[(D(s,s') + 1)^2] + \frac{w_{\mathrm{gp}}}{2}\mathbb E_{\mathcal M}\left[\lVert\nabla D(s,s')\rVert^2\right],$$

with style reward $r = \max\left(0, 1 - \tfrac14(D(s,s') - 1)^2\right)$, combined with a task reward. Least squares avoids the saturating log loss; the gradient penalty on dataset samples stabilises training by keeping the discriminator flat on the data manifold. The LSGAN objective corresponds to minimising a Pearson $\chi^2$ divergence under suitable target choices.

**Intuition.** Instead of telling the policy what to do at each instant, a discriminator tells it whether its motion looks like the data. This gives style without frame-by-frame tracking, at the price of a non-stationary reward.

**Used in:** [GAIL](papers/gail.md), [AMP](papers/amp.md), [ASE](papers/ase.md), [peng-animals](papers/peng-animals.md) (contrast), [radosavovic-humanoid](papers/radosavovic-humanoid.md) (contrast), [PULSE](papers/pulse.md).

<a id="m-latent-skills"></a>
### VAEs, conditional VAEs, skill embeddings, mutual-information objectives (ASE), VQ

**VAE ELBO.** For a latent-variable model $p_\vartheta(x, z) = p(z)p_\vartheta(x\mid z)$ and encoder $q_\varphi(z\mid x)$,

$$\log p_\vartheta(x) = \mathbb E_{q_\varphi}\left[\log\frac{p_\vartheta(x,z)}{q_\varphi(z\mid x)}\right] + D_{\mathrm{KL}}(q_\varphi(z\mid x)\Vert p_\vartheta(z\mid x)) \ge \mathbb E_{q_\varphi}[\log p_\vartheta(x\mid z)] - D_{\mathrm{KL}}(q_\varphi(z\mid x)\Vert p(z)).$$

The identity follows from $\log p(x) = \log p(x,z) - \log p(z\mid x)$ averaged under $q$; the inequality drops the non-negative posterior KL. With a Gaussian encoder, the reconstruction term is differentiated by reparameterisation ([m-reparam](#m-reparam)) and the KL to a standard normal is closed-form, $\tfrac12\sum_i(\mu_i^2 + \sigma_i^2 - 1 - \log\sigma_i^2)$. A weight $\beta$ on the KL trades reconstruction for a smoother latent space. The sequential version is [m-rssm-elbo](#m-rssm-elbo).

**Conditional VAEs and motion priors.** A conditional VAE conditions both encoder and decoder on context $c$ (e.g. the current state): $q(z\mid x, c)$, $p(x\mid z, c)$, and a learned prior $p(z\mid c)$. [ControlVAE](papers/controlvae.md) encodes transitions of a physics-based character into $z$, with a policy as decoder, and trains it through a learned world model, which makes it a direct relative of model-based analytic training. [PULSE](papers/pulse.md) distils a motion tracker ([PHC](papers/phc.md)) into a latent space with a learnable prior, and [MaskedMimic](papers/maskedmimic.md) trains a conditional VAE over partial goals.

**Mutual-information skill objectives.** [ASE](papers/ase.md) learns a low-level policy $\pi(a\mid s, z)$ whose skills $z$ are distinguishable from behaviour, by maximising $I(s, s'; z)$ alongside an AMP-style style reward. Using $I = \mathcal H(z) - \mathcal H(z\mid s, s')$ and a variational approximation $q(z\mid s, s')$,

$$I(s,s'; z) \ge \mathcal H(z) + \mathbb E_{p(z)\,p(s,s'\mid z)}[\log q(z\mid s, s')],$$

since the gap is an expected KL between the true posterior and $q$. With $z$ uniform on the unit hypersphere ($\mathcal H(z)$ constant) and $q$ a von Mises–Fisher density, the bound gives the per-step reward $\kappa\,\mu_q(s, s')^\top z$. A diversity term further encourages different latents to produce different actions.

**Vector quantisation.** VQ-VAE replaces the continuous posterior by the nearest codebook vector, $z_q = e_{k^\ast}$, $k^\ast = \arg\min_k\lVert z_e - e_k\rVert$, passes gradients straight through ($\partial z_q/\partial z_e := I$), and adds codebook and commitment losses $\lVert\mathrm{sg}(z_e) - e\rVert^2 + \beta\lVert z_e - \mathrm{sg}(e)\rVert^2$.

**Intuition.** Latent skill spaces separate how to move from what to do, so that downstream tasks search a low-dimensional, behaviourally meaningful space instead of raw actions.

**Used in:** [World Models](papers/world-models.md), [ASE](papers/ase.md), [ControlVAE](papers/controlvae.md), [PULSE](papers/pulse.md), [MaskedMimic](papers/maskedmimic.md), [SONIC](papers/sonic.md), [SuperMimic](papers/supermimic.md) (goal encoders with a shared decoder).

<a id="m-distillation"></a>
### Behaviour cloning, DAgger, policy distillation

**Behaviour cloning.** Given expert state–action pairs, minimise a supervised loss $\mathbb E_{s\sim d^{\pi^\ast}}[\ell(\pi_\theta(s), \pi^\ast(s))]$. If the learner errs with probability at most $\epsilon$ on the expert's state distribution and a mistake can lead to states the expert never visits, the expected cost over horizon $T$ can grow as $J(\pi_\theta)\le J(\pi^\ast) + T^2\epsilon$: an early mistake moves the learner off-distribution, where further mistakes are uncontrolled. This quadratic compounding is the imitation analogue of model compounding error ([m-model-learning](#m-model-learning)).

**DAgger.** Ross, Gordon and Bagnell (2011) iterate: roll out a mixture $\pi_i = \beta_i\pi^\ast + (1-\beta_i)\hat\pi_i$, label the visited states with expert actions, aggregate all data, and train $\hat\pi_{i+1}$ on the aggregate. Because training on the aggregate is a follow-the-leader strategy in an online learning game whose losses are the per-iteration imitation losses on the learner's own distribution, a no-regret argument shows that after $N$ iterations some policy in the sequence satisfies

$$J(\hat\pi)\le J(\pi^\ast) + uT\epsilon_N + O\left(\frac{1}{N}\right)$$

(for suitable $\beta_i$ and enough samples per iteration), where $\epsilon_N$ is the best achievable imitation loss on the aggregated distribution and $u$ bounds how much one deviation from the expert can increase the cost-to-go. The dependence on $T$ becomes linear because the learner is trained on its own state distribution.

**Policy distillation.** In teacher–student pipelines ([m-domain-rand](#m-domain-rand)) and multi-policy consolidation ([PHC](papers/phc.md), [omnih2o](papers/omnih2o.md), [BeyondMimic](papers/beyondmimic.md)), the student minimises $\mathbb E[D_{\mathrm{KL}}(\pi^T(\cdot\mid s)\Vert\pi^S(\cdot\mid o))]$ or the squared error between mean actions, on states from DAgger-style student rollouts.

**Intuition.** Imitate on the distribution you will actually visit. Learned world models face the same lesson for prediction, and the same remedy (train on one's own rollouts).

**Used in:** [lee-challenging-terrain](papers/lee-challenging-terrain.md), [RMA](papers/rma.md), [miki-wild](papers/miki-wild.md), [extreme-parkour](papers/extreme-parkour.md), [PHC](papers/phc.md), [PULSE](papers/pulse.md), [omnih2o](papers/omnih2o.md), [exbody](papers/exbody.md), [SONIC](papers/sonic.md), [humanoid-parkour](papers/humanoid-parkour.md).

---

## 8. Geometry and architecture

<a id="m-rotations"></a>
### SO(3), quaternions, log map, rotation errors

**SO(3).** Rotations form the group $SO(3) = \lbrace R\in\mathbb R^{3\times 3}: R^\top R = I, \det R = 1\rbrace$, a 3-dimensional manifold. Its Lie algebra consists of skew-symmetric matrices $[\omega]_\times$; the exponential map is Rodrigues' formula $\exp([\theta u]_\times) = I + \sin\theta\,[u]_\times + (1-\cos\theta)[u]_\times^2$ for unit axis $u$, and the log map inverts it for $\theta\in[0,\pi)$, with $\theta = \arccos((\mathrm{tr}R - 1)/2)$.

**Quaternions.** A unit quaternion $\mathbf q = (\cos\tfrac\theta2, \sin\tfrac\theta2 u)$ represents the same rotation as $-\mathbf q$ (double cover). The rotation error between $\mathbf q_1$ and $\mathbf q_2$ is the angle of $\mathbf q_1^{-1}\otimes\mathbf q_2$,

$$\theta_{12} = 2\arccos\left(\lvert\langle\mathbf q_1, \mathbf q_2\rangle\rvert\right) = \lVert\log(R_1^\top R_2)^\vee\rVert,$$

the geodesic distance on $SO(3)$; the absolute value removes the sign ambiguity. The derivative of $\arccos$ is unbounded at $\pm 1$, so implementations use the norm of the vector part, $2\,\mathrm{atan2}(\lVert\mathrm{vec}\rVert, \lvert w\rvert)$, or the squared chordal distance $\lVert R_1 - R_2\rVert_F^2 = 4(1 - \cos\theta_{12})$, which is smooth.

**Network inputs and outputs.** Euler angles have singularities and quaternions have the sign ambiguity; continuous representations such as the 6-D representation (first two columns of $R$, re-orthonormalised by Gram–Schmidt) avoid discontinuities in learned regressors. Locomotion policies commonly observe the gravity vector in the base frame, $R^\top(0, 0, -1)^\top$, which encodes roll and pitch without the unobservable yaw. Anchor- or heading-relative frames (yaw removed) make tracking rewards invariant to global heading.

**Intuition.** Rotations are not vectors: average, difference and regress them through the group's geometry, or the losses and gradients will contain spurious discontinuities.

**Used in:** [DeepMimic](papers/deepmimic.md), [BeyondMimic](papers/beyondmimic.md), [PHC](papers/phc.md), [h2o](papers/h2o.md), [SuperTrack](papers/supertrack.md), [legged-gym](papers/legged-gym.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-attention"></a>
### Attention and transformer map encoders

**Scaled dot-product attention.** For queries $Q\in\mathbb R^{n\times d}$, keys $K\in\mathbb R^{m\times d}$ and values $V\in\mathbb R^{m\times d_v}$,

$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V,$$

with the softmax over keys. The $\sqrt d$ keeps logits of unit scale when query and key entries have unit variance, avoiding saturated softmax gradients. Multi-head attention runs $h$ such maps on learned projections and concatenates them; a transformer layer adds residual connections, normalisation and a feed-forward block. Without positional encodings attention is permutation-equivariant in its keys, so spatial structure must be supplied by encoding the coordinates of each element.

**Map encoders.** [attention-map-encoding](papers/attention-map-encoding.md) encodes each heightmap point (its coordinates and height, or local convolutional features) as a key–value token and forms the query from proprioception and command, so that the attention weights select terrain regions relevant to the current state, typically those near upcoming footholds. Cost scales with the number of map tokens. [watch-your-step](papers/watch-your-step.md) achieves a comparable selection explicitly: a foothold prediction model predicts the next footholds and only $k\times k$ patches around them are passed to the policy, which can be viewed as hard attention with a separately trained, self-supervised query. [radosavovic-humanoid](papers/radosavovic-humanoid.md) uses a causal transformer over the observation–action history as an in-context adaptation mechanism.

**Intuition.** Attention lets the state decide which parts of the map to read. Foothold-gated sampling hard-codes the same decision with a learned predictor, at a fraction of the cost.

**Used in:** [attention-map-encoding](papers/attention-map-encoding.md), [watch-your-step](papers/watch-your-step.md), [radosavovic-humanoid](papers/radosavovic-humanoid.md), [TD-MPC2](papers/td-mpc2.md) (contrast: MLP), [MaskedMimic](papers/maskedmimic.md), [SONIC](papers/sonic.md), [DreamerV3](papers/dreamer-v3.md) (contrast: recurrent).

<a id="m-symmetry"></a>
### Symmetry augmentation and equivariance for locomotion

**Group action.** Legged robots are typically bilaterally symmetric: a reflection $g$ about the sagittal plane maps states and actions by fixed linear operators, $s\mapsto L_g s$, $a\mapsto K_g a$ (swapping left and right limbs, negating lateral and yaw quantities). If the dynamics and reward are invariant, $p(L_g s'\mid L_g s, K_g a) = p(s'\mid s, a)$ and $r(L_g s, K_g a) = r(s, a)$, then the optimal policy can be chosen equivariant, $\pi(L_g s) = K_g\pi(s)$, and the value invariant, $V(L_g s) = V(s)$.

**Three ways to use it.** Data augmentation adds mirrored transitions to every batch, which is valid for off-policy losses and for PPO with care (the importance ratio of a mirrored sample must be computed under the current policy at the mirrored state). A symmetry loss adds $\lVert\pi_\theta(L_g s) - K_g\pi_\theta(s)\rVert^2$ as a soft constraint. Equivariant architectures enforce the constraint exactly, e.g. $\pi(s) = \tfrac12[\tilde\pi(s) + K_g^{-1}\tilde\pi(L_g s)]$ for an arbitrary network $\tilde\pi$ (valid because $g$ is an involution). For a learned world model, the same augmentation doubles the effective data and imposes the physically correct invariance on the model's Jacobians, which is relevant when those Jacobians drive the policy gradient.

**Intuition.** Symmetry is free data and a free inductive bias; it also removes gait asymmetries that RL otherwise finds and the hardware dislikes.

**Used in:** [legged-gym](papers/legged-gym.md), [hwangbo-anymal](papers/hwangbo-anymal.md), [radosavovic-humanoid](papers/radosavovic-humanoid.md), [humanoid-parkour](papers/humanoid-parkour.md), [beamdojo](papers/beamdojo.md), [TGI](papers/tgi.md).
