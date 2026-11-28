# Mathematics cookbook for the TGI lineage

This document collects the mathematics that the paper and group reports of this study rely on. Each entry carries a fixed anchor (listed in [CONVENTIONS](CONVENTIONS.md)), states a precise definition, derives the central result in enough steps to be checked by hand, gives a short intuition, and lists the paper reports that use it. The entries are ordered so that each depends only on earlier ones: foundations of sequential decision making, model-free gradient estimators, model-based reinforcement learning, analytic (first-order) gradients, optimal control, legged-robot mechanics and perception, imitation, and a final group of geometric and architectural tools.

The thread that ties the entries together is the design of Terrain-Grounded Imagination ([TGI](papers/tgi.md)): a legged-robot policy trained by backpropagating 32-step imagined returns through a learned, contact-structured world model ($s' = s + f + B J^\top W$, with exact contact Jacobians and friction-cone and centre-of-pressure bounds), with terrain read from the simulator's heightmap by differentiable lookups, an event-gated return that stops at imagined touchdowns on poorly supported terrain and bootstraps from the critic there, a small likelihood-ratio term added to the pathwise gradient, and an impact penalty on model-predicted contact forces. The TGI-specific statements below were checked against the released code; file:line references are given in the [TGI](papers/tgi.md) report. Where an entry bears directly on one of these components, the connection is made explicit.

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

1. Foundations: [MDP and occupancy](#m-mdp) · [Bellman equations and TD](#m-bellman) · [time-limit truncation vs termination](#m-timeout-bootstrap) · [λ-returns](#m-lambda-return)
2. Model-free estimators: [score-function gradient and policy gradient theorem](#m-pg-theorem) · [advantage and GAE](#m-advantage-gae) · [multiple critics per reward group](#m-multi-critic) · [trust regions and TRPO](#m-trust-region) · [PPO](#m-ppo-clip) · [deterministic policy gradient](#m-dpg) · [maximum-entropy RL](#m-maxent) · [curiosity and RND](#m-intrinsic-motivation)
3. Model-based RL: [model learning and compounding error](#m-model-learning) · [delta-action vs delta-state correction](#m-residual-dynamics) · [ensembles and uncertainty](#m-ensembles) · [RSSM and the ELBO](#m-rssm-elbo) · [Dyna and MBPO](#m-dyna-mbpo) · [MPC, CEM, MPPI](#m-mpc)
4. Analytic gradients: [reparameterisation](#m-reparam) · [estimator variance and ESNR](#m-variance-scaling) · [straight-through estimators](#m-straight-through) · [BPTT through dynamics](#m-bptt) · [short-horizon objective with terminal critic](#m-short-horizon) · [gradients through contact](#m-contact-gradients) · [mixing first- and zeroth-order estimators](#m-estimator-mixing) · [stopping-time truncation and TGI's event-gated return](#m-stopping-time-truncation)
5. Optimal control: [LQR, iLQR, DDP](#m-lqr-ddp) · [shooting, collocation, real-time iteration](#m-shooting)
6. Legged robotics: [rigid-body dynamics with contact](#m-rigid-contact) · [PD control and actuators](#m-pd-actuator) · [LIPM, ZMP, capture point](#m-lipm-zmp) · [SLIP and Raibert foot placement](#m-slip) · [centroidal and single-rigid-body dynamics](#m-centroidal-srbd) · [whole-body control](#m-wbc) · [terrain lookups and SDFs](#m-terrain-lookup) · [domain randomisation and teacher–student](#m-domain-rand) · [curricula](#m-curriculum)
7. Imitation: [kinematic retargeting](#m-retargeting) · [tracking rewards](#m-tracking-reward) · [adversarial imitation](#m-adversarial-imitation) · [latent skills and VAEs](#m-latent-skills) · [composition of primitives](#m-skill-composition) · [diffusion and guidance](#m-diffusion-guidance) · [behaviour cloning, DAgger, distillation](#m-distillation)
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

<a id="m-timeout-bootstrap"></a>
### Bootstrapping at time-limit truncation vs true termination

**Two reasons an episode ends.** A *termination* is part of the task: the robot falls, the MDP enters an absorbing state with zero reward, and $V^\pi(s_{\text{term}}) = 0$. A *truncation* is part of the training procedure: the simulator resets after $T_{\max}$ steps so that data stay diverse, but the task itself would continue. If the state does not contain the elapsed time, the time limit is not part of the MDP that the critic models, and the correct TD target at a truncated step is the ordinary bootstrap.

**Targets.** With a termination flag $d_t$ and a separate truncation flag $\tau_t$, the one-step target and the GAE recursion ([m-advantage-gae](#m-advantage-gae)) are

$$y_t = r_t + \gamma(1 - d_t)\,V(s_{t+1}), \qquad \hat A_t = \delta_t + \gamma\lambda(1-d_t)(1-\tau_t)\,\hat A_{t+1}, \qquad \delta_t = y_t - V(s_t).$$

Termination removes the bootstrap; truncation keeps it but stops the trace, because $\hat A_{t+1}$ belongs to the next episode. Here $s_{t+1}$ must be the true successor state, not the reset state that the vectorised simulator returns in its place; [legged-gym](papers/legged-gym.md) implements the equivalent correction $r_T \leftarrow r_T + \gamma V(s_{T+1})$ on time-outs.

**Bias of the naive target.** Treating a time-out as a termination sets $y_t = r_t$ at the cut. The critic then sees the same state labelled with $V^\pi(s)$ in most episodes and with $r$ alone in the episodes that happen to end there, so it learns a value that depends on an unobserved variable (time left). The resulting error is not noise: it lowers values systematically by $\Pr(\text{cut at } s)\,\gamma\,\mathbb E[V^\pi(s')]$, which in locomotion biases the policy against states that are visited late in long episodes. The alternative remedy, adding remaining time to the state, is correct when the task genuinely has a deadline.

**Imagination.** A finite imagination window is a truncation by construction, and the terminal critic of [m-short-horizon](#m-short-horizon) is exactly the bootstrap prescribed above; termination inside the window is handled by the survival weight $c_t$. Because TGI's critic is fit only to imagined λ-returns, simulator time-outs never enter its targets, and the model's training windows are masked after any episode end, terminal or not. The ANYmal recipe has a genuine deadline (the [AME-2](papers/ame2.md) goal reward is active only in the last seconds of an 8 s episode); there the time left is part of the command observed by actor and critic, is decremented during imagination, and the reward is zeroed past the deadline, which is the "time in the state" remedy.

**Intuition.** Only the task should end the return; the training loop's reset schedule should not.

**Used in:** [legged-gym](papers/legged-gym.md), [PPO](papers/ppo.md), [GAE](papers/gae.md), [SHAC](papers/shac.md), [AME-2](papers/ame2.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-lambda-return"></a>
### λ-returns and TD(λ)

**Definition.** The λ-return is the exponentially weighted average of all $n$-step returns,

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)},$$

the weights $(1-\lambda)\lambda^{n-1}$ summing to one. On a finite window ending at $t + H$ with bootstrap $V(s_{t+H})$, the residual weight is assigned to the longest available return: $G_t^\lambda = (1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}G_t^{(n)} + \lambda^{H-1} G_t^{(H)}$, which is exactly the critic target of SHAC's Eq. 7 ([SHAC](papers/shac.md)).

**Recursive form.** Writing $G_t^{(n)} = r_t + \gamma G_{t+1}^{(n-1)}$ for $n \ge 2$ and $G_t^{(1)} = r_t + \gamma V(s_{t+1})$,

$$G_t^\lambda = (1-\lambda)\left(r_t + \gamma V(s_{t+1})\right) + \sum_{n \ge 2} (1-\lambda)\lambda^{n-1}\left(r_t + \gamma G_{t+1}^{(n-1)}\right) = r_t + \gamma\left[(1-\lambda)V(s_{t+1}) + \lambda G_{t+1}^\lambda\right],$$

using $\sum_{n \ge 1}(1-\lambda)\lambda^{n-1} = 1$ and reindexing the second sum. With termination this becomes $G_t^\lambda = r_t + \gamma(1 - d_{t+1})[(1-\lambda)V_{t+1} + \lambda G_{t+1}^\lambda]$, initialised at the horizon by $G_H^\lambda = V(s_H)$. This backward recursion is how Dreamer, SHAC, SuperMimic (its Eq. 3) and TGI compute critic targets over imagined windows in $O(H)$.

**TD(λ) and eligibility traces.** The backward view accumulates an eligibility trace $e_t = \gamma\lambda e_{t-1} + \nabla_\psi V_\psi(s_t)$ and updates $\psi \leftarrow \psi + \alpha \delta_t e_t$. For offline updates over a complete episode the sum of these increments equals the forward-view update towards $G^\lambda_t$; this equivalence follows from the telescoping identity $G^\lambda_t - V(s_t) = \sum_{k \ge 0}(\gamma\lambda)^k \delta_{t+k}$, which is derived in [m-advantage-gae](#m-advantage-gae).

**State-dependent λ.** Nothing in the recursion requires λ to be constant. Replacing it by $\lambda_t = \lambda(s_{t+1})\in[0,1]$ gives $G_t = r_t + \gamma(1-d_t)\big[(1-\lambda_t)V(s_{t+1}) + \lambda_t G_{t+1}\big]$, which still satisfies $\mathbb E[G_t] = V^\pi(s_t)$ when $V = V^\pi$ (each step mixes two unbiased continuations), and lets the estimator trust the critic more in states where the sampled continuation is less reliable. The extreme case $\lambda_t\in\lbrace 0,1\rbrace$ turns the λ-return into a return truncated at a random stopping time, treated in [m-stopping-time-truncation](#m-stopping-time-truncation).

**TGI's two returns.** TGI uses the recursion twice with different λ (here the index convention is $d_t$, $V_t := V_\psi(s_{t+1})$ for the transition $t\to t+1$). The critic is regressed onto ungated λ-returns with constant $\lambda = 0.95$, $\gamma = 0.99$, $R_H = V(s_H)$, computed without gradient and without a target network. The actor maximises a return of the same recursive form in which $\lambda_t = 1$ everywhere except at imagined touchdowns whose unsupported sole share exceeds 0.3 (`--pg_event 0.3`), where $\lambda_t = 0$: Dreamer's λ-return with the constant λ replaced by a 0/1 stopping indicator. Through this return the actor's gradient is pathwise; the λ-return targets also supply the advantages of TGI's small likelihood-ratio term ([m-estimator-mixing](#m-estimator-mixing)).

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

<a id="m-multi-critic"></a>
### Multiple critics per reward group with separately normalised advantages

**Decomposition.** Write the reward as a sum of groups, $r_t = \sum_{k=1}^K r^k_t$ (for example dense gait regularisers and a sparse foothold term). By linearity of expectation $V^\pi = \sum_k V^{\pi,k}$ with $V^{\pi,k}$ the value of $r^k$ alone, and because the TD residual and GAE are linear in $(r, V)$, the per-group advantages computed with the same $(\gamma,\lambda)$ sum to the advantage of the total reward: $\sum_k \hat A^k_t = \hat A_t$. Training $K$ critics instead of one therefore changes nothing in expectation if the advantages are simply added; what it changes is the regression problem (each critic fits a target of one scale and one temporal structure) and, once normalisation is applied, the weighting.

**Separately normalised combination.** [BeamDojo](papers/beamdojo.md) combines its two critics as

$$\hat A_t = \sum_k w_k\,\frac{\hat A^k_t - \mu_k}{\sigma_k},$$

with batch mean $\mu_k$ and standard deviation $\sigma_k$ of each group's advantage. Relative to normalising the summed advantage, this multiplies group $k$ by $w_k/\sigma_k$, an adaptive per-batch reweighting of the reward groups. Its purpose is to prevent rare, large events (a missed foothold) from dominating the normalised advantage and drowning the dense signal, or vice versa. The price is that the optimised objective is no longer a fixed weighted return: as a group's advantage variance falls, its effective weight rises, so the policy is optimising a moving combination.

**Contrast with the analytic route.** A pathwise gradient does not normalise advantages, so mixed reward scales enter through their derivatives instead; sparse events then matter through how steeply the reward changes near them, which is why TGI replaces sparse foothold indicators by smoothed fields ([m-terrain-lookup](#m-terrain-lookup)) and handles the remaining discontinuity by truncating the return ([m-stopping-time-truncation](#m-stopping-time-truncation)) rather than by a second critic.

**Intuition.** Separate critics give each reward its own baseline and its own voice in the advantage; the price is that the objective's weights are set by batch statistics.

**Used in:** [BeamDojo](papers/beamdojo.md), [GAE](papers/gae.md), [PPO](papers/ppo.md), [TGI](papers/tgi.md) (contrast).

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

**Intuition.** PPO is first-order trust-region optimisation with a soft wall. Its estimator is still the score function, so the per-sample gradient carries the high variance of [m-pg-theorem](#m-pg-theorem); PPO compensates with very large batches. SuperMimic replaces this estimator by a pathwise one, and TGI largely does (keeping a score-function term at weight 0.1, [m-estimator-mixing](#m-estimator-mixing)), without changing the outer actor–critic structure.

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

<a id="m-intrinsic-motivation"></a>
### Curiosity and random network distillation

**Intrinsic reward.** When the task reward rarely produces informative differences between action sequences (a successful sequence of footholds on stepping stones is seldom sampled), an intrinsic reward $r^{\text{int}}_t$ that is large in unfamiliar states is added, $r_t = r^{\text{ext}}_t + \beta\,r^{\text{int}}_t$. Its role is to change the data distribution, not the objective at convergence; $\beta$ is usually annealed or kept small.

**Forward-model curiosity.** Prediction-error curiosity uses a learned dynamics model, $r^{\text{int}}_t = \lVert\hat f_\omega(\phi(s_t), a_t) - \phi(s_{t+1})\rVert^2$, in a feature space $\phi$. It rewards states where the model is wrong, which includes aleatoric noise (the "noisy TV" failure mode) as well as genuine novelty ([m-ensembles](#m-ensembles)).

**Random network distillation (RND).** A fixed, randomly initialised network $g$ defines a target, and a predictor $\hat g_\omega$ is trained by regression on visited states; the intrinsic reward is the prediction error, $r^{\text{int}}(s) = \lVert\hat g_\omega(s) - g(s)\rVert^2$. Because the target is a deterministic function of the state, the error has no irreducible component: it decays exactly where data have accumulated and remains large elsewhere, making it a density-like novelty measure without a dynamics model. Intrinsic and extrinsic returns are usually given separate value heads, with the intrinsic one treated as non-episodic, which connects to [m-multi-critic](#m-multi-critic). [Zhang et al.](papers/risky-terrains.md) use a variant on state–action pairs, $r_{\text{curio}} = c\,\lvert M_1(s,a) - M_2(s,a)\rvert$, and report improved robustness rather than faster learning.

**Relation to analytic training.** Curiosity is a remedy for a variance-limited score-function estimator: it makes rare informative samples less rare. An imagination-based pathwise method attacks the same problem differently, by obtaining a gradient direction from the model even where no successful sample has occurred; exploration then matters mainly for covering the model's training distribution ([m-curriculum](#m-curriculum)).

**Intuition.** Pay the agent to go where its predictions are poor, so that the data contain the rare events the task reward needs.

**Used in:** [risky-terrains](papers/risky-terrains.md), [PPO](papers/ppo.md) (the estimator it supplements), [BeamDojo](papers/beamdojo.md) (contrast: densified reward instead of exploration bonus).

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

**Values are not Jacobians.** For analytic policy gradients ([m-bptt](#m-bptt), [m-short-horizon](#m-short-horizon)) the policy update uses $\partial\hat f/\partial s$ and $\partial\hat f/\partial a$, not only $\hat f$. A model whose predictions match the data can have Jacobians that do not: a function can be uniformly close to another while its derivative oscillates. The error of the analytic gradient therefore depends on Jacobian error, which a value-matching loss controls only indirectly (through smoothness priors of the network, multi-step losses that penalise mis-propagated perturbations, and the data's coverage of nearby actions). This is why SuperMimic collects data with action noise, "to widen the range of actions on which the world model is fit", and why Sobolev-style losses that also match derivatives are a natural, if rarely used, extension. Near contact events the true dynamics are discontinuous ([m-contact-gradients](#m-contact-gradients)), and a smooth network necessarily replaces the jump by a steep ramp whose slope is set by architecture and data density rather than physics; TGI's event-gated return, which stops the differentiable return at poorly supported touchdowns and substitutes the critic ([m-stopping-time-truncation](#m-stopping-time-truncation)), can be read as declining to trust the model's Jacobian chain beyond exactly those events.

**Structured models: putting known physics into $\hat f$.** A generic network must learn from data both what is uncertain (actuator behaviour, effective inertia, contact compliance) and what is already known exactly (kinematics, terrain geometry, the fact that unilateral contact cannot pull). A structured model hard-codes the second part, so that it holds both in prediction and in the Jacobians. TGI's `ImpulseWorldModel` is the instance in this study. It predicts the next state as

$$s_{t+1} = s_t + f(x) + B(x)\,J(s_t)^\top W, \qquad W = \big(F_b, M_b\big)_{b\in\text{soles}},$$

where $x$ collects the heading-invariant state, terrain features, current and previous joint set-points and the domain-randomisation context; $f$ is a learned free-motion increment and $B$ a learned matrix (an effective inverse inertia times $\Delta t$ in normalised increment space). The wrench of each sole is built from learned quantities inside exact constraints: the load is $\lambda_b = 100\,\mathrm N\cdot\sigma\big((0.04 - \phi_b)/0.01\big)\,\sigma(y_{b,0})\,\mathrm{softplus}(y_{b,1})$, a sigmoid gate on the sole's gap $\phi_b$ to the highest terrain point beneath it (4 cm offset, 1 cm temperature) times a learned load probability and magnitude, so that no force acts off the ground; the force is $F_b = \lambda_b\big(n_b + \mu\tanh(\lVert t_\perp\rVert)\,t_\perp/\lVert t_\perp\rVert\big)$ with $\mu = 1$, which keeps the tangential part inside the friction cone $\lVert F_{b,t}\rVert\le\mu\lambda_b$; and the moment $M_b = \lambda_b\,(w, l, \mu_z)\odot\tanh(y_{b,4:7})$ (rotated into the world by the sole orientation) keeps the centre of pressure inside the sole's half-width $w$ and half-length $l$. The generalised wrench $J^\top W$ (base force, base moment $\sum_b (x_b - p)\times F_b + M_b$, and joint torques $a_j^\top[(x_b - o_j)\times F_b + M_b]$ masked to each sole's kinematic chain) is assembled exactly from forward kinematics, a learned analogue of $J_c^\top\lambda$ in [m-rigid-contact](#m-rigid-contact). The model is trained with the multi-step loss above on $k = 8$ windows, with the terrain features and contact geometry recomputed from the *predicted* state at every step, plus a contact-flag cross-entropy and a Huber loss on the simulator's per-foot contact force. Two consequences matter for gradients: the contact gate's 1 cm temperature sets the width of the ramp that replaces the touchdown discontinuity, and the derivative of the next state with respect to foot position passes through exact lever arms and terrain lookups rather than through learned features.

**Intuition.** One-step error is the wrong target when the model is to be iterated. Train on the rollout distribution, build in what is known exactly, and for gradient-based policy learning remember that the policy sees the model through its derivatives.

**Used in:** [nagabandi-mbmf](papers/nagabandi-mbmf.md), [PETS](papers/pets.md), [PlaNet](papers/planet.md) (latent overshooting), [MBPO](papers/mbpo.md), [RWM](papers/rwm.md), [RWM-O](papers/rwm-o.md), [MAAC](papers/maac.md), [PWM](papers/pwm.md), [amigo-decoupled](papers/amigo-decoupled.md), [ControlVAE](papers/controlvae.md), [SuperTrack](papers/supertrack.md), [shi-mb-imitation](papers/shi-mb-imitation.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md) (contact-structured model, terrain re-queried during the multi-step fit).

<a id="m-residual-dynamics"></a>
### Delta-action vs delta-state model correction (ASAP)

**Setting.** A simulator $f^{\text{sim}}$ is available but differs from the real system $f^{\text{real}}$. Given real transitions $(s_t, a_t, s_{t+1})$, there are two places to insert a learned correction.

**Delta-state (residual dynamics).** Learn an additive correction to the predicted next state, $\hat s_{t+1} = f^{\text{sim}}(s_t, a_t) + g_\omega(s_t, a_t)$, by regression on $s_{t+1} - f^{\text{sim}}(s_t,a_t)$. This is the classical residual model; it can represent any discrepancy, including ones that no action could produce (a phantom force on the base), and its output is in state space, so its error compounds through rollouts as in [m-model-learning](#m-model-learning). Every learned world model that predicts increments, $\hat s_{t+1} = s_t + \Delta_\omega(s_t, a_t)$, is a delta-state model with the identity as "simulator"; TGI's free-motion term $f$ is of this kind, and its contact term is structured ([m-model-learning](#m-model-learning)).

**Delta-action.** [ASAP](papers/asap.md) instead learns a residual on the action fed to the simulator, $\hat s_{t+1} = f^{\text{sim}}\big(s_t,\ a_t + \pi^\Delta_\theta(s_t, a_t)\big)$, and trains $\pi^\Delta$ by RL in the simulator with a reward for matching the recorded next state, since $f^{\text{sim}}$ is not differentiable. The corrected simulator is then used to fine-tune the tracking policy, after which $\pi^\Delta$ is discarded. The parameterisation restricts the correction to what the actuators could have done: it captures torque-tracking errors, latency and joint friction well, preserves the simulator's contact handling and physical consistency (momentum is still exchanged only through contacts), but cannot express discrepancies outside the actuation channel, such as wrong terrain compliance or a mis-modelled foot geometry. The two corrections are related through the actuation Jacobian: to first order a delta-action $\Delta a$ produces the delta-state $F^a\Delta a$ ([m-bptt](#m-bptt)), so delta-action models span only the column space of $F^a$, which excludes the unactuated base directions except through contact.

**Intuition.** Correct the simulator where the error enters (the actuators) if you trust its physics; correct the state if you do not.

**Used in:** [ASAP](papers/asap.md), [hwangbo-anymal](papers/hwangbo-anymal.md) (actuator network as an action-channel correction), [shi-mb-imitation](papers/shi-mb-imitation.md) (residual state model), [RWM](papers/rwm.md), [TGI](papers/tgi.md).

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

which is vacuous at realistic error levels because model error enters with the factor $1/(1-\gamma)^2$. (The constants above, and those below, were checked against the MBPO text; MBPO writes $\eta[\pi]$ for $J(\pi)$ and $\hat\eta[\pi]$ for the model return.) For $k$-step branched rollouts with the model error measured on the data-collecting policy's distribution (Theorem 4.2, proved as Theorem A.3),

$$J(\pi) \ge J^{\mathrm{branch}}(\pi) - 2r_{\max}\left[\frac{\gamma^{k+1}\epsilon_\pi}{(1-\gamma)^2} + \frac{\gamma^k + 2}{1-\gamma}\epsilon_\pi + \frac{k}{1-\gamma}(\epsilon_m + 2\epsilon_\pi)\right].$$

The model error now enters linearly in $k$ rather than through the full effective horizon, while the first policy-shift term decays with $\gamma^k$. Taken literally this bound is maximised at $k = 0$, i.e. by not using the model, because a policy shift $\epsilon_\pi$ is pessimistically assumed to inflate the model error to $\epsilon_m + 2\epsilon_\pi$. MBPO therefore introduces the model error under the *updated* policy, $\epsilon_{m'} \ge \max_t\mathbb E_{s\sim\pi_t}[D_{\mathrm{TV}}(p(s'\mid s,a)\Vert\hat p(s'\mid s,a))]$ (Theorem A.2's statement; the main text's restatement writes the expectation under $\pi_{D,t}$), approximates it linearly as $\hat\epsilon_{m'}(\epsilon_\pi)\approx\epsilon_m + \epsilon_\pi\,d\epsilon_{m'}/d\epsilon_\pi$, and states (Theorem 4.3, proved as Theorem A.2)

$$J(\pi) \ge J^{\mathrm{branch}}(\pi) - 2r_{\max}\left[\frac{\gamma^{k+1}\epsilon_\pi}{(1-\gamma)^2} + \frac{\gamma^{k}\epsilon_\pi}{1-\gamma} + \frac{k}{1-\gamma}\,\epsilon_{m'}\right],$$

whose minimiser over $k$ is positive for sufficiently small $\epsilon_{m'}$. The empirical finding that $d\epsilon_{m'}/d\epsilon_\pi$ is small and shrinks with more training data (MBPO's Figure 1) is what licenses $k > 0$; the theory alone does not. All three proofs bound the total-variation distance between trajectory distributions under two policy–dynamics pairs step by step and sum the reward discrepancies, with the branch point weighted by $\gamma^t$.

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

<a id="m-variance-scaling"></a>
### Variance of score-function vs pathwise estimators; Gaussian smoothing; expected signal-to-noise ratio

**Setting.** Let the policy be Gaussian, $a_t = \mu_\theta(s_t) + \sigma\epsilon_t$ with $\epsilon_t\sim\mathcal N(0, I_m)$ over $H$ steps, and write the rollout objective as $f(\theta, \bar\epsilon)$ with $\bar\epsilon = \epsilon_{1:H}$. The objective actually optimised is the smoothed $F_\sigma(\theta) = \mathbb E_{\bar\epsilon}[f(\theta,\bar\epsilon)]$. The zeroth-order (score-function) and first-order (pathwise) single-sample estimators are $\hat g_0 = \frac{1}{\sigma}(f - b)\sum_t (\partial\mu_t/\partial\theta)^\top\epsilon_t$ and $\hat g_1 = \nabla_\theta f(\theta,\bar\epsilon)$ ([m-pg-theorem](#m-pg-theorem), [m-reparam](#m-reparam)).

**Score-function variance: dimension and $1/\sigma^2$.** Take one step, $\partial\mu/\partial\theta = I$ and a smooth $f$. With baseline $b = f(\theta)$ and $f(\theta + \sigma\epsilon) - f(\theta)\approx\sigma g^\top\epsilon$, $g = \nabla f$, the estimator is $\hat g_0\approx\epsilon\epsilon^\top g$. Gaussian fourth moments give $\mathbb E[\epsilon\epsilon^\top gg^\top\epsilon\epsilon^\top] = 2gg^\top + \lVert g\rVert^2 I$, hence $\mathrm{Cov}[\hat g_0] = gg^\top + \lVert g\rVert^2 I$ and $\mathrm{tr}\,\mathrm{Cov}[\hat g_0] = (m+1)\lVert g\rVert^2$: even in the most favourable, locally linear case the total variance grows linearly with the number of perturbed dimensions. In a rollout, every other step's noise also perturbs the return without being correlated with $\epsilon_t$, so the residual $f - b$ is of order $\sigma\lVert g\rVert\sqrt{Hm}$ while the score is of order $\sqrt{Hm}/\sigma$ per unit parameter Jacobian; when $f$ is not linear on the scale $\sigma$ (or is noisy), the residual no longer shrinks with $\sigma$ and the variance acquires an explicit $1/\sigma^2$. Suh et al. (Lemma 3.10 of [suh-diffsim-gradients](papers/suh-diffsim-gradients.md)) bound it as

$$\mathrm{Var}\big[\bar\nabla^{[0]}F\big] \le \frac{B_V^2 B_\pi^2}{N}\cdot\frac{Hm}{\sigma^2}, \qquad \lvert f\rvert\le B_V,\ \lVert D_\theta\mu\rVert_{\mathrm{op}}\le B_\pi,$$

linear in the horizon–dimension product, inverse in $\sigma^2$, and independent of how stiff the dynamics are.

**Pathwise variance: Jacobian products.** The pathwise estimator involves no $1/\sigma$ and, for $L$-Lipschitz $f$, its variance is bounded by $L^2\lVert\partial g_\theta/\partial\theta\rVert^2$ independently of $m$ ([m-reparam](#m-reparam)). The constant $L$ of a rollout, however, is a product of closed-loop Jacobians, so $\mathrm{Var}[\hat g_1]$ grows like $\lVert\nabla f\rVert^{2H}$ (the scaling quoted by [PWM](papers/pwm.md) after Suh et al.), i.e. like $e^{2\lambda_{\max}H}$ in terms of the Lyapunov exponent ([m-bptt](#m-bptt)), and like $1/\nu$ for a contact ramp of width $\nu$ ([m-contact-gradients](#m-contact-gradients)). The two estimators thus have complementary failure modes: $\hat g_0$ degrades with dimension and small $\sigma$, $\hat g_1$ with stiffness, chaos and horizon. AHAC's Lemma 3.2 gives the corresponding bound on the first-order sample error, $\lVert\bar\nabla^{[1]}J - \bar\nabla^{[0]}J\rVert \le H B_r B_\pi\big(\tfrac12 + B_f^{H-1}\big)$ under Lipschitz bounds $B_r, B_\pi, B_f$ on reward, policy and dynamics ([AHAC](papers/ahac.md)).

**Gaussian smoothing.** By Stein's identity, $\nabla F_\sigma(\theta) = \frac1\sigma\mathbb E[f(\theta + \sigma\epsilon)\epsilon]$, which is well defined for any bounded measurable $f$; if $f$ is Lipschitz it also equals $\mathbb E[\nabla f(\theta + \sigma\epsilon)]$. Smoothing makes $F_\sigma$ differentiable with gradient bounded by $B_V\sqrt{2/\pi}/\sigma$ per direction, and, for $L$-Lipschitz $f$, gives $\nabla F_\sigma$ a Lipschitz constant of order $L\sqrt m/\sigma$. Policy noise is therefore randomised smoothing of the return: larger $\sigma$ yields a better-conditioned objective at the price of optimising a blurred version of it. In TGI the imagination noise ($\sigma = 0.2$ on normalised actions) is drawn only when the likelihood-ratio term is enabled (`--lr_coef > 0`, the default); the pathwise gradient is then averaged over noisy rollouts, which is a Monte Carlo estimate of $\nabla F_\sigma$ rather than of the gradient of the deterministic return.

**Expected signal-to-noise ratio.** Variance alone does not say whether a gradient is useful, because a large gradient can have large variance and still point the right way. Parmas et al. (2023), as used by [PWM](papers/pwm.md), measure per-coordinate signal against noise,

$$\mathrm{ESNR}(\hat g) = \mathbb E_\theta\left[\sum_{p=1}^{P}\frac{\mathbb E[\hat g_p]^2}{\mathrm{Var}[\hat g_p]}\right],$$

with the outer expectation over parameter values visited in training. A batch of $N$ samples has per-coordinate SNR $N$ times the single-sample value, so ESNR states how many samples are needed before the sign of the average is reliable. On the chaotic double pendulum PWM finds that the true dynamics give the lowest ESNR, that SimNorm-regularised MLP models give higher ESNR, and that a model trained on 16-step autoregressive windows sustains higher ESNR at long horizons than one trained on 3-step windows, which is its argument that learned models can supply more useful first-order gradients than the system they imitate.

**Intuition.** Zeroth-order estimators pay for dimension and for small noise; first-order estimators pay for stiffness and for long Jacobian chains. ESNR is the currency in which the two can be compared.

**Used in:** [REINFORCE](papers/reinforce.md), [SVG](papers/svg.md), [PIPPS](papers/pipps.md), [metz-gradients](papers/metz-gradients.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [PWM](papers/pwm.md), [onoda-composite-gradients](papers/onoda-composite-gradients.md), [TGI](papers/tgi.md).

<a id="m-straight-through"></a>
### Straight-through estimators: exact forward value, surrogate backward gradient

**Construction.** Given a forward function $h$ whose derivative is useless (zero almost everywhere, undefined, or unavailable) and a differentiable surrogate $g$ that approximates it, the straight-through (ST) combination is

$$\tilde y(x) = g(x) + \mathrm{sg}\big(h(x) - g(x)\big), \qquad \tilde y = h(x)\ \text{in the forward pass}, \qquad \frac{\partial\tilde y}{\partial x} := \frac{\partial g}{\partial x}\ \text{in the backward pass}.$$

The downstream computation sees the exact value $h(x)$, and gradients flow as though $g$ had been computed. The original instance is quantisation ($h$ rounds, $g$ is the identity).

**What it estimates.** The ST gradient is not the gradient of the objective evaluated with $h$, nor of the objective evaluated with $g$: it is the chain rule applied with $g$'s local Jacobian at a point whose downstream values were computed with $h$. Its bias is therefore $(\partial g/\partial x - \partial h/\partial x)$ propagated through the downstream Jacobian. It is well motivated when $g$ approximates a smoothing of $h$, $g(x)\approx\mathbb E_u[h(x + u)]$, because then $\partial g/\partial x$ approximates the derivative of the smoothed objective ([m-variance-scaling](#m-variance-scaling)), and it is poorly motivated when $g$ and $h$ disagree in sign of slope.

**Instances in this study.** Categorical latents in [DreamerV2](papers/dreamer-v2.md) and [DreamerV3](papers/dreamer-v3.md) use a sampled one-hot vector forward and the softmax probabilities backward, $z = \mathrm{onehot} + p - \mathrm{sg}(p)$. VQ-VAE codes ([m-latent-skills](#m-latent-skills)), the residual VQ of [MoConVQ](papers/moconvq.md) and the finite scalar quantisation of [SONIC](papers/sonic.md) pass gradients straight through the quantiser. [Song et al.](papers/song-diffsim-quadruped.md) use the simulator's next state forward and a single-rigid-body model's Jacobian backward, and [DMO](papers/amigo-decoupled.md) evaluates a learned model's Jacobians at simulator states while taking forward values from the simulator: the same idea applied to whole dynamics steps. TGI uses it in the unsupported sole share: the forward value uses nearest-node heights, so that a foot centred on a 20 cm stone reads as fully supported, while the gradient flows through the bilinear interpolant, $\tilde h = h^{\mathrm{bil}} + \mathrm{sg}(h^{\mathrm{near}} - h^{\mathrm{bil}})$ ([m-terrain-lookup](#m-terrain-lookup)). A related but distinct pattern in TGI selects an element without gradient and differentiates the selected value, as in the sole gap to the highest terrain point, $\phi_b = p_z - h_{i^\ast}$ with $i^\ast = \arg\max_i\mathrm{sg}(h_i)$; that is the ordinary subgradient of the maximum, not a surrogate.

**Intuition.** Compute what is true, differentiate what is useful, and accept that the gradient is a heuristic whose quality depends on how well the surrogate's slope matches the smoothed truth.

**Used in:** [DreamerV2](papers/dreamer-v2.md), [DreamerV3](papers/dreamer-v3.md), [MoConVQ](papers/moconvq.md), [SONIC](papers/sonic.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [amigo-decoupled](papers/amigo-decoupled.md), [TGI](papers/tgi.md).

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

**Chaos and the variance of BPTT.** In a chaotic system ($\lambda_{\max} > 0$) the objective is a highly oscillatory function of $\theta$ at long horizons: two parameter vectors $\delta$ apart produce trajectories that separate like $\delta e^{\lambda_{\max}t}$. The exact gradient of a single rollout is then a poor guide to the gradient of the expected (or smoothed) objective. If the initial state is random, the BPTT gradient remains an unbiased estimator of the gradient of the expectation where the expectation is differentiable, but its variance grows like $e^{2\lambda_{\max}T}$. [Metz et al.](papers/metz-gradients.md) ("Gradients are Not All You Need", arXiv 2111.05803, 2021) trace this failure to the spectrum of the recurrent Jacobian $\partial s_i/\partial s_{i-1}$ and demonstrate it in three domains: policy optimisation through the Brax rigid-body simulator (Ant), where with fixed randomness the loss along a random parameter direction becomes highly curved beyond about 8–16 steps and, although averaging over policy noise smooths the loss surface, the gradient variance across seeds still grows exponentially with unroll length; meta-learning of learned optimisers; and jammed-disk packing in molecular dynamics. They survey remedies (well-behaved system or architecture choices, truncated backpropagation, gradient clipping, shadowing methods for ergodic systems, learned models, and black-box estimators) and report that on Ant raw gradients through 400 steps failed under an extensive hyperparameter search, only a narrow band of truncation lengths learned a gait, gradient clipping found no working setting, and PPO outperformed all of their gradient-based attempts. [suh-diffsim-gradients](papers/suh-diffsim-gradients.md) analyse the same phenomenon for contact-rich simulation. Legged locomotion is prone to it: contact switching, balance near unstable equilibria and the closed-loop feedback of the policy all contribute positive local expansion rates.

**Remedies.** The literature uses four. Truncation (cut gradients after $H$ steps and bootstrap with a critic, [m-short-horizon](#m-short-horizon)) bounds the chain length. Gradient clipping bounds the norm but not the direction error. Smoothing the dynamics or the objective ([m-contact-gradients](#m-contact-gradients)) reduces local expansion. Stabilising feedback in the policy itself reduces $\rho(\Phi_t)$: a policy that rejects perturbations makes its own closed-loop Jacobians contractive, so as training succeeds the gradients tend to become better conditioned. The last point also explains a failure mode early in training: an unstable initial policy produces exactly the expanding Jacobian products that make its gradients least reliable.

**Intuition.** BPTT through physics multiplies sensitivity matrices; their product grows or shrinks exponentially with the number of steps at a rate set by how fast nearby trajectories separate. Long horizons make analytic gradients exact but useless.

**Used in:** [PILCO](papers/pilco.md), [SVG](papers/svg.md), [PIPPS](papers/pipps.md), [Dreamer](papers/dreamer.md), [metz-gradients](papers/metz-gradients.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [DiffMimic](papers/diffmimic.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [PWM](papers/pwm.md), [amigo-decoupled](papers/amigo-decoupled.md), [SuperTrack](papers/supertrack.md), [ControlVAE](papers/controlvae.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md).

<a id="m-short-horizon"></a>
### Truncated-horizon objective with terminal critic (SHAC), bias–variance in horizon length

**Objective.** [SHAC](papers/shac.md) samples $N$ trajectories of short horizon $h$ from the differentiable simulator, continuing from the end states of the previous window, and minimises (its Eq. 5)

$$\mathcal L_\theta = -\frac{1}{Nh}\sum_{i=1}^N\left[\sum_{t=t_0}^{t_0+h-1}\gamma^{t-t_0}r(s^i_t, a^i_t) + \gamma^h V_{\bar\psi}(s^i_{t_0+h})\right],$$

with a reparameterised Gaussian policy, a target critic $V_{\bar\psi}$, gradients cut between windows, and the critic fit by MSE to TD(λ) targets computed over the window ([m-lambda-return](#m-lambda-return)). In this study's notation, with imagination horizon $H$, learned model $\hat f_\omega$ and survival weights, the objective used by [SuperMimic](papers/supermimic.md) (its Eq. 2) is

$$\hat J_H(\theta) = \mathbb E_{s_0\sim\mathcal D}\left[\sum_{t=0}^{H-1}\gamma^t\hat c_t\,\hat r_t + \gamma^H\hat c_H\,V_\psi(\hat s_H)\right], \qquad \hat s_{t+1} = \hat f_\omega(\hat s_t, a_t),\ a_t = \pi_\theta(\hat o_t, g_t),$$

with $s_0$ drawn from the replay buffer $\mathcal D$ ([m-dyna-mbpo](#m-dyna-mbpo)), $\hat c_{t+1} = \hat c_t(1 - \hat d_{t+1})$ a differentiable survival weight in place of SHAC's hard reset, and $\omega$ frozen during the policy update.

**TGI's variant.** TGI keeps this structure ($H = 32$ steps of 20 ms, $\gamma = 0.99$, soft terminations, replay-buffer starts, frozen model and critic) but changes two things. First, the horizon is not fixed: the return stops at the first imagined touchdown whose unsupported sole share exceeds 0.3 and bootstraps from the critic at the post-touchdown state, so $H$ is replaced by a state-dependent stopping time $\tau\le H$ ([m-stopping-time-truncation](#m-stopping-time-truncation)). This is not a detach of the dynamics Jacobian: both the value and the gradient of the remainder of the window are replaced by $V_\psi$, and actions after $\tau$ receive no pathwise gradient from that rollout. Second, the estimator is not purely pathwise. With $a_t = \mu_\theta(o_t) + \epsilon_t$, $\epsilon_t\sim\mathcal N(0, 0.2^2 I)$, the loss is

$$\mathcal L_\pi = -\overline{G_0} \;-\; 0.1\,\overline{\hat c_t\,\log\pi_\theta(a_t\mid o_t)\,\tilde A_t} \;+\; 2\,\overline{\lVert\pi(\mathcal M o) - \mathcal M\pi(o)\rVert^2},$$

with $\tilde A_t$ the batch-standardised λ-return advantages of the critic ([m-lambda-return](#m-lambda-return)), a fixed-weight score-function term (`--lr_coef 0.1`, [m-estimator-mixing](#m-estimator-mixing)) and a mirror-consistency loss ([m-symmetry](#m-symmetry)); gradients are clipped at norm 1.

**Gradient.** The adjoint recursion of [m-bptt](#m-bptt) applies with the terminal condition $\bar s_H = \gamma^H\nabla_s V_\psi(\hat s_H)$ (times $\hat c_H$). The update therefore combines two sources of gradient information: the model's Jacobians over $H$ steps and the critic's spatial gradient at the horizon. Special cases organise the family: $H\to\infty$ without critic is full BPTT; $H = 1$ is $\nabla_\theta[r(s,a) + \gamma V_\psi(\hat f(s,a))]$, i.e. SVG(1) ([m-reparam](#m-reparam)); and a critic-free finite $H$ is the reward-sum objective used by [DiffMimic](papers/diffmimic.md), [ControlVAE](papers/controlvae.md) and [SuperTrack](papers/supertrack.md).

**Bias.** If $V_\psi = V^\pi$ and the model is exact, $\hat J_H = J$ for every $H$ (by the Bellman equation unrolled $H$ times). With critic error $e = V_\psi - V^\pi$, the objective bias is $J_H - J = \gamma^H\mathbb E[c_H\, e(s_H)]$, and the gradient bias is

$$\nabla_\theta J_H - \nabla_\theta J = \gamma^H\,\mathbb E\left[c_H\left(\frac{ds_H}{d\theta}\right)^\top\nabla_s e(s_H)\right] + \gamma^H\,\mathbb E\left[e(s_H)\,\nabla_\theta c_H\right].$$

Two observations follow. The bias decays as $\gamma^H$, so short horizons rely more heavily on the critic. And what enters is the error in the critic's gradient $\nabla_s e$, which value regression does not directly control: a critic can fit values well and still have poor spatial gradients, the same issue as for model Jacobians in [m-model-learning](#m-model-learning). Model error contributes a second bias that grows with $H$ (compounding, [m-model-learning](#m-model-learning)), so with a learned model the bias is not monotone in $H$.

**Variance.** The pathwise gradient's variance across sampled start states and noise grows with the norms of the Jacobian products, roughly as $e^{2\lambda_{\max}H}$ ([m-bptt](#m-bptt)), and with the number and stiffness of contact events inside the window ([m-contact-gradients](#m-contact-gradients)). The horizon is therefore the principal bias–variance dial: small $H$ gives low variance and high critic-induced bias; large $H$ gives low critic bias but explosive variance and compounding model error. SHAC also argues a landscape effect: the terminal value replaces the rugged long-horizon return surface by a smooth surrogate, absorbing discontinuities from early termination. SuperMimic reports that the critic matters most after perturbations, where recovery rewards lie beyond the imagination window.

**Adaptive horizons (AHAC).** [AHAC](papers/ahac.md) (Georgiev et al., ICML 2024) starts from the first-order sample-error bound $\lVert\bar\nabla^{[1]}J - \bar\nabla^{[0]}J\rVert\le HB_rB_\pi(\tfrac12 + B_f^{H-1})$ (its Lemma 3.2, [m-variance-scaling](#m-variance-scaling)): since contact makes $B_f\gg B_r, B_\pi$, both long windows and prolonged stiff contact inflate the error. Its first variant, AHAC-1, truncates each single-environment rollout at the first step with stiff contact, $\lVert\nabla f(s_t,a_t)\rVert > C$ ($C = 500$ on Hopper), and beats SHAC's fixed $H = 32$ there; the authors also report that SHAC with graph truncation at stiff contacts in parallel environments "did not improve performance, likely due to gradient variances across differing trajectory lengths", and that asynchronous truncation is impractical for vectorised simulation. The scalable AHAC instead keeps one horizon for all environments and adapts it: the actor maximises $\sum_{h=t}^{t+H-1}\gamma^{h-t}r_h + \gamma^HV_\psi(s_{t+H})$ subject to $\lVert\nabla f(s_h,a_h)\rVert\le C$ for every $h$ in the window, the Lagrangian $L_\pi(\theta,\phi) = J + \phi^\top\big(C - \lVert\nabla f\rVert_{t:t+H}\big)$ is optimised by gradient steps on $\theta$ and on the multipliers $\phi\ge 0$, and $H$ itself is updated by a step proportional to $\sum_h\phi_h$, so the horizon settles where the stiffness constraint is just met. The stiffness is measured on the contact-force Jacobian, normalised per state dimension by $\hat q_t = \max(q_t, 1)$, and a double critic trained to convergence each iteration replaces SHAC's target critic. Across Hopper, Ant, ANYmal, Humanoid and a 152-actuator muscle humanoid (dflex, soft contact), AHAC reports about 40 % higher asymptotic reward than PPO, its strongest model-free baseline, in aggregate (50 % IQM over 10 seeds; 41 % on Ant), and higher reward than SHAC; an ablation on Ant finds SHAC's best fixed horizon close to the gait period ($H\approx 28$). The relevance to TGI is twofold: AHAC is the closest precedent for truncating at contact events, and its negative result on per-environment truncation is the obvious objection to TGI's per-rollout stopping time, which TGI avoids in part because its computation graph has a fixed length $H$ and the stop is implemented as a 0/1 weight inside the recursion rather than as a graph cut.

**Learned-model variant (PWM).** Replacing the simulator by $\hat f_\omega$ ([PWM](papers/pwm.md), [SuperMimic](papers/supermimic.md), TGI) changes the character of the gradients: a neural model is smooth by construction, so the Jacobian products are typically better conditioned than those of a stiff simulator, at the cost of model bias near contact. PWM (Georgiev et al., ICLR 2025) makes this an argument for learned models. On a ball-against-wall toy problem with a discontinuous objective, a SimNorm-regularised MLP yields a smoother landscape with a smaller optimality gap than a ReLU MLP, despite lower accuracy; on the chaotic double pendulum, learned models give lower gradient variance and higher ESNR than the true dynamics ([m-variance-scaling](#m-variance-scaling)). Its algorithm pre-trains a TD-MPC2-style latent model (encoder $E_\phi$, latent dynamics $F_\phi$, reward $R_\phi$ with task embeddings, SimNorm outputs, trained autoregressively on $H$-step windows with a latent consistency loss $\lVert z_{t+1} - \mathrm{sg}(E_\phi(s_{t+1}))\rVert^2$ and a two-hot reward cross-entropy) on offline multi-task data, then extracts each task's policy with exactly the objective above in latent space, $L_\pi = \mathbb E[\sum_{h<H}\gamma^hR_\phi(z_h, a_h) + \gamma^HV_\psi(z_H)]$, with an ensemble of three critics fit to TD(λ) targets. It reports policy extraction in under 10 minutes per task, tasks with up to 152 action dimensions solved better than methods using ground-truth dynamics, and up to 27 % higher reward than TD-MPC2 on an 80-task suite without online planning. PWM relies on implicit smoothing by regularisation; TGI instead puts exact structure into the model ([m-model-learning](#m-model-learning)), uses explicit smooth surrogates for terrain ([m-terrain-lookup](#m-terrain-lookup)), and handles the remaining discontinuity by the event-gated return and an impact penalty on model-predicted forces ([m-contact-gradients](#m-contact-gradients)).

**Intuition.** Differentiate where derivatives are reliable (the near future, through dynamics) and estimate where they are not (the far future, through a critic); choose the boundary by how fast trajectories diverge and how good the model and critic are.

**Used in:** [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [PWM](papers/pwm.md), [MAAC](papers/maac.md), [MVE](papers/mve.md), [Dreamer](papers/dreamer.md), [amigo-decoupled](papers/amigo-decoupled.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [SuperMimic](papers/supermimic.md), [TGI](papers/tgi.md) (event-gated horizon and hybrid estimator).

<a id="m-contact-gradients"></a>
### Gradients through contact: discontinuities, empirical bias of first-order estimators, smoothing, gradient cutting

**Smoothed objective and two estimators.** Following [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), let $f(\theta)$ be a (possibly discontinuous) rollout cost and define its Gaussian smoothing $F_\sigma(\theta) = \mathbb E_{w\sim\mathcal N(0,\sigma^2 I)}[f(\theta + w)]$, which is what a stochastic policy effectively optimises. Two Monte Carlo estimators of $\nabla F_\sigma$ are

$$\hat\nabla^{[0]}F = \frac{1}{N}\sum_{i=1}^N\frac{1}{\sigma^2}\left(f(\theta + w_i) - f(\theta)\right)w_i, \qquad \hat\nabla^{[1]}F = \frac1N\sum_{i=1}^N\nabla f(\theta + w_i),$$

the zeroth-order (score-function, with baseline $f(\theta)$) and first-order (pathwise) batched gradients. $\hat\nabla^{[0]}$ is unbiased whenever $F_\sigma$ is differentiable, which Gaussian smoothing guarantees for bounded $f$. $\hat\nabla^{[1]}$ is unbiased only if one may exchange gradient and expectation, which requires $f$ to be (locally Lipschitz) continuous.

**Bias at discontinuities.** Take a step $f(\theta) = \mathbb 1[\theta\ge 0]$, a caricature of "the foot touches down or not". Then $F_\sigma(\theta) = \Phi_{\mathcal N}(\theta/\sigma)$ and $\nabla F_\sigma(\theta) = \varphi_{\mathcal N}(\theta/\sigma)/\sigma > 0$, but $\nabla f = 0$ almost everywhere, so $\hat\nabla^{[1]} = 0$ with probability one: the first-order estimator is completely biased because the Dirac mass of the true derivative at the jump is never sampled. Physical contact produces such jumps in the state as a function of parameters, for example whether a swing foot clears an edge.

**Empirical bias at stiff continuity.** Real simulators and learned models are continuous but stiff: replace the step by a ramp of width $\nu\ll\sigma$ and slope $1/\nu$. Now $\hat\nabla^{[1]}$ is unbiased, but each sample lands on the ramp with probability $p\approx\nu\varphi_{\mathcal N}(\theta/\sigma)/\sigma$ and contributes $1/\nu$ when it does. With $N$ samples and $Np\ll 1$, the estimate is zero most of the time and enormous occasionally; its variance is of order $p/(N\nu^2)\approx\varphi/(N\sigma\nu)$, which diverges as $\nu\to 0$, while the *empirical* variance computed from the samples is usually zero.

**Empirical bias, precisely.** Suh et al. (Definition 3.4) say that a random vector $z$ with $\mathbb E\lVert z\rVert<\infty$ has $(\beta,\Delta,S)$-empirical bias if there is an event $E$ with $\Pr[E]\ge 1-\beta$ such that $\lVert\mathbb E[z\mid E] - \mathbb E[z]\rVert\ge\Delta$ and $\lVert z - \mathbb E[z\mid E]\rVert\le S$ almost surely on $E$. In words: with high probability the sample lies in a tight cluster ($S$ small) whose centre is far ($\Delta$) from the true mean. The paradigm is $z = 0$ with probability $1-\beta$ and $z = 1/\beta$ otherwise, which has $(\beta, 1, 0)$-empirical bias and variance $1/\beta - 1$. Lemma 3.5 makes the link to variance exact: $(\beta,\Delta,S)$-empirical bias implies $\mathrm{Var}[z]\ge\Delta_0^2/\beta$ with $\Delta_0 = \max\lbrace 0, (1-\beta)\Delta - \beta\lVert\mathbb E[z]\rVert\rbrace$, so small-β empirical bias forces large true variance that the empirical variance does not reveal. For the ramp $\bar H_\nu$ with smoothing $\sigma$ and $c_\sigma = 1/(\sqrt{2\pi}\sigma)$, the per-sample first-order gradient at $\theta = \nu/2$ has $(c_\sigma\nu, c_\sigma, 0)$-empirical bias: the true gradient is $c_\sigma$, yet with probability at least $1 - c_\sigma\nu$ the sample is exactly zero. Stiff springs ([m-rigid-contact](#m-rigid-contact)) and chaotic Jacobian products ([m-bptt](#m-bptt)) inflate $\hat\nabla^{[1]}$'s variance by the same mechanism, whereas $\hat\nabla^{[0]}$'s variance obeys the stiffness-independent bound $B_V^2B_\pi^2Hm/(N\sigma^2)$ of Lemma 3.10 ([m-variance-scaling](#m-variance-scaling)).

**Interpolation.** The α-order estimator $\bar\nabla^{[\alpha]} = \alpha\bar\nabla^{[1]} + (1-\alpha)\bar\nabla^{[0]}$, computed from independent samples, chooses $\alpha$ to minimise the estimated variance $\alpha^2\hat\sigma_1^2 + (1-\alpha)^2\hat\sigma_0^2$ subject to an accuracy constraint $\varepsilon + \alpha\lVert\bar\nabla^{[1]} - \bar\nabla^{[0]}\rVert\le\gamma$, where $\varepsilon$ is a confidence bound on the zeroth-order estimate's error. The solution is the inverse-variance weight $\alpha_\infty = \hat\sigma_0^2/(\hat\sigma_0^2 + \hat\sigma_1^2)$ when it satisfies the constraint and $(\gamma-\varepsilon)/\lVert\bar\nabla^{[1]} - \bar\nabla^{[0]}\rVert$ otherwise, with $\alpha = 0$ if $\varepsilon > \gamma$. A large disagreement between the two estimators, the symptom of strict or empirical bias, therefore pushes the estimate towards the zeroth-order one. The broader family of mixtures, and TGI's fixed-weight version, are treated in [m-estimator-mixing](#m-estimator-mixing).

**Smoothing the physics.** A complementary approach changes $f$. Soft contact models replace the complementarity condition ([m-rigid-contact](#m-rigid-contact)) by a penetration-dependent force, $\lambda_n = k\,\kappa\,\mathrm{softplus}(-\phi/\kappa)$, or by the convex, regularised formulation of [MuJoCo](papers/mujoco.md); barrier and randomised smoothing variants exist. Smoothing introduces a modelling bias (the robot "feels" ground before reaching it) in exchange for finite, informative gradients. A neural world model performs this smoothing implicitly and uncontrollably ([m-model-learning](#m-model-learning)).

**Gradient through a touchdown: the saltation matrix.** For a hybrid system with guard $g(x) = 0$ (e.g. foot height above terrain), reset map $R$ (the impact law) and vector fields $F^-$ before and $F^+$ after the event, the sensitivity of the post-event state to the pre-event state, including the effect of shifting the event time, is the saltation matrix

$$\Xi = D_xR + \frac{\left(F^+ - D_xR\,F^-\right)D_xg}{D_xg\,F^-}.$$

The denominator $D_xg\,F^-$ is the rate at which the guard is approached: the foot's normal velocity relative to the terrain surface. For a foot at position $p$ above a heightmap, $g = p_z - \mathcal H(p_x, p_y)$ and $D_xg = (-\partial_x\mathcal H, -\partial_y\mathcal H, 1)\,\partial p/\partial x$. Two situations make $\Xi$ ill-conditioned or ill-defined: grazing contact, where the approach velocity tends to zero and the second term diverges; and contact near a terrain discontinuity (a stair edge or gap border), where $\nabla\mathcal H$ is undefined or jumps between grid cells ([m-terrain-lookup](#m-terrain-lookup)), so that infinitesimally different touchdown locations have very different guard normals. These are precisely the touchdowns that the terrain poorly supports.

**Gradient cutting and return truncation.** Two different operations are both called "cutting" in this literature and must be distinguished. A *Jacobian cut* applies $\mathrm{sg}$ inside the computation graph, replacing $\Phi_t$ by $m_t\Phi_t$ with $m_t\in\lbrace 0,1\rbrace$ while leaving the objective's value unchanged: the estimator drops exactly the terms whose Jacobian chain crosses a cut step, its bias is the sum of those dropped terms, and the rewards after the cut still train the actions taken after it. SHAC's cut between windows is the time-triggered version, and AHAC-1's truncation at stiff contact the event-triggered one. A *return truncation* instead changes the objective: at the selected step the sampled continuation is replaced, in value and in gradient, by a critic evaluated at the post-event state. TGI does the second, not the first. Its gate (computed without gradient) marks an imagined touchdown, $\hat c_{t+1}(1-\hat c_t)$ with $\hat c$ the model's soft contact probability, on a sole whose unsupported share $u_{t+1}$ makes the product exceed 0.3; the return then reads $G_t = r_t + \gamma(1-d_t)V_\psi(s_{t+1})$ at that step instead of $r_t + \gamma(1-d_t)G_{t+1}$ ([m-stopping-time-truncation](#m-stopping-time-truncation)). Consequences: the touchdown transition itself is still differentiated (through $r_t$ and through $\nabla_sV_\psi(s_{t+1})$, so the model's Jacobian at the event does enter, once); everything the model predicts after the event is removed from the objective; and actions after the first gated event receive no pathwise gradient from that rollout, only the likelihood-ratio term. The trade is the critic's bias at post-touchdown states against the heavy-tailed variance and empirical bias of a Jacobian chain that runs through a poorly supported contact, at exactly the events where Suh et al.'s analysis predicts first-order estimators fail.

**TGI's smooth contact surrogates.** Where TGI does differentiate through contact, it does so through designed surrogates rather than through a stiff simulator. The model's contact gate is $\sigma\big((0.04 - \phi_b)/0.01\big)$ in the sole gap $\phi_b$ (a ramp of roughly 1 cm temperature, i.e. $\nu\approx$ a few centimetres in the notation above, set by design rather than by data density); friction and centre of pressure enter through $\tanh$ saturations; and the terrain quantities that rewards depend on are a straight-through unsupported share, a Gaussian-blurred support field and a bilinearly queried edge indicator ([m-terrain-lookup](#m-terrain-lookup), [m-straight-through](#m-straight-through)). These choices fix the effective $\nu$ of every contact-related ramp, and hence, by the empirical-bias analysis above, the minimum smoothing σ (imagination noise 0.2) at which pathwise samples become informative.

**Impact penalties.** A penalty on touchdown severity, for instance on the pre-impact normal foot velocity $\lVert v_{n}^-\rVert^2$ or on the change in contact force, acts on the same quantity from the reward side. Physically it reduces impulsive loads. Mathematically it drives the policy towards touchdowns with small velocity jumps $F^+ - D_xR\,F^-$, which shrinks the second term of $\Xi$ and the corresponding stiffness of the learned model's Jacobians, provided the approach velocity does not become so small that contacts become grazing. TGI's version penalises the world model's own sole forces, $r_{\text{impact}} = -\sum_b\max(\lVert\hat F_b\rVert - 450\,\mathrm N, 0)/100\,\mathrm N$ for the G1 (600 N and weight 3 for ANYmal); because $\hat F_b$ is a differentiable function of state and action through the contact gate and the learned load magnitude, the penalty's gradient reaches actions several steps before landing. An impact penalty and return truncation are complementary: the penalty makes most touchdowns benign; the truncation removes the model's continuation after the remainder that land on poorly supported terrain.

**Intuition.** First-order gradients are excellent where the cost is smooth and treacherous at contact events, where they are biased (discontinuities) or have heavy-tailed variance (stiffness). The options are to smooth the physics, mix in zeroth-order estimates, shorten horizons, or remove the offending terms; each buys variance with bias. TGI combines designed smoothing, a small zeroth-order term, and return truncation at the worst events.

**Used in:** [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [metz-gradients](papers/metz-gradients.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [DiffMimic](papers/diffmimic.md), [PWM](papers/pwm.md), [onoda-composite-gradients](papers/onoda-composite-gradients.md), [MuJoCo](papers/mujoco.md), [TGI](papers/tgi.md).

<a id="m-estimator-mixing"></a>
### Mixing first- and zeroth-order estimators: inverse-variance weighting, total propagation, α-order, fixed-weight hybrids

**Inverse-variance weighting.** Given two unbiased, independent estimators $\hat g_0$ and $\hat g_1$ of the same gradient with variances $v_0$ and $v_1$, the convex combination $\alpha\hat g_1 + (1-\alpha)\hat g_0$ has variance $\alpha^2v_1 + (1-\alpha)^2v_0$, minimised at

$$\alpha^\ast = \frac{v_0}{v_0 + v_1}, \qquad \mathrm{Var}^\ast = \frac{v_0v_1}{v_0 + v_1}\le\min(v_0, v_1).$$

The combination is never worse than the better estimator, provided both are unbiased and the variances are known. In practice the weights are estimated from the same samples (biasing the result slightly) and may be chosen per parameter coordinate.

**Total propagation (PIPPS).** [PIPPS](papers/pipps.md) observed that pathwise gradients through a learned probabilistic model could be "hopelessly inaccurate" through chaos, and that likelihood-ratio gradients through the same model, with a leave-one-out baseline and within-batch importance sampling, were better. Its total propagation algorithm performs the inverse-variance combination at every step of a single backward pass: at step $t$ the gradient with respect to the predicted distribution parameters $\zeta_t$ is computed both by reparameterisation (from the downstream combined gradient) and by the likelihood ratio, the two are weighted by $k_{\mathrm{LR}} = \hat\sigma_{\mathrm{LR}}^{-2}/(\hat\sigma_{\mathrm{LR}}^{-2} + \hat\sigma_{\mathrm{RP}}^{-2})$ with variances estimated across particles, and the combined estimate is passed to step $t-1$. This effectively selects, per step, the reparameterisation depth at which pathwise information is still trustworthy.

**α-order with an accuracy guard (Suh et al.).** [suh-diffsim-gradients](papers/suh-diffsim-gradients.md) argue that inverse-variance weighting is unsafe under discontinuities, because empirical bias makes $\hat v_1$ look small exactly when $\hat g_1$ is wrong. Their α-order estimator adds the constraint $\varepsilon + \alpha\lVert\bar g_1 - \bar g_0\rVert\le\gamma$, using the zeroth-order estimate as an unbiased reference with confidence radius $\varepsilon$ ([m-contact-gradients](#m-contact-gradients)). [Onoda et al.](papers/onoda-composite-gradients.md) revisit this and find that a simpler discontinuity test matches it on Suh et al.'s examples, and that on stiffened differentiable-robotics benchmarks plain per-step, per-dimension inverse-variance weighting is as good or better, concluding that variance control matters more than explicit bias detection there.

**Fixed-weight hybrids (TGI).** TGI adds the two estimators with a constant weight rather than an estimated one. Its loss is $-\overline{G_0} - \beta\,\overline{\hat c_t\log\pi_\theta(a_t\mid o_t)\tilde A_t}$ with $\beta = 0.1$ (`--lr_coef`), Gaussian imagination noise of standard deviation 0.2, $\log\pi$ evaluated at the detached sampled action so that only $\mu_\theta$ is differentiated, and $\tilde A_t$ the batch-standardised λ-return advantages of the critic. Three differences from the schemes above matter. The two terms are not estimators of the same quantity: the first is the pathwise gradient of the *event-gated* return ([m-stopping-time-truncation](#m-stopping-time-truncation)), the second a score-function gradient of the ungated λ-return objective, with advantages rescaled to unit variance, so the mixture is a weighted sum of two objectives' gradients rather than a variance-optimal combination of one. The weight does not adapt to local smoothness, so near discontinuities it does not shift towards the zeroth-order term as α-order or IVW would; the event gate plays that role instead, by removing the pathwise continuation where it is least reliable while the likelihood-ratio term continues to credit every action in the window. And because the gated return covers actions only up to the first gated touchdown, the score-function term is the only signal for actions after it. The weight is not ablated in the released code.

**Intuition.** Two noisy compasses can be averaged into a better one only if both point at the same target and one knows how noisy each is; mixing schemes differ in how they estimate the noise and what they do when one compass is broken.

**Used in:** [PIPPS](papers/pipps.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [onoda-composite-gradients](papers/onoda-composite-gradients.md), [metz-gradients](papers/metz-gradients.md), [DreamerV2](papers/dreamer-v2.md) (mixed REINFORCE and dynamics backpropagation for continuous control), [TGI](papers/tgi.md).

<a id="m-stopping-time-truncation"></a>
### Returns truncated at a state-dependent stopping time; λ-return with state-dependent $\lambda_t$; TGI's event-gated return

**Definition.** Let $\ell_t\in\lbrace 0,1\rbrace$ be a continuation indicator that depends only on the trajectory up to $s_{t+1}$, and define $\tau = \min(\lbrace t:\ell_t = 0\rbrace\cup\lbrace H-1\rbrace)$, a stopping time with respect to the trajectory's filtration. The stopped return with critic bootstrap is

$$G_0^\tau = \sum_{t=0}^{\tau}\gamma^t\bar c_t\,r_t + \gamma^{\tau+1}\bar c_{\tau+1}\,V(s_{\tau+1}), \qquad \bar c_t = \prod_{i<t}(1-d_i),$$

and it is generated by the backward recursion

$$G_{H-1} = r_{H-1} + \gamma(1-d_{H-1})V(s_H), \qquad G_t = r_t + \gamma(1-d_t)\big[(1-\ell_t)V(s_{t+1}) + \ell_t\,G_{t+1}\big].$$

This is the λ-return recursion of [m-lambda-return](#m-lambda-return) with the constant λ replaced by $\lambda_t = \ell_t$. A constant-$H$ truncated objective ([m-short-horizon](#m-short-horizon)) is the special case $\ell_t\equiv 1$; Dreamer's λ-return is the constant case $\lambda_t\equiv\lambda$; a soft gate $\lambda_t\in(0,1)$ interpolates.

**Unbiasedness of the value.** If $V = V^\pi$, then $\mathbb E[G_0^\tau] = V^\pi(s_0)$ for any stopping time. Proof by the backward recursion: conditional on the trajectory up to $s_{t+1}$, $\ell_t$ is fixed and $\mathbb E[G_{t+1}\mid s_{t+1}] = V^\pi(s_{t+1})$ by induction, so the bracket has conditional mean $V^\pi(s_{t+1})$ whatever $\ell_t$ is, and the Bellman equation closes the step ([m-bellman](#m-bellman)). This is the optional-stopping argument applied to the martingale $M_t = \sum_{i<t}\gamma^i\bar c_ir_i + \gamma^t\bar c_tV^\pi(s_t)$. The condition matters: an indicator that peeked at future rewards would bias the estimate.

**Gradient.** With a differentiable model, $\nabla_\theta G_0^\tau$ is computed by BPTT through the recursion with $\ell_t$ treated as a constant (it is computed without gradient). The adjoint recursion of [m-bptt](#m-bptt) runs only back from $\tau+1$, with terminal condition $\bar s_{\tau+1} = \gamma^{\tau+1}\bar c_{\tau+1}\nabla_sV(s_{\tau+1})$. Two things follow. Since $\tau$ is piecewise constant in $\theta$, the objective is discontinuous where the event flips, and the pathwise derivative ignores that jump (the derivative of the indicator is not represented), which is an instance of the strict bias of [m-contact-gradients](#m-contact-gradients) at the event boundary itself. And with critic error $e = V - V^\pi$, the gradient bias becomes $\mathbb E\big[\gamma^{\tau+1}\bar c_{\tau+1}(ds_{\tau+1}/d\theta)^\top\nabla_se(s_{\tau+1})\big]$ plus a term from $\nabla_\theta\bar c_{\tau+1}$: the critic's spatial-gradient error is now incurred at the event states rather than at a fixed horizon, which is acceptable only if the critic is accurate there.

**TGI's event-gated return.** TGI sets $\ell_t = 0$ exactly when an imagined touchdown on a poorly supported sole occurs in the transition $t\to t+1$: $\max_{\text{feet}}\ \hat c_{t+1}(1-\hat c_t)\,u_{t+1} > 0.3$, with $\hat c$ the model's soft contact probability, $\hat c_0$ the logged contact flag of the start state, and $u$ the straight-through unsupported sole share ([m-terrain-lookup](#m-terrain-lookup)); `--pg_event 0.3`, and `--pg_event 0` recovers the plain 32-step return. The critic is evaluated on $s_{t+1}$, the first state after the touchdown. Several properties of this design follow from the analysis above. It is a return truncation, not a Jacobian detach ([m-contact-gradients](#m-contact-gradients)). The gate is computed without gradient from the trajectory up to $s_{t+1}$, so the value of $G_0$ remains an unbiased estimate given an exact critic, while the gradient ignores the dependence of $\tau$ on $\theta$. The critic that closes the return is trained on ungated λ-returns ($\lambda = 0.95$) from the same imagined rollouts, so in post-touchdown states its targets themselves rest on model predictions beyond the event, which partly reintroduces the model's continuation through the critic's fitting rather than through its gradient. Because the graph always has $H$ steps and the stop enters only as a 0/1 weight, all rollouts in a batch share one computation graph, which sidesteps the asynchronous-truncation difficulty reported for AHAC-1 ([m-short-horizon](#m-short-horizon)). Finally, actions after $\tau$ in a rollout are trained only by the likelihood-ratio term ([m-estimator-mixing](#m-estimator-mixing)); the share of gated steps is logged during training but not reported.

**Intuition.** Differentiate the imagined future until the model reaches an event it cannot be trusted to continue through; from there, ask the critic. As long as the decision to stop looks only at the past, the value estimate stays honest; the gradient does not see the stopping rule.

**Used in:** [Dreamer](papers/dreamer.md), [SHAC](papers/shac.md), [AHAC](papers/ahac.md), [suh-diffsim-gradients](papers/suh-diffsim-gradients.md), [SuperMimic](papers/supermimic.md), [BeamDojo](papers/beamdojo.md) (contrast: second critic instead of truncation), [TGI](papers/tgi.md).

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

<a id="m-shooting"></a>
### Single vs multiple shooting, collocation, real-time iteration

**Problem.** Discretised optimal control seeks $\min_{u_{0:N-1}}\sum_k\ell(x_k,u_k) + \ell_N(x_N)$ subject to $x_{k+1} = F(x_k,u_k)$ (one integrator step of the continuous dynamics), $x_0 = \hat x_0$, and path constraints $g(x_k,u_k)\le 0$. Transcriptions differ in which variables the optimiser sees.

**Single shooting.** Only the controls are decision variables; states are eliminated by forward simulation, $x_k = x_k(u_{0:k-1})$. The problem is small and always dynamically feasible, but the objective's dependence on early controls passes through the products of Jacobians of [m-bptt](#m-bptt), so unstable or stiff dynamics make it badly conditioned, and a poor initial guess for the controls gives a poor trajectory. iLQR/DDP ([m-lqr-ddp](#m-lqr-ddp)) is a single-shooting method with a feedback-corrected forward pass, and policy training by BPTT through a model is single shooting in the policy parameters.

**Multiple shooting.** States $x_{0:N}$ become decision variables too, and the dynamics are imposed as equality constraints $x_{k+1} - F(x_k,u_k) = 0$, solved by sequential quadratic programming (SQP). Each constraint involves one step, so the Jacobian of the constraints is block-banded and every sensitivity is local; instability no longer compounds inside the objective but appears as constraint violations that the solver closes gradually. A state trajectory can be initialised independently (for example from the previous solution or a reference), and the Karush–Kuhn–Tucker system has the stagewise structure that a Riccati recursion solves in $O(N)$, as in [perceptive NMPC](papers/perceptive-nmpc.md) with its Riccati-structured QP solver. Infeasible intermediate iterates are the price.

**Collocation.** Direct collocation also treats states as variables but represents each interval by a polynomial and imposes the dynamics at collocation points, $\dot x(t_j) = f(x(t_j),u(t_j))$, rather than through a numerical integrator. It is attractive for long, stiff trajectories and for problems with free final time, and is common in offline motion planning; the resulting nonlinear programme is larger and is usually solved by interior-point methods.

**Real-time iteration.** In MPC the problem is solved again every control period from a slightly shifted initial state. The real-time iteration scheme performs only one SQP iteration per period, warm-started from the shifted previous solution, and splits it into a preparation phase (linearisation and condensing, done before the new state arrives) and a feedback phase (solving the QP once $\hat x_0$ is measured). Because consecutive problems are close, a single Newton-type step tracks the optimum, and the feedback delay is the time of one QP solve. [Perceptive NMPC](papers/perceptive-nmpc.md) uses multiple shooting with SQP and real-time iteration.

**Relevance to learning.** An imagined rollout trained by BPTT is single shooting with a neural policy as the control parameterisation; SHAC's restart of each window from the end state of the previous one is closer to multiple shooting in spirit (short, local sensitivities stitched along a trajectory), but without the continuity constraints that make multiple shooting consistent ([m-short-horizon](#m-short-horizon)). Seeding imagined rollouts from replay-buffer states, as SuperMimic and TGI do, is likewise a way of initialising many short shooting intervals on the true state distribution.

**Intuition.** Single shooting asks the optimiser to see through the whole dynamics chain; multiple shooting and collocation give it the intermediate states as handles and let it repair the chain gradually.

**Used in:** [perceptive-nmpc](papers/perceptive-nmpc.md), [ilqr-online-to](papers/ilqr-online-to.md), [TAMOLS](papers/tamols.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [SHAC](papers/shac.md) (analogy), [TGI](papers/tgi.md) (analogy).

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

<a id="m-slip"></a>
### Spring-loaded inverted pendulum and Raibert foot placement

**SLIP.** The spring-loaded inverted pendulum models a running robot as a point mass $m$ on a massless leg of rest length $r_0$ and stiffness $k$. In stance, with the foot pinned at the origin and the mass at polar coordinates $(r,\vartheta)$ measured from vertical,

$$m\ddot r = k(r_0 - r) + mr\dot\vartheta^2 - mg\cos\vartheta, \qquad mr^2\ddot\vartheta + 2mr\dot r\dot\vartheta = mgr\sin\vartheta,$$

and in flight the mass follows a ballistic parabola. Touchdown occurs when the foot, placed at a commanded angle of attack $\vartheta_{\text{td}}$, reaches the ground; lift-off when the spring returns to $r_0$. The model is conservative, so energy is fixed and the stance phase maps apex height and forward speed to the next apex. Unlike the LIPM ([m-lipm-zmp](#m-lipm-zmp)), whose height is held constant, SLIP captures the vertical bounce of running and hopping, and the stance dynamics are not integrable in closed form, so apex-return maps are computed numerically.

**Self-stability and the role of the angle of attack.** For fixed $(k, r_0, \vartheta_{\text{td}})$ the apex-to-apex map has fixed points (periodic running gaits), and for a band of angles of attack these fixed points are stable without any feedback: a perturbation in speed changes the stance trajectory so as to return the next apex towards the fixed point. The foot-placement angle is therefore the principal control input, which is the template behind both Raibert's controller and the capture point.

**Raibert foot placement.** [Raibert](papers/raibert.md) decomposed hopping control into three decoupled loops: hop height by the thrust delivered in stance, body attitude by hip torque in stance, and forward speed by foot placement in flight. The neutral point is the foothold for which stance is symmetric and speed is unchanged, approximately $x_{\text{neutral}} = \dot xT_s/2$ ahead of the hip for stance duration $T_s$. The placement law adds a correction proportional to the speed error,

$$x_f = \frac{\dot xT_s}{2} + k_{\dot x}(\dot x - \dot x_d),$$

so placing the foot ahead of the neutral point decelerates the body and placing it behind accelerates it. In LIPM terms this is a discrete-time version of stepping towards the capture point, since $\xi - x = \dot x/\omega$ is likewise proportional to velocity ([m-lipm-zmp](#m-lipm-zmp)). Quadruped trots and bounds are handled by the virtual-leg idea (legs moving in unison act as one leg), and the same heuristic survives as the footstep reference in convex MPC and as the default foothold that learned planners correct.

**Relevance to learned locomotion.** Learned policies are not given these templates, yet their gaits often exhibit them (speed-proportional foot placement, compliant stance). On sparse terrain the template breaks down because the neutral point may lie over a gap, and foot placement must be traded against terrain support; this is the decision that foothold-prediction models and terrain-dependent rewards in [Watch Your Step](papers/watch-your-step.md) and [TGI](papers/tgi.md) push into the learned policy.

**Intuition.** Running is bouncing on a spring; balance and speed are controlled almost entirely by where the next foot goes.

**Used in:** [raibert](papers/raibert.md), [capture-point](papers/capture-point.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [deepgait](papers/deepgait.md), [watch-your-step](papers/watch-your-step.md).

<a id="m-centroidal-srbd"></a>
### Centroidal and single-rigid-body dynamics, convex-MPC linearisation

**Centroidal dynamics.** The first six rows of the floating-base equation of motion ([m-rigid-contact](#m-rigid-contact)) state that the rate of change of the robot's centroidal momentum equals the external wrench. With centre of mass $c$, total mass $m$, linear momentum $m\dot c$ and centroidal angular momentum $k$,

$$m\ddot c = \sum_i f_i + mg, \qquad \dot k = \sum_i (p_i - c)\times f_i,$$

for point contacts at $p_i$ with forces $f_i$. The relation $h_G = A_G(q)\dot q$ maps joint velocities to centroidal momentum through the centroidal momentum matrix. These equations are exact for any robot, but the angular momentum depends on the full configuration and velocity; the joints influence the base only through contact forces and through internal momentum redistribution.

**Single-rigid-body model (SRBD).** If the legs are light relative to the body, $k\approx I_w\omega$ with a world-frame inertia $I_w = RI_bR^\top$, and the robot becomes one rigid body actuated by contact forces:

$$\dot p = v,\quad \dot v = \tfrac1m\sum_i f_i + g,\quad \dot R = R[\omega_b]_\times,\quad I_w\dot\omega + \omega\times I_w\omega = \sum_i (p_i - p)\times f_i.$$

The model is nonlinear through the rotation, the cross products and the $\omega\times I\omega$ term, and has the contact forces, not joint torques, as inputs.

**Convex-MPC linearisation.** [Convex MPC on Cheetah 3](papers/convex-mpc-cheetah.md) makes the SRBD problem a quadratic programme by three approximations: small roll and pitch, so that the Euler-angle rates depend on yaw only, $\dot\Theta\approx R_z(\psi)^\top\omega$; neglect of $\omega\times I\omega$; and foothold positions $p_i$ and the body trajectory taken from a reference over the horizon, so that $(p_i - p)\times f_i$ is linear in $f_i$. The result is a linear time-varying system $x_{k+1} = A_kx_k + B_kf_k + g$ in the state $x = (\Theta, p, \omega, v)$, a quadratic tracking cost and linear constraints: the linearised friction pyramid $\lvert f_{x}\rvert, \lvert f_{y}\rvert\le\mu f_z$, force bounds $f_{\min}\le f_z\le f_{\max}$, and $f_i = 0$ for swing legs given a gait schedule. The QP is solved every control period; joint torques follow from $\tau = J^\top f$ for stance legs, and swing legs track Raibert-style footholds ([m-slip](#m-slip)).

**Uses beyond MPC.** The same model serves as a differentiable surrogate: [Song et al.](papers/song-diffsim-quadruped.md) backpropagate through SRBD dynamics with torques mapped to foot forces by $f = (J^\top)^{-1}\tau$, and [TAMOLS](papers/tamols.md) builds its stability conditions on it. TGI's world model can be read as a learned generalisation of the centroidal equations: its exact contact Jacobian $J^\top W$ supplies the base force and moment $\sum_b F_b$ and $\sum_b(x_b - p)\times F_b + M_b$, while the learned matrix $B$ replaces the analytic inverse inertia of the SRBD ([m-model-learning](#m-model-learning)).

**Intuition.** Whatever the legs do, the body moves only by the contact wrench; treat the body as one rigid object and the problem of where to push becomes convex.

**Used in:** [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [song-diffsim-quadruped](papers/song-diffsim-quadruped.md), [TAMOLS](papers/tamols.md), [perceptive-nmpc](papers/perceptive-nmpc.md), [kajita-zmp-preview](papers/kajita-zmp-preview.md), [capture-point](papers/capture-point.md), [DTC](papers/dtc.md), [TGI](papers/tgi.md).

<a id="m-wbc"></a>
### Hierarchical task-space whole-body control

**Tasks.** Whole-body control converts desired task-space motions (base orientation, CoM acceleration, swing-foot trajectories, stance-foot no-slip) into joint torques and contact forces consistent with the full rigid-body dynamics ([m-rigid-contact](#m-rigid-contact)). A task $i$ is a linear equation in the decision variables $z = (\dot v, \lambda, \tau)$, typically an acceleration task $J_i\dot v + \dot J_iv = \ddot x_i^{\text{des}}$ with $\ddot x_i^{\text{des}} = \ddot x_i^{\text{ref}} + K_d(\dot x_i^{\text{ref}} - \dot x_i) + K_p(x_i^{\text{ref}} - x_i)$.

**Weighted vs strict hierarchies.** A weighted QP minimises $\sum_i w_i\lVert A_iz - b_i\rVert^2$ subject to the equations of motion and inequality constraints (friction cone, torque limits, unilateral contact). A strict hierarchy instead solves a sequence of QPs: task 1 is optimised first; task 2 is optimised within the set of optimal solutions of task 1; and so on. With equality tasks only, this is achieved by null-space projection: if $z_1$ solves task 1 and $N_1 = I - A_1^+A_1$ projects onto its null space, then $z = z_1 + N_1\zeta$ leaves task 1 unaffected for any $\zeta$, and the next task is solved for $\zeta$. With inequalities, each level adds the previous levels' optimal residuals as constraints (hierarchical QP). Typical orderings put the equations of motion and contact constraints first, then base and CoM motion, then swing feet, then posture regularisation.

**Floating base and contacts.** Stance contacts appear as constraints $J_c\dot v + \dot J_cv = 0$ (no acceleration of the contact point), and contact forces must lie in the friction cone; only the bottom $n$ rows of the dynamics involve $\tau$, so a common simplification solves for $(\dot v, \lambda)$ from the first six (unactuated) rows and the tasks, then recovers $\tau$ from the remaining rows.

**Place in the lineage.** WBC is the lower layer of model-based legged control stacks: an MPC on a reduced model ([m-centroidal-srbd](#m-centroidal-srbd)) or a trajectory optimiser such as [TAMOLS](papers/tamols.md) plans base motion and footholds, and WBC tracks the plan at kilohertz rates. Hybrid methods such as [DTC](papers/dtc.md) replace the WBC by a learned policy that tracks optimised footholds. In learned humanoid motion tracking, "whole-body control" denotes the task rather than the method: [ExBody](papers/exbody.md), [ExBody2](papers/exbody2.md) and [SONIC](papers/sonic.md) learn a single policy that tracks all body links, and the prioritisation that a hierarchy would enforce is expressed instead by reward weights ([m-tracking-reward](#m-tracking-reward)). TGI has no explicit task hierarchy; priorities between base tracking, footholds and regularisation are set by reward weights in the imagined return.

**Intuition.** Ask for what matters most first, and let less important goals use only the freedom that remains.

**Used in:** [TAMOLS](papers/tamols.md), [DTC](papers/dtc.md), [perceptive-nmpc](papers/perceptive-nmpc.md), [convex-mpc-cheetah](papers/convex-mpc-cheetah.md), [ExBody](papers/exbody.md), [ExBody2](papers/exbody2.md), [SONIC](papers/sonic.md).

<a id="m-terrain-lookup"></a>
### Heightmaps, bilinear interpolation, differentiable lookups, signed distance fields

**Heightmap.** A 2.5-D terrain representation stores heights $\mathsf H_{ij} = \mathcal H(x_0 + i\Delta, y_0 + j\Delta)$ on a grid of resolution $\Delta$ (typically 0.04–0.1 m), in the world frame or, as an elevation map, in a robot-centric frame. It cannot represent overhangs; it is the standard exteroceptive input for perceptive locomotion. A policy typically observes heights sampled at a fixed pattern of points around the base, or, in [watch-your-step](papers/watch-your-step.md), $k\times k$ patches around predicted footholds, located by rounding to the nearest cell.

**Bilinear interpolation.** For a query $(x, y)$ let $u = (x - x_0)/\Delta$, $v = (y - y_0)/\Delta$, $i = \lfloor u\rfloor$, $j = \lfloor v\rfloor$, $\alpha = u - i$, $\beta = v - j$. Then

$$\mathcal H(x,y) \approx (1-\alpha)(1-\beta)\mathsf H_{ij} + \alpha(1-\beta)\mathsf H_{i+1,j} + (1-\alpha)\beta\,\mathsf H_{i,j+1} + \alpha\beta\,\mathsf H_{i+1,j+1},$$

continuous across cell boundaries. Its gradient within a cell is

$$\frac{\partial\mathcal H}{\partial x} = \frac{1}{\Delta}\left[(1-\beta)(\mathsf H_{i+1,j} - \mathsf H_{ij}) + \beta(\mathsf H_{i+1,j+1} - \mathsf H_{i,j+1})\right],\qquad \frac{\partial\mathcal H}{\partial y} = \frac{1}{\Delta}\left[(1-\alpha)(\mathsf H_{i,j+1} - \mathsf H_{ij}) + \alpha(\mathsf H_{i+1,j+1} - \mathsf H_{i+1,j})\right].$$

Three properties matter for differentiable use. The gradient is discontinuous across cell boundaries (bilinear interpolation is $C^0$, not $C^1$). At a step of height $\Delta h$ (a stair riser), the gradient is $\Delta h/\Delta$ inside the one cell that straddles the edge and zero elsewhere, an artificial ramp whose slope depends on grid resolution, not on the terrain; for $\Delta h = 0.15$ m and $\Delta = 0.05$ m it is 3, a 72° slope. And nearest-neighbour lookup has zero gradient almost everywhere. Bicubic interpolation or pre-blurring the map yields $C^1$ surfaces at the price of rounding edges, i.e. bias in where the robot believes support ends.

**Differentiable terrain terms in imagination (TGI).** In TGI the terrain is not learned; it is read from the simulator. For the G1, the training mesh is ray-cast once on the GPU into a 4 cm heightmap, queried bilinearly and differentiably in $(x,y)$. Quantities computed analytically from the predicted state and this map, rather than predicted by the network, are: the world model's terrain input (pelvis height above terrain, a 7×5 base scan, and 5×3 per-sole relative heights); the contact geometry of each sole, where the gap $\phi_b = p_z - h_{i^\ast}$ is taken to the highest of the 15 sole points ($i^\ast$ selected without gradient, the selected height differentiated) and the normal is fixed to $(0,0,1)$ because heightmap surfaces face up; rewards; soft terminations; and the critic's scan, which is bilinear "so the value gradient reaches the base position". The policy therefore receives gradients through $\partial\mathcal H/\partial(p_x,p_y)$, which enter the guard derivative $D_xg$ of the saltation matrix ([m-contact-gradients](#m-contact-gradients)) and every terrain-dependent reward. Three derived fields supply gradients where the raw lookup's gradient is a resolution artefact:

- The *unsupported sole share* $u_b = \frac1{15}\sum_i\sigma\big((\lvert\tilde h_i\rvert - 0.03)/0.01\big)$, with $\tilde h_i$ the height of sole point $i$ relative to the sole centre, uses nearest-node heights in the forward pass and bilinear heights in the backward pass ([m-straight-through](#m-straight-through)), because bilinear values smear stone edges over a cell and would make a foot centred on a 20 cm stone read as unsupported. This is the quantity that gates TGI's return ([m-stopping-time-truncation](#m-stopping-time-truncation)).
- The *support field* is a binary map (1 on cells within 25 cm of the local maximum in a 0.5 m window, so stair treads count and gaps between stones do not; lava floors excluded) blurred by a separable Gaussian with σ = 5 cm and queried bilinearly; its gradient points towards the middle of supports over a band of a few σ, and $1 - \bar S_b$ under the sole enters the foothold reward.
- The *edge field* is a binary indicator of a height range above 3 cm within a 4 cm radius, queried bilinearly but *not* blurred; its gradient is non-zero only in the one-cell band at the indicator's border and pushes stance soles off step nosings and stone rims.

The policy's own exteroception is deliberately non-differentiable: the foothold-gated scan uses rounded indices and nearest-node heights and the foothold predictor runs under `no_grad`, so terrain enters the policy gradient only through the model, rewards, terminations and critic. For ANYmal-D, each heightmap cell is treated as a column from its top downward, and a signed distance to the nearest column is computed for collision spheres on knees, shanks, thighs and belly: the nearest column in a 7×7 window is selected without gradient and the distance to it is re-evaluated with gradient (Euclidean above the column, lateral beside it, negative inside), which gives the model clearances and normals for whole-body contact. The design principle is that the forward pass is exact and the backward pass uses smooth surrogates whose bandwidth (σ = 5 cm, 1 cm sigmoid temperatures) determines which footholds are reachable by gradient.

**Surface normal and slope.** The upward normal of the height surface is $n = (-\partial_x\mathcal H, -\partial_y\mathcal H, 1)/\sqrt{1 + \lVert\nabla\mathcal H\rVert^2}$, needed to decompose contact forces into normal and tangential parts for the friction cone ([m-rigid-contact](#m-rigid-contact)).

**Signed distance fields.** A signed distance field $\phi:\mathbb R^3\to\mathbb R$ gives the distance to the nearest surface, positive outside and negative inside, and satisfies the eikonal equation $\lVert\nabla\phi\rVert = 1$ almost everywhere, with $\nabla\phi$ the outward normal at the surface. The vertical gap $p_z - \mathcal H$ is not a distance: for a locally planar surface it relates to the true distance by $\phi\approx(p_z - \mathcal H)/\sqrt{1 + \lVert\nabla\mathcal H\rVert^2}$, so on steep slopes and near edges the vertical gap overstates clearance and has the wrong gradient direction. SDFs represent full 3-D geometry (overhangs, walls), give $C^1$ fields away from medial axes, and are used for collision costs in trajectory optimisation; their cost is storage and the need to compute or learn them from sensor data.

**Intuition.** A heightmap lookup is cheap and differentiable almost everywhere, but its derivatives inherit the grid: they are informative on smooth slopes and fictitious at edges. Using them for gradients requires deciding where they can be believed.

**Used in:** [elevation-mapping-gpu](papers/elevation-mapping-gpu.md), [TAMOLS](papers/tamols.md), [lee-challenging-terrain](papers/lee-challenging-terrain.md), [miki-wild](papers/miki-wild.md), [legged-gym](papers/legged-gym.md), [anymal-parkour](papers/anymal-parkour.md), [extreme-parkour](papers/extreme-parkour.md), [humanoid-parkour](papers/humanoid-parkour.md), [beamdojo](papers/beamdojo.md), [attention-map-encoding](papers/attention-map-encoding.md), [watch-your-step](papers/watch-your-step.md), [deepgait](papers/deepgait.md), [DTC](papers/dtc.md), [TGI](papers/tgi.md).

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

<a id="m-retargeting"></a>
### Kinematic retargeting: keypoint IK, SMPL shape fitting, scaling

**Problem.** A reference motion recorded on a human or animal (motion capture, SMPL fits to video) must be expressed in the robot's configuration space before it can be tracked. Morphologies differ in limb lengths, joint axes and limits, and number of degrees of freedom, so retargeting is an optimisation, and its errors (foot sliding, penetration, self-collision, infeasible contacts) become tracking targets that the policy cannot meet.

**Keypoint inverse kinematics.** Choose corresponding keypoints $j\in\mathcal K$ on source and robot (feet, hands, hips, head) and solve per frame, or for a whole clip,

$$q_t^\ast = \arg\min_{q\in\mathcal Q}\sum_{j\in\mathcal K}w_j\lVert \mathrm{FK}_j(q) - \hat p_{j,t}\rVert^2 + \sum_{j}w^R_j\,d_{SO(3)}\big(R_j(q),\hat R_{j,t}\big)^2 + \eta\lVert q - q_{t-1}^\ast\rVert^2,$$

subject to joint limits, with orientation errors measured on $SO(3)$ ([m-rotations](#m-rotations)) and a temporal smoothness term. [Peng et al.](papers/peng-animals.md) solve such a problem with feet and hip keypoints for dog-to-quadruped transfer; differential IK linearises it, $\min_{\dot q}\lVert e(q) + J(q)\dot q\rVert_W^2$, and integrates, as in [GMR](papers/gmr.md).

**Shape fitting and scaling.** Target positions must first be made reachable. [H2O](papers/h2o.md) fits an SMPL body shape $\beta'$ whose rest pose matches the robot skeleton over 12 corresponding joints, re-poses each sequence with $\beta'$, and then optimises robot joint angles to match the joint positions; fitting the shape first removes the in-toed artefact caused by the robot's wider stance. GMR instead scales source positions relative to the root, $p^{\text{target}}_b = \frac{h}{h_{\text{ref}}}\big[s_b(p^{\text{src}}_b - p^{\text{src}}_{\text{root}}) + s_{\text{root}}\,p^{\text{src}}_{\text{root}}\big]$, with per-body factors $s_b$ and a uniform root factor that its authors identify as essential against foot sliding, followed by IK on orientations and end-effector positions and a final height correction.

**Feasibility filtering.** Kinematic retargeting ignores dynamics, so some clips remain physically infeasible. "Sim-to-data" filtering trains a privileged tracker in simulation and discards clips it cannot follow ([H2O](papers/h2o.md), [ASAP](papers/asap.md)); [ExBody2](papers/exbody2.md) automates curation by the tracking error of an initial policy. Learned alternatives fold retargeting into the policy's latent space, as in [SONIC](papers/sonic.md)'s human encoder.

**Intuition.** Retargeting decides what "imitating the human" means for a body that is not human; errors here are not noise for the tracker to average but targets it is asked to reproduce.

**Used in:** [peng-animals](papers/peng-animals.md), [H2O](papers/h2o.md), [OmniH2O](papers/omnih2o.md), [ExBody](papers/exbody.md), [ExBody2](papers/exbody2.md), [ASAP](papers/asap.md), [GMR](papers/gmr.md), [BeyondMimic](papers/beyondmimic.md), [SONIC](papers/sonic.md), [SuperMimic](papers/supermimic.md).

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

**IRL and occupancy matching are dual (GAIL's derivation).** Maximum-causal-entropy IRL with a convex cost regulariser $\psi$ seeks $\tilde c = \arg\max_c\,-\psi(c) + \big(\min_\pi -\mathcal H(\pi) + \mathbb E_{\rho^\pi}[c]\big) - \mathbb E_{\rho^E}[c]$, after which RL finds the policy for $\tilde c$. GAIL shows that this two-stage procedure returns

$$\mathrm{RL}\circ\mathrm{IRL}_\psi(\pi_E) = \arg\min_\pi\ -\mathcal H(\pi) + \psi^\ast\big(\rho^\pi - \rho^E\big), \qquad \psi^\ast(x) = \sup_c\ x^\top c - \psi(c),$$

by writing the problem as a saddle point over $(\pi, c)$, using that occupancy measures are in one-to-one correspondence with policies and that the Lagrangian is convex in $\rho$ and concave in $c$, so min and max may be exchanged. The convex conjugate $\psi^\ast$ is therefore a penalty on the occupancy difference. Two choices make this concrete. With $\psi\equiv 0$ (no regularisation, all costs allowed), $\psi^\ast$ is the indicator of $\rho^\pi = \rho^E$: exact occupancy matching, which is impractical with finite data. With GAIL's regulariser $\psi_{\mathrm{GA}}$, $\psi^\ast_{\mathrm{GA}}(\rho^\pi - \rho^E) = \sup_D\ \mathbb E_{\rho^\pi}[\log D(s,a)] + \mathbb E_{\rho^E}[\log(1 - D(s,a))]$, which equals the Jensen–Shannon divergence up to constants and yields the adversarial objective above. Adversarial imitation is thus inverse RL with the cost restricted by $\psi$, solved in its dual.

**State-only discriminators.** When the demonstrator's actions are unavailable or meaningless for the robot (motion capture, video, a different morphology), the discriminator is applied to transitions, $D(s, s')$, which matches the state-transition occupancy $\rho^\pi(s,s') = \sum_a\rho^\pi(s,a)p(s'\mid s,a)$ instead of $\rho^\pi(s,a)$. Under deterministic, injective-in-action dynamics the two are equivalent; otherwise state-only matching is weaker, admitting any policy that produces the same transitions. AMP, [ASE](papers/ase.md) and [PULSE](papers/pulse.md)'s adversarial terms use features of $(s,s')$, often a hand-designed observation map $\Phi(s)$ of body positions and velocities so that the discriminator ignores irrelevant state components.

**Gradient penalty.** Discriminators trained to separate two distributions with disjoint supports become sharp, their gradients vanish or explode, and the policy's reward becomes uninformative. Penalising $\lVert\nabla_xD(x)\rVert^2$ on data samples (the zero-centred "R1" penalty used by AMP) or on interpolates (WGAN-GP's $(\lVert\nabla D\rVert - 1)^2$) keeps the discriminator smooth. In AMP's form the penalty makes $D$ flat on the dataset manifold, so that small deviations of the policy's transitions from the data produce graded, not saturated, rewards; AMP reports it as essential for stable training.

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

**Learned and conditional priors.** With a fixed prior $p(z) = \mathcal N(0, I)$ the KL term pulls every posterior towards the same point regardless of context, which is wrong when the plausible next skills depend on the current state. A conditional prior $p_\vartheta(z\mid s)$ learned jointly makes the KL $D_{\mathrm{KL}}(q(z\mid s, x)\Vert p_\vartheta(z\mid s))$ measure only the information in $x$ (the future motion or goal) beyond what the state already predicts. Sampling from $p_\vartheta(z\mid s)$ then generates plausible behaviour without a goal, and a downstream task policy can act in $z$ with the prior as a behavioural regulariser. [ControlVAE](papers/controlvae.md), [PULSE](papers/pulse.md) and [MaskedMimic](papers/maskedmimic.md) all use state-conditional learned priors.

**Posterior as a residual on the prior.** ControlVAE and MaskedMimic parameterise the posterior relative to the prior: $\mu_q = \mu_p(s) + \Delta\mu(s, x)$ (MaskedMimic also with its own variance), so that the encoder outputs only a correction. For Gaussians with shared variance the KL then reduces to $\lVert\Delta\mu\rVert^2/(2\sigma^2)$, a penalty on the correction alone; the encoder no longer has to reproduce the prior's mean, the KL cannot be lowered by moving the prior towards an encoder that is itself wrong, and at test time dropping the residual (setting $\Delta\mu = 0$) recovers the prior's behaviour. The cost is that the prior must be good enough for small residuals to suffice. ControlVAE fixes $\sigma_q = \sigma_p = 0.3$, so its KL is exactly $\lVert\Delta\mu\rVert^2/(2\sigma_p^2)$; MaskedMimic reports that removing the residual prior drops success at sitting on held-out objects from 96.9 % to 21.1 %.

**Vector quantisation.** VQ-VAE replaces the continuous posterior by the nearest codebook vector, $z_q = e_{k^\ast}$, $k^\ast = \arg\min_k\lVert z_e - e_k\rVert$, passes gradients straight through ($\partial z_q/\partial z_e := I$, [m-straight-through](#m-straight-through)), and adds codebook and commitment losses $\lVert\mathrm{sg}(z_e) - e\rVert^2 + \beta\lVert z_e - \mathrm{sg}(e)\rVert^2$.

**Residual vector quantisation.** A single codebook of size $K$ represents at most $K$ distinct latents per code and loses detail. Residual VQ quantises in $L$ stages: $r_0 = z_e$, $e^{(l)} = \mathrm{VQ}_l(r_{l-1})$, $r_l = r_{l-1} - e^{(l)}$, and $z_q = \sum_{l=1}^L e^{(l)}$, so that each layer encodes the error left by the previous ones and the representable set grows as $K^L$ with only $LK$ stored vectors. Coarse layers capture the gist and later layers detail, which also gives a natural ordering for autoregressive generation over tokens. [MoConVQ](papers/moconvq.md) uses $L = 8$ layers of 512 entries per code and reports that residual layers recover detail lost by a single VQ layer and that the discrete latent is markedly more robust to input noise than a continuous VAE latent.

**Intuition.** Latent skill spaces separate how to move from what to do, so that downstream tasks search a low-dimensional, behaviourally meaningful space instead of raw actions.

**Used in:** [World Models](papers/world-models.md), [ASE](papers/ase.md), [ControlVAE](papers/controlvae.md), [PULSE](papers/pulse.md), [MaskedMimic](papers/maskedmimic.md), [MoConVQ](papers/moconvq.md), [npmp](papers/npmp.md), [SONIC](papers/sonic.md), [BeyondMimic](papers/beyondmimic.md), [SuperMimic](papers/supermimic.md) (goal encoders with a shared decoder).

<a id="m-skill-composition"></a>
### Multiplicative composition of primitives (MCP), progressive networks

**Additive vs multiplicative mixtures.** Given $K$ primitive policies $\pi_i(a\mid s)$ and state-dependent gating weights $w_i(s)\ge 0$, a mixture of experts samples one primitive, $\pi(a\mid s) = \sum_iw_i(s)\pi_i(a\mid s)$, so at each step only one primitive acts. Multiplicative composition (MCP) instead forms a product of experts,

$$\pi(a\mid s) = \frac1{Z(s)}\prod_{i=1}^K\pi_i(a\mid s)^{w_i(s)},$$

so that all primitives act simultaneously and each can veto actions it finds implausible. For Gaussian primitives with diagonal covariances $\mathcal N(\mu_i, \sigma_i^2)$ the product is Gaussian in closed form, per action dimension $j$:

$$\sigma_j^{-2} = \sum_i\frac{w_i}{\sigma_{i,j}^2}, \qquad \mu_j = \sigma_j^2\sum_i\frac{w_i}{\sigma_{i,j}^2}\mu_{i,j},$$

a precision-weighted average in which a confident primitive (small $\sigma_{i,j}$) dominates the dimensions it cares about. Composition can therefore specialise primitives to subsets of the body and blend them, which an additive mixture cannot.

**Progressive networks and capacity growth.** When a single network trained on a growing or hard-mined dataset forgets earlier skills, progressive networks freeze the trained column and add a new column with lateral connections from the frozen activations, so new capacity reuses old features without overwriting them. [PHC](papers/phc.md) combines both ideas: primitives $P^{(1)}, P^{(2)},\dots$ are trained in sequence on successively harder subsets of motion clips (each frozen when its success plateaus, the next initialised from it or connected laterally), a fail-state recovery primitive is added, and a composer outputs non-negative weights that combine all frozen Gaussian primitives multiplicatively. The design addresses catastrophic forgetting under hard-negative mining; the price is inference cost linear in $K$ and a composer that must learn when each primitive applies.

**Intuition.** Let every specialist vote on every action dimension, weighted by its confidence, and add specialists rather than retraining one generalist.

**Used in:** [PHC](papers/phc.md), [PULSE](papers/pulse.md) (distils the composed tracker into a latent space), [ASE](papers/ase.md) (contrast: single latent-conditioned policy).

<a id="m-diffusion-guidance"></a>
### DDPM, latent diffusion, classifier and cost guidance

**Forward and reverse processes.** A denoising diffusion probabilistic model (DDPM) defines a fixed forward chain that gradually adds Gaussian noise to data $x_0$, $q(x_k\mid x_{k-1}) = \mathcal N(\sqrt{1-\beta_k}\,x_{k-1}, \beta_kI)$, with closed-form marginal $x_k = \sqrt{\bar\alpha_k}\,x_0 + \sqrt{1-\bar\alpha_k}\,\epsilon$, $\bar\alpha_k = \prod_{i\le k}(1-\beta_i)$, $\epsilon\sim\mathcal N(0, I)$. A network $\epsilon_\theta(x_k, k)$ is trained to predict the noise,

$$\mathcal L(\theta) = \mathbb E_{x_0, k, \epsilon}\left[\lVert\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_k}x_0 + \sqrt{1-\bar\alpha_k}\epsilon,\ k)\rVert^2\right],$$

a reweighted form of the ELBO ([m-latent-skills](#m-latent-skills)). Sampling runs the learned reverse chain from $x_K\sim\mathcal N(0, I)$, $x_{k-1} = \frac{1}{\sqrt{1-\beta_k}}\big(x_k - \frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta(x_k,k)\big) + \sigma_kz$. The noise predictor is, up to scale, the score of the noised marginal: $\nabla_{x_k}\log q(x_k)\approx -\epsilon_\theta(x_k,k)/\sqrt{1-\bar\alpha_k}$.

**Latent and trajectory diffusion.** Diffusing in a learned latent space (latent diffusion) reduces dimension and lets the decoder enforce structure. For control, $x_0$ is a trajectory window of states, actions or latents, and the model learns the distribution of plausible futures given the recent past; executing only the first action and re-sampling each step gives a receding-horizon controller ([m-mpc](#m-mpc)). Per-token noise levels allow conditioning on observed (noise-free) past tokens while denoising future ones.

**Classifier and cost guidance.** To sample from $p(\tau\mid y)\propto p(\tau)\,p(y\mid\tau)$ without retraining, Bayes' rule adds the gradient of the log-likelihood to the score, $\nabla\log p(\tau_k\mid y) = \nabla\log p(\tau_k) + \nabla\log p(y\mid\tau_k)$, so each denoising step is shifted by $s\,\Sigma_k\nabla_{\tau_k}\log p(y\mid\tau_k)$ with a guidance scale $s$. Choosing $p(y\mid\tau)\propto\exp(-G(\tau))$ for a differentiable cost $G$ turns guidance into gradient descent on $G$ interleaved with denoising: the prior keeps samples on the data manifold while the cost steers them. [BeyondMimic](papers/beyondmimic.md) trains a transformer DDPM on state–latent windows of a VAE distilled from its trackers and guides it at test time with costs for joystick velocity, waypoints, keyframe inpainting and obstacle avoidance. Guidance applies the cost gradient to noised samples, where $G$ is evaluated off-manifold, and strong guidance trades fidelity to the data for cost reduction; classifier-free guidance avoids an external gradient by mixing conditional and unconditional score estimates.

**Relation to analytic policy gradients.** Cost guidance and BPTT through a model both use $\nabla G$ with respect to a trajectory; guidance applies it at inference to individual samples under a learned prior, whereas TGI and SuperMimic amortise it into policy parameters during training.

**Intuition.** Learn what plausible motion looks like by learning to remove noise from it; steer generation by nudging each denoising step down the gradient of the cost you care about.

**Used in:** [BeyondMimic](papers/beyondmimic.md), [ExBody](papers/exbody.md) (motions generated by a text-to-motion diffusion model), [MaskedMimic](papers/maskedmimic.md) (contrast: conditional VAE).

<a id="m-distillation"></a>
### Behaviour cloning, DAgger, policy distillation

**Behaviour cloning.** Given expert state–action pairs, minimise a supervised loss $\mathbb E_{s\sim d^{\pi^\ast}}[\ell(\pi_\theta(s), \pi^\ast(s))]$. If the learner errs with probability at most $\epsilon$ on the expert's state distribution and a mistake can lead to states the expert never visits, the expected cost over horizon $T$ (costs in $[0,1]$, $\epsilon$ an upper bound on the 0-1 loss) satisfies $J(\pi_\theta)\le J(\pi^\ast) + T^2\epsilon$ (Ross and Bagnell 2010, restated as Theorem 2.1 of the DAgger paper), and the bound is tight: an early mistake moves the learner off-distribution, where further mistakes are uncontrolled. This quadratic compounding is the imitation analogue of model compounding error ([m-model-learning](#m-model-learning)).

**DAgger.** Ross, Gordon and Bagnell ("A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011, arXiv 1011.0686) iterate: roll out the mixture $\pi_i = \beta_i\pi^\ast + (1-\beta_i)\hat\pi_i$ for $T$ steps, label every visited state with the expert's action, aggregate all datasets, and train $\hat\pi_{i+1}$ on the aggregate (with $\beta_1 = 1$; the parameter-free choice $\beta_i = \mathbb 1[i = 1]$ often works best). Training on the aggregate is Follow-The-Leader in an online learning game whose loss at round $i$ is the surrogate imitation loss $\ell$ under $d_{\pi_i}$, and their analysis uses only the no-regret property of that learner. In the standard form of their main results (infinite samples per iteration; $\ell$ strongly convex, bounded, and an upper bound on the 0-1 loss with respect to $\pi^\ast$; $\beta_i\le(1-\alpha)^{i-1}$ for a constant $\alpha$ independent of $T$), with

$$\epsilon_N = \min_{\pi\in\Pi}\frac1N\sum_{i=1}^N\mathbb E_{s\sim d_{\pi_i}}[\ell(s,\pi)]$$

the loss of the best policy in hindsight on the aggregated distributions, they show:

$$\text{if } N = \tilde O(T):\ \exists\,\hat\pi\in\hat\pi_{1:N}\ \text{with}\ \mathbb E_{s\sim d_{\hat\pi}}[\ell(s,\hat\pi)]\le\epsilon_N + O(1/T);$$

$$\text{if } N = \tilde O(uT):\ \exists\,\hat\pi\in\hat\pi_{1:N}\ \text{with}\ J(\hat\pi)\le J(\pi^\ast) + uT\epsilon_N + O(1),$$

where task costs lie in $[0,1]$ and $u$ bounds the increase in the expert's cost-to-go caused by one deviating action, $Q^{\pi^\ast}_{T-t+1}(s,a) - Q^{\pi^\ast}_{T-t+1}(s,\pi^\ast)\le u$ for all $a$, $t$ and reachable $s$ (their Theorem 2.2, due to Ross and Bagnell 2010, converts the first statement into the second). The general version (their Theorem 4.1) reads $\mathbb E_{s\sim d_{\hat\pi}}[\ell(s,\hat\pi)]\le\epsilon_N + \gamma_N + \frac{2\ell_{\max}}{N}\big[n_\beta + T\sum_{i=n_\beta+1}^N\beta_i\big]$ with $\gamma_N$ the average regret and $n_\beta$ the last iteration with $\beta_i > 1/T$; finite-sample versions add an Azuma–Hoeffding term. The dependence on $T$ is linear, against the tight $T^2\epsilon$ of behaviour cloning, because the loss is measured on the learner's own state distribution. Two caveats: $\epsilon_N$ is a loss on the learner-induced distributions and can exceed the behaviour-cloning $\epsilon$ on the expert's distribution, and $u$ can be of order $T$ when single mistakes are unrecoverable, in which case the bound is again quadratic.

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
