# GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation

| | |
|---|---|
| Authors | John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel |
| Venue / year | ICLR 2016 (arXiv v6, 2018) |
| Link | https://arxiv.org/abs/1506.02438 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Policy-gradient variance grows with the horizon because each action's score is weighted by rewards caused by other actions. Actor–critic methods reduce variance by substituting a learned value function, at the cost of bias that, unlike variance, does not vanish with more samples. GAE frames the choice of the weight $\Psi_t$ in $\mathbb{E}[\sum_t\Psi_t\nabla\log\pi(a_t\mid s_t)]$ as a bias–variance problem and proposes a one-parameter family, controlled by $\lambda$, that interpolates between the one-step TD residual and the Monte Carlo return. It is the advantage estimator used by [PPO](ppo.md) and by essentially every legged-locomotion PPO pipeline in [g5](../groups/g5-learned-locomotion.md), and its λ-return form is the critic target of [SuperMimic](supermimic.md) and [TGI](tgi.md).

## 2. Method

The paper treats $\gamma$ as a variance-reduction parameter of an undiscounted problem and defines an estimator $\hat A_t$ as *γ-just* if it yields an unbiased estimate of the discounted gradient $g^\gamma$; any $\hat A_t = Q_t - b_t$ with $Q_t$ unbiased for $Q^{\pi,\gamma}$ and $b_t$ independent of $a_t$ qualifies (Proposition 1). With the TD residual $\delta^V_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, the $k$-step estimator is the telescoping sum $\hat A^{(k)}_t = \sum_{l<k}\gamma^l\delta^V_{t+l}$, and GAE is their exponential average

$$\hat A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta^V_{t+l}.$$

$\lambda = 1$ gives the Monte Carlo return minus a baseline, γ-just for any $V$ but high-variance; $\lambda = 0$ gives $\delta_t$, γ-just only if $V = V^{\pi,\gamma}$. The paper stresses that $\gamma < 1$ biases the gradient regardless of $V$, whereas $\lambda < 1$ biases it only through value error, which explains why good $\lambda$ is lower than good $\gamma$. Section 4 reinterprets GAE as reward shaping with potential $\Phi = V$ followed by a steeper discount $\gamma\lambda$, and introduces the *response function* $\chi(l; s_t,a_t) = \mathbb{E}[r_{t+l}\mid s_t,a_t] - \mathbb{E}[r_{t+l}\mid s_t]$ to quantify how far credit must propagate. The value function is fitted by a trust-region regression (average KL between Gaussian predictive distributions $\le\epsilon$, solved by conjugate gradient with a Gauss–Newton matrix), and the policy by [TRPO](trpo.md). The policy step uses $V_{\phi_i}$ before it is refitted, to avoid driving advantages to zero by overfitting. See [GAE](../math-cookbook.md#m-advantage-gae) and [λ-returns](../math-cookbook.md#m-lambda-return).

## 3. Evidence

Policies and value functions were 100–50–25 tanh MLPs (over $10^4$ parameters) acting on raw kinematics and outputting joint torques, simulated in MuJoCo at 0.01 s. On cart-pole (21 seeds) the best settings were $\gamma\in[0.96,0.99]$, $\lambda\in[0.92,0.99]$. On 3D biped locomotion (33-D state, 10 actuators; 50,000 steps per batch; 9 seeds; about 2 h per run on 16 cores) the best results were at $\gamma\in[0.99,0.995]$, $\lambda\in[0.96,0.99]$, yielding a stable running gait after 1000 iterations; the authors compute that this corresponds to 5.8 days of real time. Quadruped locomotion (29-D state, 8 actuators) and biped standing-up used 200,000 steps per batch, 5 seeds and about 4 h on 32 cores; $\lambda = 0.96$ beat both $\lambda = 0$ and no value function for the quadruped, and for standing the value function always helped but $\lambda = 0.96$ and $1$ were similar. A "No VF" baseline using a time-dependent average return performed markedly worse in all plotted tasks.

## 4. Critical analysis

The bias analysis is informal: the paper shows *when* each estimator is γ-just but gives no bound on the bias of intermediate $\lambda$ as a function of value error, and the authors list the relation between value-fitting error and gradient error as open. The empirical support for the $\lambda$ range comes from a modest number of seeds (5–21) on four tasks, all with a single policy optimiser (TRPO). The robustness of $\lambda\approx0.95$ across later work is better evidence than the paper's own sweep. The 5.8-day real-time figure is an argument by arithmetic, not a demonstration; real-robot training at this sample count was not attempted. GAE also hard-codes an *on-policy* estimator: the δ terms assume the actions were drawn from the current policy, which is why off-policy and model-based methods use different targets. Finally, the discussion's remark that one-step critics ($\lambda = 0$) caused "excessive bias" while [DDPG](ddpg.md) made them work foreshadows the on-policy/off-policy split that the [g1 report](../groups/g1-rl-foundations.md) discusses.

## 5. Cross-references

- Builds on: [A3C](a3c.md)'s n-step advantages, TD(λ), and the [PG theorem](pg-theorem.md); uses [TRPO](trpo.md) for the policy step.
- Extended by: [PPO](ppo.md), which uses truncated GAE over fixed-length segments with $\lambda = 0.95$.
- Contrasts with: [TD3](td3.md) and [SAC](sac.md), whose one-step off-policy Q targets avoid importance corrections but incur the bias GAE warns about.
- Related in other groups: [Dreamer](dreamer.md) uses λ-returns as the target for its imagined-trajectory critic; [SHAC](shac.md) uses TD(λ) targets for its terminal critic.

## 6. Relevance to TGI

TGI's critic is fitted to exactly GAE's recursion: in `trainer.py` the targets are computed backward over the 32 imagined steps as $R_t = r_t + \gamma(1-d_t)\big((1-\lambda)V(s_{t+1}) + \lambda R_{t+1}\big)$ with $\gamma = 0.99$, $\lambda = 0.95$ (the PPO defaults), and the auxiliary likelihood-ratio term weights the score by the normalised advantage $R_t - V(s_t)$, which is the truncated GAE estimate. GAE's reward-shaping reading also clarifies what TGI's gradient cut does: at an imagined touchdown on unsupported terrain the pathwise return is replaced by $V(s_{t+1})$, i.e. the trace parameter is locally set to zero for the *policy* gradient, accepting critic bias in exchange for removing an unreliable Jacobian. See [TGI](tgi.md) and [contact gradients](../math-cookbook.md#m-contact-gradients).
