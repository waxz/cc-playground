# TRPO: Trust Region Policy Optimization

| | |
|---|---|
| Authors | John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel |
| Venue / year | ICML 2015 (arXiv v5, 2017) |
| Link | https://arxiv.org/abs/1502.05477 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

In 2015 derivative-free methods (CEM, CMA-ES) were still competitive with gradient-based policy search on control problems, and learning locomotion typically required hand-designed low-dimensional policy classes. Schulman et al. diagnosed the problem with gradient methods as step-size selection: the [policy gradient](pg-theorem.md) is a local quantity, and a step that is too large changes the state distribution and destroys performance. TRPO derives a surrogate objective whose improvement provably lower-bounds the true improvement, then approximates the resulting algorithm with a KL-divergence trust region. It is the bridge between the [natural policy gradient](natural-pg.md) and [PPO](ppo.md), and it supplied the policy update for [GAE](gae.md)'s 3D locomotion results.

## 2. Method

Using the identity of Kakade and Langford, $\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tilde\pi}\big[\sum_t\gamma^tA_\pi(s_t,a_t)\big]$, the paper defines the local surrogate $L_\pi(\tilde\pi) = \eta(\pi) + \sum_s\rho_\pi(s)\sum_a\tilde\pi(a\mid s)A_\pi(s,a)$, which uses the *old* state distribution and matches $\eta$ to first order. Theorem 1 extends the conservative-policy-iteration bound from mixture policies to arbitrary stochastic policies:

$$\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D^{\max}_{\mathrm{KL}}(\pi,\tilde\pi),\qquad C = \frac{4\epsilon\gamma}{(1-\gamma)^2},\ \epsilon = \max_{s,a}|A_\pi(s,a)|.$$

Maximising the right-hand side each iteration is a minorisation–maximisation scheme with guaranteed monotone improvement (Algorithm 1). Because the theoretical $C$ yields tiny steps, the practical algorithm replaces the penalty with a constraint and the max-KL with the state-averaged KL:

$$\max_\theta\ \mathbb{E}_{s\sim\rho_{\theta_{\rm old}},a\sim q}\!\left[\frac{\pi_\theta(a\mid s)}{q(a\mid s)}Q_{\theta_{\rm old}}(s,a)\right]\quad\text{s.t.}\quad \mathbb{E}_s\big[D_{\mathrm{KL}}(\pi_{\theta_{\rm old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s))\big]\le\delta.$$

The step is computed by linearising the objective and quadratising the constraint, solving $Fx = g$ with conjugate gradients using Fisher-vector products (the Fisher matrix obtained analytically as the Hessian of the KL, not as the empirical covariance of scores), and a backtracking line search that enforces the constraint and improvement. Two sampling schemes are given: *single path* (ordinary trajectories) and *vine* (branch rollouts from a rollout set with common random numbers, which needs state resets and is thus simulator-only). Background: [trust region and Fisher information](../math-cookbook.md#m-trust-region).

## 3. Evidence

With $\delta = 0.01$ throughout, TRPO was evaluated on MuJoCo swimmer (10-D state, 364 policy parameters), hopper (12-D, 4806 parameters) and walker (20-D, 8206 parameters), with 50 k (swimmer) to 1 M (hopper, walker) simulation steps per iteration for 200 iterations, averaged over five runs. Single-path and vine TRPO solved all three, learning hopping and walking gaits from general-purpose networks and simple forward-velocity rewards; natural gradient with a tuned fixed step solved only cart-pole and swimmer; CEM and CMA failed on the larger problems; the max-KL variant (only tractable on cart-pole) learned somewhat more slowly than the average-KL version. Single-path wall time was 5 min (swimmer), 35 min (hopper) and 100 min (walker). On seven Atari games with a 33,500-parameter CNN, 500 iterations took about 30 h on a 16-core machine; results were mixed against DQN (e.g. single-path Pong 20.9 vs DQN 20.0; Breakout 10.8 vs 168.0), with one run per game and no error statistics.

## 4. Critical analysis

The monotonic-improvement theorem holds only for the penalised, max-KL, exact-advantage algorithm. The practical method breaks all three conditions (constraint instead of penalty, average KL, estimated advantages), so its observed near-monotone improvement is empirical, as the authors acknowledge. The comparison isolates the step-size rule well (fixed-coefficient NPG vs per-step KL constraint), which is the paper's most convincing result. Data efficiency is poor: walking required on the order of 200 M simulation steps. The second-order machinery couples the update to the network architecture: [PPO](ppo.md) lists incompatibility with dropout and with policy–value parameter sharing as TRPO's main practical limitations. Later work (Engstrom et al., 2020) showed that much of the stability attributed to trust regions in TRPO and PPO depends on implementation details such as observation and advantage normalisation.

## 5. Cross-references

- Builds on: [Natural PG](natural-pg.md) (same step direction) and [PG theorem](pg-theorem.md); conservative policy iteration (Kakade and Langford, 2002).
- Extended by: [GAE](gae.md), which supplies the advantage estimates and a trust-region value-function fit; [PPO](ppo.md), its first-order approximation.
- Contrasts with: [DDPG](ddpg.md), whose authors characterise TRPO as "significantly less data efficient" because it does not learn an action-value function.
- Related in other groups: [MBPO](mbpo.md) and model-ensemble TRPO reuse the trust-region update inside learned models.

## 6. Relevance to TGI

TGI does not use a trust region; its update is a pathwise gradient step with gradient-norm clipping. TRPO's lasting relevance is indirect: the KL-controlled step made zeroth-order training on legged robots reliable enough that PPO became the default against which [SuperMimic](supermimic.md) and [TGI](tgi.md) measure themselves. TRPO's conclusion anticipated model learning as the way to reduce sample complexity, the direction [g2](../groups/g2-world-models.md) and TGI take. Its advantage-based surrogate also reappears in the policy-improvement bounds of model-based methods ([Dyna/MBPO](../math-cookbook.md#m-dyna-mbpo)).
