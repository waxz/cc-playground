# PG theorem: Policy gradient methods for reinforcement learning with function approximation

| | |
|---|---|
| Authors | Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour |
| Venue / year | NIPS 12 (1999; proceedings 2000) |
| Link | no-arxiv (NeurIPS proceedings) |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

By the late 1990s value-function methods with function approximation were known to diverge or chatter, because a small change in the value estimate can cause a discontinuous change in the greedy policy. Sutton et al. argued for representing the policy explicitly and following the gradient of performance, and asked what that gradient looks like when it must be estimated with an approximate value function. Their answer, the policy gradient theorem, turns [REINFORCE](reinforce.md) from a Monte Carlo recipe into a framework in which a critic can be inserted. Konda and Tsitsiklis's actor–critic analysis appeared concurrently. Every actor–critic method in this group, and the critic-bootstrapped objectives of [SHAC](shac.md) and [TGI](tgi.md), rest on this result.

## 2. Method

For either the average-reward or the start-state discounted objective $\rho(\theta)$, the theorem states

$$\frac{\partial \rho}{\partial \theta} = \sum_s d^{\pi}(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta}\, Q^{\pi}(s,a),$$

where $d^\pi$ is the (discounted) state distribution under $\pi$. The key property is that no derivative of $d^\pi$ appears, although $d^\pi$ depends on $\theta$; the gradient can therefore be estimated by sampling states from on-policy experience. Rewriting $\partial\pi/\partial\theta = \pi\,\nabla\log\pi$ recovers the score-function form $\mathbb{E}[\nabla_\theta\log\pi\,Q^\pi]$. The second result concerns *compatible function approximation*: if an approximator $f_w(s,a)$ satisfies $\nabla_w f_w = \nabla_\theta \log \pi_\theta$ and $w$ minimises the on-policy squared error to $Q^\pi$, then substituting $f_w$ for $Q^\pi$ leaves the gradient exact. Third, the paper proves that policy iteration with such approximators and suitable step sizes converges to a locally optimal policy. See [policy gradient theorem](../math-cookbook.md#m-pg-theorem) and [Bellman equations](../math-cookbook.md#m-bellman).

## 3. Evidence

The contribution is theoretical: the theorem, the compatibility condition and a convergence proof. There is no large-scale experiment; the paper's influence derives from the generality of the result.

## 4. Critical analysis

The theorem is exact but its practical premises are not met by deep RL. Compatible features are linear in $\nabla\log\pi$ and therefore cannot represent $Q^\pi$ in general, only its projection onto the advantage subspace; [GAE](gae.md)'s appendix notes that this theory gives no guidance on temporal credit assignment, which is where most variance originates. Modern implementations also routinely drop the discount from the state weighting (DDPG says so explicitly), so the gradient they follow is not the gradient of any fixed objective (a point later made formally by Nota and Thomas, 2020). The theorem's lasting value is structural: it licenses replacing returns by any unbiased estimate of $Q^\pi$ or $A^\pi$, and it is the stochastic counterpart of the [deterministic policy gradient](dpg.md), which shows the same "no derivative of the state distribution" property for deterministic actors. Compatible features also lead directly to the [natural policy gradient](natural-pg.md), whose direction equals the compatible weights $w$.

## 5. Cross-references

- Builds on: [REINFORCE](reinforce.md), whose estimator it generalises to critic-weighted forms.
- Extended by: [Natural PG](natural-pg.md), which shows compatible weights equal the natural gradient.
- Extended by: [A3C](a3c.md), [GAE](gae.md), [TRPO](trpo.md), [PPO](ppo.md), all of which estimate $A^\pi$ rather than $Q^\pi$.
- Contrasts with: [DPG](dpg.md), the deterministic analogue that differentiates $Q$ with respect to the action instead of weighting the score.

## 6. Relevance to TGI

TGI does not use the theorem's estimator as its main gradient; it differentiates the imagined return through the world model. The theorem nevertheless underwrites two of its components: the critic $V$ used to bootstrap the 32-step imagined return at the horizon and at cut touchdowns, and the auxiliary likelihood-ratio term whose weight is the λ-return advantage $\hat A_t = G^\lambda_t - V(s_t)$. Both assume that an approximate value function may stand in for Monte Carlo returns without destroying the gradient, the assumption this paper made respectable. See [TGI](tgi.md) and [short-horizon objective](../math-cookbook.md#m-short-horizon).
