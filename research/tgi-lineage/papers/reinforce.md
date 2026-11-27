# REINFORCE: Simple statistical gradient-following algorithms for connectionist reinforcement learning

| | |
|---|---|
| Authors | Ronald J. Williams |
| Venue / year | Machine Learning 8(3–4):229–256, 1992 |
| Link | no-arxiv (Springer, doi:10.1007/BF00992696) |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Williams asked how a network of stochastic units can adjust its weights to increase expected reinforcement when the only feedback is a scalar reward and no gradient of the environment is available. The answer, a family of algorithms he named REINFORCE ("REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility"), is the root of every likelihood-ratio policy-gradient method in this study. A3C, TRPO, GAE and PPO all compute the same estimator; they differ in what they multiply the score by and how far they let a single batch move the policy. The paper is also the canonical reference for the claim, repeated in SuperMimic and TGI, that model-free policy gradients treat the dynamics as a black box.

## 2. Method

For a stochastic unit with output distribution $g(y; w, x)$, the REINFORCE update is

$$\Delta w = \alpha\,(r - b)\,\frac{\partial \ln g(y; w, x)}{\partial w},$$

where $r$ is the received reinforcement, $b$ a reinforcement baseline that may depend on the input but not on the sampled output, and $\partial \ln g/\partial w$ the *characteristic eligibility*. Williams proves that the expected update lies in the direction of $\nabla_w \mathbb{E}[r]$, i.e. the update is an unbiased (up to positive scaling) estimate of the gradient of expected reward, for any baseline that does not depend on the action. An episodic variant accumulates eligibilities over a trajectory and multiplies by the episode return, which is the form now called REINFORCE in deep RL: $\hat g = \sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,(R - b)$. The paper derives eligibilities for Bernoulli units and for Gaussian units whose mean *and* standard deviation are learned, the parameterisation later used by TRPO, PPO and most locomotion policies. The background is collected in the cookbook under [score-function gradient and baselines](../math-cookbook.md#m-pg-theorem).

## 3. Evidence

The paper is primarily theoretical. Its contribution is the unbiasedness theorem and the general recipe for constructing eligibilities from any differentiable output distribution; the empirical content is limited to small illustrative tasks. Its evidential weight comes from three decades of subsequent use, not from its own experiments.

## 4. Critical analysis

The estimator requires only that $\log \pi$ be differentiable in $\theta$; it needs no derivative of reward or dynamics. This generality is also its weakness. The variance of $\hat g$ grows with the horizon (each action's score is multiplied by rewards it did not cause) and with action dimension (the score of a Gaussian policy is $(a-\mu)/\sigma^2$ per dimension, so noise from all dimensions enters every component). Williams' own baseline addresses the first problem only partially; the later sequence of fixes, reward-to-go, state-dependent value baselines, the advantage function and [GAE](gae.md), is the history of trying to reduce this variance without adding too much bias. The convergence statement concerns the expected update; it says nothing about step size, which became the concern of [natural gradients](natural-pg.md), [TRPO](trpo.md) and [PPO](ppo.md). The paper also introduced (with Peng, 1991) entropy regularisation, retained in [A3C](a3c.md) and [PPO](ppo.md) and elevated to an objective in [SAC](sac.md).

## 5. Cross-references

- Extended by: [Policy gradient theorem](pg-theorem.md), which replaces the Monte Carlo return with $Q^\pi$ and justifies learned critics.
- Extended by: [A3C](a3c.md), which pairs the estimator with a learned value baseline and n-step returns.
- Extended by: [GAE](gae.md), which gives a principled bias–variance family of advantage estimates to multiply the score by.
- Contrasts with: [SVG](svg.md) and [SHAC](shac.md), which differentiate the return with respect to actions through a model or simulator rather than weighting the score by sampled returns (see [reparameterisation](../math-cookbook.md#m-reparam)).

## 6. Relevance to TGI

TGI's policy loss contains a REINFORCE term explicitly. In `tgi/training/trainer.py` the imagined rollout injects reparameterised Gaussian noise ($\sigma = 0.2$) into the actions, computes $\log \pi$ of the noisy action under a fixed-variance Gaussian, and adds $-0.1\,\mathbb{E}[\log\pi(a_t\mid s_t)\,\hat A_t]$ to the pathwise loss (`--lr_coef 0.1`), with $\hat A_t$ the normalised λ-return advantage. The score-function term therefore survives in [TGI](tgi.md) as an auxiliary signal that does not depend on the world model's Jacobians, which matters precisely where those Jacobians are least trustworthy (see [contact gradients](../math-cookbook.md#m-contact-gradients)).
