# DPG: Deterministic Policy Gradient Algorithms

| | |
|---|---|
| Authors | David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller |
| Venue / year | ICML 2014 (PMLR 32) |
| Link | no-arxiv (proceedings.mlr.press/v32/silver14) |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) (depth-3 addition) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Stochastic policy gradients integrate over both states and actions; in high-dimensional continuous action spaces the action integral is the dominant source of variance. Silver et al. asked whether the policy gradient exists for a deterministic policy $a = \mu_\theta(s)$, for which the action integral disappears. The answer is the deterministic policy gradient theorem, the direct parent of [DDPG](ddpg.md) and [TD3](td3.md), and, through its interpretation as a pathwise derivative through a critic, a conceptual bridge to the first-order methods of [g3](../groups/g3-analytic-gradients.md). It is added at depth 3 because DDPG states that its contribution is to make DPG work with neural networks; without DPG the deterministic branch of the lineage has no root.

## 2. Method

For a deterministic policy and discounted state distribution $\rho^\mu$,

$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s\sim\rho^{\mu}}\!\left[\nabla_\theta \mu_\theta(s)\,\nabla_a Q^{\mu}(s,a)\big|_{a=\mu_\theta(s)}\right].$$

As in the stochastic [policy gradient theorem](pg-theorem.md), no derivative of the state distribution appears. The paper shows this gradient is the limit of the stochastic policy gradient as the policy variance tends to zero, so the two are one family. Because the expectation is over states only, the gradient can be estimated off-policy from a behaviour policy $\beta$ without importance weights over actions (the critic is learned off-policy by Q-learning). A compatible function approximator $Q^w(s,a) = (a-\mu_\theta(s))^\top \nabla_\theta\mu_\theta(s)^\top w + V^v(s)$ yields an exact gradient, analogous to Sutton et al.'s result. The resulting algorithm, COPDAC-Q (compatible off-policy deterministic actor–critic), uses linear features and gradient-TD critics. See [deterministic policy gradient](../math-cookbook.md#m-dpg).

## 3. Evidence

Experiments use linear or tile-coded approximators: continuous bandits with up to 50 action dimensions, mountain car, pendulum, puddle world, and an octopus-arm striking task. As reported in the paper and summarised by DDPG, the deterministic estimator showed clear data-efficiency gains over stochastic on- and off-policy actor–critic, with the margin widening as action dimension grew in the bandit experiments.

## 4. Critical analysis

DPG moves the burden of variance into the critic: the actor gradient is exactly as good as $\nabla_a Q$, the action-gradient of a learned function, which is not what Q-learning trains for. Errors in $\nabla_a Q$ are systematic rather than zero-mean, and the actor actively seeks regions where $Q$ is overestimated; this is the failure [TD3](td3.md) later analysed. Exploration is decoupled from the policy and must be supplied by added noise. The paper's linear-approximator theory did not survive the move to neural networks: DDPG found that naive DPG with networks is unstable without target networks. The deterministic limit also removes the entropy-driven exploration that [SAC](sac.md) reinstates.

## 5. Cross-references

- Builds on: [PG theorem](pg-theorem.md), of which it is the zero-variance limit.
- Extended by: [DDPG](ddpg.md), deep-network DPG with replay and target networks; [TD3](td3.md).
- Contrasts with: [SVG](svg.md), which generalises DPG to stochastic, reparameterised policies and to gradients through a learned model (DDPG notes that DPG is the deterministic limit of SVG(0)).

## 6. Relevance to TGI

The DPG actor update is a one-step pathwise gradient: $\nabla_\theta \mu$ chained with $\nabla_a Q$. TGI's objective is the multi-step version of the same chain rule, with the learned world model supplying $\partial s_{t+1}/\partial(s_t,a_t)$ for 32 steps and a state-value critic supplying the terminal $\nabla_s V(s_H)$; at a cut touchdown the chain stops and the critic gradient takes over, which locally reduces TGI's update to a DPG-like "differentiate through the critic" step. See [TGI](tgi.md), [SHAC](shac.md) and [reparameterisation](../math-cookbook.md#m-reparam).
