# SAC: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

| | |
|---|---|
| Authors | Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine |
| Venue / year | ICML 2018 (PMLR 80) |
| Link | https://arxiv.org/abs/1801.01290 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2017 the field faced a dilemma. On-policy methods ([TRPO](trpo.md), [PPO](ppo.md), [A3C](a3c.md)) were stable but discarded data after each update; off-policy [DDPG](ddpg.md) reused data but was brittle and failed on high-dimensional tasks such as the 21-action-dimension rllab Humanoid. Prior maximum-entropy methods (soft Q-learning) were off-policy but needed approximate inference to sample from an energy-based policy. SAC resolves the dilemma with an off-policy actor–critic whose stochastic actor maximises reward plus entropy. In the lineage it is the stochastic, entropy-regularised completion of the deterministic branch and, through its reparameterised actor gradient, the closest model-free relative of the pathwise estimators in [g3](../groups/g3-analytic-gradients.md).

## 2. Method

The objective augments return with policy entropy, $J(\pi) = \sum_t\mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t) + \alpha\mathcal H(\pi(\cdot\mid s_t))]$; the temperature $\alpha$ is folded into a reward scale. *Soft policy iteration* alternates soft policy evaluation, with backup $\mathcal T^\pi Q(s_t,a_t) = r + \gamma\mathbb{E}[V(s_{t+1})]$ and $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a) - \log\pi(a\mid s)]$, and soft policy improvement, an information projection

$$\pi_{\rm new} = \arg\min_{\pi'\in\Pi}D_{\mathrm{KL}}\!\left(\pi'(\cdot\mid s_t)\,\Big\|\,\frac{\exp Q^{\pi_{\rm old}}(s_t,\cdot)}{Z^{\pi_{\rm old}}(s_t)}\right).$$

Lemmas 1–2 and Theorem 1 prove convergence to the optimal policy within $\Pi$ for finite action spaces. The practical algorithm learns a soft value $V_\psi$ (squared residual to $\mathbb{E}_{a\sim\pi}[Q_\theta - \log\pi_\phi]$, with $a$ drawn from the current policy), two soft Q-functions $Q_{\theta_i}$ (squared residual to $r + \gamma V_{\bar\psi}(s')$ with a Polyak target $\bar\psi$), and a policy minimising the expected KL. Because the target density is a differentiable network, the policy is reparameterised, $a_t = f_\phi(\epsilon_t; s_t)$, giving

$$\hat\nabla_\phi J_\pi = \nabla_\phi\log\pi_\phi(a_t\mid s_t) + \big(\nabla_{a_t}\log\pi_\phi(a_t\mid s_t) - \nabla_{a_t}Q(s_t,a_t)\big)\nabla_\phi f_\phi(\epsilon_t;s_t),$$

which the authors describe as extending DDPG-style gradients to any tractable stochastic policy. Actions are $\tanh$-squashed Gaussians, with the log-likelihood corrected by $\sum_i\log(1-\tanh^2 u_i)$. Following [TD3](td3.md), the minimum of the two Q-functions is used in the value and policy gradients. See [maximum-entropy RL](../math-cookbook.md#m-maxent) and [reparameterisation](../math-cookbook.md#m-reparam).

## 3. Evidence

On Gym Hopper, Walker2d, HalfCheetah, Ant, Humanoid and rllab Humanoid (5 seeds, one evaluation rollout per 1000 steps, curves showing min–max), SAC matched baselines on easier tasks and exceeded DDPG, PPO, soft Q-learning and the concurrent TD3 on the harder ones in both speed and final return, with HalfCheetah returns approaching 15,000 within 3 M steps and Humanoid tasks learned over 10 M steps where DDPG made no progress on Ant, Humanoid and rllab Humanoid. The paper attributes PPO's slower learning to the large batches it needs for stability on high-dimensional tasks. Ablations on rllab Humanoid show that a deterministic variant (closer to DDPG) has much larger seed-to-seed variability; on Ant, evaluating the mean action outperforms stochastic evaluation, reward scale is the single sensitive hyperparameter (too small gives a near-uniform policy, too large a near-deterministic one), and target smoothing $\tau$ has a wide stable range (0.005 used everywhere).

## 4. Critical analysis

The convergence theory is tabular; the deep algorithm inherits no guarantee, and the separate value network, which the paper found to stabilise training, was dropped in the authors' follow-up together with the manual reward scale, replaced by automatic temperature tuning against a target entropy. The comparison with TD3 is complicated by the mutually inconsistent baselines in the two papers (TD3 reports SAC far below SAC's own numbers), a reminder that implementation and evaluation protocol dominate many published gaps. SAC's evidence, like TD3's, is from single-environment, one-step-per-update training measured in environment steps; it does not address wall-clock efficiency with thousands of parallel simulators, the regime that decided algorithm choice for legged robots ([g5](../groups/g5-learned-locomotion.md)). In [SHAC](shac.md)'s GPU-parallel comparison SAC and PPO were both far behind SHAC in samples and closer in wall time. Entropy maximisation changes the objective: the evaluated return is not what SAC optimises, and robustness benefits attributed to max-ent policies are asserted from prior work rather than tested here. SAC was, however, the algorithm with which the same group later trained a real quadruped from scratch (Haarnoja et al., 2019), the strongest evidence for its sample efficiency on hardware.

## 5. Cross-references

- Builds on: [DDPG](ddpg.md) (off-policy actor–critic), [TD3](td3.md) (minimum of two critics), soft Q-learning; [SVG](svg.md)(0), of which SAC is a max-ent variant with a separate value network.
- Extended by: [MBPO](mbpo.md), which uses SAC as the policy optimiser on short branched model rollouts; [DayDreamer](daydreamer.md) contrasts its world-model learning with SAC-style model-free robot learning.
- Contrasts with: [PPO](ppo.md), which uses entropy only as a regulariser and requires on-policy data; [SHAC](shac.md), which differentiates through multi-step dynamics rather than a one-step critic.

## 6. Relevance to TGI

SAC's actor update is the clearest model-free antecedent of TGI's gradient. Both reparameterise a Gaussian action, $a = \mu_\phi(s) + \sigma\epsilon$, and differentiate a learned value with respect to the action; SAC stops after one step at $\nabla_aQ$, while TGI continues through the world model for up to 32 steps before reaching the critic $V(s_H)$, which places TGI in the SVG($H$)/[MAAC](maac.md) family. TGI's noise is fixed ($\sigma = 0.2$) and its objective carries no entropy term, so it does not inherit SAC's exploration mechanism; exploration in the simulator comes from Gaussian action noise of 0.1. Like MBPO-with-SAC, TGI is a Dyna-style method that trains the policy on model rollouts branched from real states, but it replaces SAC's critic-gradient update with backpropagation through those rollouts. See [TGI](tgi.md) and [Dyna/MBPO](../math-cookbook.md#m-dyna-mbpo).
