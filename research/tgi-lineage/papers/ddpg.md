# DDPG: Continuous control with deep reinforcement learning

| | |
|---|---|
| Authors | Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra |
| Venue / year | ICLR 2016 |
| Link | https://arxiv.org/abs/1509.02971 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

DQN learned from pixels but required a maximisation over actions at every step, which is infeasible for continuous, high-dimensional action spaces; naive discretisation of a 7-DoF arm with three levels per joint already gives $3^7 = 2187$ actions. Lillicrap et al. adapt DQN's stabilisers (replay, target networks) to the [deterministic policy gradient](dpg.md), producing Deep DPG (DDPG), a model-free, off-policy actor–critic for continuous control. DDPG opens the deterministic/off-policy branch of this group, which continues through [TD3](td3.md) and, with a stochastic actor, [SAC](sac.md). Its actor update is the first deep-RL instance of a pathwise derivative, taken through a learned critic rather than through dynamics.

## 2. Method

A critic $Q(s,a\mid\theta^Q)$ is trained off-policy on the Bellman target $y_i = r_i + \gamma Q'(s_{i+1},\mu'(s_{i+1}\mid\theta^{\mu'})\mid\theta^{Q'})$, and the actor follows the DPG chain rule:

$$\nabla_{\theta^\mu}J \approx \frac{1}{N}\sum_i \nabla_a Q(s,a\mid\theta^Q)\big|_{s=s_i,a=\mu(s_i)}\,\nabla_{\theta^\mu}\mu(s\mid\theta^\mu)\big|_{s_i}.$$

Four ingredients make this work with networks. A replay buffer of $10^6$ transitions, sampled uniformly in minibatches of 64. *Soft* target networks for both actor and critic, $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ with $\tau = 0.001$, which the ablation shows to be crucial. Batch normalisation of inputs and layers, to handle features with different physical units. Exploration by adding temporally correlated Ornstein–Uhlenbeck noise ($\theta = 0.15$, $\sigma = 0.2$) to the deterministic action. Networks are 400–300 ReLU MLPs (about 130,000 parameters) with a tanh-bounded actor, Adam learning rates $10^{-4}$ (actor) and $10^{-3}$ (critic), $L_2$ critic weight decay $10^{-2}$, $\gamma = 0.99$. For pixels, three convolutional layers and action repeats of 3 frames. The paper notes that, as is common, the discount in the state-visitation distribution is ignored. See [DPG/DDPG](../math-cookbook.md#m-dpg) and [Bellman equations](../math-cookbook.md#m-bellman).

## 3. Evidence

More than 20 MuJoCo tasks (cart-pole swing-up, reaching, gripper, puck striking, monoped balancing, cheetah, hopper, walker2d, the HyQ quadruped model) plus TORCS were trained with identical hyperparameters for at most 2.5 M steps, 5 replicas each. Scores were normalised so that a uniform-random policy scores 0 and the iLQG planner of Tassa et al. 2012 (with full access to dynamics and derivatives) scores 1. Average/best low-dimensional scores include cheetah 0.903/1.206, hardCheetah 1.311/1.990, walker2d 0.705/1.573, hopper 0.676/0.936, hyq 0.416/0.722; from pixels, walker2d 0.944/1.476 and cheetah 0.457/0.792. The original DPG with replay and batch normalisation but without target networks scored far lower on most tasks (e.g. cheetah −0.008 average). Learned Q-values tracked true returns on pendulum and cart-pole but were inaccurate on cheetah. On TORCS some replicas completed laps while others failed. The authors observe that most problems were solved in 2.5 M steps, about 20× fewer than DQN needed on Atari.

## 4. Critical analysis

The normalisation against iLQG is informative but not decisive: iLQG here is a receding-horizon planner whose performance depends on its own horizon and cost tuning, so scores above 1 indicate a weak baseline on some tasks as much as a strong learner. Five replicas with large gaps between average and best (walker2d 0.705 vs 1.573; TORCS −393 vs 1840) show seed sensitivity, later documented systematically by Henderson et al. (2018) and cited by both TD3 and SAC as DDPG's main weakness. The critic's value estimates are acknowledged to be poor on harder tasks, but the paper does not investigate the consequence that [TD3](td3.md) later proved: an actor maximising an approximate $Q$ by gradient ascent induces overestimation, which then propagates through the target. Batch normalisation and OU noise were dropped by later re-implementations without loss. In robotics, DDPG's sample efficiency did not translate into adoption for locomotion, because the bottleneck there moved from samples to wall-clock time once GPU simulation made samples cheap; replay-based updates scale poorly with thousands of parallel environments unless the update-to-data ratio is raised, which reintroduces instability.

## 5. Cross-references

- Builds on: [DPG](dpg.md) (actor gradient), DQN (replay, target networks); NFQCA (batch neural DPG).
- Extended by: [TD3](td3.md) (twin critics, delayed updates, target smoothing); [SAC](sac.md) (stochastic, entropy-regularised actor). [MAAC](maac.md) and [SVG](svg.md) extend the critic-gradient actor update to multi-step model rollouts.
- Contrasts with: [TRPO](trpo.md), which DDPG's authors describe as significantly less data-efficient because it learns no action-value function; [GAE](gae.md), whose authors found one-step critics too biased for their 3D tasks.
- Related in other groups: [PILCO](pilco.md), discussed in DDPG's related work as an analytic-gradient model-based alternative "impractical for high-dimensional problems".

## 6. Relevance to TGI

DDPG supplies three ingredients that recur in TGI in altered form. First, the actor gradient $\nabla_a Q\,\nabla_\theta\mu$ is the one-step special case of TGI's pathwise objective; TGI extends the chain rule through 32 world-model steps and a state-value critic at the horizon. Second, Polyak averaging: DDPG's soft target update is mirrored by TGI's exponential moving average of policy weights (decay 0.995), though TGI uses it for the *deployed* policy rather than for targets. Third, replay: TGI stores simulator transitions in a buffer (200 steps × number of environments) and samples imagination start states from it, so its policy update is off-policy with respect to the simulator data even though each imagined rollout is on-policy. See [TGI](tgi.md) and [Dyna-style rollouts](../math-cookbook.md#m-dyna-mbpo).
