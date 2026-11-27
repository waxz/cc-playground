# DPPO: Emergence of Locomotion Behaviours in Rich Environments

| | |
|---|---|
| Authors | Nicolas Heess, Dhruva TB, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, Yuval Tassa, Tom Erez, Ziyu Wang, S. M. Ali Eslami, Martin Riedmiller, David Silver |
| Venue / year | arXiv preprint, 2017 |
| Link | https://arxiv.org/abs/1707.02286 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Reward design for locomotion was known to be brittle: small changes to shaping terms yield unexpected gaits. Heess et al. hypothesise that rich, procedurally varied terrain, rather than reward engineering, can induce robust skills from a simple forward-progress reward, and they need a policy optimiser that scales to many workers and recurrent networks to test it. Their answer, Distributed PPO (DPPO), appeared within days of the [PPO](ppo.md) preprint, which cites it as concurrent work using the adaptive-KL variant for 3D locomotion. The paper is added at depth 3 because it is the first demonstration that PPO plus parallel data collection plus terrain diversity and curricula yields perceptive locomotion over gaps, hurdles and platforms, the template that [legged_gym](legged-gym.md) and the [g5](../groups/g5-learned-locomotion.md) literature later moved onto GPUs and real robots. Without it the group's account of why PPO became the locomotion default would jump from MuJoCo benchmarks to ANYmal with no intermediate.

## 2. Method

The optimiser is PPO with the adaptive KL penalty (Algorithm 1): maximise $\sum_t\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}\hat A_t - \lambda\,\mathrm{KL}[\pi_{\rm old}\|\pi_\theta]$ for $M$ steps, fit the baseline for $B$ steps, and multiply or divide $\lambda$ by $\alpha$ when the KL leaves $[\beta_{\rm low},\beta_{\rm high}]\times\mathrm{KL_{target}}$. For recurrent policies, truncated backpropagation through time over windows of $K$ steps is used, with $K$-step bootstrapped advantages $\hat A_t = \sum_{i=1}^{K}\gamma^{i-1}r_{t+i} + \gamma^{K-1}V_\phi(s_{t+K}) - V_\phi(s_t)$. Data collection and gradient computation are distributed over workers that synchronise with a parameter server after every gradient step; synchronous gradient averaging worked better than asynchronous updates. Normalisation statistics for inputs and rewards are shared across workers. Batches were 64,000 (walker), 128,000 (humanoid) and 24,000 (reacher) time steps. See [PPO](../math-cookbook.md#m-ppo-clip).

The environments are MuJoCo obstacle courses (hurdles, gaps, variable terrain, slalom walls, platforms), regenerated every episode, for a planar walker (9 DoF, 6 actuators), a quadruped (12 DoF, 8 actuators) and a humanoid (28 DoF, 21 actuators). The reward is essentially forward velocity minus a torque penalty (plus pose constraints or track-centring and an alive bonus). Observations split into proprioception and exteroception, the latter a height profile sampled on a body-fixed grid whose density falls with distance, processed by a separate subnetwork. "Curriculum" courses increase obstacle difficulty along the track. See [heightmaps](../math-cookbook.md#m-terrain-lookup) and [curricula](../math-cookbook.md#m-curriculum).

## 3. Evidence

On benchmark tasks (planar walker, humanoid, memory reacher), DPPO matched TRPO's final performance, scaled with worker count in wall-clock time, trained recurrent policies, and was faster in wall-clock time than the authors' A3C with the same number of workers; results are shown as curves only. On the obstacle courses, the walker learned to run, jump hurdles nearly as tall as itself, leap gaps and crouch under platforms, and the quadruped learned to jump hurdles and gaps and turn around walls, all from forward-progress reward. A walker trained on a curriculum course improved faster on both easy and hard hurdle validation courses than one trained on stationary statistics. Policies trained on hurdles were more robust than flat-terrain policies to unobserved friction changes, rubble, actuator-strength changes and inclines in some conditions. Humanoid results were diverse and seed-sensitive, with some runs reaching local optima with "visually unsatisfactory" gaits.

## 4. Critical analysis

The evidence for emergence is largely qualitative (videos, time-lapse figures, normalised returns) with few seeds reported, and the humanoid results explicitly depend on algorithm, exploration, reward, termination and initialisation. The claim that rewards were not tuned is partly undermined by the per-body differences in reward terms and termination conditions, which the authors acknowledge. The robustness study shows a trend, not a uniform effect. What the paper does demonstrate convincingly is the engineering recipe: an on-policy, KL-controlled policy gradient whose stability does not degrade as data collection is parallelised, combined with terrain randomisation and a difficulty curriculum. The open problems it leaves, gait quality, seed sensitivity and reliance on enormous sample counts, are those that motion priors ([AMP](amp.md)), privileged teachers ([RMA](rma.md), [Lee et al.](lee-challenging-terrain.md)) and, later, model-based first-order methods such as [TGI](tgi.md) attempt to address.

## 5. Cross-references

- Builds on: [PPO](ppo.md) (adaptive-KL variant), [TRPO](trpo.md), [A3C](a3c.md) (parallel workers).
- Extended by: [legged_gym](legged-gym.md) (massively parallel PPO with game-inspired terrain curriculum on GPU), [Lee et al.](lee-challenging-terrain.md), [Miki et al.](miki-wild.md) in [g5](../groups/g5-learned-locomotion.md).
- Contrasts with: [DeepMimic](deepmimic.md), which obtains natural gaits through motion-capture tracking rewards rather than environmental richness.

## 6. Relevance to TGI

TGI's task design descends from this line: velocity or goal commands over procedurally generated gaps, pits, stairs, boxes and stepping stones, a terrain heightmap read around the robot, and a success-guided curriculum over terrain type and difficulty. Where Heess et al. let terrain diversity shape behaviour through a zeroth-order optimiser that needs only reward samples, TGI reads the same terrain through differentiable lookups, so that edges and gaps enter the policy gradient analytically, and cuts that gradient where an imagined foot lands on unsupported terrain. See [TGI](tgi.md) and [terrain lookups](../math-cookbook.md#m-terrain-lookup).
