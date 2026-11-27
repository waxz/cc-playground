# RMA: Rapid Motor Adaptation for Legged Robots

| | |
|---|---|
| Authors | Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik |
| Venue / year | Robotics: Science and Systems (RSS), 2021 |
| Link | https://arxiv.org/abs/2107.04034 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

RMA addresses the same sim-to-real problem as [Lee et al.](lee-challenging-terrain.md), but on the low-cost Unitree A1 (about 12 kg, direct-drive-like actuators) and without an actuator network, a trajectory generator or reference motions. Its framing is online adaptation: a robot should infer the properties of its current environment (payload, friction, motor strength, terrain height) from its recent state–action history within fractions of a second and modulate its behaviour accordingly. RMA turns the teacher–student split into a structured **latent adaptation** scheme that became the default in the Berkeley/CMU strand of the lineage, including [Extreme Parkour](extreme-parkour.md), whose phase-1 policy uses RMA's descendant, regularised online adaptation.

## 2. Method

**Base policy and encoder (phase 1).** A privileged environment vector $e_t \in \mathbb{R}^{17}$ (payload mass and its position, twelve motor strengths, friction and a scalar local terrain height) is encoded by an MLP $\mu$ into an extrinsics vector $z_t = \mu(e_t) \in \mathbb{R}^8$. The base policy $a_t = \pi(x_t, a_{t-1}, z_t)$ takes the 30-dimensional proprioceptive state (joint positions and velocities, roll, pitch, binary foot contacts), the previous action and $z_t$, and outputs joint position targets for a PD controller ($K_p = 55$, $K_d = 0.8$). $\pi$ and $\mu$ are trained jointly end-to-end with **PPO** ([clipped surrogate](../math-cookbook.md#m-ppo-clip)) for 15 000 iterations of 80 000 samples, about 1.2 billion steps in 24 hours on a desktop with one GPU, in RaiSim on fractal terrain.

**Adaptation module (phase 2).** A 1-D CNN $\phi$ estimates the extrinsics from the last $k = 50$ steps (0.5 s) of states and actions, $\hat z_t = \phi(x_{t-k:t-1}, a_{t-k:t-1})$, trained by regression $\|\hat z_t - z_t\|^2$. Crucially, data are collected on-policy with the *estimated* $\hat z_t$ driving the base policy, starting from a randomly initialised $\phi$, in the spirit of DAgger; this exposes $\phi$ to the off-nominal trajectories its own errors produce ([latent adaptation](../math-cookbook.md#m-domain-rand), [DAgger](../math-cookbook.md#m-distillation)). At deployment the adaptation module runs at 10 Hz and the base policy at 100 Hz, asynchronously.

**Reward.** The reward is described as bioenergetics-inspired: forward progress capped at 0.35 m/s, penalties on lateral motion and yaw, mechanical work $|\tau\cdot(q_t - q_{t-1})|$, ground impact, torque smoothness, action magnitude, joint speed, orientation, vertical acceleration and foot slip. As in Hwangbo et al., the penalty coefficients start small and are raised by a fixed curriculum to prevent the stand-still optimum; terrain difficulty is *not* curriculated.

Note the design choice that RMA regresses the *behaviourally relevant* latent $z_t$ rather than the physical parameters $e_t$: the encoder only has to preserve what changes the optimal action.

## 3. Evidence

In simulation (Table II, test ranges wider than training ranges, parameters resampled within episodes), RMA reached a 73.5 % success rate against 62.4 % for a robust domain-randomised policy, 56.5 % for explicit system identification, 41.7 % for AWR-based adaptation and 52.1 % without the adaptation module; the privileged "expert" with true $z_t$ reached 76.2 %. On hardware, RMA climbed steps, stepped down 15 cm with 80 % success, walked on a memory-foam mattress and uneven foam (100 %), carried up to 12 kg (100 % of body mass), and crossed an oily plastic sheet with plastic-wrapped feet in 90 % of trials, with the extrinsics estimate visibly changing at slip onset. Outdoors it walked on sand, mud, grass and dirt without failure, descended hiking-trail stairs in 70 % of trials and crossed cement and pebble piles in 80 %. Trials were small (typically five per condition).

## 4. Critical analysis

RMA shows that a low-dimensional latent, trained end-to-end to be useful to the policy, is a better adaptation target than identified physical parameters, and that a compact system with no actuator model transfers to cheap hardware. Its limitations are those of all blind controllers: the authors note that sudden falls on stairs and multi-leg obstructions cause failures, and that "a truly reliable walking robot" needs exteroception. The local terrain height in $e_t$ is a single discretised scalar, deliberately coarse so that the policy does not depend on fast, accurate terrain sensing; the method therefore says nothing about choosing footholds.

There is a subtler point. The adaptation module is trained to imitate a latent produced by an encoder that saw the true parameters. Whenever the parameters are not identifiable from 0.5 s of history, the regression target is ambiguous and $\hat z_t$ collapses towards a conditional mean; performance then depends on the base policy being robust to latents near that mean. The small gap between RMA and the expert (73.5 % vs 76.2 %) suggests this is benign for dynamics parameters, but terrain geometry ahead of the robot is fundamentally non-identifiable from proprioception, which is why later work ([Extreme Parkour](extreme-parkour.md)) keeps RMA-style adaptation for dynamics while distilling exteroception separately.

## 5. Cross-references

- Builds on: [PPO](ppo.md); the privileged-learning idea of [Lee et al. 2020](lee-challenging-terrain.md); [DAgger](../math-cookbook.md#m-distillation).
- Extended by: [Extreme Parkour](extreme-parkour.md), which uses regularised online adaptation (a single-phase variant) plus depth distillation; [Radosavovic et al.](radosavovic-humanoid.md), which replaces the explicit adaptation module with in-context adaptation in a transformer.
- Contrasts with: [Hwangbo et al.](hwangbo-anymal.md), which closes the reality gap with a learned actuator model rather than online adaptation; [system identification baselines](../math-cookbook.md#m-domain-rand) that estimate $e_t$ directly.

## 6. Relevance to TGI

TGI does not include an adaptation module, but the RMA design is relevant to its world model in two ways. TGI's world model predicts next state and contact forces conditioned on state, action and terrain; it is fitted to simulator transitions under domain randomisation, so unobserved parameters such as friction or mass enter it only as averaged noise, much as $\hat z_t$ collapses to a conditional mean when parameters are not identifiable. Second, RMA's insight that the policy should depend only on behaviourally relevant compressions of the environment reappears in TGI's foothold-gated perception (inherited from [Watch Your Step](watch-your-step.md)), which feeds the policy only the terrain around predicted footholds.
