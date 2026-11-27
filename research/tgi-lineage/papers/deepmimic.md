# DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills

| | |
|---|---|
| Authors | Xue Bin Peng, Pieter Abbeel, Sergey Levine, Michiel van de Panne |
| Venue / year | ACM Transactions on Graphics 37(4), SIGGRAPH 2018 |
| Link | https://arxiv.org/abs/1804.02717 |
| Group | [g6-character-imitation](../groups/g6-character-imitation.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2017 deep RL could make simulated humanoids locomote, but the motions had well-known artefacts (flailing arms, odd postures); sampling-based trackers such as SAMCON reproduced acrobatic mocap but were complex and did not extend naturally to task objectives or perception. DeepMimic proposes the simplest workable combination: a policy conditioned on state, a phase variable and an optional goal, trained with PPO on a reward that sums a per-frame motion-tracking term and a task term. Its central empirical finding is that two episode-design choices, reference state initialisation (RSI) and early termination (ET), are what make highly dynamic skills (backflips, spin-kicks) learnable.

DeepMimic is the anchor of the explicit-tracking branch. Its reward structure, RSI and ET reappear almost unchanged in [PHC](phc.md), [MaskedMimic](maskedmimic.md), the robot trackers of [g7](../groups/g7-humanoid-tracking.md) ([BeyondMimic](beyondmimic.md), [SONIC](sonic.md)) and therefore in [SuperMimic](supermimic.md), whose tracking reward is BeyondMimic's.

## 2. Method

The policy $\pi(a_t|s_t,g_t)$ outputs PD targets ([PD actuation](../math-cookbook.md#m-pd-actuator)) at 30 Hz; spherical joints use axis-angle targets. The state contains link positions relative to the pelvis, link quaternions and linear and angular velocities in a heading-aligned frame, plus a phase $\phi\in[0,1]$. The policy is a Gaussian with fixed diagonal covariance and a 1024–512 ReLU MLP mean; terrain tasks add a convolutional heightmap encoder.

The reward is $r_t=\omega^I r^I_t+\omega^G r^G_t$ with $\omega^I=0.7$, $\omega^G=0.3$, and the imitation term

$$r^I_t = 0.65\,r^p_t + 0.1\,r^v_t + 0.15\,r^e_t + 0.1\,r^c_t,$$

$$r^p_t = \exp\Big(-2\sum_j \|\hat q^j_t\ominus q^j_t\|^2\Big),\quad r^v_t=\exp\Big(-0.1\sum_j\|\hat{\dot q}^j_t-\dot q^j_t\|^2\Big),$$

$$r^e_t=\exp\Big(-40\sum_e\|\hat p^e_t-p^e_t\|^2\Big),\quad r^c_t=\exp\big(-10\|\hat p^c_t-p^c_t\|^2\big),$$

where $\ominus$ is the quaternion difference and $\|\cdot\|$ its rotation angle ([rotation errors](../math-cookbook.md#m-rotations)). Each term is a Gaussian kernel on a tracking error, bounded in $[0,1]$; see [tracking rewards](../math-cookbook.md#m-tracking-reward). Training is PPO with the clipped surrogate ([PPO](../math-cookbook.md#m-ppo-clip)), advantages from GAE($\lambda$) and value targets from TD($\lambda$) ([GAE](../math-cookbook.md#m-advantage-gae)).

RSI samples the initial state uniformly along the reference clip, so the agent sees high-reward states (a mid-air flip) before it can reach them; the authors interpret it as a second channel, besides the reward, through which the reference informs learning. ET ends an episode with zero subsequent reward when the torso or head touches the ground; it both shapes the reward and removes uninformative "struggling on the ground" samples from the batch. Three multi-clip schemes are evaluated: a max over per-clip imitation rewards, a one-hot skill selector, and a composite policy that samples among independently trained experts with Boltzmann weights on their value functions, $p^i(s)\propto\exp(V^i(s)/T)$.

## 3. Evidence

The humanoid (197-D state, 36-D action) learns more than 20 skills from 0.5–5 s clips; simulation runs in Bullet at 1.2 kHz with stable PD. Normalised returns (32 episodes, RSI and ET applied in evaluation) range from 0.485 (frontflip) to 0.985 (walk), with backflip at 0.729; Atlas, a T-Rex and a 418-D dragon are trained from retargeted or keyframed data. Each skill needed 41–191 million samples (backflip 72 M, spin 191 M), which the authors describe as often several days per skill.

The ablation (Table 5, evaluation without ET and from a fixed initial state) gives normalised returns for RSI+ET / ET only / RSI only: backflip 0.791 / 0.730 / 0.379, sideflip 0.823 / 0.717 / 0.355, spinkick 0.848 / 0.858 / 0.358, walk 0.980 / 0.981 / 0.974. ET is therefore decisive for dynamic skills; RSI matters qualitatively, since the policy trained without RSI reaches a comparable return on the backflip by hopping backwards and never learns the flip. Task objectives change behaviour: strike success is 99% with $r^I+r^G$ against 19% with imitation only, and throw success 75% against 5%. Policies withstand pelvis pushes of up to 720 N for 0.2 s (run) without perturbation training, which the authors attribute to exploration noise. Policies transfer across morphology only when retrained: humanoid policies applied to Atlas score 0.013–0.014.

## 4. Critical analysis

The paper demonstrates that a dense, hand-weighted tracking reward plus two episode-level heuristics suffice for single-clip imitation of dynamic skills. It does not demonstrate scalability: each skill is a separate policy, multi-clip rewards only work for similar motions (the max-reward fails to mix a sideflip and frontflip), and the composite policy is a hand-assembled ensemble. The phase variable ties the policy to a clock that advances linearly, so it cannot slow down or re-time a motion for recovery; later trackers replaced the phase by future reference frames ([PHC](phc.md), [MaskedMimic](maskedmimic.md), [SuperMimic](supermimic.md)'s $K=10$ future frames). The ablations are mostly single runs. Robustness numbers come without domain randomisation and in a simulator with stable PD, so they say nothing about sim-to-real. The reward weights and kernel scales are tuned by hand and partly character-specific, which is the criticism that motivates [AMP](amp.md).

Conceptually, RSI and ET are exploration devices for a model-free estimator: they move the start-state distribution onto the reference manifold and truncate trajectories whose returns would otherwise dominate the variance. Methods that use analytic gradients inherit them for a different reason, namely to keep short imagined rollouts inside the region where a learned model is accurate (see [ControlVAE](controlvae.md) and [SuperMimic](supermimic.md)).

## 5. Cross-references

- Builds on: [PPO](ppo.md) and [GAE](gae.md) for optimisation; SAMCON and DeepLoco for the problem setting; contrasts its tracking reward with [GAIL](gail.md).
- Extended by: [AMP](amp.md) (replaces the tracking reward by an adversarial style reward but keeps RSI and ET); [PHC](phc.md) (scales tracking to AMASS with RSI, relaxed ET and hard-negative mining); [peng-animals](peng-animals.md) (transfers DeepMimic to a real quadruped); [BeyondMimic](beyondmimic.md) and [SONIC](sonic.md) (humanoid robot tracking).
- Contrasts with: [SuperTrack](supertrack.md) and [ControlVAE](controlvae.md), which minimise a similar tracking error by backpropagation through a learned world model rather than by PPO.

## 6. Relevance to TGI

TGI is a locomotion method, but its training recipe descends from the tracking recipe through [SuperMimic](supermimic.md). Three DeepMimic ingredients survive. First, the dense exponential-kernel reward family (SuperMimic uses BeyondMimic's version of it), which provides a smooth, differentiable objective well suited to backpropagation through a model. Second, the idea that the start-state distribution is a design variable: SuperMimic's return-based reset sampling and TGI's success-guided terrain curriculum are descendants of RSI's observation that resets steer exploration. Third, early termination, which SuperMimic and TGI convert into a differentiable survival weight $\hat c_t$ so that termination can enter an analytic gradient.
