# legged_gym: Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

| | |
|---|---|
| Authors | Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter |
| Venue / year | Conference on Robot Learning (CoRL), PMLR 164, 2022 (arXiv 2021) |
| Link | https://arxiv.org/abs/2109.11978 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

The earlier ANYmal controllers ([Hwangbo et al.](hwangbo-anymal.md), [Lee et al.](lee-challenging-terrain.md), [Miki et al.](miki-wild.md)) needed hours to days of CPU simulation per policy (the paper cites 12, 82, 88 and 120 hours for prior rough-terrain results). This paper moves the entire on-policy loop, simulation, reward and observation computation, PPO rollouts and updates, onto a single GPU using [Isaac Gym](isaac-gym.md), and studies how PPO's hyperparameters must change when thousands of robots are simulated in parallel. The open-source `legged_gym` and `rsl_rl` code that accompanied it became the default substrate of the field; nearly every later paper in this group, and both of TGI's parent works, train in it or in its successor Isaac Lab. It is the paper that defines the "PPO in massively parallel simulation" paradigm in its canonical form.

## 2. Method

**Batch geometry.** With $n_\text{robots}$ parallel environments each contributing $n_\text{steps}$ transitions, the PPO batch is $B = n_\text{robots}\, n_\text{steps}$. Increasing $n_\text{robots}$ by orders of magnitude forces $n_\text{steps}$ down to keep $B$ reasonable, but $n_\text{steps}$ cannot be arbitrarily small: because the advantage is estimated with [GAE](../math-cookbook.md#m-advantage-gae), which needs multiple consecutive rewards, learning degraded below about 25 steps (0.5 s at 50 Hz). Episodes (20 s) span many policy updates. Mini-batches of tens of thousands of samples were found to stabilise learning.

**Time-out bootstrapping.** Because episodes are reset on a time limit that the critic cannot observe, the authors distinguish terminal failures from time-outs and, on time-out, bootstrap the critic target with its own value prediction, $r_T \leftarrow r_T + \gamma V(s_{T+1})$. They note that the standard Gym interface and popular implementations such as Stable-Baselines ignored this distinction ([TD targets](../math-cookbook.md#m-bellman)).

**Game-inspired curriculum.** All terrains (flat, slopes up to 25°, rough, discrete obstacles up to ±0.2 m, stairs 5–20 cm) are tiled in one mesh with rows of increasing difficulty. A robot that walks past its 8 m tile's border is promoted; one that covers less than half the distance implied by its command is demoted; robots solving the hardest level are sent to a random level to avoid forgetting. This replaces the particle-filter curriculum of Lee et al. with a rule of negligible cost that uses the population of robots itself as the estimate of the policy's competence distribution ([terrain curricula](../math-cookbook.md#m-curriculum)).

**Task.** Observations are base linear and angular velocity, projected gravity, joint positions and velocities, the previous action and **108 height samples** on a grid around the base, each the distance from the terrain to the base. Actions are joint position targets tracked by PD (through an LSTM actuator network for ANYmal's SEAs, a simplification of [Hwangbo et al.](hwangbo-anymal.md)'s history-MLP). The reward has nine terms: command tracking, penalties on other base velocities, torques, joint accelerations, action rate and collisions, and a feet-air-time term. There is no gait prior, no trajectory generator and no teacher. Sim-to-real additions are friction randomisation in [0.5, 1.25], observation noise and random pushes every 10 s of up to ±1 m/s.

## 3. Evidence

With $n_\text{robots} = 20\,000$ and $n_\text{steps} = 50$ ($B$ = 1 M) as the quality reference, the authors sweep 128 to 16 384 robots at three batch sizes. Performance drops sharply when $n_\text{steps}$ becomes too small and degrades slowly when too few robots make samples correlated; wall-clock time scales almost linearly up to about 4000 robots. The recommended setting, 4096 robots and $B = 98\,304$, trains a rough-terrain policy in 1500 updates in under 20 minutes on an RTX A6000 (flat ground in under four minutes). In simulation the policy climbs 0.2 m stairs with nearly 100 % success, declines on randomised obstacles as height increases, and fails to climb slopes beyond 25°. The same setup trains ANYmal-B, ANYmal-C with an arm, Unitree A1 and, with one extra reward, Cassie. On hardware, ANYmal-C walks stairs and obstacles using a lidar elevation map, but the maximum command was reduced to 0.6 m/s because map imperfections reduced robustness at higher speed.

## 4. Critical analysis

The contribution is infrastructural and empirical. The scaling study is specific to one task and one reward, and its conclusion, that there is an upper limit to useful parallelism for an on-policy algorithm at fixed batch, is stated as an observation rather than explained beyond the GAE-horizon argument. The paper is candid that the policy "always converges to a trotting gait" with artefacts such as a dragging leg, removed by reward tuning, and that the perceptive policy is fragile to the real elevation map because it was trained on perfect height samples, the gap that [Miki et al.](miki-wild.md)'s noise model closes.

For the argument of this study, the most important feature of the paper is what massive parallelism does *not* change. PPO's gradient is still a likelihood-ratio estimate whose variance scales with the horizon over which GAE propagates credit; parallelism reduces that variance by brute force (a batch of about $10^5$ samples per update, about $1.5\times 10^8$ samples per policy) but leaves the estimator's structure untouched ([score-function gradients](../math-cookbook.md#m-pg-theorem)). The curriculum also presupposes that difficulty is a scalar along which competence transfers smoothly; on sparse terrain this assumption fails, which is why [Zhang et al.](risky-terrains.md) had to relax the demotion rule and [BeamDojo](beamdojo.md) disabled demotion altogether.

## 5. Cross-references

- Builds on: [Isaac Gym](isaac-gym.md); [PPO](ppo.md) and [GAE](gae.md); the curriculum idea of [Lee et al.](lee-challenging-terrain.md); the actuator network of [Hwangbo et al.](hwangbo-anymal.md).
- Extended by: [Miki et al.](miki-wild.md) (contemporaneous, CPU-based but merged later); [ANYmal parkour](anymal-parkour.md); [Extreme Parkour](extreme-parkour.md), [humanoid parkour](humanoid-parkour.md), [BeamDojo](beamdojo.md) and [attention-based map encoding](attention-map-encoding.md), all built on this pipeline; [RWM](rwm.md) and [SuperMimic](supermimic.md), which collect their world-model data in the same kind of parallel simulator.
- Contrasts with: [SHAC](shac.md), which exploits GPU parallelism for first-order gradients through a differentiable simulator instead of for larger PPO batches.

## 6. Relevance to TGI

TGI is a legged_gym descendant in nearly every engineering respect: it runs in Isaac Lab with 8192 (G1) or 4096 (ANYmal-D) parallel environments, uses heightmap lookups around the robot as terrain input, a terrain grid with a success-driven curriculum, PD joint targets and, for ANYmal-D, the ANYdrive LSTM actuator model. What TGI changes is the use of those parallel environments. Instead of feeding PPO's likelihood-ratio estimator, the simulator provides start states and transitions to fit a world model, and the policy gradient is obtained by backpropagating a 32-step (0.64 s) imagined return through the model and differentiable terrain lookups ([short-horizon objective](../math-cookbook.md#m-short-horizon)). The 0.5 s minimum horizon that legged_gym found necessary for GAE and TGI's 0.64 s imagination horizon are of the same order, a reminder that both methods rely on a learned critic for credit beyond about one gait cycle.
