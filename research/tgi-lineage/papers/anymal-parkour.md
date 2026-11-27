# ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots

| | |
|---|---|
| Authors | David Hoeller, Nikita Rudin, Dhionis Sako, Marco Hutter |
| Venue / year | Science Robotics 9(88), eadi7566, 2024 (arXiv 2023) |
| Link | https://arxiv.org/abs/2306.14874 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Perceptive velocity-tracking controllers ([legged_gym](legged-gym.md), [Miki et al.](miki-wild.md)) walk over stairs and rubble but cannot climb a 1 m box, jump a 1 m gap, crawl under a table, or decide which of these to do. ANYmal parkour assembles a fully learned hierarchy that does all three on ANYmal-D (about 55 kg, 85 N m per actuator) at up to 2 m/s: a perception module that reconstructs the 3D scene from occluded point clouds, a catalogue of specialised locomotion skills, and a navigation policy that selects and commands those skills. It is the ETH counterpart of [Extreme Parkour](extreme-parkour.md) and the direct precursor of the goal-reaching, box-and-gap ANYmal-D task that [TGI](tgi.md) uses for its 3D experiments.

## 2. Method

**Locomotion skills.** Five PPO policies (walk, jump, climb up, climb down, crouch) share observation and action spaces: proprioception, a 2.5D height map extracted from the perception module's reconstruction, and a command; outputs are joint position targets at 50 Hz. They extend the **position-based formulation** of Rudin et al. (IROS 2022): rather than tracking a velocity, the robot must reach a target position and heading within a time budget, and the task reward is active only near the end of the episode, leaving the policy free to choose its own velocity profile. Each skill has its own obstacle and curriculum (gaps up to 1 m, boxes up to 1 m, passages down to 0.4 m), trained on 80 % task obstacle and 20 % rough terrain. Climb-down terminates on high foot impact forces, because otherwise the policy learns to jump down in a way that would damage hardware; climb-up allows knee and base contacts by reducing their penalties. Height maps are perturbed by point noise and shifts of up to 7.5 cm. Symmetry augmentation is used ([symmetry](../math-cookbook.md#m-symmetry)) and, per the authors, removes asymmetries reported for the earlier formulation.

**Perception.** Point clouds from six depth cameras and a lidar are voxelised and processed by dense 3D convolutional encoder–decoders at two resolutions (12.5 cm voxels over 4 m with autoregressive feedback of the previous output; 6.25 cm over 2 m around the robot). The decoder predicts per-voxel occupancy (binary cross-entropy) and centroid positions (Euclidean loss), trained on 2000 simulated trajectories with augmentation (point perturbation, blobs, removed patches, pose noise). A dense rather than sparse implementation was chosen to scale to 4000 robots during RL, at about 45 GB of GPU memory.

**Navigation.** A 5 Hz policy receives goal position, remaining time, base velocity, orientation and the perception latent, and outputs a skill choice plus a local position, heading and time command, executed by the frozen 50 Hz skills. PPO is modified with a **hybrid actor**: a categorical distribution over skills and a Gaussian over commands ([PPO](../math-cookbook.md#m-ppo-clip)). The distance-to-goal penalty is sparse, applied only at the final step, and a curriculum moves goals from near the start to further away.

## 3. Evidence

The evidence is mostly qualitative and from hardware demonstrations. The robot traverses randomised arrangements of stairs, boxes, gaps and tables at up to 2 m/s, saturates motor torque during a 0.9 m climb, deflects the hip by more than 160° to catch a landing, and reaches a 0.8 m-wide box (narrower than its standing footprint) at 1.5 m/s and places its feet precisely enough to jump from it, although the jump skill was trained with boxes twice that size. The walking skill climbs 40° slopes and 0.25 m steps and runs at 2 m/s on flat ground. In simulation each skill keeps high success up to about 90 % of its training difficulty and drops sharply beyond, at the robot's physical limits (Fig. 4F). The navigation policy learns non-obvious skill uses, for instance invoking the jump skill to turn on the spot in tight passages. Reconstructions are compared qualitatively with a GPU elevation-mapping baseline and recover box tops, table surfaces and drift-corrupted regions that the elevation map misses.

## 4. Critical analysis

The system shows that hierarchical PPO with learned perception can reach agility well beyond velocity tracking. Its costs are explicit in the paper: eight networks, each separately tuned, with interdependencies (the navigation policy works only with the perception latent and skills it was trained with, and new skills require retraining perception). The navigation policy "requires many iterations to converge" and fails to discover correct behaviour without a goal-distance curriculum; the authors suggest pre-training from brute-force search. These are symptoms of sparse, delayed reward under a likelihood-ratio estimator: a single decision at 5 Hz influences success only seconds later, through frozen low-level policies whose dynamics PPO treats as a black box.

The position-based formulation is the most consequential idea here for the rest of the lineage. By rewarding arrival rather than velocity, it removes an arbitrary constraint that makes agile manoeuvres infeasible, but it also makes the task reward sparse in time, so learning depends on auxiliary shaping ("move in direction", "don't wait" in Rudin et al.) and curricula. [Zhang et al.](risky-terrains.md) carried this formulation to stepping stones and found exploration under it the central difficulty.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md) and [Isaac Gym](isaac-gym.md); the GPU elevation map of [Miki et al.](elevation-mapping-gpu.md) as baseline; [PPO](ppo.md); the perceptive policies of [Miki et al.](miki-wild.md).
- Extended by: [Zhang et al. 2023](risky-terrains.md), which applies the navigation formulation to stepping stones and beams; [TGI](tgi.md)'s 3D ANYmal-D goal-reaching task over climbs, gaps, boxes and stepping stones.
- Contrasts with: [Extreme Parkour](extreme-parkour.md), a single end-to-end depth policy on a small robot; [DTC](dtc.md), which supplies footholds from a model-based planner.

## 6. Relevance to TGI

TGI's 3D experiment is defined as "Isaac Lab's ANYmal-D rough-terrain task ... reach a goal within 8 s over climbs, gaps, pits, boxes, stairs and stepping stones", which is the position-based, time-budgeted formulation and obstacle family of this paper (via Rudin et al. 2022) collapsed into a single policy. ANYmal parkour needed five specialised skills, each with its own curriculum, plus a navigation layer, largely because one PPO policy could not discover all behaviours under sparse arrival rewards. TGI's claim, to be tested in [g0](../groups/g0-anchor.md), is that first-order gradients through a world model with impact-force prediction make a single policy feasible on this task family. TGI's 3D terrain input (distances and directions from feet and body spheres to the nearest surface, walls included) also addresses the limitation that motivated this paper's 3D reconstruction: 2.5D height maps cannot represent walls and overhangs.
