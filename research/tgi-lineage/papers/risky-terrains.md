# Risky-terrain ANYmal: Learning Agile Locomotion on Risky Terrains

| | |
|---|---|
| Authors | Chong Zhang, Nikita Rudin, David Hoeller, Marco Hutter |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024 (arXiv 2023) |
| Link | https://arxiv.org/abs/2311.10484 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

Added at depth 3 because it is the first end-to-end RL controller for a quadruped on genuinely sparse footholds, and the explicit "baseline RL" in [attention-based map encoding](attention-map-encoding.md) and a source of terrain designs for [BeamDojo](beamdojo.md). Before it, sparse terrain was the domain of model-based planners and hybrids ([DTC](dtc.md)). The authors state the core difficulty: on risky terrain "exploration is difficult due to the numerous ways to fail", and feasible solutions "are quite rare and hard to explore compared to nonrisky terrains". The paper documents what it takes to make PPO succeed there.

## 2. Method

All training uses PPO in Isaac Gym with legged_gym's parallel setup, on ANYmal-D, with a heightmap scan, proprioception and position/heading/time-left commands.

**Navigation formulation.** Following Rudin et al. 2022 and [ANYmal parkour](anymal-parkour.md), the robot must reach a target within the episode; the task reward $r_\text{task} = \frac{1}{T_r}\frac{1}{1+\|\chi_t - \chi^*_t\|^2}$ is active only for $t > T - T_r$, with auxiliary "move in direction" and "don't wait" shaping. Velocity commands are abandoned because the robot must change speed rapidly on stones.

**Exploration strategy.** (i) A relaxed curriculum: robots are demoted only when the policy cannot beat random actions in remaining distance, so they keep practising a level until solved; the standard rule caused large reward gaps between levels that a "difficulty-agnostic optimizer" misattributed to actions. (ii) Random-network-distillation curiosity, $r_\text{curio} = c\,|M_1(s,a) - M_2(s,a)|$, which improved robustness rather than learning speed. (iii) Symmetry augmentation: each transition is mirrored left–right and front–back into four, with shared advantages and returns ([symmetry](../math-cookbook.md#m-symmetry)).

**Generalist–specialist training.** A generalist is trained for 8000 iterations on "Stones-Everywhere" (sparsity up to 81 % at level 9, random heights and shifts); specialists are fine-tuned on two-row stones (2000 iterations), balance beams evolving from two rows of stones (3000) and stepping beams (1000). Training specialists from scratch for 10 000 iterations failed.

## 3. Evidence

On Stones-Everywhere, the velocity-tracking formulation fell below 90 % success beyond level 4 (about 40 % sparsity) and declined rapidly, while navigation-based variants stayed high; symmetry augmentation was the most important single component on the hardest level. Specialists transferred poorly to each other's terrains (transferability matrix, Fig. 9), so no single policy covered all terrains. A beam curriculum of progressively narrower beams produced a crawling local optimum; stones merging into a beam avoided it. On hardware, with motion capture and a known terrain map, the robot crossed 20 cm stones covering about 10 % of the area at peak 2.7 m/s (8/10 success, 10/10 survival) and 20 cm beams with gaps at 2.5 m/s (5/5). A 5 cm odometry drift could cause a fall.

## 4. Critical analysis

The paper proves feasibility and is candid about cost: navigation formulation, curiosity, symmetry, a modified curriculum, generalist pre-training and per-terrain specialists were all needed, rewards were described as "an equilibrium of multiple motivations" that is tedious to tune, and a unified policy could not be trained because learning on different terrains interfered. Perception was bypassed. Each device addresses the same underlying fact, that random Gaussian exploration rarely produces a successful sequence of foot placements, so the sampled returns contain little information about which action perturbations help. That the most effective device was symmetry augmentation, which simply multiplies the data, is consistent with a variance-limited estimator.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md), [ANYmal parkour](anymal-parkour.md) (navigation formulation), [PPO](ppo.md), [Isaac Gym](isaac-gym.md).
- Extended by: [attention-based map encoding](attention-map-encoding.md), which generalises across sparse terrains with one policy; [BeamDojo](beamdojo.md), which reuses terrain types and two-stage training for a humanoid.
- Contrasts with: [DTC](dtc.md), which tracks planner footholds.

## 6. Relevance to TGI

TGI's ANYmal-D task (reach a goal within 8 s over gaps, boxes and stepping stones) is the navigation formulation used here, and TGI's G1 environments include stepping stones. This paper is the clearest evidence of how much exploration machinery PPO needs on such terrain, and therefore of the gap TGI claims to close with first-order gradients through a world model. Symmetry augmentation of the kind found decisive here is retained in TGI's direct parent, [Watch Your Step](watch-your-step.md), which mirrors each PPO batch rather than adding a symmetry loss; whether TGI's imagination-based training keeps it is not stated in the README.
