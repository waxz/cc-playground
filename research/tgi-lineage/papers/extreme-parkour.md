# Extreme Parkour: Extreme Parkour with Legged Robots

| | |
|---|---|
| Authors | Xuxin Cheng, Kexin Shi, Ananye Agarwal, Deepak Pathak |
| Venue / year | IEEE International Conference on Robotics and Automation (ICRA), 2024 (arXiv 2023) |
| Link | https://arxiv.org/abs/2309.14341 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Extreme Parkour asks how far a single end-to-end policy can push a low-cost quadruped (Unitree A1, hip height 26 cm, body length 40 cm) using only a front-facing depth camera running at 10 ± 2 Hz with jitter and artefacts. It is the Berkeley/CMU counterpart of [ANYmal parkour](anymal-parkour.md), appeared within weeks of it and of Zhuang et al.'s "Robot Parkour Learning", and differs from both by refusing hierarchy and elevation maps: one network maps depth and proprioception to joint targets, and the policy chooses its own heading. It combines the [RMA](rma.md) family's adaptation with the depth-distillation recipe of Agarwal et al. (CoRL 2022), and its terrain grid and promotion rule come directly from [legged_gym](legged-gym.md).

## 2. Method

**Unified inner-product reward.** Instead of random velocity commands, waypoints placed on each terrain define a direction $\hat d_w = (p - x)/\|p - x\|$, and tracking is rewarded in the world frame,

$$r_\text{tracking} = \min\big(\langle v, \hat d_w\rangle,\ v_\text{cmd}\big).$$

The authors use the world rather than the base frame specifically to stop the policy from walking around obstacles. Because the policy learned to step close to edges to save energy, a clearance penalty $r_\text{clearance} = -\sum_i c_i\,M[p_i]$ is added, where $c_i$ marks foot contact and $M$ is 1 within 5 cm of a terrain edge. A third inner-product term, $r_\text{stylized} = W\cdot 0.5\,(\langle \hat v_\text{fwd}, \hat c\rangle + 0.5)^2$ with $\hat c = [0,0,-1]^\top$ and a binary switch $W$, produces a handstand on two front legs. Regularisation terms come from legged_gym.

**Phase 1.** A policy is trained with PPO ([clipped surrogate](../math-cookbook.md#m-ppo-clip)) on proprioception, **scandots** (height samples of the terrain, the same privileged exteroception as legged_gym's height grid), the oracle heading $\hat d_w$, the handstand flag and commanded speed. Environment parameters are estimated with regularised online adaptation, a single-phase variant of RMA ([latent adaptation](../math-cookbook.md#m-domain-rand)). Terrains (hurdles, high steps, gaps, tilted ramps) are arranged by difficulty with legged_gym's promotion/demotion rule ([curriculum](../math-cookbook.md#m-curriculum)).

**Phase 2: dual distillation.** Scandots are replaced by a ConvNet–GRU on 58 × 87 depth images, trained by DAgger against the phase-1 actions, with the actor initialised from phase 1 ([distillation](../math-cookbook.md#m-distillation)). The student must also predict the heading. Feeding the untrained heading prediction back as an input causes distribution drift, so a mixture of teacher and student is used: the observed yaw is the prediction if within 0.6 rad of the oracle, and the oracle otherwise. Depth latency is fixed at 0.08 s and proprioceptive latency at 0.016 s on the onboard Jetson NX; the deployable policy trains in under 20 hours on one RTX 3090.

## 3. Evidence

On hardware the A1 jumps onto a 0.5 m box (twice its hip height), long-jumps a 0.8 m gap (twice the foot separation), crosses tilted ramps of 37° and walks on its front legs, including down stairs without vision. In a simulated obstacle course (256 robots, Table 2), mean normalised x-displacement was 0.98 for the full method against 0.75 without the inner-product reward (which learned to collide and retry on steps) and 0.82 for a policy trained on a simulated noisy elevation map; removing the clearance penalty kept displacement (0.99) but raised mean edge violation from 0.03 to 0.08. For distillation (Table 3), the mixture scheme reached 0.92 displacement against 0.94 for oracle headings and 0.12 or 0.05 for the alternatives. In real-world trials (five per difficulty), the method had 20–80 % higher success than baselines on the hardest instance of each terrain; a human joystick operator providing headings failed on ramps.

## 4. Critical analysis

The paper shows that dense, geometrically meaningful shaping and a single policy can produce emergent, highly dynamic skills, and that heading should be a policy output when obstacle geometry dictates it. Its evidence is thin by experimental standards (five real trials per condition, a simulated benchmark designed by the authors), and comparisons with the concurrent ANYmal work across robots are not controlled.

Two features matter for the sparse-terrain argument. First, the clearance penalty and the observation that the policy otherwise hugs edges are evidence that PPO's optimum on discontinuous terrain sits close to the discontinuity: the return is flat until the foot slips off, so nothing in the gradient estimate pushes the foot away from the edge until failures are sampled. The fix is a hand-designed dense surrogate of a constraint. Second, the terrains here (hurdles, steps, gaps, ramps) still offer contiguous landing surfaces; none requires choosing among isolated footholds, and the waypoint heading does part of the planning. The paper therefore demonstrates agility rather than foothold precision.

## 5. Cross-references

- Builds on: [legged_gym](legged-gym.md) (terrain curriculum, regularisers, scandots); [RMA](rma.md); [PPO](ppo.md); [DAgger](../math-cookbook.md#m-distillation).
- Extended by: [humanoid parkour](humanoid-parkour.md), which adopts the scandot-teacher/depth-student recipe for the Unitree H1.
- Contrasts with: [ANYmal parkour](anymal-parkour.md) (hierarchical, 3D reconstruction, elevation maps); [Miki et al.](miki-wild.md) (elevation-map belief encoder); [BeamDojo](beamdojo.md), whose sampling-based foothold reward is a finer-grained descendant of the edge penalty.

## 6. Relevance to TGI

The edge-clearance penalty is an early, hand-crafted answer to the problem TGI addresses structurally. TGI keeps terrain exact inside imagination through differentiable lookups and penalises predicted impact forces, so a policy update can respond to the geometric consequence of a foot placement before a fall is sampled. TGI also cuts the backpropagated gradient at imagined touchdowns on poorly supported terrain, precisely the near-edge contacts that Extreme Parkour's policy gravitated towards: there the pathwise gradient through contact is unreliable ([contact gradients](../math-cookbook.md#m-contact-gradients)) and TGI defers to the critic. The scandot representation, a heightmap sampled at fixed points, is the ancestor of TGI's foothold-gated terrain samples.
