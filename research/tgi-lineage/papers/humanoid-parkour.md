# Humanoid Parkour: Humanoid Parkour Learning

| | |
|---|---|
| Authors | Ziwen Zhuang, Shenzhe Yao, Hang Zhao |
| Venue / year | Conference on Robot Learning (CoRL), 2024 |
| Link | https://arxiv.org/abs/2406.10759 |
| Group | [g5-learned-locomotion](../groups/g5-learned-locomotion.md) |
| Source | full text read |

## 1. Problem and position in the lineage

This paper carries the quadruped parkour recipe ([Extreme Parkour](extreme-parkour.md), and Zhuang et al.'s own "Robot Parkour Learning") to the Unitree H1 humanoid. It claims a single vision-based whole-body policy that jumps onto 0.42 m platforms, leaps 0.8 m gaps, clears hurdles, climbs stairs and runs at 1.8 m/s outdoors, without motion references. It occupies the middle step of the humanoid strand of this group, between blind walking ([Radosavovic et al.](radosavovic-humanoid.md)) and sparse-foothold traversal ([BeamDojo](beamdojo.md)), and it provides an explicit admission, in the form of a footstep reward, that PPO did not place feet precisely on stairs unaided.

## 2. Method

Training proceeds in three stages in Isaac Gym, all with PPO ([clipped surrogate](../math-cookbook.md#m-ppo-clip)).

**Plane walking.** A walking policy follows $(v_x, v_y, \omega_\text{yaw})$ commands in $[-0.8, 2.0]$ m/s, $[-0.8, 0.8]$ m/s and $[-1, 1]$ rad/s. Rather than a feet-air-time reward or reference motions, the terrain height field carries **fractal noise**, so that the robot learns to lift its feet because the ground demands it. Rewards are task (velocity tracking, energy), regularisation (torque, joint acceleration and velocity, contacts) and safety terms (arm and waist motion, foot separation, action rate).

**Oracle parkour policy.** Ten terrain types (jump up/down, leap, hurdle, stairs up/down, tilted ramp, and others) are laid out in a 10 × 40 grid with difficulty interpolated per row; robots are promoted after covering 3/4 of a sub-terrain and demoted when failing before 1/2 ([curriculum](../math-cookbook.md#m-curriculum)). The policy is initialised from the walking policy, and an automatic yaw command steers robots initialised in random orientations back to the straight track so that turning is not forgotten. The actor is an MLP scandot encoder (an 11 × 19 grid of height samples), a GRU-based velocity estimator from proprioception, and a GRU–MLP backbone; the critic sees true base velocity. Two terms are added: a virtual-obstacle penetration penalty $r_\text{penetrate} = \alpha\sum_p d(p)\,\|v(p)\|$ over points attached to the robot's links, discouraging edge exploitation on leaps and jumps, and, on stairs, a **footstep reward** $r_\text{step} = 6\cdot(-\ln\|d_x\|)$ on the forward offset between the foot and the middle of each tread, because RL "does not generate accurate enough action without explicit footstep guidance".

**Depth distillation.** The scandot encoder is replaced by a CNN on 48 × 64 depth images; GRU and MLP weights are copied from the oracle. Simulated depth receives clipping, multi-level Gaussian noise and artefacts; real depth is filtered with the RealSense pipeline. DAgger with an L1 action loss is run across four RTX 3090 GPUs, one trainer and three collectors ([distillation](../math-cookbook.md#m-distillation)). The policy runs onboard on an Intel i7 CPU.

## 3. Evidence

In simulation (Table 2; three sub-tracks of increasing difficulty, ten robots per terrain), the fractal-noise policy reached 100 % success on jump-up and leap, against much lower success with a feet-air-time reward, and longer average distances on stairs and hurdles; the extracted table does not report all success entries consistently, and the appendix holds further obstacles. In the real world (ten trials per configuration on four tasks: jump-up, leap, stairs, hurdle), the vision policy outperformed a blind walking policy and Unitree's built-in controller with raised foot height (Fig. 5). Distillation from scratch failed to balance within 24 hours, and single-GPU distillation reached about half the performance of the four-GPU version. Teleoperating the arms during parkour did not destabilise the robot.

## 4. Critical analysis

The paper shows that the scandot-teacher/depth-student pipeline scales to humanoids, and that terrain noise can replace foot-lifting rewards. The quantitative evidence is limited: success rates on hardware are reported in a figure for four tasks, and simulation comparisons concern reward design rather than alternatives to the method. Generalisation beyond the ten manually built terrain types is explicitly not claimed.

The stair footstep reward is the most informative detail for this study. Stairs are a sparse-foothold terrain in disguise: each tread is a support region of 0.2–0.3 m depth, and stepping on the nosing leads to a slip several steps later. The authors found that PPO's credit assignment did not discover tread-centred footholds from survival and velocity rewards alone and injected a dense log-distance reward towards a hand-chosen target. This is the same pattern seen in [Extreme Parkour](extreme-parkour.md)'s edge penalty and [BeamDojo](beamdojo.md)'s foothold reward: when the return is a step function of foot position, the score-function estimator receives no signal until failures are frequent, and designers substitute a dense, differentiable-looking surrogate in the reward.

## 5. Cross-references

- Builds on: [Extreme Parkour](extreme-parkour.md) (scandots, depth distillation, curriculum); [legged_gym](legged-gym.md); [PPO](ppo.md); [Radosavovic et al.](radosavovic-humanoid.md) for humanoid RL feasibility.
- Extended by: [BeamDojo](beamdojo.md), which shares the Isaac Gym humanoid setting and tackles stepping stones and beams.
- Contrasts with: motion-prior humanoid locomotion via imitation ([AMP](amp.md), [g7](../groups/g7-humanoid-tracking.md)); [attention-based map encoding](attention-map-encoding.md), which learns foothold attention without hand-placed targets.

## 6. Relevance to TGI

TGI's G1 training terrains (stairs, gaps, pits, stepping stones, inherited from [Watch Your Step](watch-your-step.md)) overlap heavily with this paper's parkour set. The footstep reward illustrates the problem TGI's design aims to remove: TGI neither scripts target footholds nor rewards distance to them; instead, a foothold prediction model selects where the policy looks, and the policy gradient flows through a world model that reads exact terrain, so the effect of shifting a foot towards the tread centre is, in principle, available as a gradient rather than as a hand-crafted reward. Whether that gradient is informative near tread edges is exactly what TGI's touchdown gradient cut acknowledges as doubtful ([contact gradients](../math-cookbook.md#m-contact-gradients)).
