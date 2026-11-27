# DTC: Deep Tracking Control

| | |
|---|---|
| Authors | Fabian Jenelten, Junzhe He, Farbod Farshidian, Marco Hutter |
| Venue / year | Science Robotics 9(86), eadh5401, 2024 (arXiv 2023) |
| Link | https://arxiv.org/abs/2309.15462 |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2023 ANYmal could be controlled either by trajectory optimisation (TO) with whole-body control, which places feet accurately and generalises across terrain geometries because it plans against the map, or by RL policies trained in simulation ([Lee et al.](lee-challenging-terrain.md), [Miki et al.](miki-wild.md)), which are far more robust to slip, map errors and contact mismatch but, in the authors' words, "struggle with sparse rewards emerging from environments where valid footholds are rare, such as gaps or stepping stones". DTC is the explicit attempt to obtain both: a model-based planner, [TAMOLS](tamols.md), computes footholds and a base trajectory online, and an RL policy learns to *track* the planner's footholds with joint-level commands in a physics simulator. It differs from [DeepGait](deepgait.md), which it cites, by using TO rather than a learned planner for the high level, and from earlier imitation-from-TO work by generating references online during RL training rather than from an offline dataset, so that the policy observes the planner's *current* intention.

## 2. Method

**Planner in the loop.** During training, TAMOLS (parallelised on CPU and exposed to Python) runs beside the GPU simulation and, from the ground-truth state and a noise-free elevation map, returns footholds $p^*_{i}$, a base pose trajectory $b^*(t)$ with its derivatives, and joint positions $q^*$ at the next touchdown (via analytical IK). The gait is a fixed trot (stride 0.93 s). The planner is re-solved only at each touchdown and after falls, pushes or command changes; re-solving at every policy step was found to produce a "lazy" local optimum, because the optimiser re-plans around a policy that fails to follow it.

**Observations.** The policy (an MLP, 50 Hz, PD joint targets as actions) sees proprioception, planar footholds, desired joint positions at touchdown, desired contact state and time left in phase, the reference twist, and a 40-point height scan sampled along the line from each foot to its target foothold. It does *not* see the optimised base pose, which the authors found reduces sensitivity to mapping errors and makes the policy planner-agnostic. The critic additionally receives privileged information (optimised base pose, twist and acceleration, external wrenches, friction, map drift) in an asymmetric actor–critic arrangement.

**Rewards.** Base tracking uses "soft" exponential kernels, $r_{B_n} = \exp(-\sigma_{B_n}\|b^{*(n)}(t+\Delta t) \ominus b^{(n)}(t)\|^2)$ for derivative orders $n=0,1,2$ (cookbook: [tracking rewards](../math-cookbook.md#m-tracking-reward)). Foothold tracking uses a "hard" logarithmic reward granted once per stance at contact,

$$r_{p_i} = -\ln\!\left(\|p_i^* - p_i\|^2 + \epsilon\right),$$

whose gradient grows as the error shrinks. A *consistency* reward penalises the distance between consecutive planner solutions on their overlapping horizon, discouraging the hesitation commonly seen when RL agents refuse to attempt hard terrain. Regularisation penalises foot power, joint power, action rate and joint acceleration.

**Robustness.** Observation noise, Laplace-distributed map noise and per-foot map drift (up to 0.2 m), pushes, friction in $[0.1, 1.2]$ and external wrenches are randomised (cookbook: [domain randomisation](../math-cookbook.md#m-domain-rand)); the planner always sees ground truth, so the policy learns to realise the motion the planner *would* produce with perfect information. A 12-type, 10-level terrain curriculum covers 76 800 m² ([curriculum](../math-cookbook.md#m-curriculum)). Training used 4096 robots, 90 000 PPO epochs and two weeks, generating about 23 years of optimised trajectories.

## 3. Evidence

On hardware (ANYmal C and D), the policy tracked footholds on flat ground with a mean error of 2.3 cm (standard deviation 0.48 cm) over heading speeds of 0.4–1.0 m/s. It crossed four consecutive 0.6 m gaps and a 1.8 m beam with 100 % success over four trials each, and a 0.1 m gap with 100 % over ten trials, terrains on which the teacher–student RL baseline of [Miki et al.](miki-wild.md) failed. It crossed fields of 0.2 × 0.2 m stepping stones, including loose and height-varying ones. Against model-based baselines, it survived a hidden collapsing floor, a pitching cover plate (map–odometry error up to 0.4 m) and slippery, deformable obstacles where TAMOLS alone fell. Deployed zero-shot with a different planner (Grandia et al.'s [perceptive NMPC](perceptive-nmpc.md)) it climbed a 0.48 m box, which the authors report as 50 % higher than the RL baseline; the TAMOLS-based pipeline was limited to about 0.40 m by TAMOLS's simplified kinematics. In simulation, raising the planner rate from 1 Hz to 50 Hz reduced failures by 7.11 percentage points; DTC outperformed an RL baseline retrained in the same environment chiefly on sparse terrains (stepping stones, beams, gaps, pallets), and matched it on dense ones. Providing desired joint positions from IK markedly accelerated learning of foothold tracking.

## 4. Critical analysis

The paper is candid about two negatives that deserve emphasis. First, DTC was "not more sample efficient than similar unifying RL approaches"; the authors conclude that foothold accuracy "is something intrinsically complicated to learn". Two weeks of training with a planner in the loop is the price of the hybrid. Second, the policy's capabilities are bounded by the planner's: the fixed trot, the 0.40 m climbing limit from TAMOLS's kinematics, and the risk that a deployed planner produces foothold patterns not seen in training are all inherited. The asymmetry in information (planner sees truth during training, noisy maps at deployment) is handled by noise on the desired joint positions, a heuristic that prevents the policy from reverse-engineering the drift but has no guarantee. The reported comparisons to RL baselines also mix architecture and training environment; the fairer retrained baseline (baseline-rl-3) closes most of the gap on dense terrain, so the advantage is specifically on sparse footholds, which is exactly where the planner supplies information that exploration alone fails to find. Conceptually, DTC uses TO as a *dense reward generator* and an observation, not as a gradient source: the policy is still optimised by PPO with score-function gradients ([g1](../groups/g1-rl-foundations.md)), and the planner's own gradients of cost with respect to footholds are discarded.

## 5. Cross-references

- **Builds on:** [TAMOLS](tamols.md) (reference generator), [perceptive NMPC](perceptive-nmpc.md) (deployment planner), [DeepGait](deepgait.md) and DeepLoco (planner/tracker hierarchy), [legged_gym](legged-gym.md) (training environment and curriculum), [PPO](ppo.md).
- **Contrasts with:** [Miki et al.](miki-wild.md) and [ANYmal Parkour](anymal-parkour.md)-style end-to-end perceptive RL; imitation of TO trajectories offline (Brakel et al., Bogdanovic et al.), which DTC replaces by online references.
- **Extended by:** [Watch Your Step](watch-your-step.md), which cites DTC and keeps the idea that footholds are the right interface between perception and control, but *learns* the foothold predictor from the policy's own rollouts instead of obtaining footholds from TO; [BeamDojo](beamdojo.md) and related sparse-foothold humanoid work in [g5](../groups/g5-learned-locomotion.md).

## 6. Relevance to TGI

DTC and [TGI](tgi.md) address the same failure of model-free RL, sparse footholds, from opposite directions. DTC injects model-based knowledge as a *target*: TO says where to step, and PPO learns to comply. TGI injects model-based knowledge as a *gradient*: the policy's return is differentiated through a learned world model and through exact, differentiable terrain lookups, so that the gradient of a foothold-support term pushes the predicted touchdown towards supported terrain without any planner. DTC's logarithmic foothold reward, whose gradient sharpens as the error shrinks, has a counterpart in TGI's smooth support field, a Gaussian-blurred supportability map whose spatial gradient "pulls the soles toward stone centres"; in DTC the gradient is only felt through PPO's advantage estimates, in TGI it flows analytically. TGI also inherits DTC's height-scan philosophy through [Watch Your Step](watch-your-step.md): sample the terrain only where the feet will go. Finally, DTC's finding that re-planning at every step invites lazy behaviour is the TO analogue of a concern in TGI's design: the model and the policy co-adapt, which is why TGI re-anchors imagined rollouts at real simulator states every iteration.
