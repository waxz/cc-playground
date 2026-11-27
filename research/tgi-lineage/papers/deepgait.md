# DeepGait: Planning and Control of Quadrupedal Gaits using Deep Reinforcement Learning

| | |
|---|---|
| Authors | Vassilios Tsounis, Mitja Alge, Joonho Lee, Farbod Farshidian, Marco Hutter |
| Venue / year | IEEE Robotics and Automation Letters (RA-L) 5(2), 2020 (arXiv 2019) |
| Link | https://arxiv.org/abs/1909.08399 |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | full text read |

## 1. Problem and position in the lineage

In 2019 perceptive quadruped locomotion on ANYmal was dominated by modular model-based pipelines: a foothold planner searched a scored elevation map near a nominal (Raibert-type) foothold, and a separate optimiser generated a base motion for the chosen contacts. DeepGait states the limitation of these methods precisely: approaches that jointly optimise discrete contact choices and continuous motion are kinodynamic but too slow to run online, while those that run online are *kinostatic*, decoupling foothold selection from the base dynamics and relying on hand-designed terrain qualification. End-to-end deep RL, on the other hand, had shown emergent locomotion but suffered from sample complexity and undesired emergent behaviour. DeepGait is the first ETH work in this lineage to place RL *inside* a model-based decomposition: a learned high-level planner proposes support phases and footholds from a heightmap, a model-based feasibility check replaces physics during its training, and a learned low-level controller tracks the footholds. It is therefore the earliest member of the "hybrid TO + RL" branch that leads through [DTC](dtc.md) to the foothold-centric perception of [Watch Your Step](watch-your-step.md) and [TGI](tgi.md).

## 2. Method

**Gait parameterisation.** A gait is a sequence of *support phases* $\Phi = \langle R_B, r_B, v_B, r_F, c_F, t_E, t_S\rangle$: base orientation, position and velocity, the four foot positions $r_F \in \mathbb{R}^{3\times4}$, the contact state $c_F \in \{0,1\}^4$, and the elapsed time and time-to-switch of the phase.

**Gait Planner (GP).** The GP is a Gaussian policy $\pi_{\theta_P}(a\mid o_P)$ queried at roughly 2 Hz. Its observation contains the heading error to the goal, the CoM velocity, the foot positions and contact states, and a robot-centric $32\times32$ heightmap at 4 cm resolution processed by three convolutional layers. Its action edits the current phase into a candidate successor $\Phi^*_{t+1}$ (base rotation and translation, CoM velocity, foot displacements, a 3-bit code for the contact state, and phase timings). The key design choice is that the GP's MDP contains **no physics simulation**. Its transition is deterministic bookkeeping (stance feet are held, heights are snapped to the terrain) followed by three termination checks: footholds must be clear of edges and gaps on an eight-point grid around each foot, the base must not collide with the terrain, and the transition must be *dynamically feasible* according to a linear programme, $F_{\mathrm{croc}}(\Phi_t,\Phi^*_{t+1}) \in \{0,1\}$. The LP follows the CROC formulation (convex resolution of centroidal dynamic trajectories): centroidal dynamics with zero angular-momentum rate, a Bézier-parameterised CoM trajectory, unilateral friction-cone constraints on contact forces, and kinematic bounds on the feet relative to the base. Infeasible proposals end the episode. The reward combines multiplicative and additive terms,

$$r_P = r_p \cdot r_h^2 \cdot r_k - r_c,$$

with $r_p$ rewarding progress of the *average foothold* towards the goal (so that leaning does not pay), $r_h$ facing the goal, $r_k$ staying close to nominal footholds under the shoulders, and $r_c$ penalising feet that stay in contact for many phases. The GP is trained with PPO and a GAE critic (cookbook: [PPO](../math-cookbook.md#m-ppo-clip), [GAE](../math-cookbook.md#m-advantage-gae)).

**Gait Controller (GC).** The GC runs at 100 Hz, observes only proprioception plus foothold tracking errors, desired contact states and a phase variable, and outputs joint position targets tracked by a PD controller at 400 Hz (cookbook: [PD action spaces](../math-cookbook.md#m-pd-actuator)). It is trained with TRPO in the RaiSim physics engine on randomised foothold sequences, with rewards on foothold and contact-state tracking, foot slip, swing velocity, torque and base motion, and terminations on large tilt and base collision (cookbook: [trust region](../math-cookbook.md#m-trust-region)). The GC does not require the GP during training; any feasible foothold sequence can serve as reference.

## 3. Evidence

All results are in simulation. On the Flat-World, Random-Stairs and Temple-Ascent (stairs, gaps, stepping stones, narrow bridges) scenarios, the combined GP+GC reaches episodic success rates (goal within 0.5 m) of 99.8 % on flat ground, 96.4 % over gaps, 96.8 % on stepping stones and 90.6 % on stairs, with foothold tracking scores (fraction of footholds hit within 5 cm) of 0.985, 0.967, 0.970 and 0.751 respectively. Under test-time perturbations (base mass +25 %, shank length ±10 %) flat and gap performance changes little, while stairs drop to 76–90 % with large variance. GP training took about 82 hours and two billion samples per terrain on 14 CPU workers; GC training took 58 hours (flat) to 116 hours (3D terrain). In a qualitative comparison with the model-based Free-Gait planner, only DeepGait crossed a 40 cm gap: the learned planner selected a foothold across the gap in one shot and the controller pitched the base to extend reach, whereas Free-Gait's greedy, statically stable search could not.

## 4. Critical analysis

The paper's central quantitative claim, an "18-fold effective reduction in sample complexity" for the planner, is an estimate rather than a measurement. It is derived from throughputs (1 kHz for the LP versus 60 kHz for the simulator) and the simulated time spanned by an episode, and the arithmetic in the text is internally inconsistent (130 s of simulated time at a 2.5 ms step is 52 000 steps, not the stated 24 000; the 18-fold figure matches the former). More importantly, no planner was trained against full physics for comparison, so the claim is about wall-clock cost per transition, not about the number of transitions needed to reach a given performance. Two billion samples and 82 hours remain expensive.

The feasibility-as-dynamics idea has a sharper consequence that the authors acknowledge: the CROC model is conservative (zero angular-momentum rate, contact states restricted to three or four feet), so the planner learns conservative sequences and "the burden of realizing fully dynamic and optimal motions" falls on the controller. The interface between the two policies is a fixed, hand-designed phase parameterisation, which caps the gaits the system can express. Tracking quality on stairs (FTS 0.751) shows that the low-level controller does not fully realise the planner's intent in 3D, which is the problem [DTC](dtc.md) later attacked with a far larger training budget. Finally, the absence of hardware experiments means that robustness claims rest on mass and limb-length perturbations only.

## 5. Cross-references

- **Builds on:** DeepLoco (Peng et al., 2017), from which the two-level, two-time-scale hierarchy is taken; CROC for the feasibility LP; [PPO](ppo.md), [TRPO](trpo.md) and [GAE](gae.md) as optimisers ([g1](../groups/g1-rl-foundations.md)).
- **Contrasts with:** Free-Gait and other kinostatic planners built on the [Raibert](raibert.md) heuristic; [convex MPC](convex-mpc-cheetah.md), which takes footholds as given.
- **Extended by:** [DTC](dtc.md), which keeps the planner/tracker split and the foothold-tracking objective but replaces the learned planner by trajectory optimisation ([TAMOLS](tamols.md)) and deploys on hardware; [Watch Your Step](watch-your-step.md), which cites DeepGait and learns a foothold predictor whose output gates perception.
- **Related (other groups):** [Hwangbo et al.](hwangbo-anymal.md) and [Lee et al.](lee-challenging-terrain.md) (end-to-end learned ANYmal controllers of the same lab and period).

## 6. Relevance to TGI

Three ideas in [TGI](tgi.md) have a precursor here. First, DeepGait replaces expensive physics with a cheap surrogate during training of the planner; TGI replaces the simulator with a learned world model during policy optimisation. The motivation is the same (sample cost), but DeepGait's surrogate is an exact feasibility *classifier* on a coarse model, whereas TGI's is a learned *regressor* of the next state whose gradients are used. Second, DeepGait reasons about locomotion as a sequence of footholds read from a heightmap and checks each foothold against nearby edges and gaps; TGI's foothold-gated perception and its "unsupported sole share" and smooth support field, used both as reward and as the trigger of the gradient cut, are continuous, differentiable descendants of DeepGait's discrete foothold-validity check. Third, DeepGait shows the cost of a fixed hierarchical interface; TGI deliberately has no planner–tracker split, and lets a single policy, informed by predicted footholds, be shaped by gradients through the model.
