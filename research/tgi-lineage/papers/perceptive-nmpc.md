# Perceptive NMPC: Perceptive Locomotion through Nonlinear Model Predictive Control

| | |
|---|---|
| Authors | Ruben Grandia, Fabian Jenelten, Shaohui Yang, Farbod Farshidian, Marco Hutter |
| Venue / year | IEEE Transactions on Robotics, 2023 (arXiv 2022) |
| Link | https://arxiv.org/abs/2208.08373 |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | full text read (depth-3 addition) |

## 1. Problem and position in the lineage

This paper represents the state of model-based perceptive locomotion at ETH immediately before hybrid TO+RL controllers took over: a nonlinear MPC that optimises all degrees of freedom of ANYmal at 100 Hz, with terrain exposed to the optimiser as local convex foothold regions and a signed distance field. It enters this study for three reasons. It is "baseline-to-2" in [DTC](dtc.md) and the planner with which DTC climbed a 0.48 m box; its limitations section explicitly proposes RL tracking controllers as the remedy for hard-coded reactive behaviours; and it contains a direct experimental comparison between single-shooting iLQR and multiple shooting that bears on the stability of backpropagating through long rollouts, the core concern of [g3](../groups/g3-analytic-gradients.md) and of [TGI](tgi.md).

## 2. Method

**Perception.** An elevation map (4 cm) is updated at 20 Hz. Each update is classified into steppable and non-steppable cells by inclination and roughness, segmented into planes, and converted into a signed distance field with non-steppable regions inflated. For each leg and each future contact phase, a convex polygon around the Raibert-type nominal foothold is extracted from the selected plane and enforced as linear inequality constraints on the foothold; the SDF constrains swing legs and knees away from the terrain.

**Model and optimal control problem.** The model is kinodynamic: centroidal dynamics of the base obtained from the top six rows of the full rigid-body equations, $\dot v_B = M_B^{-1}(\tau_B^{\mathrm{dist}} - M_{Bj}\ddot q_j - n_B + \sum_i J_{B,i}^\top \lambda_i)$, with joint velocities and contact forces as inputs, so changes of inertia and CoM location with leg configuration are captured (cookbook: [rigid-body dynamics](../math-cookbook.md#m-rigid-contact)). The problem $\min \Phi(x(T)) + \int_0^T L\,dt$ over a 1 s horizon includes friction cones, foothold polygons, collision avoidance and joint limits, with an LQR-based terminal cost approximating the infinite-horizon tail (cookbook: [MPC](../math-cookbook.md#m-mpc), [LQR/DDP](../math-cookbook.md#m-lqr-ddp)).

**Solver.** Multiple shooting with SQP, a Riccati-structured QP (HPIPM), a filter line search and the real-time iteration scheme (one SQP step per control cycle). A whole-body torque controller and reactive behaviours for early and late touchdown run at 400 Hz.

## 3. Evidence

Mean MPC iteration time is 6.6 ms (maximum 9.8 ms); classification and segmentation take 38.8 ms per map update on average. The SRBD model, compared with the kinodynamic one during a step-up at 30° pitch, shows a 2.6 cm CoM error, a 13.3 N m bias equivalent to an unmodelled 3.6 kg payload at the robot's tip. In the solver comparison, iLQR implemented with the same constraint handling and Riccati backward pass satisfies dynamics exactly on flat ground but its forward rollout diverges on a stepping-stone course, at 63 % of the traversal at 0.5 m/s and at 78 % at 0.75 m/s (a leap and a near-singular step-down); SLQ fails similarly, while multiple shooting with real-time iteration stays within 10 % of the fully converged cost. Hardware experiments include a ramp–gap–step course, uneven stepping stones 20 cm apart in height, and indoor and outdoor stairs.

## 4. Critical analysis

The convex foothold regions trade the freedom of optimising over the raw map for numerical reliability; the selection of a plane per leg is independent and discrete, so all feet can end up on one tread when walking sideways, and small regions can flicker in and out across map updates. The gait is externally fixed and adapted only at early or late touchdown, and the reactive behaviours are hard-coded; the authors themselves name RL tracking controllers as a route to robustness, which [DTC](dtc.md) then took. The solver comparison is the paper's most transferable result: a single-shooting method that propagates through the full nonlinear rollout at each iteration is "sudden, unpredictable" in failure on hard terrain, whereas a method that allows small, temporary dynamics defects between shooting nodes is robust. The comparison involves a handful of scenarios, so it is evidence of a mechanism rather than a general ranking.

## 5. Cross-references

- **Builds on:** [online iLQG-MPC](ilqr-online-to.md) (the DDP family it moves away from), [Raibert](raibert.md) (nominal footholds), and the SRBD and convex-MPC tradition of [Di Carlo et al.](convex-mpc-cheetah.md).
- **Contemporary:** [TAMOLS](tamols.md), the smoothed-height-map alternative from the same group.
- **Extended by:** [DTC](dtc.md), which deploys its trained policy zero-shot with this planner.
- **Related (other groups):** [SHAC](shac.md) truncates the horizon and bootstraps with a critic for the same reason this paper abandons long single-shooting rollouts: gradients propagated through long contact-rich rollouts are unreliable (cookbook: [BPTT](../math-cookbook.md#m-bptt)).

## 6. Relevance to TGI

[TGI](tgi.md) differentiates through rollouts of a learned model, a single-shooting computation in the sense of this paper. The iLQR divergence reported here is the optimal-control face of the problem TGI handles with three devices: a short horizon (32 steps, 0.64 s) with a critic at the end, as in [SHAC](shac.md); start states drawn from the simulator, which plays the role of re-anchoring shooting nodes on the true trajectory; and the gradient cut at imagined touchdowns on poorly supported terrain, which removes exactly the events (leaps onto stones, edge contacts) where this paper's iLQR diverged. TGI's 3D ANYmal variant also adopts the signed-distance representation used here for collision avoidance: its world model receives distance and direction from feet and from spheres on the knees, shanks, thighs and body to the nearest surface, the learned counterpart of this paper's SDF constraints.
