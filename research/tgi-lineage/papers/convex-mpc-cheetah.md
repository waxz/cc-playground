# Convex MPC: Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control

| | |
|---|---|
| Authors | Jared Di Carlo, Patrick M. Wensing, Benjamin Katz, Gerardo Bledt, Sangbae Kim |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018 |
| Link | no-arxiv |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Template controllers in the tradition of [Raibert](raibert.md) stabilise running gaits with decoupled heuristics but do not reason about the coupled body dynamics over a horizon, and full nonlinear MPC on the whole-body model ([Tassa et al.](ilqr-online-to.md)) was, in 2018, too slow and too fragile for a 45 kg-class quadruped on hardware. Di Carlo and co-authors showed that a *convex* MPC on a single-rigid-body model can be solved fast enough on board to stabilise a wide range of dynamic gaits on the MIT Cheetah 3. The paper fixed a template for quadruped control that was reproduced on many platforms (notably the open-source Mini Cheetah software) and is the reference point against which learned quadruped controllers were later compared.

## 2. Method

The robot is modelled as a single rigid body (SRBD) acted on by ground reaction forces $f_i$ at the feet: $m\ddot p = \sum_i f_i + m g$ and $\frac{d}{dt}(I\omega) = \sum_i r_i \times f_i$, with the legs treated as massless. Three approximations make the problem convex: roll and pitch are assumed small, so the rotational kinematics are linearised about the current yaw; the gyroscopic term $\omega \times I\omega$ is neglected; and the foot positions $r_i$ over the horizon are taken from the planned footsteps rather than optimised. The result is a linear time-varying system in the state $(\Theta, p, \omega, \dot p)$ (with gravity appended as a constant state) and the input $u = (f_1,\dots,f_4)$. The MPC solves, at a rate of tens of hertz, the quadratic programme

$$\min_{x,u}\ \sum_{k} \|x_{k+1} - x^{\mathrm{ref}}_{k+1}\|_{Q}^2 + \|u_k\|_{R}^2 \quad \text{s.t.}\quad x_{k+1} = A_k x_k + B_k u_k,\ \ f_{\min} \le f_{i,z} \le f_{\max},\ \ |f_{i,x}|, |f_{i,y}| \le \mu f_{i,z},$$

with the friction cone linearised to a pyramid and swing legs constrained to zero force by the gait schedule (cookbook: [receding-horizon control](../math-cookbook.md#m-mpc); [rigid-body dynamics and friction cones](../math-cookbook.md#m-rigid-contact)). Only the first force is applied, mapped to joint torques by $\tau = J^\top f$ and combined with swing-leg PD control at a high rate. Footholds come from a Raibert-type heuristic with an additional velocity-dependent term related to the [capture point](capture-point.md).

## 3. Evidence

The paper demonstrates on hardware a set of gaits (trotting, bounding, pronking, galloping and related patterns) under a single controller with gait-specific schedules, and reports running speeds of the order of 3 m/s and robustness to disturbances and to unmodelled payloads. Because the full text was not read, other quantitative details (solve times, horizon length) are not quoted here; the commonly cited configuration in follow-up work uses a horizon of roughly one gait cycle and a dense QP solver.

## 4. Critical analysis

The convexity is bought with assumptions that hold in the regime the paper targets and fail outside it. Small roll and pitch excludes flips and large recoveries. Massless legs are acceptable for the Cheetah's light legs and less so for robots with heavy limbs; [Grandia et al.](perceptive-nmpc.md) later quantified the SRBD error on ANYmal during a step-up as a 2.6 cm CoM offset, equivalent to a 3.6 kg unmodelled payload at the robot's tip. Fixed footholds and a fixed gait schedule make the problem convex but remove the two decisions that matter most on rough terrain; the controller is blind. The paper therefore marks a boundary: convex MPC is a robust *balance and force distribution* layer on flat or mildly rough ground, and everything terrain-related must come from elsewhere. Its later extension with whole-body impulse control (Kim et al., 2019) addressed tracking fidelity, not terrain.

## 5. Cross-references

- **Builds on:** [Raibert](raibert.md) (foot placement) and [capture point](capture-point.md) (velocity correction of the foothold); SRBD planning in earlier humanoid and quadruped work.
- **Extended by:** [TAMOLS](tamols.md) and [perceptive NMPC](perceptive-nmpc.md), which keep an SRBD-like or kinodynamic model but optimise footholds against an elevation map; [DTC](dtc.md), which tracks the output of such optimisers with a learned policy.
- **Related (other groups):** [Song et al.](song-diffsim-quadruped.md) backpropagate through a simplified rigid-body model aligned with a whole-body simulator, effectively turning the convex-MPC model into a differentiable training model for a policy; [TD-MPC](td-mpc.md) replaces the hand-built linear model by a learned latent model with sampling-based MPC.
- **Contrasts with:** [Hwangbo et al.](hwangbo-anymal.md) and [Lee et al.](lee-challenging-terrain.md), whose learned policies output joint targets directly and were shown to exceed model-based baselines in robustness on ANYmal.

## 6. Relevance to TGI

[TGI](tgi.md) does not run MPC at deployment; its policy maps observations to PD joint targets in a single network evaluation. The convex-MPC paradigm survives in the *structure of TGI's world model*. TGI's contact-impulse model predicts the next state as free motion plus the effect of contact wrenches through the exact contact Jacobian, $s_{t+1} = s_t + f(x) + B(x)\,J(s_t)^\top W$, and it enforces, by construction, that tangential forces lie inside a friction cone $|f_t| \le \mu\, p\, m$ and that no force acts when the foot is above the terrain. These are the same physical constraints that the convex MPC places on its decision variables; the difference is that in TGI they constrain a learned predictor rather than an optimiser. Training the policy through 32-step rollouts of this model is also structurally a receding-horizon objective, amortised into a policy instead of re-solved online.
