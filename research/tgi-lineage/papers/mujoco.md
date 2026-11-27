# MuJoCo: A Physics Engine for Model-Based Control

| | |
|---|---|
| Authors | Emanuel Todorov, Tom Erez, Yuval Tassa |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012 |
| Link | no-arxiv |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

Online trajectory optimisation of the kind demonstrated in the companion paper [Tassa, Erez and Todorov 2012](ilqr-online-to.md) evaluates the dynamics, and finite-difference derivatives of the dynamics, thousands of times per control step. Existing engines in 2012 (ODE, Bullet, PhysX and robotics simulators built on them) were designed for games or for forward simulation, used maximal coordinates or complementarity-based contact solvers with non-smooth behaviour, and were too slow or too irregular for this use. MuJoCo ("Multi-Joint dynamics with Contact") was designed for the opposite requirement: fast, deterministic, smooth evaluation of forward and inverse dynamics in generalised coordinates, so that a model-based controller can treat the simulator as a differentiable function. The engine later became the default benchmark substrate of continuous-control RL, which places it at a junction of this study: it is a tool of the classical model-based paradigm that became the testbed of model-free learning.

## 2. Method

MuJoCo represents articulated bodies in generalised (joint) coordinates and computes the equations of motion with recursive algorithms (composite rigid body for the mass matrix, recursive Newton–Euler for bias forces), $M(q)\ddot q + h(q,\dot q) = \tau + J_c^\top \lambda$ (cookbook: [rigid-body dynamics and contact](../math-cookbook.md#m-rigid-contact)). Its distinguishing design choice is the contact model. Instead of solving a hard linear complementarity problem, contact impulses are obtained from a convex optimisation that admits a small, controllable amount of constraint violation ("soft" contacts). This makes the solution unique, continuous in the state, and, crucially for control, makes *inverse* dynamics well defined in the presence of contact, because the soft model gives a one-to-one map between accelerations and forces. The engine is written for speed (multithreaded finite differencing, sparse factorisations) and uses an XML model format, MJCF, that has since become a common interchange format for robot descriptions.

## 3. Evidence

The 2012 paper reports timing benchmarks showing that MuJoCo evaluates dynamics and finite-difference derivatives for humanoid-scale models fast enough to support online trajectory optimisation on a desktop computer; the downstream evidence is the online iLQG controller of the companion paper. The full text was not read for this study, so the specific timing numbers are not reproduced. The broader evidence of its influence is circumstantial but strong: the continuous-control benchmarks used by [DDPG](ddpg.md), [TRPO](trpo.md), [PPO](ppo.md), [SAC](sac.md), [MBPO](mbpo.md) and [SHAC](shac.md) are MuJoCo tasks, and the engine was made freely available in 2021 and open-sourced in 2022 after its acquisition by DeepMind, followed by accelerator-based variants (MJX in JAX, MuJoCo Warp).

## 4. Critical analysis

The soft-contact model is a deliberate approximation. It yields smooth dynamics and useful derivatives at the cost of small penetrations and creep, and its parameters (solver impedance and reference time constants) affect behaviour in ways that matter for sim-to-real transfer. This is the same bias-for-smoothness trade-off that the differentiable-simulation literature later analysed explicitly (cookbook: [gradients through contact](../math-cookbook.md#m-contact-gradients); [Suh et al.](suh-diffsim-gradients.md)). A second point is easy to miss: MuJoCo's original purpose was model-based control with an *exact* model, yet its most visible use became model-free RL, in which the model is used only as a sample generator. The design goals that made it suitable for iLQG (speed, determinism, smoothness) are also what made massive sampling feasible, which illustrates how infrastructure built for one paradigm enabled the one that displaced it. Finally, finite differencing through a soft-contact simulator gives gradients of the *smoothed* system; they are informative near contact but can be misleading about the hard-contact system the robot actually experiences.

## 5. Cross-references

- **Companion:** [Online iLQG-MPC](ilqr-online-to.md), the controller that motivated the engine.
- **Used by (other groups):** the continuous-control benchmarks of [g1](../groups/g1-rl-foundations.md) and [g2](../groups/g2-world-models.md); [SHAC](shac.md) and the differentiable-simulation work of [g3](../groups/g3-analytic-gradients.md) compare against MuJoCo-style tasks; [RWM](rwm.md) cites it among simulation platforms.
- **Contrasts with:** the GPU simulators used for legged RL ([legged_gym](legged-gym.md) on Isaac Gym, RaiSim in [DeepGait](deepgait.md) and [Hwangbo et al.](hwangbo-anymal.md)), which favour throughput over differentiability.

## 6. Relevance to TGI

MuJoCo appears inside [TGI](tgi.md) in a concrete, if modest, role. The repository loads MJCF models of the Unitree G1 and ANYmal-D with the MuJoCo Python bindings and builds from them a *differentiable* forward-kinematics module in PyTorch, which computes foot-sole and body-part positions from the world model's predicted joint state; these positions feed the differentiable terrain lookups and the contact Jacobian $J(s)^\top W$ used in TGI's contact-impulse world model. TGI's sim-to-sim evaluation also runs exported G1 policies in MuJoCo with Unitree's model, as a check against a second physics engine. More fundamentally, TGI's whole premise, optimising a controller by differentiating a model of the dynamics, is the premise MuJoCo was built for; the difference is that TGI differentiates a *learned* model fitted to a (non-differentiated) GPU simulator, rather than the simulator itself.
