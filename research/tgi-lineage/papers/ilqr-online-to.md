# Online iLQG-MPC: Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization

| | |
|---|---|
| Authors | Yuval Tassa, Tom Erez, Emanuel Todorov |
| Venue / year | IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012 |
| Link | no-arxiv |
| Group | [g4-classical-control](../groups/g4-classical-control.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

By 2012, differential dynamic programming (DDP, Jacobson and Mayne 1970) and its Gauss–Newton variant iLQR/iLQG (Li and Todorov 2004–2005) were established offline tools for computing locally optimal trajectories. Tassa, Erez and Todorov asked whether the same machinery could run *online*, inside a receding-horizon loop, fast enough to synthesise whole behaviours for high-dimensional simulated robots from simple cost functions, without motion capture, templates or hand-designed controllers. The paper is the bridge between classical trajectory optimisation and modern model-based control in two senses: it is a direct ancestor of model predictive control with full-body dynamics on legged robots, and its authors simultaneously built [MuJoCo](mujoco.md) to make the required dynamics derivatives cheap.

## 2. Method

At each control step the controller solves a finite-horizon optimal control problem
$\min_{u_{0:T-1}} \sum_t \ell(x_t,u_t) + \ell_f(x_T)$ subject to $x_{t+1} = f(x_t,u_t)$, starting from the measured state and warm-started with the shifted previous solution, applies the first control, and repeats (cookbook: [receding-horizon control](../math-cookbook.md#m-mpc)). Each optimisation is an iLQG sweep. The *forward pass* rolls the nonlinear dynamics out under the current controls. The *backward pass* propagates a quadratic model of the value function from the end of the horizon to the start:

$$Q_x = \ell_x + f_x^\top V'_x,\quad Q_u = \ell_u + f_u^\top V'_x,\quad Q_{uu} = \ell_{uu} + f_u^\top V'_{xx} f_u,\quad Q_{ux} = \ell_{ux} + f_u^\top V'_{xx} f_x,$$

yielding a feed-forward correction $k = -Q_{uu}^{-1}Q_u$ and a feedback gain $K = -Q_{uu}^{-1}Q_{ux}$; a line search over closed-loop rollouts $u_t = \hat u_t + \alpha k_t + K_t(x_t - \hat x_t)$ completes the iteration (cookbook: [LQR, Riccati recursion, iLQR/DDP](../math-cookbook.md#m-lqr-ddp)). The paper's algorithmic contributions are practical: a regularisation that adds $\mu I$ to $V'_{xx}$ (penalising deviation in state rather than in control), which keeps $Q_{uu}$ positive definite and improves robustness of the backward pass, together with an adaptive schedule for $\mu$ and an improved line search. Derivatives $f_x, f_u$ are obtained by finite differences through MuJoCo, whose smooth, invertible contact model makes them well defined across contact.

## 3. Evidence

The demonstrations are in simulation: a humanoid with on the order of twenty degrees of freedom stands up from arbitrary poses on the ground and remains balanced under perturbations, and simpler systems (swimmers, acrobot-like mechanisms) exhibit behaviours that emerge from the cost alone. The paper reports that the controller runs close to real time on the computing hardware of the time. As the full text was not available, specific timing figures are not quoted here.

## 4. Critical analysis

The method inherits the strengths and weaknesses of local, gradient-based optimal control. It needs an accurate, differentiable model, and on hardware no such model exists; the demonstrations are therefore in simulation, where the model is exact by construction. Contacts are handled by MuJoCo's smooth soft-contact model, which gives usable gradients but trades physical fidelity for differentiability; the same trade-off reappears in differentiable-simulation RL (cookbook: [gradients through contact](../math-cookbook.md#m-contact-gradients)). The solution is local and depends on the warm start. Most importantly for this lineage, iLQR is a *single-shooting* method: every iteration rolls the full nonlinear dynamics forward, and the line search proceeds over closed-loop rollouts. [Grandia et al.](perceptive-nmpc.md) later reported that on ANYmal stepping-stone scenarios this forward rollout becomes unstable and diverges where the robot leaps or approaches singularities, which led them to multiple shooting. That failure mode is the trajectory-optimisation counterpart of exploding gradients when backpropagating through long rollouts (cookbook: [BPTT through dynamics](../math-cookbook.md#m-bptt)).

## 5. Cross-references

- **Builds on:** DDP (Jacobson and Mayne) and iLQG (Li and Todorov).
- **Companion:** [MuJoCo](mujoco.md), built by the same authors to supply fast dynamics derivatives.
- **Extended by:** [perceptive NMPC](perceptive-nmpc.md) (SQP with Riccati-structured QPs and multiple shooting, on hardware); [convex MPC on Cheetah 3](convex-mpc-cheetah.md) (the convex, simplified-model branch).
- **Related (other groups):** [SVG](svg.md) and [SHAC](shac.md) backpropagate through dynamics to improve a *policy* rather than an open-loop control sequence; [TD-MPC](td-mpc.md) and [PETS](pets.md) run receding-horizon planning on learned models; [PILCO](pilco.md) computes analytic policy gradients through a learned probabilistic model.

## 6. Relevance to TGI

This is the most direct classical ancestor of TGI's optimisation mechanism. TGI rolls a model forward for 32 steps from states visited in the simulator, sums rewards, adds a critic value at the end, and differentiates this return with respect to the policy parameters by backpropagation through the model ([SHAC](shac.md)'s objective, cookbook: [short-horizon objective](../math-cookbook.md#m-short-horizon)). The backward pass of iLQR computes, at second order, the same chain of Jacobian products $f_x^\top, f_u^\top$ that reverse-mode autodiff computes at first order in TGI; $V'_x$ in the iLQR recursion is the adjoint that backpropagation carries from step to step, and the terminal cost $\ell_f$ is the counterpart of TGI's critic. Three differences matter. TGI optimises a shared, closed-loop policy over thousands of start states instead of one control sequence from one state, which amortises the optimisation and removes the online solve. TGI's model is learned, so its Jacobians are only as good as the world model, whereas iLQR differentiates the true simulator. And TGI's gradient cut at poorly supported touchdowns is a learned-model analogue of the regularisation and line-search safeguards in this paper: both exist because the backward pass is untrustworthy exactly where contact makes the dynamics stiff or discontinuous.
