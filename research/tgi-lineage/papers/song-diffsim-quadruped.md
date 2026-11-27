# Song et al.: Learning Quadruped Locomotion Using Differentiable Simulation

| | |
|---|---|
| Authors | Yunlong Song, Sangbae Kim, Davide Scaramuzza |
| Venue / year | CoRL 2024 |
| Link | https://arxiv.org/abs/2403.14864 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Through 2023, policy learning with differentiable simulation on legged robots stayed in simulation. Whole-body dynamics with intermittent contact give the discontinuous, stiff landscapes analysed by [Suh et al.](suh-diffsim-gradients.md), and the robots that walked in the real world were trained with PPO in massively parallel non-differentiable simulators ([legged_gym](legged-gym.md)). Song et al. sidestep the problem of differentiating contact. They run a high-fidelity, non-differentiable simulator (IsaacGym) forward and backpropagate through a deliberately simplified, smooth surrogate: single-rigid-body dynamics of the trunk driven by leg forces. This is one of the first real-robot demonstrations (on Mini Cheetah) of a policy trained with first-order gradients. It also introduces the forward/backward decoupling that [DMO](amigo-decoupled.md) then generalises to learned models.

## 2. Method

The surrogate is the single-rigid-body model familiar from [convex MPC](convex-mpc-cheetah.md):

$$\dot p = v,\quad \dot v = \tfrac1m\sum_i f_i + g,\quad \dot q = \Lambda(\omega)\,q,\quad \dot\omega = I^{-1}(\eta - \omega\times I\omega).$$

The policy outputs joint targets, a PD law gives torques, and the torques map to foot forces via $f = (J^\top)^{-1}\tau$, treating the PD controller as a differentiable layer ([m-pd-actuator](../math-cookbook.md#m-pd-actuator)). The policy is deterministic, and the gradient is plain BPTT of a differentiable loss ([m-bptt](../math-cookbook.md#m-bptt)):

$$\nabla_\theta L = \frac1N\sum_k\Big(\sum_i \frac{\partial l_k}{\partial x_k}\Big(\prod_{j=i}^k\frac{\partial x_j}{\partial x_{j-1}}\Big)\frac{\partial x_i}{\partial\theta} + \frac{\partial l_k}{\partial u_k}\frac{\partial u_k}{\partial\theta}\Big).$$

**State alignment** keeps the surrogate on the true trajectory:

$$\hat x^{\rm diff}_{t+1} = x^{\rm non\text{-}diff}_{t+1} + \alpha\big(x^{\rm diff}_{t+1} - \mathrm{sg}(x^{\rm diff}_{t+1})\big).$$

In the forward pass this sets the state to IsaacGym's value. In the backward pass it passes $\alpha\,\partial x^{\rm diff}_{t+1}/\partial x_t$, with $0<\alpha\le 1$ acting as a gradient decay. Episodes of 240 steps are split into 24-step windows (short-horizon training, as in [SHAC](shac.md) but without a critic). The loss combines velocity tracking, height, body rate, action magnitude, orientation and a swing-foot term toward Raibert-heuristic footholds ([Raibert](raibert.md)) under a fixed gait schedule.

## 3. Evidence

On a double integrator, the differentiable-simulation policy approaches the LQR optimum within a few iterations, which PPO with 1024 environments does not. A single simulated robot learns to walk at 0.2 m/s from 24 samples per iteration over 1000 iterations, where PPO fails. Table 1 compares final reward and wall-clock time at equal iterations. With 4 robots the reward is −3.83 against PPO's −25.82. With 1024 robots the two are close (−2.69 against −2.80) at about the same time (11.6 and 11.4 min). The authors report solving the task with about 64 robots and 200 iterations. Trot, pace, bound and gallop at 1–4 Hz transfer zero-shot to Mini Cheetah with a blind policy trained with 64 robots. In the ablation, removing state alignment prevents learning entirely. An appendix experiment shows that a non-differentiable termination penalty, which helps PPO, has no effect on differentiable simulation, because "the gradient of a constant value is equal to zero, and we do not leverage a state value function".

## 4. Critical analysis

The paper's insight is that the backward model need not be the forward model. A smooth, physically motivated surrogate supplies useful descent directions while an accurate simulator keeps the visited states realistic. This is a deliberate application of analytic smoothing, the alternative to randomised smoothing noted by Suh et al. The cost is bias of unknown size. The gradient of a single-rigid-body model with prescribed gait and Raibert footholds says nothing about where to step, and the authors acknowledge that foot placement cannot be discovered from the loss. The method is therefore a learned tracking controller inside a heavily structured locomotion template, not a general first-order RL method. The comparison with PPO uses a dense differentiable loss and no termination penalties, conditions that favour the first-order method. As the termination appendix shows, PPO benefits from exactly the terms that first-order gradients cannot use. The absence of a critic is a limitation the authors themselves suggest addressing.

## 5. Cross-references

- Builds on: [SHAC](shac.md) (short windows); [convex MPC](convex-mpc-cheetah.md) (single-rigid-body model); [Raibert](raibert.md) (foothold heuristic); [Metz et al.](metz-gradients.md) (cited for the BPTT expression and chaos).
- Extended by: [DMO](amigo-decoupled.md), which keeps simulator-forward and model-backward but learns the backward model, adds a critic, and cites this work as the model-free precursor of decoupling.
- Contrasts with: [SuperMimic](supermimic.md) and [TGI](tgi.md), which use one learned model for both passes.

## 6. Relevance to TGI

Two lessons from Song et al. carry into TGI. The first is structural: TGI's world model is a hybrid between Song's analytic surrogate and a black-box network. It keeps the exact contact Jacobian and terrain geometry, $s_{t+1} = s_t + f(x) + B(x)J(s_t)^\top W$, and learns the free motion, inverse inertia and contact wrenches. Physics therefore shapes the backward pass while data supplies accuracy. The second concerns rewards: TGI uses soft, differentiable terminations with survival weights and a critic, avoiding the zero-gradient termination penalty documented here. Unlike Song et al., TGI places no Raibert template on the feet. Footholds must be discovered through gradients of terrain-dependent rewards, which is why the reliability of gradients at touchdowns, and the event-gated cut, matter.
