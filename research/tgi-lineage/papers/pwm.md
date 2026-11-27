# PWM: Policy Learning with Multi-Task World Models

| | |
|---|---|
| Authors | Ignat Georgiev, Varun Giridhar, Nicklas Hansen, Animesh Garg |
| Venue / year | ICLR 2025 |
| Link | https://arxiv.org/abs/2407.02466 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

After [AHAC](ahac.md), the same first author took a different route around the contact problem. Instead of truncating gradients at stiff contact in a differentiable simulator, PWM replaces the simulator with a learned, regularised world model and argues that such a model gives *better* first-order gradients than the true dynamics. The paper's claim is that, for first-order optimisation, a world model should be smooth and have a small optimality gap rather than be accurate. PWM combines this with a multi-task framing: pre-train one [TD-MPC2](td-mpc2.md) world model offline on many tasks, then extract a per-task policy in minutes with a [SHAC](shac.md)-style actor-critic. It is the clearest statement in this group of why the move from differentiable simulators to differentiable learned models can improve gradient quality, the premise on which [SuperMimic](supermimic.md) and [TGI](tgi.md) rest.

## 2. Method

**Smoothness argument.** In a ball-at-wall toy problem the ball "sticks" to the wall, which makes the objective $J(\theta)$ discontinuous in the launch angle. Gradient descent from $\theta=-\pi$ is run on the true objective and on MLP approximations with different activations. The model errors and optimality gaps $|J(\theta^*)-J(\hat\theta)|$ are 0.0 and 16.85 for the true function, 0.71 and 16.05 for a ReLU MLP, and 1.13 and 3.47 for an MLP with SimNorm activations (Fig. 2). The least accurate model is the best to optimise through. On the chaotic double pendulum, citing the scaling $\mathrm{Var}[\nabla^{[1]}J]\propto\|\nabla f\|^{2H}$, the authors measure gradient variance and expected signal-to-noise ratio (ESNR) over horizons. MLP models have lower variance and higher ESNR than the true dynamics, and a model trained autoregressively on 16-step windows sustains higher ESNR at long horizons than one trained on 3-step windows (Fig. 3; [m-model-learning](../math-cookbook.md#m-model-learning)).

**Algorithm.** The world model ($E_\phi$, $F_\phi$, $R_\phi$, with task embedding $e$) is TD-MPC2's, made differentiable, with LayerNorm, Mish and SimNorm outputs. It is trained on $H$-step autoregressive windows with a latent-consistency loss against a stop-gradient encoder target and a discrete-regression reward loss. The actor minimises

$$L_\pi(\theta) = -\mathbb{E}\Big[\sum_{h=1}^{H-1}\gamma^h R_\phi(z_h,a_h) + \gamma^H V_\psi(z_H)\Big],\quad z_1 = E_\phi(s_1),\; z_{h+1}=F_\phi(z_h,a_h),$$

by backpropagation through the latent rollout ([m-short-horizon](../math-cookbook.md#m-short-horizon)). An ensemble of three critics is trained on TD(λ) targets over the imagined window. Start states come from the offline buffer.

## 3. Evidence

On five dflex tasks (hopper, ant, ANYmal, humanoid, SNU humanoid with 152 actions), PWM, TD-MPC2 and DreamerV3 start from pre-trained world models and fine-tune online, while SHAC uses the simulator. PWM reaches higher asymptotic reward than SHAC and TD-MPC2, but "does not scale well to the highest-dimensional task" (Fig. 5). With stiffer contact on hopper, SHAC's reward falls by 48 % while PPO and PWM are unaffected, and PWM stays 17 % above PPO (Fig. 7a). On the multi-task suites MT30 and MT80, with a 48 M-parameter model and per-task policy training for 10 k gradient steps (9.3 min on an RTX 6000), PWM scores up to 27 % higher than TD-MPC2 without online planning (Fig. 6), and comparably to single-task SAC and DreamerV3 on MT30. An activation ablation on ant (ReLU, Mish, Mish+SimNorm) shows an inverse relation between world-model loss and policy reward (Fig. 8a). Smaller policy batches are better per unit time (Fig. 7b). A single multi-task policy trained the same way performs poorly (appendix).

## 4. Critical analysis

PWM turns an observation into a design principle: model regularisation acts as analytic smoothing of the objective, so first-order gradients through the model see a landscape free of the discontinuities and chaos that defeat them in the simulator. The pedagogical examples make the mechanism concrete, and the stiffness ablation is the most direct published evidence that learned models are robust to stiff contact where differentiable simulators are not.

The limits are substantial. "Smoother is better" is shown on a 1D toy and one activation ablation. Smoothing also biases the optimum, and the ball-wall example works because the smoothed landscape happens to keep the global optimum's basin. Nothing in the method controls that bias, and the authors' term "optimality gap" can only be measured when the true optimum is known. The single-task comparison with SHAC mixes several factors: offline pre-training data from other agents, latent representations and critic ensembles. It does not isolate model smoothness. The data requirement is large (120 k trajectories per dm_control task), and the authors list dependence on pre-existing data, per-task retraining and the difficulty of training autoregressive TD-MPC2 models as limitations. There is no hardware and no terrain.

## 5. Cross-references

- Builds on: [SHAC](shac.md) (actor-critic structure); [AHAC](ahac.md) (contact as the source of gradient error); [TD-MPC2](td-mpc2.md) (world model, SimNorm); [Suh et al.](suh-diffsim-gradients.md) and [PIPPS](pipps.md) / Parmas et al. (bias, variance, ESNR).
- Contrasts with: [Dreamer](dreamer.md) / [DreamerV3](dreamer-v3.md), which learn latent models by reconstruction, and [TD-MPC2](td-mpc2.md), which extracts policies with DDPG-style gradients and online planning; [DMO](amigo-decoupled.md), which keeps the simulator for forward passes.
- Extended by: [SuperMimic](supermimic.md), which applies the same principle to humanoid motion tracking at scale and on hardware.

## 6. Relevance to TGI

PWM supplies the argument for why TGI backpropagates through a learned model rather than the simulator: a regularised model fitted on autoregressive windows can give higher-ESNR gradients through contact than the true dynamics. TGI applies this in a constrained form. Its contact model is smooth by construction (a sigmoid gate on the sole–terrain gap with a 1 cm temperature, softplus force magnitudes, tanh-bounded friction and centre of pressure), and it is fitted on 8-step autoregressive windows. TGI departs from PWM in one respect that bears on PWM's bias concern. Terrain is not learned. It is read from the simulator through differentiable lookups, so edges and gaps stay exact, and the smoothing does not erase the discontinuities that matter for footholds. The price is that gradients near those edges are again unreliable. TGI's event-gated cut handles exactly that residual case, using smoothing where it is harmless and truncation where it is not.
