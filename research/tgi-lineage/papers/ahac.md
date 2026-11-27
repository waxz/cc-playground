# AHAC: Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation

| | |
|---|---|
| Authors | Ignat Georgiev, Krishnan Srinivasan, Jie Xu, Eric Heiden, Animesh Garg |
| Venue / year | ICML 2024 (PMLR 235) |
| Link | https://arxiv.org/abs/2405.17784 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[SHAC](shac.md) made first-order policy learning competitive on dflex locomotion tasks, but it was brittle, sensitive to hyperparameters, and its fixed horizon $h=32$ was a tuned constant. Georgiev et al. connect SHAC's residual instability to the finite-sample analysis of [Suh et al.](suh-diffsim-gradients.md): stiff contact makes first-order batched gradients (FOBG) inaccurate relative to zeroth-order ones (ZOBG), and the inaccuracy compounds with horizon. Their remedy is to stop backpropagating at stiff contact, first literally (AHAC-1, single environment) and then, for parallel simulation, by learning the horizon length under a stiffness constraint. AHAC is the closest published antecedent of [TGI](tgi.md)'s event-gated gradient cut: both truncate the pathwise chain at contact events and hand over to a critic.

## 2. Method

The analysis uses the soft-Heaviside friction model $\bar H(x)$ of Suh et al. and the FOBG/ZOBG definitions ([m-contact-gradients](../math-cookbook.md#m-contact-gradients)). Defining the finite-sample error $B = \|\bar\nabla^{[1]}J - \bar\nabla^{[0]}J\|$ and assuming Lipschitz policy, reward and dynamics with constants $B_\pi, B_r, B_f$, Lemma 3.2 bounds

$$B \;\le\; H B_r B_\pi\big(c + B_f^{\,H-1}\big)$$

for a constant $c$ (Appendix B). Since contact makes $B_f \gg B_r, B_\pi$, the authors draw two conclusions: long rollouts increase FOBG error, and prolonged stiff contact compounds it. A ball-against-wall toy problem with $N = 1024$ samples and $H = 40$ shows the FOBG sample error staying low until contact and growing afterwards, and the expected signal-to-noise ratio ESNR $=\mathbb{E}[\sum\mathbb{E}[\nabla J]^2/\sum\mathrm{Var}[\nabla J]]$ (Parmas et al. 2023) dropping after contact.

**AHAC-1** rolls a single environment until $\|\nabla f(s_t,a_t)\| > C$ (stiff contact), truncates there, and bootstraps with the critic, so the stiff Jacobian $\nabla f(s_{t-1},a_{t-1})$ never enters the gradient (Fig. 4). With $C=500$ on hopper it beats SHAC ($H = 32$). It cannot be vectorised, because asynchronous truncation across environments produces unbounded computation graphs.

**AHAC** therefore keeps synchronous windows but adapts their length. An Ant sweep with SHAC shows the best and least variable reward at $H\approx 28$, close to the gait period, motivating a learned horizon. The actor solves

$$\max_\theta\; \sum_{h=t}^{t+H-1}\gamma^{h-t}r(s_h,a_h) + \gamma^H V_\psi(s_{t+H})\quad\text{s.t.}\quad \|\nabla f(s_h,a_h)\| \le C\;\;\forall h,$$

via the Lagrangian $L_\pi(\theta,\phi) = J(\theta) + \phi^\top(C - [\|\nabla f(s_h,a_h)\|]_h)$, and the horizon is updated as $H \leftarrow H + \alpha_\phi\sum_h\phi_h$. In practice $\nabla f$ is replaced by the Jacobian of the contact forces only, normalised by a per-dimension acceleration scaling so that one threshold $C$ serves all tasks. A double critic trained to convergence each iteration replaces SHAC's target critic.

## 3. Evidence

Tasks are dflex hopper, ant, ANYmal, humanoid and the 152-muscle SNU humanoid, with ten seeds and interquartile-mean statistics; baselines are PPO, SAC, SHAC and SVG (Amos et al. 2021, untuned). Aggregated over tasks, AHAC obtains about 40 % higher asymptotic reward than PPO (Fig. 9); on ant 41 % more than PPO, which is reported to remain below AHAC even after 3 B steps, i.e. with about 100× more samples and 10× more time (Fig. 8). On SNU humanoid SHAC and AHAC obtain 44 % and 64 % more reward than PPO but with wide confidence intervals (Fig. 10). The ablation on ant (Fig. 11) attributes most of the gain to the adaptive-horizon objective; fixing SHAC to AHAC's converged $H=29$ helps but does not match AHAC, which the authors read as evidence that the best horizon changes during training; the double critic mainly reduces run-to-run variance.

## 4. Critical analysis

The paper's practical insight is sound and transferable: gradient error originates at identifiable events (stiff contact), so the horizon should depend on where those events occur rather than be a global constant. The supporting evidence is weaker than the framing. Lemma 3.2 is a worst-case bound that is exponential in $H$ for any $B_f > 1$, contact or not; it explains why long horizons hurt but does not single out contact beyond the assumption that $B_f$ is large there. The empirical validation of the lemma is one toy problem. AHAC's constraint is enforced softly through a scalar horizon shared by all environments, so in the vectorised version it no longer truncates *at* contact but shortens every rollout when contact is frequent, a qualitatively different mechanism from AHAC-1; the paper acknowledges that synchronous truncation at stiff contacts "did not improve performance, likely due to gradient variances across differing trajectory lengths", which is a negative result worth noting. All results are on dflex with its soft contact; there is no hardware and no rough terrain. Gains over PPO are reported relative to PPO's maximum reward per task, which compresses absolute differences.

## 5. Cross-references

- Builds on: [SHAC](shac.md) (algorithm and simulator); [Suh et al.](suh-diffsim-gradients.md) (empirical bias of FOBG under stiff contact); [PIPPS](pipps.md) / Parmas et al. 2023 (ESNR).
- Extended by: [PWM](pwm.md) (same first author; moves from differentiable simulation to learned world models, arguing that smoothness of the model rather than truncation should address contact).
- Contrasts with: [Onoda et al.](onoda-composite-gradients.md), who find that per-step variance control suffices on similar dflex tasks.

## 6. Relevance to TGI

TGI's event-gated cut realises the AHAC-1 idea in a form that vectorises, which AHAC itself did not achieve. Rather than shortening a shared horizon, TGI keeps a fixed 32-step window and, per rollout and per step, replaces the continuation return $G_{t+1}$ by the critic $V(s_{t+1})$ wherever an imagined touchdown occurs with unsupported sole share above 0.3 (`G_t = r_t + γ(1−d_t)((1−l_t)V(s_{t+1}) + l_t G_{t+1})`, with $l_t=0$ at such events). Because the mask is applied inside a fixed-length recursion, graph length is constant and no asynchrony arises. Two further differences are substantive. The trigger is geometric and semantic (a foot landing on poorly supported terrain) rather than a threshold on the contact-force Jacobian; TGI does not cut at ordinary touchdowns on supported terrain, relying on its learned model's smoothed contact there, in line with [PWM](pwm.md). And TGI keeps the touchdown transition itself in the gradient through the critic's input, whereas AHAC-1 omits the stiff Jacobian entirely.
