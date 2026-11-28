# Neural CDE: Neural Controlled Differential Equations for Irregular Time Series

| | |
|---|---|
| Authors | Patrick Kidger, James Morrill, James Foster, Terry Lyons |
| Venue / year | NeurIPS 2020 |
| Link | https://arxiv.org/abs/2005.08926 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

An ODE's solution is fixed by its initial condition, so ODE-based sequence models ([Latent ODE](latent-ode.md), [GRU-ODE-Bayes](gru-ode-bayes.md)) must interrupt the solve to inject each observation, which also prevents adjoint backpropagation across observations. Controlled differential equations, from rough-path theory, let a data path drive the state continuously. The Neural CDE is to an RNN what a Neural ODE is to a ResNet.

## 2. Method

Given observations $(t_i,x_i)$, build the natural cubic spline $X:[t_0,t_n]\to\mathbb R^{v+1}$ with $X_{t_i}=(x_i,t_i)$ and solve

$$z_t=z_{t_0}+\int_{t_0}^tf_\theta(z_s)\,dX_s=z_{t_0}+\int_{t_0}^tf_\theta(z_s)\frac{dX}{ds}\,ds,\qquad z_{t_0}=\zeta_\theta(x_0,t_0),$$

with $f_\theta:\mathbb R^w\to\mathbb R^{w\times(v+1)}$. Time must be a channel because CDEs are invariant to reparameterisation of the path ("tree-like invariance"): without $t$ in $X$, the model would be blind to the speed at which data arrive. The initial network breaks translation invariance. Partially observed channels are interpolated separately, and cumulative observation counts can be appended as "observational intensity" channels. Training uses the adjoint with $O(L+H)$ memory. Theorems: universal approximation, and any $\dot z=h_\theta(z,X_t)$ is representable as a CDE but not conversely.

## 3. Evidence

CharacterTrajectories with 30/50/70 % of points dropped: 98.7 / 98.8 / 98.6 % accuracy vs ODE-RNN 95.4 / 96.0 / 95.3 %, GRU-D 94.2 / 90.2 / 91.9 %, at 1.3 MB vs ≈15–17 MB memory (Table 1). PhysioNet 2019 sepsis (40,335 stays, 72 h, 5 % positive): AUC 0.880 with observational intensity vs GRU-Δt 0.878, ODE-RNN 0.874, GRU-D 0.871; without intensity the Neural CDE falls to 0.776, below GRU-D's 0.850 (Table 2). Speech Commands MFCC (regular): 89.8 ± 2.5 % vs ODE-RNN 65.9 ± 35.6 %, with baselines failing to train in some runs.

## 4. Critical analysis

The PhysioNet result is the 2019 sepsis challenge, not the 2012 mortality task that [DAWIN](dawin.md) benchmarks; the paper reports no PhysioNet-2012 AUROC. Its own numbers show the model's dependence on intensity channels: without them it loses to every baseline except GRU-ODE, and the authors note that the RNN baselines always receive $\Delta t$, so the comparison is not like-for-like. Natural cubic splines use future knots, so the model is non-causal as presented; online use needs other interpolants (later work by the same group). The authors report Neural CDEs to be about five times slower than RNNs and parameter-hungry ($O(\omega vw)$ in the final layer of $f$). Subsequent benchmarks in this corpus place Neural CDE low: average UEA accuracy 0.714 at 30 % dropped in [ContiFormer](contiformer.md) (below GRU-D), 60.2 % average and 13.54 MSE on PPG-DaLiA in [LinOSS](linoss.md). In [DAWIN](dawin.md)'s table Neural CDE gets 0.6974 mortality AUROC and NRMSE above 1 on several tasks, values that suggest under-tuning rather than intrinsic weakness, given its competitive literature results. As a time embedding, the CDE makes time a *driving coordinate*: the state changes in proportion to $dX$, which includes $dt$, and the Riemann–Stieltjes form makes it invariant to how the path is sampled, apart from interpolation error.

## 5. Cross-references

- Builds on: [Neural ODE](neural-ode.md), [Latent ODE](latent-ode.md) (ODE-RNN as the interrupted alternative).
- Contrasts with: [GRU-ODE-Bayes](gru-ode-bayes.md) (proved less expressive in the "GRU-ODE with input" form), [GRU-D](gru-d.md).
- Related: [ContiFormer](contiformer.md) uses the same spline for its continuous query; [S5](s5.md) and [LinOSS](linoss.md) outperform it at lower cost.

## 6. Relevance to the study question

Question 3: family (3), the input-driven variant. It contributes the cleanest statement of why time must be an explicit channel in a reparameterisation-invariant model and the memory-efficient adjoint across observations; its benchmarks show the price in speed and causality.
