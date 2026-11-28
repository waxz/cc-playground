# Latent ODE: Latent ODEs for Irregularly-Sampled Time Series

| | |
|---|---|
| Authors | Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud |
| Venue / year | NeurIPS 2019 |
| Link | https://arxiv.org/abs/1907.03907 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Between observations a standard RNN's state is constant or undefined, and the decay RNNs ([GRU-D](gru-d.md)) force it to relax monotonically toward zero. This paper generalises decay to learned dynamics: the ODE-RNN evolves the hidden state by a [Neural ODE](neural-ode.md) between observations and updates it with a GRU at each one; used as the encoder of the latent ODE of Chen et al., it gives a fully ODE-based sequence-to-sequence model. It also provides the most widely reused PhysioNet-2012 protocol (minute resolution, 8,000 stays).

## 2. Method

ODE-RNN (Algorithm 1): $h_i'=\mathrm{ODESolve}(f_\theta,h_{i-1},(t_{i-1},t_i))$, $h_i=\mathrm{RNNCell}(h_i',x_i)$. The paper makes explicit that the decay RNN $h_{i-1}e^{-\tau\Delta t}$ is the exact solution of $\dot h=-\tau h$, so ODE-RNN strictly contains it (Table 2). Latent ODE: $z_0\sim p(z_0)$, $z_{0..N}=\mathrm{ODESolve}(f_\theta,z_0,(t_0..t_N))$, $x_i\sim p(x_i|z_i)$, with $q(z_0|\{x_i,t_i\})$ produced by an ODE-RNN run backward, trained by the ELBO; a Poisson-process likelihood $\sum_i\log\lambda(t_i)-\int\lambda(t)dt$ with $\lambda=\lambda(z(t))$ can model observation times jointly.

## 3. Evidence

MuJoCo Hopper (10,000 trajectories, 100 steps), interpolation MSE ×10⁻² with 10 % observed: latent ODE (ODE encoder) 0.360, ODE-RNN 1.647, GRU-D 1.968, RNN-Δt 2.454 (Table 3). In extrapolation all autoregressive models are poor (ODE-RNN 13.5, GRU-D 38.1 at 10 %), while encoder–decoder models are good (latent ODE 1.441). PhysioNet-2012 interpolation MSE ×10⁻³: latent ODE 2.118, ODE-RNN 2.361, RNN-Decay 3.215, GRU-D 3.384. In-hospital mortality AUC (Table 6): RNN-Δt 0.787, RNN-Impute 0.764, RNN-Decay 0.807, GRU-D 0.818, ODE-RNN 0.833, latent ODE (ODE encoder) 0.829, latent ODE + Poisson 0.826, RNN-VAE 0.515 (13.75 % positive class). Human Activity per-time-point accuracy: latent ODE 0.846, ODE-RNN 0.829. Cost: ODE-RNN takes 60 % longer than a GRU and the latent ODE about twice the ODE-RNN.

## 4. Critical analysis

The authors themselves observe that ODE-RNN, latent ODE and GRU-D reach similar mortality AUC and suggest that "modelling dynamics between time points does not make a difference for binary classification of the full time series." That is a sober and important result for the theme: the benefit of continuous-time state shows up in interpolation and sparse-data regression, not in sequence-level classification. The Poisson term did not help classification either. Extrapolation results show that the choice between autoregressive and encoder–decoder training matters more than the time embedding; ODE-RNN extrapolates worse than a plain RNN in two of four columns. Rounding PhysioNet times to the minute still yields 2,880 possible times per stay, and the ODE must be solved over the union of batch time points, so cost does not shrink with sparsity, unlike decay RNNs. The latent ODE assumes a deterministic trajectory given $z_0$, which is apt for MuJoCo free flight but not for systems driven by unobserved inputs; [GRU-ODE-Bayes](gru-ode-bayes.md) and [Neural CDE](neural-cde.md) relax this.

## 5. Cross-references

- Builds on: [Neural ODE](neural-ode.md), [GRU-D](gru-d.md) (decay as a special case).
- Extended by / contemporaneous: [GRU-ODE-Bayes](gru-ode-bayes.md), [Neural CDE](neural-cde.md) (removes the interrupt-at-observation structure), [mTAN](mtan.md) (reuses the PhysioNet protocol and beats it at 0.858).
- Contrasts with: [S5](s5.md) (linear closed-form transitions with per-step $\Delta t$).

## 6. Relevance to the study question

Question 3: family (3). Its PhysioNet table (0.764–0.833 AUROC) is the main external reference for judging DAWIN's 0.9993, and its finding that continuous dynamics do not help whole-sequence classification bounds what any time embedding can contribute on that benchmark.
