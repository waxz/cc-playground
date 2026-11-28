# Neural ODE: Neural Ordinary Differential Equations

| | |
|---|---|
| Authors | Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud |
| Venue / year | NeurIPS 2018 (arXiv v6, October 2026, with a correction) |
| Link | https://arxiv.org/abs/1806.07366 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

A residual block $h_{t+1}=h_t+f(h_t,\theta_t)$ is an Euler step. Neural ODEs take the limit and define the hidden state as the solution of $\dot h(t)=f(h(t),t,\theta)$ computed by a black-box adaptive solver. For time series the consequence is that the state is defined at every real time, so irregular observation times need no binning. This paper is the root of the solver-based branch of the taxonomy ([Latent ODE](latent-ode.md), [GRU-ODE-Bayes](gru-ode-bayes.md), [LTC](ltc.md), [Neural CDE](neural-cde.md), [ContiFormer](contiformer.md)).

## 2. Method

For a loss $L(z(t_1))$ with $z(t_1)=z(t_0)+\int_{t_0}^{t_1}f(z,t,\theta)\,dt$, the adjoint $a(t)=\partial L/\partial z(t)$ obeys

$$\dot a(t)=-a(t)^\top\frac{\partial f}{\partial z},\qquad \frac{dL}{d\theta}=-\int_{t_1}^{t_0}a(t)^\top\frac{\partial f}{\partial\theta}\,dt,$$

solved backward together with a re-integration of $z$, so memory is $O(1)$ in the number of solver steps (Algorithm 1). When the loss depends on several observation times, the backward solve is split at each and the adjoint jumps by $\partial L/\partial z(t_i)$. Continuous normalising flows follow from $\partial\log p/\partial t=-\mathrm{tr}(\partial f/\partial z)$. The time-series model is generative: $z_{t_0}\sim p(z_{t_0})$, $z_{t_1..t_N}=\mathrm{ODESolve}(z_{t_0},f,\theta,t_0..t_N)$, $x_{t_i}\sim p(x|z_{t_i})$, with an RNN recognition network run backward in time and a VAE objective, optionally with an inhomogeneous Poisson-process likelihood for observation times.

## 3. Evidence

MNIST: ODE-Net 0.42 % error with 0.22 M parameters vs ResNet 0.41 % (0.60 M), at $O(1)$ memory. Error is controllable through solver tolerance, and the number of function evaluations (NFE) grows during training. The 2026 revision corrects an earlier claim: the backward pass needs about 0.8 times as many evaluations as the forward pass (median, tolerances $10^{-5}$–$10^{-1}$), not about half; the original figure came from a logging bug. Spirals with irregular samples: predictive RMSE 0.1642 (latent ODE) vs 0.3937 (RNN) with 30 of 100 points, 0.1346 vs 0.1813 with all 100.

## 4. Critical analysis

The time-series evidence is a toy (2-D spirals, a 4-D latent). The paper itself lists the costs that later work confirms: solver tolerances must be chosen for both passes, minibatching forces solving at the union of all time points, and reconstructing the trajectory backward can drift (checkpointing is the remedy). Adjoint gradients are only as accurate as the backward solve; [LTC](ltc.md) abandons them for BPTT for this reason. The key structural limitation for sequence data, pointed out by [Neural CDE](neural-cde.md), is that an ODE trajectory is determined by its initial condition: new observations cannot enter except by interrupting the solve, as in [Latent ODE](latent-ode.md)'s ODE-RNN. The correction of the NFE ratio is a reminder that claimed efficiency advantages of continuous models need independent re-measurement. In the taxonomy, a Neural ODE embeds time not as a feature but as the *integration variable*: elapsed time determines how far the state flows, and extrapolation in $\Delta t$ is handled by integrating longer, which is principled only if $f$ is correct outside the training range.

## 5. Cross-references

- Extended by: [Latent ODE](latent-ode.md) (ODE-RNN encoder), [GRU-ODE-Bayes](gru-ode-bayes.md) (GRU-shaped vector field with Bayesian jumps), [Neural CDE](neural-cde.md) (data-driven vector field, adjoint across observations), [ContiFormer](contiformer.md).
- Contrasts with: [S4](s4.md)/[S5](s5.md) (linear dynamics solved in closed form, no solver), [LTC](ltc.md) (fused fixed-step solver, BPTT).
- Background: adjoint sensitivity, in the theme report's Mathematical background.

## 6. Relevance to the study question

Question 3: the founding instance of family (3), continuous-time hidden states. It supplies the adjoint method used in the theme report and defines the solver cost that diagonal SSMs and DAWIN avoid by restricting the dynamics to be linear.
