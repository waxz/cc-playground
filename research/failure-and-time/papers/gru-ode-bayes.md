# GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series

| | |
|---|---|
| Authors | Edward De Brouwer, Jaak Simm, Adam Arany, Yves Moreau |
| Venue / year | NeurIPS 2019 |
| Link | https://arxiv.org/abs/1905.12374 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

"Sporadic" data are irregular both in time and across dimensions. The paper frames the observations as noisy samples of a process driven by an SDE $dY=\mu(Y)dt+\sigma(Y)dW$ and learns the evolution of the mean and covariance of $Y(t)$ with a filter: a continuous-time GRU propagates the state between observations and a GRU-based Bayesian update absorbs each observation. It is contemporaneous with [Latent ODE](latent-ode.md)'s ODE-RNN, and differs by its GRU-shaped vector field and its explicitly probabilistic update loss.

## 2. Method

Rewriting the GRU update $h_t=z_t\odot h_{t-1}+(1-z_t)\odot g_t$ as a difference equation and taking the continuum limit gives GRU-ODE:

$$\frac{dh}{dt}=(1-z(t))\odot(g(t)-h(t)),$$

with $z,g,r$ the continuous counterparts of the gates. Because $z\in(0,1)$ and $g\in(-1,1)$, $h$ stays in $[-1,1]$ and the field is Lipschitz with constant 2 (a "continuity prior"). At an observation, GRU-Bayes jumps $h(t^-)\to h(t^+)=\mathrm{GRU}(h(t^-),f_{\text{prep}}(y,m,h(t^-)))$. The loss is the masked negative log-likelihood before the jump plus $\lambda$ times the KL divergence between an analytic Gaussian Bayes update of the pre-jump prediction with the observation and the post-jump prediction. Minibatching iterates over the sorted union of observation times; Euler, midpoint or Dormand–Prince solvers are supported.

## 3. Evidence

USHCN daily climate (1,114 stations, ≈60 observations per station over 4 years) and MIMIC-III (21,250 patients, 96 variables, forecast the next three measurements after 36 h). MSE / NegLL (Table 1): GRU-ODE-Bayes 0.43 / 0.84 on USHCN and 0.48 / 0.83 on MIMIC-III, vs GRU-D 0.53 / 0.99 and 0.79 / 1.16, T-LSTM 0.59 / 1.67 and 0.62 / 1.02, NeuralODE-VAE 0.96 / 1.46 and 0.89 / 1.35. Small-sample ablation (Table 2, MIMIC-III MSE): with 1,000 patients GRU-ODE-Bayes 0.77 vs a discretised autonomous GRU 0.87; with the full data the discretised version is as good (0.46 vs 0.48, not significantly different). The model represents the Fokker–Planck dynamics of a multivariate Ornstein–Uhlenbeck process exactly (Appendix I).

## 4. Critical analysis

The most informative result is the ablation: the continuous-time cell helps only when data are scarce, and with full MIMIC-III a discretised GRU matches it. This parallels [Latent ODE](latent-ode.md)'s observation on classification and suggests that the advantage of continuous time is mainly an inductive bias (continuity, boundedness) whose value shrinks with data. The comparison set omits [Latent ODE](latent-ode.md)'s ODE-RNN, the closest competitor. The Gaussian observation model and diagonal covariance limit the claimed SDE generality, and the analytic KL target assumes the observation noise variance is known or learnt consistently. Cost scales linearly with the number of observations, but the ODE must still be integrated across all gaps for every series in the batch. As a time embedding, GRU-ODE uses elapsed time only as integration length; the vector field is autonomous unless a continuous input is supplied.

## 5. Cross-references

- Builds on: [Neural ODE](neural-ode.md), [GRU-D](gru-d.md) (baseline).
- Contemporaneous: [Latent ODE](latent-ode.md) (ODE-RNN).
- Contrasted by: [Neural CDE](neural-cde.md), which proves that ODEs of the form $\dot z=h(z,X_t)$, including "GRU-ODE" with $X$ as input, are strictly less expressive than CDEs, and finds GRU-ODE consistently weak.
- Related: [LTC](ltc.md) (bounded input-dependent time constants).

## 6. Relevance to the study question

Question 3: family (3), the filtering variant (continuous prediction, discrete Bayesian correction), which is the structure of a Kalman filter and the natural template for asynchronous robot sensors. Its small-data ablation is evidence for recommending continuous-time cells when data are scarce rather than by default.
