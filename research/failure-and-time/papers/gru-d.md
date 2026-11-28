# GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values

| | |
|---|---|
| Authors | Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu |
| Venue / year | arXiv 2016; Scientific Reports 8:6085, 2018 |
| Link | https://arxiv.org/abs/1606.01865 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (arXiv v2) |

## 1. Problem and position in the lineage

Clinical time series are sampled irregularly and with "informative missingness": whether and when a variable was measured correlates with the label. Before GRU-D the common remedies were imputation (mean, forward fill) or concatenating the mask and the elapsed time to the RNN input. GRU-D is the earliest paper in this corpus to turn elapsed time into a learned *decay gate* acting on both inputs and hidden state, and it is the reference "time-aware RNN" baseline in [Latent ODE](latent-ode.md), [GRU-ODE-Bayes](gru-ode-bayes.md), [Neural CDE](neural-cde.md), [mTAN](mtan.md) and [ContiFormer](contiformer.md).

## 2. Method

With mask $m_t^d$ and per-variable time since last observation $\delta_t^d$ (accumulated over missing steps), the decay rates are

$$\gamma_t=\exp\{-\max(0,\,W_\gamma\delta_t+b_\gamma)\}\in(0,1].$$

The input decay (diagonal $W_{\gamma_x}$) pulls a missing variable from its last observation toward the empirical mean, $x_t^d\leftarrow m_t^dx_t^d+(1-m_t^d)\big(\gamma_{x,t}^dx_{t'}^d+(1-\gamma_{x,t}^d)\tilde x^d\big)$; the hidden decay (full $W_{\gamma_h}$) shrinks the state before the GRU update, $h_{t-1}\leftarrow\gamma_{h,t}\odot h_{t-1}$; and the mask is fed to all gates. The baselines it defines are the standard taxonomy of "time as a feature": GRU-mean, GRU-forward and GRU-simple $[x_t;m_t;\delta_t]$.

## 3. Evidence

PhysioNet Challenge 2012 (set A, 4,000 ICU stays, 33 variables, hourly binning for the non-RNN baselines) and MIMIC-III (19,714 admissions, 99 features), 5-fold cross-validation. In-hospital mortality AUC (Table 2): GRU-D 0.8424 ± 0.012 on PhysioNet and 0.8527 ± 0.003 on MIMIC-III, against GRU-simple 0.8155 and 0.8380, GRU-forward 0.8162 and 0.8252, and the best non-RNN baseline SVM-simple 0.8277 on PhysioNet. On four PhysioNet tasks jointly GRU-D reaches 0.8370 average AUC vs 0.8249 for GRU-simple (Table 1). On synthetic Gesture data GRU-simple fails when missingness is weakly correlated with the label, while GRU-D does not. Early prediction on MIMIC-III: GRU-D at 36 h matches the best non-RNN model at 48 h.

## 4. Critical analysis

The decay is monotone and fixed in form: the hidden state can only relax toward zero (and inputs toward a dataset mean), which encodes homeostasis but cannot represent oscillation or growth between observations, the limitation [Latent ODE](latent-ode.md) targets. The $\max(0,\cdot)$ makes the decay exactly 1 for small $W\delta+b$, so short gaps are ignored entirely and the gradient vanishes there. The gains over GRU-simple are 0.01–0.03 AUC with standard deviations of about 0.01, and later papers measure a lower PhysioNet AUC for GRU-D under their own protocols (0.818 ± 0.008 in [Latent ODE](latent-ode.md) and [mTAN](mtan.md)), which shows that preprocessing (hourly binning vs minute resolution, train/test split) moves results as much as the architecture. Mozer et al. (2017), cited in [Latent ODE](latent-ode.md), found that exponential decay alone did not improve over standard RNNs; GRU-D's benefit appears to come as much from the mask features and mean-reverting input imputation as from time itself. Mathematically the hidden decay is a first-order linear ODE $\dot h=-\lambda h$ solved exactly over $\delta$, i.e. the real-valued special case of the diagonal transition $e^{\lambda\Delta t}$ used by [S5](s5.md) and [DAWIN](dawin.md), but with $\lambda$ an input-dependent function of the gap, foreshadowing [Mamba](mamba.md)'s selective step.

## 5. Cross-references

- Extended by: [Latent ODE](latent-ode.md) (ODE-RNN replaces fixed decay with learned dynamics), [GRU-ODE-Bayes](gru-ode-bayes.md) (continuous GRU between observations).
- Contrasts with: [Neural CDE](neural-cde.md) (data enter through the driving path, not through gates), [mTAN](mtan.md) (attention over time embeddings instead of recurrence).
- Related: [DAWIN](dawin.md) adds rotation $e^{-i\omega\Delta t}$ and a ZOH input gain to the same exponential decay.

## 6. Relevance to the study question

Question 3: GRU-D is the canonical instance of family (2), learned decay gates, and its baselines define family (1), time as an input feature. Its PhysioNet-2012 AUROC (0.8424) is one of the reference values used in the theme report to assess DAWIN's 0.9993 claim.
