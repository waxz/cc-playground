# mTAN: Multi-Time Attention Networks for Irregularly Sampled Time Series

| | |
|---|---|
| Authors | Satya Narayan Shukla, Benjamin M. Marlin |
| Venue / year | ICLR 2021 |
| Link | https://arxiv.org/abs/2101.10318 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (corpus addition) |
| Source | full text read |

## 1. Problem and position in the lineage

ODE-based models ([Latent ODE](latent-ode.md)) handle irregular sampling but are slow, and RNN variants depend on imputation or decay heuristics. mTAN re-represents an irregular multivariate series at a fixed set of reference times by attention whose queries and keys are learned continuous-time embeddings, then processes the regular representation with a standard network. It was added because it is the strongest PhysioNet-2012 mortality result in the corpus under the [Latent ODE](latent-ode.md) protocol, needed to assess [DAWIN](dawin.md)'s 0.9993 claim, and because it is the canonical attention-based time embedding.

## 2. Method

Each of $H$ embeddings maps time to $d_r$ dimensions,

$$\phi_h(t)[0]=\omega_{0h}t+\alpha_{0h},\qquad \phi_h(t)[i]=\sin(\omega_{ih}t+\alpha_{ih}),\;0<i<d_r,$$

and for query time $t$ and the $L_d$ observations of dimension $d$,

$$\kappa_h(t,t_{id})=\mathrm{softmax}_i\!\Big(\frac{\phi_h(t)wv^\top\phi_h(t_{id})^\top}{\sqrt{d_k}}\Big),\qquad \hat x_{hd}(t)=\sum_i\kappa_h(t,t_{id})x_{id},\qquad \mathrm{mTAN}(t)[j]=\sum_{h,d}\hat x_{hd}(t)U_{hdj}.$$

$\hat x_{hd}$ is a learned kernel smoother in time per channel. An encoder (mTAN to reference points, then a bidirectional GRU) and decoder form a VAE (mTAND-Full); mTAND-Enc is the encoder plus a classifier.

## 3. Evidence

PhysioNet-2012 (37 variables, times rounded to the minute, 13.8 % positive): interpolation MSE ×10⁻³ 4.139 vs latent ODE (ODE encoder) 6.721 at 50 % observed (Table 1). Mortality AUC (Table 2): mTAND-Full 0.858 ± 0.004, mTAND-Enc 0.854 ± 0.001, Phased-LSTM 0.836, ODE-RNN 0.833, L-ODE-ODE 0.829, IP-Nets 0.819, GRU-D 0.818, RNN-Decay 0.807, SeFT 0.795, RNN-Δt 0.787, RNN-Impute 0.764, with 0.2 min per epoch vs 16.5 (ODE-RNN) and 22.0 (L-ODE-ODE). MIMIC-III mortality: 0.8544 (ODE-RNN 0.8561, not significantly different). Human Activity: 0.910 accuracy.

## 4. Critical analysis

mTAN's attention weights depend only on time embeddings, not on values; [ContiFormer](contiformer.md) points out that this makes its patterns input-independent and reports lower UEA accuracy for mTAN (0.738 at 30 % dropped). On the pendulum task in [S5](s5.md), mTAND's MSE is an order of magnitude worse than state-space models (65.64 vs 3.41 ×10⁻³), consistent with a kernel smoother being poorly suited to tracking dynamical state. The reference-point resampling sets a resolution, so very fine timing is smoothed out. The sinusoid-plus-linear embedding is Time2Vec's, so the same periodic-aliasing caveat applies. As evidence for the theme, Table 2 provides the most complete set of PhysioNet-2012 mortality AUROCs under one protocol: thirteen models span 0.515–0.858, and the best is 0.14 below DAWIN's reported 0.9993.

## 5. Cross-references

- Builds on: [Time2Vec](time2vec.md) (embedding form), [Transformer](transformer.md) (scaled dot-product attention), [Latent ODE](latent-ode.md) (protocol and baselines).
- Contrasts with: [ContiFormer](contiformer.md) (value-dependent continuous attention), [GRU-D](gru-d.md).
- Evaluated in: [S5](s5.md) (pendulum), [ContiFormer](contiformer.md) (UEA).

## 6. Relevance to the study question

Question 3: family (5) used inside attention. It is the reference point for PhysioNet-2012 AUROC, and shows that a cheap periodic time embedding plus attention is competitive for sequence-level clinical classification, where continuous dynamics add little.
