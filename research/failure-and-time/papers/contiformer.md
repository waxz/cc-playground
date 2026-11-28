# ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

| | |
|---|---|
| Authors | Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li |
| Venue / year | NeurIPS 2023 (arXiv February 2024) |
| Link | https://arxiv.org/abs/2402.10635 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (main text, efficiency appendix) |

## 1. Problem and position in the lineage

Transformers for irregular series usually add a time encoding to tokens ([Time2Vec](time2vec.md)-style or kernel-based, as in [mTAN](mtan.md)); ODE models capture continuous dynamics but are recurrent. ContiFormer puts Neural-ODE trajectories inside attention: each key and value becomes a function of time, and attention is an integral over time rather than a dot product at one instant. It is DAWIN's representative of "continuous-time attention."

## 2. Method

For observation $i$ with projected key $K_i$ and value $V_i$ at time $t_i$,

$$k_i(t)=K_i+\int_{t_i}^tf(\tau,k_i(\tau);\theta_k)\,d\tau,\qquad v_i(t)=V_i+\int_{t_i}^tf(\tau,v_i(\tau);\theta_v)\,d\tau,$$

a natural cubic spline gives a continuous query $q(t)$ through the $Q_i$, and

$$\alpha_i(t)=\frac{\int_{t_i}^tq(\tau)k_i(\tau)^\top d\tau}{t-t_i},\qquad \hat v_i(t)=\frac{\int_{t_i}^tv_i(\tau)d\tau}{t-t_i},\qquad \mathrm{CT\text{-}ATTN}(t)=\sum_i\mathrm{softmax}_i\big(\alpha_i(t)/\sqrt{d_k}\big)\hat v_i(t),$$

with $\alpha_i(t_i)=q(t_i)k_i(t_i)^\top$ by continuity. Integrals are evaluated at reference times by mapping each interval to $[-1,1]$ and using Gauss–Legendre quadrature with all $N^2$ ODEs solved in one RK4 call (step 0.1, so $S=80$ evaluations), giving $O(N^2Sd^2)$ work with $O(S)$ sequential depth. A theorem shows that suitable $k_i(\cdot)$ reproduce any attention matrix, so time-embedding and kernelised Transformers are special cases.

## 3. Evidence

Spirals: interpolation RMSE ×10⁻² 0.49 vs Transformer 1.37 and latent ODE 2.09. Twenty UEA datasets with 30/50/70 % dropped (Table 3): average accuracy 0.8126 / 0.7997 / 0.7749 vs TST 0.8089 / 0.7793 / 0.7297, S5 0.7854 / 0.7638 / 0.7401, mTAN 0.7381 / 0.7118 / 0.6955, Neural CDE 0.7142 / 0.6929 / 0.6753. Marked point processes: best log-likelihood on most of six datasets (e.g. Traffic 0.635 vs SAHP 0.372). Regular forecasting: best on ILI only (MSE 2.632 vs TimesNet 2.874). Cost on RacketSports (Table 6): 1.88× S5's training time at step 0.1 (TST 0.14×, ODE-RNN 2.47×, Neural CDE 3.48×); about four times slower than ODE-RNN at input length 1,000 and markedly higher GPU memory.

## 4. Critical analysis

The quadratic number of ODE systems makes ContiFormer the most expensive model in this theme; the authors' own efficiency appendix shows memory growing sharply with length. Its gains over TST (a discrete Transformer with time features) are 0.4–4.5 points of average accuracy, which bounds what the continuous machinery adds over a well-tuned time-as-feature Transformer. The "continuous" output holds only up to solver error and with dropout disabled (footnote 3). The query spline is non-causal, as in [Neural CDE](neural-cde.md). There is no PhysioNet-2012 mortality evaluation, so it cannot serve as an AUROC reference for [DAWIN](dawin.md). Normalising integrals by $t-t_i$ removes dependence on gap length for the averaged quantities, which stabilises training but also removes one way elapsed time could matter.

## 5. Cross-references

- Builds on: [Neural ODE](neural-ode.md), [Neural CDE](neural-cde.md) (spline query), [Transformer](transformer.md).
- Contrasts with: [mTAN](mtan.md) (attention driven only by time embeddings), [S5](s5.md) (linear, cheap, used as its efficiency reference).
- Cited by: [DAWIN](dawin.md) as the continuous-time attention baseline (not run in its experiments).

## 6. Relevance to the study question

Question 3: family (3), continuous-time attention. It establishes the upper end of the cost range and shows that a discrete Transformer with time features is a strong, much cheaper baseline.
