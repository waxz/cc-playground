# LinOSS: Oscillatory State-Space Models

| | |
|---|---|
| Authors | T. Konstantin Rusch, Daniela Rus |
| Venue / year | ICLR 2025 (arXiv v3, June 2025) |
| Link | https://arxiv.org/abs/2410.03943 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

First-order diagonal SSMs need careful constraints on their eigenvalues ($\mathrm{Re}\,\lambda<0$, HiPPO or ring initialisation, normalisation) to be stable and to remember long histories. LinOSS builds an SSM from forced *linear* harmonic oscillators, the linear and scan-compatible version of [UnICORNN](unicornn.md), and shows that stability then needs only a non-negative diagonal stiffness.

## 2. Method

$$y''(t)=-Ay(t)+Bu(t)+b,\qquad x(t)=Cy(t)+Du(t),\qquad A=\mathrm{diag}(A_{kk}\ge0).$$

With $z=y'$, the implicit discretisation $z_n=z_{n-1}+\Delta t(-Ay_n+Bu_n)$, $y_n=y_{n-1}+\Delta tz_n$ gives $x_n=M^{IM}x_{n-1}+F_n^{IM}$ with $M^{IM}$ computed in $O(m)$ via the Schur complement $S=(I+\Delta t^2A)^{-1}$. Proposition 3.1: the eigenvalues of $M^{IM}$ have $|\lambda|^2=S_{kk}=1/(1+\Delta t^2A_{kk})\le1$. The implicit–explicit (IMEX) variant uses $y_{n-1}$ in the first equation, is symplectic and time-reversible, and has $|\lambda|=1$. Both are computed with the associative scan $(a_1,a_2)\bullet(b_1,b_2)=(b_1a_1,\,b_1a_2+b_2)$ on $2\times2$ blocks of diagonals. Universality for continuous causal operators is proved.

## 3. Evidence

Six long UEA classification sets (lengths 405–17,984), same tuning protocol for all models (Table 1): average accuracy LinOSS-IM 67.8 %, LinOSS-IMEX 65.0 %, Log-NCDE 64.3 %, S6 62.0 %, S5 61.8 %, LRU 61.7 %, NCDE 60.2 %, Mamba 58.6 %; EigenWorms 95.0 % for LinOSS-IM vs 87.8 % (LRU). PPG-DaLiA heart-rate regression with windows of 49,920 samples (Table 2): MSE ×10⁻² 6.4 (IM) and 7.5 (IMEX) vs Log-NCDE 9.56, Mamba 10.65, LRU 12.17, S5 12.63, NCDE 13.54. Weather, 720-step forecasting: MAE 0.508 (IMEX) vs S4 0.578.

## 4. Critical analysis

The damping in LinOSS-IM is a numerical property of the implicit integrator: $|\lambda|^2=1/(1+\Delta t^2A)$ depends on the step, so the same continuous oscillator forgets at a rate that changes with $\Delta t$. Unlike the exact exponential transition of [S5](s5.md) or [DAWIN](dawin.md), LinOSS is therefore not a consistent representation of one continuous system across sampling rates, and the paper fixes $\Delta t$ as a (learned or tuned) hyperparameter without testing irregular intervals. The evidence is strong within its protocol (same tuning grid for all models, five splits), and it is one of the few benchmarks in the corpus where Mamba, S5, LRU and Neural CDEs are compared fairly on real long time series; Mamba is last on the UEA average. The oscillatory inductive bias suits signals with rhythmic structure (PPG, EEG, gait), and second-order dynamics are the natural form of mechanical systems, but the paper evaluates no control or robotics task.

## 5. Cross-references

- Builds on: [UnICORNN](unicornn.md), [S5](s5.md) (parallel scan).
- Compared with: [Mamba](mamba.md), [LRU](lru.md), [S5](s5.md), [Neural CDE](neural-cde.md).
- Contrasts with: [DAWIN](dawin.md) (first-order complex oscillator with exact decay $e^{-\gamma\Delta t}$ rather than integrator-induced damping).

## 6. Relevance to the study question

Question 3: oscillatory recurrences in family (4). It provides the fairest recent benchmark of the scan-based models on long real-valued series, and its step-dependent damping illustrates why exact (ZOH) discretisation is preferable when $\Delta t$ varies.
