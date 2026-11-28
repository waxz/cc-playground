# DSS: Diagonal State Spaces are as Effective as Structured State Spaces

| | |
|---|---|
| Authors | Ankit Gupta, Albert Gu, Jonathan Berant |
| Venue / year | NeurIPS 2022 |
| Link | https://arxiv.org/abs/2203.14343 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (DAWIN's "S4D" citation [7]) |
| Source | full text read |

## 1. Problem and position in the lineage

[S4](s4.md)'s diagonal-plus-low-rank machinery is hard to implement and analyse. DSS shows that dropping the low-rank term and using a purely diagonal complex state matrix, initialised from the eigenvalues of the normal part of HiPPO, matches S4. DAWIN cites this paper for "S4D"; the diagonal-SSM line (DSS, S4D, [S5](s5.md), [LRU](lru.md)) is the structure that DAWIN's recurrence instantiates.

## 2. Method

ZOH discretisation of $\dot x=Ax+Bu$ with step $\Delta$ (input held constant over the sample interval):

$$\bar A=e^{A\Delta},\qquad \bar B=(\bar A-I)A^{-1}B,\qquad \bar K_k=Ce^{Ak\Delta}(e^{A\Delta}-I)A^{-1}B.$$

Proposition 1: any diagonalisable $A$ with nonzero eigenvalues gives a kernel expressible by a diagonal system, (a) $\bar K=\tilde w\,\Lambda^{-1}(e^{\Lambda\Delta}-I)\,\mathrm{elementwise\text{-}exp}(P)$ with $P_{ik}=\lambda_ik\Delta$, or (b) with a row-softmax normalisation. DSS-EXP constrains $\mathrm{Re}\,\lambda=-e^{\Lambda_{re}}<0$; DSS-SOFTMAX allows any sign and normalises. The recurrent view is $x_{i,k}=e^{\lambda_i\Delta}x_{i,k-1}+\lambda_i^{-1}(e^{\lambda_i\Delta}-1)u_k$. $\Delta=e^{\Delta_{\log}}$ per channel, initialised log-uniform in $[0.001,0.1]$; $\Lambda$ uses "Skew-HiPPO" initialisation.

## 3. Evidence

LRA average 81.88 (DSS-SOFTMAX) and 81.18 (DSS-EXP) vs S4 80.21 in the same pipeline; Path-X 87.8 / 85.6 vs 88.0 (Table 1). Raw Speech Commands 98.2 % (DSS-EXP) vs S4 98.1 %. Random initialisation of $\Lambda$ is worse, and truncating kernels to length 128 hurts long tasks. Crucially, **DSS-EXP-NO-SCALE**, which omits the ZOH input factor $\Lambda^{-1}(e^{\Lambda\Delta}-I)$, falls to 65.03 average and fails Path-X.

## 4. Critical analysis

The NO-SCALE ablation is direct evidence that the ZOH input scaling is not cosmetic: without it the effective input gain is not matched to the decay rate, the state's magnitude grows as $1/(1-|\bar\lambda|)$ for slowly decaying modes, and long-range training fails. [LRU](lru.md) later reached the same conclusion ("discretisation performs normalisation") from the RNN side. For [DAWIN](dawin.md), which uses $1-e^{-\gamma\Delta t}$ (the ZOH numerator for the real part of the pole, without division by $\lambda$), the ablation implies that this factor carries most of the normalising effect; the missing $1/\lambda$ only rescales by a learnable constant per unit, but the missing imaginary part changes the steady-state gain with $\Delta t$ (see the theme report). DSS uses a single $\Delta$ per channel and a convolutional kernel, so it handles neither irregular sampling nor input-dependent steps. The S4 comparison uses official S4 hyperparameters while DSS used a larger learning rate for $\Delta_{\log}$, which the authors note explains the Text gap.

## 5. Cross-references

- Builds on: [S4](s4.md).
- Parallel / extended by: [S5](s5.md) (diagonal MIMO, scan, variable $\Delta t$), [LRU](lru.md) (direct discrete parameterisation and $\gamma$ normalisation), [Mamba](mamba.md) (selective diagonal SSM).
- Related: [DAWIN](dawin.md) (diagonal complex transition with a real-pole ZOH input factor).

## 6. Relevance to the study question

Question 3: family (4). It gives the cleanest formula for ZOH discretisation of a diagonal complex system and the decisive ablation showing that the input factor $(e^{\lambda\Delta}-1)/\lambda$ is necessary for long-range learning; this is the formula against which DAWIN's $\alpha$ is compared.
