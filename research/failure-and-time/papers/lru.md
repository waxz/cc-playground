# LRU: Resurrecting Recurrent Neural Networks for Long Sequences

| | |
|---|---|
| Authors | Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De |
| Venue / year | ICML 2023 |
| Link | https://arxiv.org/abs/2303.06349 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Deep SSMs ([S4](s4.md), [DSS](dss.md), [S5](s5.md)) are motivated by continuous-time theory, HiPPO initialisation and discretisation. LRU asks which of these ingredients is actually responsible for their long-range performance by starting from a vanilla RNN and adding changes one at a time. The answer is relevant to the theme because it separates the role of the continuous-time interpretation (and hence of $\Delta t$) from the role of parameterisation and normalisation.

## 2. Method

Steps: remove the recurrent nonlinearity; diagonalise the (complex) recurrence; parameterise eigenvalues as

$$\lambda_j=\exp\big(-\exp(\nu_j^{\log})+i\exp(\theta_j^{\log})\big),$$

which guarantees $|\lambda_j|<1$ and decouples magnitude and phase; initialise uniformly on a ring $r_{\min}\le|\lambda|\le r_{\max}$ near the unit circle (small phase for Path-X); and normalise the input by

$$x_k=\mathrm{diag}(\lambda)x_{k-1}+\gamma\odot(Bu_k),\qquad \gamma_j=\sqrt{1-|\lambda_j|^2}.$$

Proposition 3.3 explains the normalisation: under white-noise input $\mathbb E|x_\infty|^2=\mathbb E|Bu|^2/(1-|\lambda|^2)$, which blows up as $|\lambda|\to1$. No discretisation or $\Delta$ is used.

## 3. Evidence

Removing the recurrent nonlinearity improves accuracy (sCIFAR 69.9 → 72.2 %, ListOps 43.9 → 50.4 %; Table 1). The full LRU reaches sCIFAR 89.0, ListOps 60.2, Text 89.4, Retrieval 89.9, PathFinder 95.1 and Path-X 94.2 %, similar to S4D (91.5, 60.2, 86.4, 89.5, 94.2, 97.5) and S5 (88.8, 58.5, 86.2, 88.9, 95.7, 96.0) reproduced in the same pipeline (Table 3). Section 4 reports that replacing S4D's ZOH factor $(e^{\Delta A}-I)A^{-1}B$ by its first-order approximation $\Delta B$ matches performance, and argues that "discretisation performs normalisation": without the $\Delta$ factor the state grows as $O(\Delta^{-1})$.

## 4. Critical analysis

LRU shows that, for regularly sampled long-range benchmarks, the continuous-time story is not needed: an RNN parameterised directly in the discrete domain with stable exponential parameterisation and explicit normalisation matches the SSMs. Two consequences follow for time embedding. First, the ZOH input factor's main function on regular data is to normalise the state; DAWIN's $\alpha=1-e^{-\gamma\Delta t}$ is such a normaliser, for constant input, whereas LRU's $\gamma$ normalises for white-noise input. Second, a model that has no $\Delta$ cannot be resampled or fed irregular intervals: the price of LRU's simplicity is the loss of the zero-shot rate change shown by [S4](s4.md) and [S5](s5.md). Mapping LRU to irregular data requires reintroducing $\lambda^{\Delta t}$, i.e. returning to S5. The ablations use only LRA, and Path-X needed hand-restricted phase initialisation, which shows sensitivity to how oscillation frequencies are initialised, a concern shared by [DAWIN](dawin.md)'s $\omega_m\in[0.1,2\pi]$ initialisation.

## 5. Cross-references

- Builds on / analyses: [S4](s4.md), [DSS](dss.md), [S5](s5.md).
- Contrasts with: [Mamba](mamba.md) (input-dependent transition), [UnICORNN](unicornn.md) and [LinOSS](linoss.md) (stability from second-order structure rather than explicit normalisation).
- Related: [DAWIN](dawin.md) (DC-gain normalisation via a ZOH-like factor).

## 6. Relevance to the study question

Question 3: family (4) without discretisation. It provides the normalisation argument used in the theme report's analysis of DAWIN, and the counterpoint that continuous-time parameterisation is justified only when $\Delta t$ varies or may change at deployment.
