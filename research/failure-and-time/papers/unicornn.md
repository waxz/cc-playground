# UnICORNN: A Recurrent Model for Learning Very Long Time Dependencies

| | |
|---|---|
| Authors | T. Konstantin Rusch, Siddhartha Mishra |
| Venue / year | ICML 2021 |
| Link | https://arxiv.org/abs/2103.05487 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

The exploding and vanishing gradient problem limits RNNs on very long sequences, and BPTT memory grows with length. UnICORNN derives a recurrent cell from an undamped second-order ODE of independent driven oscillators and discretises it with a structure-preserving (symplectic) integrator, so that the hidden-state gradients obey a Hamiltonian system that cannot have an attracting fixed point at zero. It is the nonlinear predecessor of [LinOSS](linoss.md) and represents the oscillatory branch of family (4) in DAWIN's related work.

## 2. Method

$$y''=-[\sigma(w\odot y+Vu+b)+\alpha y],\qquad \text{i.e.}\quad y'=z,\;z'=-[\sigma(w\odot y+Vu+b)+\alpha y],$$

a Hamiltonian system with $H=\frac\alpha2\|y\|^2+\frac12\|z\|^2+\sum_iw_i^{-1}\log\cosh(w_iy_i+(Vu)_i+b_i)$. Symplectic Euler with a learned per-neuron step $\Delta t\,\hat\sigma(c)$, $\hat\sigma(c)=\tfrac12+\tfrac12\tanh(c/2)\in(0,1)$, gives, per layer $\ell$,

$$y_n^\ell=y_{n-1}^\ell+\Delta t\,\hat\sigma(c^\ell)\odot z_n^\ell,\qquad z_n^\ell=z_{n-1}^\ell-\Delta t\,\hat\sigma(c^\ell)\odot[\sigma(w^\ell\odot y_{n-1}^\ell+V^\ell y_n^{\ell-1}+b^\ell)+\alpha y_{n-1}^\ell].$$

The map is invertible, so states are recomputed backward and memory is $O(Nbd+Lbm)$. Proposition 3.1 bounds gradients polynomially, as $(N\Delta t)^3$; the vanishing-gradient analysis uses volume preservation.

## 3. Evidence

psMNIST 98.4 % (256 units, 135 k parameters) vs coRNN 97.3 % and LSTM 92.9 % (Table 1). EigenWorms (length 17,984): 90.3 ± 3.0 % with 1.5 k parameters, vs coRNN 86.7 %, signature Neural CDE 77.8 %, expRNN 40.0 % (Table 3). Vital-sign regression from PPG and noise-padded CIFAR-10 also favour UnICORNN, and the CUDA implementation is reported faster than cuDNN LSTM.

## 4. Critical analysis

UnICORNN uses a fixed nominal $\Delta t$ (e.g. $2.81\times10^{-5}$ on the first EigenWorms layer) chosen as a hyperparameter; the "time step" is a numerical device to set the oscillators' effective frequency and timescale, not physical elapsed time. With symplectic Euler, a variable physical step would be admissible (the method remains symplectic per step), but the paper never tests irregular sampling. The recurrence is nonlinear, so the parallel scan of [S5](s5.md) does not apply; speed comes from per-neuron independence. The absence of damping is deliberate for long memory, but it means information is never forgotten except by phase mixing, which suits classification over very long sequences more than filtering of nonstationary signals. Baselines are RNNs; no SSM or Transformer comparisons appear. The gradient bounds require $\Delta t\ll1$ and bounded weights.

## 5. Cross-references

- Builds on: coRNN (Rusch and Mishra 2021), IndRNN.
- Extended by: [LinOSS](linoss.md) (linear oscillators, implicit and IMEX steps, parallel scan).
- Contrasts with: [LRU](lru.md) (stability by $|\lambda|<1$ plus normalisation), [DAWIN](dawin.md) (first-order complex damped rotation rather than second-order real oscillators).

## 6. Relevance to the study question

Question 3: oscillatory recurrences in family (4). It shows that second-order structure gives long memory with bounded gradients, and highlights that "$\Delta t$" in such cells is usually a learned timescale rather than a channel for physical time.
