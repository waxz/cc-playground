# WFB: Wave Function Backpropagation with Explicit Temporal-Interval Dynamics

| | |
|---|---|
| Authors | Byunggu Yu, Justin (Junwhan) Kim (University of the District of Columbia) |
| Venue / year | arXiv September 2026; cited by DAWIN as IEEE ICTAI 2026 |
| Link | https://arxiv.org/abs/2609.00503 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (the conventions anticipated no arXiv version; one exists) |

## 1. Problem and position in the lineage

WFB is the first of the DAWIN authors' three wave formulations ([WFB](wfb.md) → [WSP](wsp.md) → [DAWIN](dawin.md)). It replaces an affine-plus-activation unit by a parameterised wave whose phase contains both the state and the elapsed interval, and tests it in a deliberately feed-forward trajectory predictor. Sequence modelling is explicitly out of scope.

## 2. Method

For state $x$ and interval $\Delta t$, unit $n$ outputs

$$f_n(x,\Delta t)=A_{x,n}A_{t,n}\cos(k_n^\top x-\omega_n\Delta t-\theta_n),$$

with learnable amplitudes, wavenumber, angular frequency and phase; gradients are the standard chain rule (Eqs. 15–19), noting that $\partial L/\partial\omega_n$ scales with $\Delta t$. A spatial-Laplacian "curvature" penalty $P=\frac\lambda2\sum_nk_n^4f_n^2$ is applied, not as a loss term but as a custom gradient correction to $(A_t,\omega,\theta)$ with $k$ held fixed (Eqs. 27–28), either replacing ("Laplacian-only") or adding to ("STD-Laplacian") the task gradient. The predictor projects each of 8 observations to 128 dimensions, applies the wave layer, flattens and decodes 12 future positions.

## 3. Evidence

ETH/UCY and JAAD-style pseudo-trajectories split by agent (453,338 windows; $\Delta t\in[0.398,10.8]$ s at 2.5 FPS), five seeds, ADE in normalised image coordinates. With motion features, real-$\Delta t$ WFB lowers ADE by 20.4 % relative to the original (smaller) FFN, and constant-$\Delta t$ WFB still by 12.2 %. Position-only, capacity-matched (≈401k parameters): WFB real $\Delta t$ 0.011787 ± 0.001123, matched FFN with $\Delta t$ 0.012043 ± 0.000518, matched FFN 0.012194, **WFB with shuffled $\Delta t$ 0.011418 ± 0.000274 (best)**, constant $\Delta t$ 0.012374. STD-Laplacian with $\lambda=10^{-5}$ improves ADE by 1.9 %; larger $\lambda$ degrades it and Laplacian-only fails (ADE ≈0.19–0.23). Inference-time doubling of $\Delta t$ raises ADE by 30.9 %. A linear decoder after the wave layer (26,520 parameters) gives the lowest ADE in the decoder ablation.

## 4. Critical analysis

The paper is unusually candid: it states that the evidence "does not attribute the gain to interval alignment," since shuffled intervals do best, and that differences among matched models are within five-seed variability. With motion features, velocities and accelerations are divided by $\Delta t$, so interval information leaks into the state anyway. The "curvature correction" is a heuristic gradient modification, not the gradient of any objective, and it helps only when negligible. Sensitivity of a trained model to perturbed $\Delta t$ (Fig. 6) shows that $\Delta t$ is used, not that it is used correctly. In the taxonomy, WFB is time-as-feature (family 1) passed through a sinusoid (family 5): $\cos(k^\top x-\omega\Delta t-\theta)$ is a Fourier feature of the concatenated vector $[x;\Delta t]$ with frequency vector $[k;-\omega]$, which is exactly how [DAWIN](dawin.md)'s Remark 1 recovers it.

## 5. Cross-references

- Builds on: [Fourier features](fourier-features.md), [SIREN](siren.md) (periodic representations), [GRU-D](gru-d.md), [Latent ODE](latent-ode.md), [Neural CDE](neural-cde.md) (cited alternatives).
- Extended by: [WSP](wsp.md) (persistent complex recurrence), [DAWIN](dawin.md) (damped integration, scan).
- Contrasts with: [Time2Vec](time2vec.md) (periodic embedding of time alone, concatenated to inputs).

## 6. Relevance to the study question

Question 3: WFB is the clearest example in the corpus of the failure mode of family (1) combined with (5): a sinusoidal map of $[x;\Delta t]$ adds capacity and a useful nonlinearity, but its own ablation shows no benefit from correct timing. This finding should temper the interpretation of DAWIN's interval-intervention results on the same pedestrian data.
