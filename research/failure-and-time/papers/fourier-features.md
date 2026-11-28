# Fourier Features: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains

| | |
|---|---|
| Authors | Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng |
| Venue / year | NeurIPS 2020 |
| Link | https://arxiv.org/abs/2006.10739 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Coordinate-based MLPs (pixel → colour, point → occupancy, NeRF) produce blurry outputs unless the coordinates are first passed through sinusoids, a trick NeRF called "positional encoding." This paper explains the trick with neural-tangent-kernel (NTK) theory and generalises it to random Fourier features. It is the theoretical basis for sinusoidal encodings of any low-dimensional continuous input, including time, and for DAWIN's spatial excitation $e^{ik^\top x}$.

## 2. Method

Map $v\in[0,1)^d$ to $\gamma(v)=[a_1\cos(2\pi b_1^\top v),a_1\sin(2\pi b_1^\top v),\dots,a_m\cos(2\pi b_m^\top v),a_m\sin(2\pi b_m^\top v)]^\top$. The induced kernel is stationary:

$$k_\gamma(v_1,v_2)=\gamma(v_1)^\top\gamma(v_2)=\sum_ja_j^2\cos\big(2\pi b_j^\top(v_1-v_2)\big)=h_\gamma(v_1-v_2).$$

In the NTK regime a trained network is kernel regression with $h_{\text{NTK}}\circ h_\gamma$; training error in the NTK eigenbasis decays as $e^{-\eta\lambda_it}$, so components with small eigenvalues (high frequencies, for a standard MLP whose NTK spectrum falls off fast) are learned extremely slowly: spectral bias. Fourier features make the composed kernel shift-invariant with a bandwidth set by the scale of $b_j$, trading underfitting (narrow spectrum) against aliasing (wide spectrum). Gaussian $b_j\sim\mathcal N(0,\sigma^2)$ with tuned $\sigma$ is recommended; the distribution's scale matters far more than its shape.

## 3. Evidence

1-D regression experiments match the NTK prediction of per-frequency convergence. Table 1 (PSNR; IoU for 3-D shape): 2-D natural images 19.32 (no mapping), 21.71 (basic), 24.95 (positional encoding), 25.57 (Gaussian); text images 18.40 → 30.47; 3-D shape IoU 0.864 → 0.973; CT 16.75 → 28.33; MRI 15.44 → 19.88; NeRF 26.14 → 34.51 (Gaussian best throughout).

## 4. Critical analysis

The argument assumes infinite width, small learning rates and MSE; in practice it predicts trends rather than numbers, though the 1-D experiments align closely. The kernel view states directly the cost of periodic encodings: they impose stationarity (only differences matter) and periodicity, so the network cannot distinguish inputs that differ by a period of every feature, and frequencies above the sampling density alias. For time embedding this has two implications. Sinusoidal encodings of *absolute* time interpolate well within the training range but extrapolate periodically, not monotonically, beyond it, which is why [Time2Vec](time2vec.md) and [mTAN](mtan.md) add a linear term. And the bandwidth $\sigma$ must match the timescales present: in [DAWIN](dawin.md), $k_m\sim\mathcal N(0,1/d)$ applied to layer-normalised inputs gives phases with unit standard deviation, a low-bandwidth regime further pushed down by its Laplacian penalty, quite unlike the high-frequency regime ($\sigma$ tuned per task, often ≫1) where Fourier features help most.

## 5. Cross-references

- Related: [SIREN](siren.md) (sine activations throughout, with an initialisation that controls frequency), [Transformer](transformer.md) positional encoding (axis-aligned log-spaced frequencies, a special case).
- Used by: [Time2Vec](time2vec.md) and [mTAN](mtan.md) (learned frequencies on time), [WFB](wfb.md) and [DAWIN](dawin.md) (sinusoidal maps of the state).
- Background: the NTK/spectral-bias argument is reproduced in the theme report's Mathematical background.

## 6. Relevance to the study question

Question 3: family (5), periodic encodings. It supplies the reason sinusoidal maps help (kernel bandwidth) and their failure modes (aliasing, periodic extrapolation), which govern when a periodic time embedding is appropriate.
