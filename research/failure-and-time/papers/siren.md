# SIREN: Implicit Neural Representations with Periodic Activation Functions

| | |
|---|---|
| Authors | Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein |
| Venue / year | NeurIPS 2020 |
| Link | https://arxiv.org/abs/2006.09661 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (main text; initialisation proof summarised from the main text) |

## 1. Problem and position in the lineage

Implicit neural representations map coordinates $x\in\mathbb R^m$ (space, or space and time) to signal values. ReLU networks are piecewise linear with zero second derivative, so they represent fine detail and derivatives poorly. SIREN uses the sine as activation in every layer, which yields smooth, derivative-friendly representations and makes it possible to fit signals through constraints on their gradients and Laplacians. DAWIN claims SIREN as its high-damping limit.

## 2. Method

$$\Phi(x)=W_n(\phi_{n-1}\circ\dots\circ\phi_0)(x)+b_n,\qquad \phi_i(x_i)=\sin(W_ix_i+b_i).$$

Any derivative of a SIREN is a SIREN (phase-shifted sines). The initialisation is essential: hidden weights $w\sim\mathcal U(-\sqrt{6/n},\sqrt{6/n})$ make each pre-activation approximately $\mathcal N(0,1)$ and each activation arcsine-distributed, independent of depth; the first layer is $\sin(\omega_0Wx+b)$ with $\omega_0=30$ so that it spans many periods on $[-1,1]$. The authors note that a sine layer with $|w^\top x|<\pi$ roughly preserves spatial frequency and increases it above that.

## 3. Evidence

Fitting a 300-frame 512×512 video: about 30 dB PSNR vs about 25 dB for a ReLU MLP. Images, audio and their gradients and Laplacians are represented accurately where ReLU with positional encoding represents the image but not its derivatives. Poisson image reconstruction and editing from gradients only; signed distance functions via the Eikonal constraint; Helmholtz and wave equations solved from source terms; hypernetwork priors over SIREN functions.

## 4. Critical analysis

Without the prescribed initialisation, the authors report that SIRENs train poorly, so the method is a pairing of activation and initialisation rather than "use a sine." The frequency content is set by $\omega_0$ and grows slowly with depth, which makes SIREN a learned high-frequency basis whose bandwidth, like that of [Fourier features](fourier-features.md), must suit the signal; too high produces noisy interpolation. SIREN treats time, when present, as one more input coordinate: it represents a fixed signal on a fixed domain and has no notion of state or causality, so it cannot by itself handle sequences with unknown future. [DAWIN](dawin.md)'s Remark 1 (as $\gamma\to\infty$, DAWIN "recovers SIREN on $\mathbb R^d$") holds only for a single layer: DAWIN's blocks interleave LayerNorm, linear output projections and residual connections, and its spatial frequencies are initialised at $k\sim\mathcal N(0,1/d)$, not $\omega_0=30$, so the limit is a sinusoidal feature layer rather than SIREN's network and initialisation.

## 5. Cross-references

- Related: [Fourier features](fourier-features.md) (input mapping rather than activation), [WFB](wfb.md) (sinusoid with an elapsed-time phase term).
- Claimed as a limit by: [DAWIN](dawin.md).
- Contrasts with: [Time2Vec](time2vec.md) (sine only on time, the rest of the network standard).

## 6. Relevance to the study question

Question 3: family (5). SIREN demonstrates what periodic activations buy (smooth derivatives, high-frequency detail) and what they require (careful initialisation, bandwidth control); as a time embedding it is appropriate for representing a known signal over a bounded domain, not for causal filtering of irregular streams.
