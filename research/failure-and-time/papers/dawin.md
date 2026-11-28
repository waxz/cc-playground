# DAWIN: Damped Atomic Wave Integration Neurons — A Universal Primitive for Continuous-Time Spatiotemporal Learning

| | |
|---|---|
| Authors | Byunggu Yu, Justin (Junwhan) Kim (University of the District of Columbia) |
| Venue / year | arXiv preprint, November 2026 |
| Link | https://arxiv.org/abs/2611.30025 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) (seed paper) |
| Source | full text read (including the PyTorch reference listing) |

## 1. Problem and position in the lineage

DAWIN proposes to replace the perceptron $y=\sigma(w^\top x+b)$ by a unit with an internal complex state that evolves in continuous time between irregularly spaced events. It positions itself against three families: solver-based continuous-time models ([Neural ODE](neural-ode.md), [Neural CDE](neural-cde.md), [ContiFormer](contiformer.md)), discretised state-space models ([S4](s4.md), [DSS](dss.md), [S5](s5.md), [LRU](lru.md), [Mamba](mamba.md)), and the authors' own wave formulations [WFB](wfb.md) (a memoryless sinusoid with $\Delta t$ in the phase) and [WSP](wsp.md) (a serial complex recurrence). The claim of novelty is that spatial projection and elapsed-interval scaling are bound into one "atomic" complex excitation, which the authors contrast with the "separable" $Bx_i$ of SSMs.

## 2. Method

For an input $x_i\in\mathbb R^d$ arriving after interval $\Delta t_i=\tau_i-\tau_{i-1}\ge 0$ ($\Delta t_1=0$), unit $m$ with parameters $(k_m,\omega_m,\theta_m,A_m>0,\gamma_m>0)$ computes (Eqs. 1–3)

$$\phi_{i,m}=k_m^\top x_i-\theta_m,\qquad \alpha_m(\Delta t)=\begin{cases}1-e^{-\gamma_m\Delta t}&\Delta t>0\\ 1&\Delta t=0\end{cases},\qquad u_{i,m}=A_m\,\alpha_m(\Delta t_i)\,e^{i\phi_{i,m}},$$

$$\psi_{i,m}=D_m(\Delta t_i)\,\psi_{i-1,m}+u_{i,m},\qquad D_m(\Delta t)=e^{-(\gamma_m+i\omega_m)\Delta t}.$$

The state is carried as the real pair $(p,q)$ (Eq. 4), and stacked layers pass both quadratures. Closed-form adjoint and parameter gradients are given (Eqs. 5–8); $\gamma_m=\mathrm{softplus}(\tilde\gamma_m)+\gamma_{\min}$ with $\gamma_{\min}=10^{-4}$. Proposition 1 states $|D_m|=e^{-\gamma_m\Delta t}\le 1$; Theorem 1 bounds $|\psi_{n,m}|\le A_m/(1-e^{-\gamma_m\Delta t_{\min}})$ under the hypothesis $\Delta t_i\ge\Delta t_{\min}>0$. A "spatiotemporal Laplacian" regulariser $\frac1M\sum_m A_m^2(\|k_m\|^2+\omega_m^2+\gamma_m^2)^2$ penalises high frequencies (Eq. 13). Proposition 2 states associativity of $(d_2,u_2)\oplus(d_1,u_1)=(d_2d_1,\,d_2u_1+u_2)$, which permits a Blelloch scan (a three-stage chunked Triton kernel, FP32 accumulation). DAWINLinear is inserted into a pre-norm residual MLP, into a Transformer's FFN slot, and into a GNN node update.

## 3. Evidence

Ten seeds, parameter budgets matched within ±2 %, eleven tasks, NVIDIA B200, BF16 autocast. Reported test results (Table 2): damped oscillator with out-of-range test intervals, DAWIN 0.7589 NRMSE vs Mamba 0.7373 (not significant), S5 0.8177, Neural CDE 1.2799; HalfCheetah, Hopper and Reacher: DAWIN loses to Time-Transformer (e.g. 0.3995 vs 0.3538 on HalfCheetah) and to S5 on HalfCheetah; Walker2d: Neural CDE best (1.6220 vs 1.7137); PhysioNet-2012 imputation: DAWIN 0.3655 masked MSE, worst except Neural CDE (ReLU-MLP 0.2355); PhysioNet-2012 dense: DAWIN 0.9531 NRMSE vs Time-Transformer 0.9553 (10/10 seeds, Holm $p=0.010$); PhysioNet-2012 mortality: DAWIN 0.9993 AUROC, tied with Time-Transformer (0.9993), ReLU-MLP 0.9989, S5 0.9987, Mamba 0.9967, Neural CDE 0.6974; pedestrians: Time-Transformer, Neural CDE and Mamba beat DAWIN on the position-only split. Interval interventions (Table 3): shuffling $\Delta t$ raises error by 61 %, 58 %, 77 % and 72 % on the oscillator, HalfCheetah, Hopper and Reacher, but only 0.43 % on Walker2d, 0.89 % on PhysioNet dense and −0.38 % on pedestrian motion. Throughput (Table 4): at $L=16384$, $B=1$ the scan takes 0.716 ms vs 1876.5 ms serially (2630×); at $L=512$, $B=256$ DAWIN uses 443 MiB vs 18,713 MiB for Mamba.

## 4. Critical analysis

*Relation to SSMs.* $D_m(\Delta t)$ is exactly the ZOH transition $e^{\lambda\Delta t}$ of a diagonal SSM with eigenvalue $\lambda_m=-\gamma_m-i\omega_m$, and $\alpha_m(\Delta t)=1-e^{-\gamma_m\Delta t}=\gamma_m\cdot\frac{1-e^{-\gamma_m\Delta t}}{\gamma_m}$ is $(-\lambda)$ times the ZOH input factor $(e^{\lambda\Delta t}-1)/\lambda$ of the *real* pole $-\gamma_m$. DAWIN is therefore a diagonal complex SSM, as in [S5](s5.md) with per-step $\Delta t_i$, whose input matrix is diagonal and whose input is a random-Fourier-feature map $e^{i(k_m^\top x-\theta_m)}$ ([Fourier features](fourier-features.md)). The "separable $Bx$" criticism applies equally to DAWIN: its excitation is the product of a time gain and a spatial feature, exactly the structure of $\bar B(\Delta t)x$. Because the input gain uses the real pole while the transition uses the complex pole, the stationary response to a held input is $A_m\alpha_m/(1-D_m)$, which tends to $A_m\gamma_m/(\gamma_m+i\omega_m)$ as $\Delta t\to0$ and to $A_m$ for $\Delta t\gg1/\gamma_m$; exact ZOH gives the rate-independent $-b/\lambda$. DAWIN is thus not sampling-rate invariant whenever $\omega_m\neq0$ (the theme report derives this).

*Theory.* The associativity proof writes $(d_2d_1,\,d_2u_1+u_1)$ where $d_2u_1+u_2$ is meant, twice; the code (`_dawin_binary_op`) is correct. Theorem 1 discards $\alpha$ and needs $\Delta t_{\min}>0$; keeping $\alpha$, the sum telescopes, $\sum_j(1-e^{-\gamma\Delta t_j})e^{-\gamma(\tau_n-\tau_j)}\le1$, so $|\psi_n|\le A_m$ for any positive intervals, a stronger and $\Delta t_{\min}$-free bound. The real gap is the convention $\alpha(0)=D(0)=1$: every same-timestamp event adds up to $A_m$ undamped, so $k$ simultaneous events give $|\psi|$ up to $kA_m$, and $\alpha$ jumps from $\approx\gamma\epsilon$ at $\Delta t=\epsilon$ to 1 at $\Delta t=0$. The theorem's own hypothesis excludes the model's first event ($\Delta t_1=0$). The "$\Delta t_{\text{zero}}$" probe announced in RQ3 is never reported; it is precisely the undamped-accumulation regime.

*Evidence.* The validation AUROC on PhysioNet-2012 mortality is $1.0000\pm0.0000$ for five of six models, and test AUROC is ≥0.9967 for all but Neural CDE. In-hospital mortality on this dataset is reported in the cited literature at 0.764–0.858 ([GRU-D](gru-d.md) 0.8424 under its own protocol; [Latent ODE](latent-ode.md) table 0.764–0.833; [mTAN](mtan.md) 0.858), so 0.999 indicates target leakage or a mis-specified task, and DAWIN's Neural CDE (0.6974) is below the weakest RNN baseline in those papers. The dense NRMSE of 0.9531 means roughly 9 % of variance explained if NRMSE is RMSE/σ, and the "best" margin is 0.0022. Every Holm $p=0.010$ is the floor attainable with ten seeds and five comparators, and reflects seed variance on one split. Interval sensitivity is large only where $\Delta t$ matters by construction, and no comparator was subjected to the same intervention. The VRAM figure for Mamba (18,713 MiB at both $B{=}4096,L{=}32$ and $B{=}256,L{=}512$, i.e. 131,072 tokens, ≈146 kB per token, for models of roughly 0.2 M parameters if the Table 2 sizes apply; the benchmark model size is not stated) is consistent with a reference implementation that materialises $(B,L,D,N)$ tensors, which Mamba's own fused kernel avoids; latencies are reported with a standard deviation of exactly 0.000. Whether S5 and Mamba received $\Delta t$ is not stated.

## 5. Cross-references

- Builds on: [WFB](wfb.md) (sinusoid with $\Delta t$ in phase), [WSP](wsp.md) (complex carry), [S5](s5.md) (diagonal ZOH with variable $\Delta t$ and associative scan), [Fourier features](fourier-features.md) and [SIREN](siren.md) (sinusoidal input maps).
- Contrasts with: [Mamba](mamba.md) (input-dependent $\Delta$, which DAWIN lists as future work), [LRU](lru.md) (normalisation by $\sqrt{1-|\lambda|^2}$ rather than DC gain), [Neural CDE](neural-cde.md), [GRU-D](gru-d.md) (real decay gate without rotation).
- Related: [RoPE](rope.md): with $\gamma_m=0$, $D_m$ is a continuous-time rotary encoding applied inside a recurrence.

## 6. Relevance to the study question

DAWIN informs question 3 as a combination of families (4) and (5) of the taxonomy. Its useful content is a cheap closed-form parameterisation (exact exponential transition, a DC-normalised input gain, bounded sinusoidal input) that runs as a scan; its claims of novelty relative to diagonal SSMs, of strict stability, and of state-of-the-art clinical accuracy are not supported by its own equations and tables.
