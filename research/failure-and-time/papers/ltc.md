# LTC: Liquid Time-constant Networks

| | |
|---|---|
| Authors | Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu |
| Venue / year | AAAI 2021 (arXiv 2020) |
| Link | https://arxiv.org/abs/2006.04439 |
| Theme | [t3-time-embedding](../themes/t3-time-embedding.md) |
| Source | full text read (main text and appendix tables) |

## 1. Problem and position in the lineage

LTCs take the leaky-integrator neuron of computational neuroscience, $\dot v=-g_lv+S(t)$ with conductance-based synapses $S=f(v,I)(A-v)$, as a recurrent cell. The resulting time constant depends on the input and state, which the authors call "liquid." Within the solver branch of the taxonomy LTC sits between [Neural ODE](neural-ode.md) (unconstrained vector field) and the later linear SSMs (input-independent time constants), and anticipates [Mamba](mamba.md)'s input-dependent step size.

## 2. Method

$$\frac{dx}{dt}=-\Big[\frac1\tau+f(x,I,t,\theta)\Big]\odot x+f(x,I,t,\theta)\odot A,\qquad \tau_{\text{sys}}=\frac{\tau}{1+\tau f}.$$

A fused explicit–implicit Euler step, replacing only the linearly occurring $x(t_i)$ by $x(t_{i+1})$, gives the closed update

$$x(t+\Delta t)=\frac{x(t)+\Delta t\,f(x,I,t,\theta)\odot A}{1+\Delta t\,(1/\tau+f(x,I,t,\theta))},$$

unrolled $L$ times per input. Training is by BPTT rather than the adjoint, trading memory for gradient accuracy (Table 1). With $f$ sigmoidal and bounded, Theorem 1 bounds $\tau_i/(1+\tau_iW_i)\le\tau_{\text{sys},i}\le\tau_i$, and Theorem 2 bounds $\min(0,A_i^{\min})\le x_i\le\max(0,A_i^{\max})$. Universality (Theorem 3) and lower bounds on latent trajectory length (Theorems 4–5) are given.

## 3. Evidence

Seven tasks, five runs each (Table 3): LTC is best on Gesture (69.55 % vs CT-GRU 68.31 %), Occupancy (94.63 %), Traffic (0.099 squared error vs LSTM 0.169) and Ozone (F1 0.302), but not on activity recognition (Neural ODE 97.26 % vs 95.67 %), sequential MNIST (LSTM 98.41 % vs 97.57 %) or Power (CT-GRU 0.586 vs 0.642). Person activity, second setting with the [Latent ODE](latent-ode.md) protocol: 0.882 vs latent ODE 0.846. HalfCheetah kinematics (5 % random actions): MSE 2.308 vs LSTM 2.500 and Neural ODE 3.805. Trajectory length in a 2-D PCA projection grows faster for LTC than for Neural ODEs and CT-RNNs.

## 4. Critical analysis

The baselines are older RNNs and an RK4 Neural ODE; no SSM or attention baseline appears, and several wins are within one or two standard deviations. The authors acknowledge three limitations that matter for the theme: LTCs suffer vanishing gradients and "would not be the obvious choice for learning long-term dependencies"; performance depends heavily on the solver (explicit Euler degrades it); and time and memory are higher than for Neural ODEs. The trajectory-length expressivity measure is evaluated at random initialisation and does not establish trained performance. The fused step is a semi-implicit integrator whose effective decay $1/(1+\Delta t(\cdot))$ depends on $\Delta t$ nonlinearly; changing the sampling interval changes the discrete dynamics in a way that is only first-order consistent with the ODE, unlike the exact exponential transition of a ZOH-discretised linear system. For irregular data the step must be set per gap, which the paper does not evaluate. The input-dependent time constant is LTC's lasting contribution: the same idea, made linear and scan-compatible, is Liquid-S4 and the selective $\Delta$ of [Mamba](mamba.md).

## 5. Cross-references

- Builds on: [Neural ODE](neural-ode.md), CT-RNNs (Funahashi and Nakamura, 1993).
- Contrasts with: [GRU-ODE-Bayes](gru-ode-bayes.md) (also bounded, but with jumps at observations), [S4](s4.md)/[S5](s5.md) (input-independent linear dynamics).
- Extended by: Liquid-S4 (cited in [S5](s5.md)), and conceptually [Mamba](mamba.md).

## 6. Relevance to the study question

Question 3: family (3) with an input-dependent time constant. It shows that bounded, state-dependent decay is achievable inside a cheap fixed-step update, and its stated limitations (long-range memory, solver dependence) explain why later work moved to linear recurrences with exact discretisation.
