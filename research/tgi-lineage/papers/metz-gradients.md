# Metz et al.: Gradients are Not All You Need

| | |
|---|---|
| Authors | Luke Metz, C. Daniel Freeman, Samuel S. Schoenholz, Tal Kachman |
| Venue / year | arXiv preprint, 2021 |
| Link | https://arxiv.org/abs/2111.05803 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) (depth-3 addition) |
| Source | full text read |

## 1. Problem and position in the lineage

This short report from Google Brain (two of its authors built Brax) generalises the "curse of chaos" of [PIPPS](pipps.md) from model-based policy search to differentiable programming at large: rigid-body simulation, learned optimisers, meta-learning and recurrent networks. It argues that gradients computed through iterated dynamical systems can be exactly correct and still useless, and it ties this failure to the spectrum of the system's Jacobian. It is cited by [Suh et al.](suh-diffsim-gradients.md), [Song et al.](song-diffsim-quadruped.md) and [PWM](pwm.md) as the reference for chaos-induced gradient variance, and it fills the gap in this group between PIPPS's GP cart-pole and the deep-RL differentiable-simulation papers.

## 2. Method

For $s_{t+1} = f(s_t,\theta)$ and loss $l = \frac1N\sum_t l_t(s_t,\theta)$,

$$\frac{dl}{d\theta} = \frac1N\sum_{t=0}^N\Big[\frac{\partial l_t}{\partial\theta} + \sum_{k=1}^t\frac{\partial l_t}{\partial s_t}\Big(\prod_{i=k}^t\frac{\partial s_i}{\partial s_{i-1}}\Big)\frac{\partial s_k}{\partial\theta}\Big].$$

When the Jacobians $\partial s_i/\partial s_{i-1}$ have eigenvalues above one in magnitude (positive Lyapunov exponents), the products grow exponentially in $N$, and per-sample gradients explode ([m-bptt](../math-cookbook.md#m-bptt)). Stochasticity smooths the *expected* loss, but reparameterised gradient estimates still multiply the per-sample Jacobians, so their variance can grow exponentially while black-box (evolution-strategies-style) estimates of the same smoothed gradient remain bounded. A one-dimensional example, $l(x) = 0.1\sin(xw/\pi)+(x/10)^2+0.1$ with Gaussian smoothing, shows the crossover: for low frequency $w$ the reparameterisation gradient has lower variance, for high frequency the black-box gradient does ([m-reparam](../math-cookbook.md#m-reparam)).

## 3. Evidence

In Brax Ant with a stochastic MLP policy in double precision, loss slices along a random parameter direction become increasingly rough as the unroll grows from 2 to 1024 steps. Averaging over policy noise smooths them, yet gradient variance grows exponentially with unroll length at all four test points. Backpropagating the reward through 400 steps fails to learn despite extensive tuning. Inserting a stop-gradient every $t$ steps reveals a narrow band, around $t\approx10$, in which a walking gait is learned. Shorter truncations give a lunging policy, and longer ones fail (Fig. 6). Gradient clipping over a sweep of strengths and learning rates gives no positive result. Analogous behaviour is shown for learned optimisers and other iterated systems.

## 4. Critical analysis

The paper is a position statement backed by illustrative experiments, not a systematic study, but its central point is robust and has become consensus. It also surveys remedies: well-behaved systems (LSTM-like architectures, contact softening), proxy objectives, truncation (biased), gradient clipping (biased, ineffective here), shadowing methods for ergodic systems, and black-box estimators. Two claims deserve qualification for robotics. First, contact makes rigid-body dynamics locally chaotic only in some regions; the authors note that restricting to contact-free regions restores stability. Second, the remedies it favours for physics (black-box methods) scale poorly with parameter dimension, which is why the deep-RL line ([SHAC](shac.md), [AHAC](ahac.md), [PWM](pwm.md)) chose truncation plus a critic and smoother models instead. The truncation result is notable because it shows the bias–variance trade-off in the horizon directly. Without a critic, the useful band of horizons is narrow, the problem SHAC's terminal value widens.

## 5. Cross-references

- Builds on: [PIPPS](pipps.md) (curse of chaos, LR versus RP).
- Extended by: [Suh et al.](suh-diffsim-gradients.md) (adds discontinuity and empirical bias); [SHAC](shac.md) (truncation plus critic); [PWM](pwm.md) (learned models with lower variance and higher ESNR than chaotic true dynamics).

## 6. Relevance to TGI

The paper explains two fixed choices in TGI: the short 32-step window with a critic bootstrap, which keeps Jacobian products short, and the global gradient-norm clipping at 1.0 together with skipping non-finite updates, which Metz et al. show is a safeguard rather than a cure. TGI's event cut can also be read as an instance of the paper's observation that chaotic sensitivity is concentrated in particular regions of state space. In legged locomotion, the region where small perturbations decide between a supported and an unsupported foot is the touchdown near an edge, and TGI truncates there rather than everywhere.
