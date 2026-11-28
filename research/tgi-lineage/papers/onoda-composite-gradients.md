# Onoda et al.: Does "Do Differentiable Simulators Give Better Policy Gradients?" Give Better Policy Gradients?

| | |
|---|---|
| Authors | Ku Onoda, Paavo Parmas, Manato Yaguchi, Yutaka Matsuo |
| Venue / year | ICLR 2026 |
| Link | https://arxiv.org/abs/2604.18161 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) (depth-3 addition) |
| Source | full text read (main text and robotics experiments) |

## 1. Problem and position in the lineage

[Suh et al.](suh-diffsim-gradients.md) argued that discontinuities make first-order gradients biased, and that empirical bias makes this bias invisible to variance estimates. They proposed the α-order estimator (AoBG), which guards inverse-variance mixing with a confidence interval around the REINFORCE estimate. Onoda et al., with the first author of [PIPPS](pipps.md) as co-author, re-examine this and ask whether bias is the main practical obstacle. They answer in two parts. In the constructed discontinuous settings of Suh et al., a simpler test (DDCG) detects nonsmooth regions more robustly than AoBG. On differentiable robotics benchmarks, plain per-step inverse-variance weighting (IVW-H) suffices, and empirical bias plays a minor role. The paper is the most recent word on estimator mixing and is directly relevant to whether TGI's mixture of a pathwise and a likelihood-ratio term needs a bias detector.

## 2. Method

Both estimators target a Gaussian-smoothed objective with $\zeta = \theta + \sigma\epsilon$, and mixtures take the form $\hat g = \alpha\hat g_1 + (1-\alpha)\hat g_0$ ([m-contact-gradients](../math-cookbook.md#m-contact-gradients), [m-pg-theorem](../math-cookbook.md#m-pg-theorem)). Inverse-variance weighting (IVW) sets $\alpha = \mathbb{V}[\hat g_0]/(\mathbb{V}[\hat g_0]+\mathbb{V}[\hat g_1])$. DDCG (Discontinuity Detection Composite Gradient) uses a lightweight statistical test with a single hyperparameter to decide when to distrust $\hat g_1$, motivated by the observation that rare, large first-order samples near a discontinuity cause systematic underestimation of variance. IVW-H applies IVW per time step and per dimension within the trajectory, computing variances without extra simulator calls, much as total propagation does in PIPPS.

## 3. Evidence

Reproducing all experiments of Suh et al. under the same settings, DDCG matches or exceeds AoBG, is less sensitive to hyperparameters and remains reliable with small sample sizes. On dflex cart-pole, ant and hopper, with contact stiffness raised to 10× (ant) and 50× (hopper) the values used by GIPPO, pure first-order and pure zeroth-order gradients struggle on ant. IVW, AoBG and GIPPO perform clearly better, and IVW-H attains the best or comparable reward. On hopper IVW-H improves on IVW, and AoBG does not outperform plain IVW. The authors conclude that "variance control via stepwise IVW-H is often more critical than explicit bias detection on these benchmarks".

## 4. Critical analysis

The paper usefully separates two questions that Suh et al. left entangled: whether empirical bias exists (it does, and DDCG detects it more cheaply), and whether it dominates practical learning (on these benchmarks, apparently not). The second conclusion is bounded by its testbed: three dflex tasks with penalty contact, flat ground and dense rewards. Geometric discontinuities such as terrain edges, where a small change in foot position flips between support and fall, are absent, and these are where empirical bias would be expected to matter most. The result therefore supports variance control as a default but does not show that bias handling is unnecessary in sparse-terrain locomotion.

## 5. Cross-references

- Builds on: [Suh et al.](suh-diffsim-gradients.md) (AoBG, empirical bias); [PIPPS](pipps.md) (total propagation and IVW); [SHAC](shac.md) (dflex tasks).
- Contrasts with: [AHAC](ahac.md), which truncates the horizon rather than mixing estimators.

## 6. Relevance to TGI

TGI mixes a pathwise return gradient with a likelihood-ratio term at a fixed weight (0.1), with neither IVW nor a bias test. This paper suggests the more principled per-step IVW as a candidate refinement. TGI's setting, stepping stones and gaps, is however exactly the regime its benchmarks omit. That is consistent with TGI handling the discontinuity explicitly: a geometric event detector cuts the pathwise chain at unsupported touchdowns, which plays the role of DDCG's discontinuity test without needing a statistical estimate.
