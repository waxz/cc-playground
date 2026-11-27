# Suh et al.: Do Differentiable Simulators Give Better Policy Gradients?

| | |
|---|---|
| Authors | H. J. Terry Suh, Max Simchowitz, Kaiqing Zhang, Russ Tedrake |
| Venue / year | ICML 2022 (PMLR 162) |
| Link | https://arxiv.org/abs/2202.00817 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Differentiable simulators (Brax, DiffTaichi, Nimble, dflex) were promoted on the premise that exact gradients of the dynamics yield lower-variance policy gradients than zeroth-order, REINFORCE-style estimates, and hence faster learning. Suh et al. ask under which properties of the physical system that premise holds. They compare the two estimators on equal footing, both estimating the gradient of the same *randomised-smoothed* objective, and identify three regimes in which the first-order estimator is worse: discontinuities, where it is biased; stiff continuous approximations of discontinuities, where it exhibits what they call *empirical bias*; and stiffness or chaos, where it has higher variance. The paper is the central reference for why gradients through contact must be treated with suspicion, and it frames the problem that [AHAC](ahac.md), [PWM](pwm.md) and [TGI](tgi.md)'s event-gated cut respond to.

## 2. Method

The objective is $F(\theta) = \mathbb{E}_{w}\,V_1(x_1, w_{1:H}, \theta)$ with Gaussian action noise $u_h = \pi(x_h,\theta) + w_h$, $w_h\sim\mathcal N(0,\sigma^2 I)$. The two batched estimators are

$$\bar\nabla^{[0]}F = \frac1N\sum_i \frac{1}{\sigma^2}\big(V_1(x_1,w^i,\theta)-b\big)\sum_h D_\theta\pi(x^i_h,\theta)^\top w^i_h,\qquad \bar\nabla^{[1]}F = \frac1N\sum_i \nabla_\theta V_1(x_1,w^i,\theta),$$

the zeroth-order (score-function, ZoBG) and first-order (pathwise, FoBG) batched gradients, with the zero-noise rollout as baseline $b$ (see [m-pg-theorem](../math-cookbook.md#m-pg-theorem), [m-reparam](../math-cookbook.md#m-reparam)). Lemma 3.1: ZoBG is unbiased under mild growth conditions, even with discontinuous dynamics and costs. Lemma 3.2: FoBG is unbiased only if the dynamics are locally Lipschitz and the costs $C^1$.

**Bias under discontinuity.** For $f(\theta,w) = H(\theta+w)$ (Heaviside), every sample gradient is zero almost surely, so FoBG returns zero with zero variance, while the true gradient of $F(\theta)=\mathrm{erf}(-\theta;\sigma^2)$ is non-zero.

**Empirical bias.** A random vector $z$ has $(\beta,\Delta,S)$-empirical bias if an event of probability at least $1-\beta$ exists on which $z$ is concentrated (within $S$) around a conditional mean that differs from $\mathbb{E}[z]$ by at least $\Delta$ (Definition 3.4). Lemma 3.5 shows this implies $\mathrm{Var}[z]\ge \Delta'^2/\beta$. For a Coulomb-friction relaxation $\bar H_\nu$ with slip width $\nu$, the FoBG at $\theta=\nu/2$ has $(c_\sigma\nu, c_\sigma, 0)$-empirical bias, $c_\sigma = 1/\sqrt{2\pi}\sigma$, and variance scaling as $1/\nu$. Small-sample estimates therefore look precise and are wrong; stiffening the relaxation (more realistic contact) makes it worse. The same holds for geometric discontinuities (corners) smoothed by stiff fillets.

**Variance under stiffness and chaos.** For ZoBG, $\mathrm{Var}\le B_V^2B_\pi^2 Hn/(N\sigma^2)$ (Lemma 3.10), independent of derivative magnitudes; FoBG variance scales with $\|\nabla\phi\|$, which grows with contact stiffness and, through products of Jacobians, exponentially with horizon in chaotic systems.

**α-order estimator.** $\bar\nabla^{[\alpha]}F = \alpha\bar\nabla^{[1]}F + (1-\alpha)\bar\nabla^{[0]}F$, with $\alpha$ chosen to minimise $\alpha^2\hat\sigma_1^2+(1-\alpha)^2\hat\sigma_0^2$ subject to $\epsilon + \alpha\|\bar\nabla^{[1]}F-\bar\nabla^{[0]}F\|\le\gamma$, where $\epsilon$ is a Bernstein confidence radius on the ZoBG. The closed form is the inverse-variance weight $\alpha_\infty = \hat\sigma_0^2/(\hat\sigma_1^2+\hat\sigma_0^2)$ unless the constraint binds, in which case $\alpha = (\gamma-\epsilon)/B$. The constraint uses the unbiased ZoBG as a check on the possibly biased FoBG: a large disagreement signals (empirical) bias and pushes $\alpha$ toward zero.

## 3. Evidence

The evidence consists of constructed systems. On a ball thrown at a wall (inelastic impact) the FoBG stalls in a flat region while ZoBG and AoBG reach the minimum. On angular-momentum transfer to a pivoting bar, FoBG follows the gradient off a cliff (the ball misses the pivot) and cannot recover, while ZoBG and AoBG converge to the robust optimum (Fig. 7). In a 1D pushing task with penalty contact over $H=200$ steps, FoBG outperforms ZoBG for spring constant $k=10$ and ZoBG wins for $k=1000$ (Fig. 8). With Coulomb friction, FoBG converges quickly and then degrades when the box slides off its support. On a breakout-like "tennis" task with a 21-parameter linear policy and $H=200$, using time-of-impact event detection for correct gradients, FoBG misses many balls while ZoBG and AoBG succeed, AoBG slightly faster. Double-pendulum variance plots (Fig. 6) show FoBG variance overtaking ZoBG as the horizon grows. All experiments are low-dimensional, with linear or open-loop policies.

## 4. Critical analysis

The paper's main achievement is conceptual precision. "Gradients through contact are bad" becomes three separable claims with different remedies: true bias at discontinuities (cannot be fixed by more samples), empirical bias at stiff smoothings (fixed only asymptotically, and diagnosable only by comparison with an unbiased estimator, not by variance), and high variance from stiffness or chaos (fixable by shorter horizons or softer models). The observation that the two estimators converge to *different local minima* rather than at different rates is important: it means that learning-curve comparisons between first-order and zeroth-order RL measure landscape selection as much as efficiency.

What the paper does not show is equally important. No experiment involves a neural policy, a high-dimensional robot or a learned critic, so the relevance of empirical bias to practical locomotion training is argued, not demonstrated. The AoBG requires both estimators on independent rollouts and a prior bound on value and policy-Jacobian magnitudes for the confidence radius; [Onoda et al.](onoda-composite-gradients.md) report that it needs task-specific tuning and is sample-inefficient because REINFORCE confidence intervals are wide, and that on dflex Ant and Hopper with stiffened contact plain per-step inverse-variance weighting performs as well, suggesting variance, not empirical bias, is the dominant obstacle there. The paper also mentions, without testing, analytic smoothing of the dynamics as an alternative to randomised smoothing; [PWM](pwm.md) and TGI's learned models can be read as that alternative.

## 5. Cross-references

- Builds on: [PIPPS](pipps.md) (chaos and inverse-variance mixing, which AoBG extends with a bias guard); [Metz et al.](metz-gradients.md) (chaos in differentiable physics).
- Extended by: [AHAC](ahac.md) (empirical bias under stiff contact motivates truncating the horizon at contact); [PWM](pwm.md) (learned smooth models as a remedy); [Onoda et al.](onoda-composite-gradients.md) (re-examination of AoBG).
- Contrasts with: [SHAC](shac.md), which addresses the same symptoms with a critic and truncation rather than estimator mixing; [DiffMimic](diffmimic.md), which reports strong first-order results on contact-rich motion.

## 6. Relevance to TGI

TGI's event-gated gradient cut is a targeted response to the failure modes catalogued here. A foot landing on a stone edge or gap rim is a geometric discontinuity of the kind in Example 3.7: whether the sole is supported changes discontinuously with foot position, and the world model's smoothed contact gate (a sigmoid of the sole–terrain gap with a 1 cm temperature) is a stiff relaxation of it, which is precisely the setting where Suh et al. predict empirical bias. TGI's response follows the logic of the α-order estimator but replaces its statistical detector with a geometric one: where the unsupported sole share at an imagined touchdown exceeds 0.3, the pathwise continuation is dropped and the critic, trained model-free on λ-returns, supplies the value. TGI also adds a small fixed-weight likelihood-ratio term on imagined actions, which supplies zeroth-order credit across such events.
