# PILCO: A Model-Based and Data-Efficient Approach to Policy Search

| | |
|---|---|
| Authors | Marc Peter Deisenroth, Carl Edward Rasmussen |
| Venue / year | ICML 2011; extended in "Gaussian Processes for Data-Efficient Learning in Robotics and Control", IEEE TPAMI 2015 |
| Link | no-arxiv |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | secondary (no full text read) |

## 1. Problem and position in the lineage

PILCO (probabilistic inference for learning control) is the first model-based policy search method that combined three ingredients still visible in [TGI](tgi.md): a learned dynamics model, a parametric policy, and an *analytic* gradient of the predicted long-horizon return with respect to the policy parameters, computed through the model. Its starting diagnosis was model bias: a policy optimised against a point estimate of the dynamics exploits the model's errors and fails on the real system. PILCO's answer was to make the model probabilistic and to propagate its uncertainty through the entire prediction horizon, so that the optimiser sees the expected cost under the posterior over dynamics rather than under one guess.

Within this study PILCO marks the start of the "differentiable imagination" branch, which runs through [SVG](svg.md), [Dreamer](dreamer.md), [MAAC](maac.md) and SHAC to [SuperMimic](supermimic.md) and TGI. The later deep-learning papers in this group cite it repeatedly as the benchmark for data efficiency and as the method that did not scale: [Nagabandi et al.](nagabandi-mbmf.md) note that the highest-dimensional PILCO task they could find had 11 state dimensions, against 49 in their own locomotion work.

## 2. Method

The dynamics are modelled by independent Gaussian processes, one per state dimension, with squared-exponential kernels, predicting the state difference $\Delta_t = s_{t+1}-s_t$ from $(s_t,a_t)$. Given a Gaussian distribution over the current state, $p(s_t)=\mathcal N(\mu_t,\Sigma_t)$, the predictive distribution of $s_{t+1}$ under the GP posterior is not Gaussian; PILCO approximates it by *moment matching*, computing its exact mean and covariance in closed form and replacing it by a Gaussian with those moments. Iterating gives Gaussian state marginals $p(s_1),\dots,p(s_T)$ over the full task horizon $T$.

With a saturating cost $c(s)=1-\exp(-\lVert s-s^\ast\rVert^2_W/2)$, the expected cost $\mathbb E[c(s_t)]$ is again available in closed form, and the objective is

$$J(\theta)=\sum_{t=0}^{T}\mathbb E_{s_t\sim p(s_t\mid\theta)}\big[c(s_t)\big].$$

Because each moment-matching step is a differentiable map $(\mu_t,\Sigma_t,\theta)\mapsto(\mu_{t+1},\Sigma_{t+1})$, the chain rule yields $\mathrm dJ/\mathrm d\theta$ exactly, by propagating derivatives of means and covariances forward in time. The policy (a linear controller or a radial-basis-function network, squashed to respect actuator limits) is optimised with a quasi-Newton method (L-BFGS) on this deterministic objective. The outer loop alternates real rollouts, GP refitting, and policy optimisation. In cookbook terms this is a pathwise gradient ([m-reparam](../math-cookbook.md#m-reparam)) through a model, but taken on the moments of the state distribution rather than on sampled trajectories, which removes Monte-Carlo variance entirely ([m-bptt](../math-cookbook.md#m-bptt)).

## 3. Evidence

The ICML paper reports learning a cart-pole swing-up, both in simulation and on a real cart-pole, from very few trials, with total interaction time of the order of tens of seconds, and a simulated unicycle balancing task. Later work by the same authors applied PILCO to a low-cost robotic manipulator. These results established a data-efficiency standard that deep model-based methods were measured against for most of the following decade ([PETS](pets.md) compares to GP baselines with moment matching and finds them best on cart-pole but infeasible on its higher-dimensional tasks).

## 4. Critical analysis

PILCO's strengths and limits both come from the GP. The GP posterior is well calibrated in low dimension and with little data, which is why model bias is controlled; but exact GP inference scales cubically with the number of data points, the moment-matching computations scale poorly with state dimension, and the squared-exponential kernel assumes smooth dynamics. Contact-rich locomotion violates the last assumption outright. [PETS](pets.md) states the point directly: GP kernels are ill suited to the discontinuous dynamics typical of robots interacting through contacts.

A second limitation, less often stated, is that moment matching is not merely an approximation of convenience. Later analysis of particle-based gradients through probabilistic models ([PIPPS](pipps.md)) showed that replacing moment matching by sampled trajectories yields policy gradients whose variance can explode with horizon, because the pathwise derivative of a long chaotic rollout is ill conditioned. PILCO's deterministic moment propagation smooths the objective and sidesteps this, at the cost of a unimodal Gaussian assumption on the state distribution. [PETS](pets.md) reports in its conclusion that its own attempts to backpropagate through neural-network ensembles for policy learning "did not yield an effective algorithm", which it attributes to the same chaotic-gradient phenomenon.

The full-horizon objective without a value function is the third limitation. PILCO optimises returns over the whole episode, which is feasible for short tasks but couples the horizon of differentiation to the task horizon. [Dreamer](dreamer.md) and SHAC break this coupling with a learned critic at the end of a short imagined segment ([m-short-horizon](../math-cookbook.md#m-short-horizon)).

## 5. Cross-references

**Extended by.**
[SVG](svg.md) and [PIPPS](pipps.md) generalise PILCO's analytic gradients to sampled trajectories through learned models.
[Dreamer](dreamer.md) backpropagates value gradients through a neural latent model with a critic beyond the horizon.

**Contrasts with.**
[PETS](pets.md) keeps PILCO's emphasis on uncertainty but replaces GPs by neural ensembles and the policy by MPC.
[ME-TRPO](me-trpo.md) abandons model gradients for likelihood-ratio updates because BPTT through neural models proved unstable.
[Dyna](dyna.md) uses the model only to sample transitions.

## 6. Relevance to TGI

TGI's policy objective is a direct descendant of PILCO's: a return predicted by a learned dynamics model and differentiated analytically with respect to the policy parameters. Three of PILCO's choices are reversed. The model is a deterministic neural network conditioned on the simulator's randomisation vector $\xi$ rather than a GP posterior, so uncertainty is not propagated at all. The horizon is short (32 steps) and closed by a critic rather than equal to the task horizon. And the gradient is taken on sampled imagined trajectories rather than on moments, which reintroduces the chaotic-gradient and contact-discontinuity problems that PILCO's smoothing avoided; TGI's event-gated return and smooth terrain surrogates ([TGI](tgi.md), [m-contact-gradients](../math-cookbook.md#m-contact-gradients)) are a targeted response to exactly those problems.
