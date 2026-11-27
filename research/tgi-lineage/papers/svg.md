# SVG: Learning Continuous Control Policies by Stochastic Value Gradients

| | |
|---|---|
| Authors | Nicolas Heess, Greg Wayne, David Silver, Timothy Lillicrap, Yuval Tassa, Tom Erez |
| Venue / year | NeurIPS 2015 |
| Link | https://arxiv.org/abs/1510.09142 |
| Group | [g3-analytic-gradients](../groups/g3-analytic-gradients.md) |
| Source | full text read |

## 1. Problem and position in the lineage

By 2015 two families of policy-gradient estimators coexisted. Likelihood-ratio methods ([REINFORCE](reinforce.md), the [policy gradient theorem](pg-theorem.md)) needed no model but had high variance; "value-gradient" methods, which backpropagate reward through a differentiable model or a learned critic (Werbos, Jordan, Fairbank; [DPG/DDPG](ddpg.md) on the model-free side), had low variance but were restricted to deterministic policies and deterministic environments. Heess et al. removed that restriction by writing stochasticity as a deterministic function of exogenous noise, the reparameterisation trick then being popularised for variational autoencoders. The result is a spectrum of algorithms, SVG(∞), SVG(1) and SVG(0), that interpolates between pure model-based backpropagation through a whole trajectory and the model-free deterministic policy gradient. SVG is the first deep-RL formulation of the pathwise policy gradient through a learned model, and the template that [MAAC](maac.md), [Dreamer](dreamer.md), [SHAC](shac.md), [PWM](pwm.md), [SuperMimic](supermimic.md) and [TGI](tgi.md) instantiate with different models and horizons.

## 2. Method

For a deterministic model $s' = f(s,a)$ and policy $a = \pi(s;\theta)$, differentiating the Bellman equation $V(s) = r(s,a) + \gamma V'(f(s,a))$ gives the coupled backward recursions

$$V_s = r_s + r_a\pi_s + \gamma V'_{s'}(f_s + f_a\pi_s),\qquad V_\theta = r_a\pi_\theta + \gamma V'_{s'} f_a \pi_\theta + \gamma V'_\theta .$$

The paper's key step is to reparameterise the stochastic policy and dynamics as $a = \pi(s,\eta;\theta)$ and $s' = f(s,a,\xi)$ with $\eta,\xi$ drawn from fixed distributions. The stochastic Bellman equation then becomes an expectation over noise of a deterministic composition, and the same recursions hold inside $\mathbb{E}_{\eta,\xi}$ (Eqs. 7–8). A Monte-Carlo estimate along one sampled trajectory is the pathwise estimator described in the cookbook under [reparameterisation](../math-cookbook.md#m-reparam); its relation to the score-function estimator is in [m-pg-theorem](../math-cookbook.md#m-pg-theorem).

Two design choices distinguish SVG from later work. First, gradients are evaluated *along real trajectories*: given an observed transition $(s,a,s')$, the noise $\xi$ that would have produced $s'$ under the learned model $\hat f$ is inferred (trivially, for additive noise $s' = \hat f(s,a)+\xi$, as the residual), so the model is used only for its Jacobians $\hat f_s, \hat f_a$, never to generate states. Second, a learned critic truncates the recursion. SVG(∞) runs the recursion over the whole episode; SVG(1) expands one step and uses $\hat V_{s'}$ thereafter,

$$\hat V_\theta \approx \mathbb{E}\big[w\,(r_a + \gamma \hat V_{s'} \hat f_a)\,\pi_\theta\big],\qquad w = \frac{p(a\mid s;\theta)}{q(a\mid s)},$$

with an importance weight $w$ that makes SVG(1) usable with experience replay (SVG(1)-ER); SVG(0) uses $Q_a\pi_\theta$ and is the stochastic analogue of [DPG](../math-cookbook.md#m-dpg). Models are small tanh MLPs predicting $s'-s$, one sub-network per state dimension.

## 3. Evidence

On a hand–ball task requiring credit assignment over 1000 steps of 10 ms, SVG(∞) evaluated on real trajectories learns, while an otherwise identical planner that backpropagates through trajectories *sampled from the learned stochastic model* "shows little improvement" because prediction error accumulates (Fig. 2); REINFORCE fails on the same task. On a 3-link swimmer, shrinking the model from 20 to 5 hidden units per state dimension degrades SVG(∞) sharply while SVG(1) is "undisturbed" (Fig. 3); SVG(1) also beats a TD-error actor-critic on 3-, 5- and 7-link swimmers. Reducing critic capacity on cart-pole (200 → 50 units) hurts the Q-based DPG more than V-based SVG(1). SVG(1)-ER then solves reacher, gripper, monoped, half-cheetah and a planar walker with contact, though the paper notes that walkers with similar reward did not always produce equally good gaits. All results are learning curves in arbitrary reward units; no numerical tables are given.

## 4. Critical analysis

SVG establishes three ideas that recur through this group: the pathwise estimator through a learned model, the use of a critic to shorten the differentiated horizon, and the observation that "model gradient error can compound during backpropagation" just as forward prediction error compounds. What the evidence shows is narrower than these ideas. The tasks are low-dimensional and short; the claim that a critic compensates for a poor model is shown on a 3-link swimmer only. The paper does not analyse *why* long recursions fail beyond model error; the later diagnosis that even exact Jacobians of a chaotic or stiff system yield exploding variance ([PIPPS](pipps.md), [Metz et al.](metz-gradients.md), [Suh et al.](suh-diffsim-gradients.md)) is absent. The importance-weighted SVG(1)-ER is off-policy only in the action marginal; state-distribution mismatch is ignored, as the authors acknowledge.

The real-trajectory evaluation is the paper's most distinctive and least imitated idea. [MAAC](maac.md) explicitly rejected it, arguing that inferring noise reintroduces likelihood-ratio-like variance and choosing to roll out the model instead; most subsequent work (Dreamer, SHAC-style model methods, SuperMimic, TGI) follows MAAC. A decade later [DMO](amigo-decoupled.md) revived exactly SVG's decoupling (simulator forward, model Jacobians backward) and reported roughly double the asymptotic return of the model-forward ablation. SVG thus marks one pole of a still-open design choice: whether the model should generate the states at which it is differentiated.

## 5. Cross-references

- Builds on: [REINFORCE](reinforce.md) and [policy gradient theorem](pg-theorem.md), the score-function alternative SVG positions itself against; [PILCO](pilco.md), which computes analytic gradients through a GP model by moment matching.
- Contrasts with: [DDPG](ddpg.md), which SVG(0) generalises to stochastic policies.
- Extended by: [PIPPS](pipps.md), which shows pathwise gradients through long particle rollouts suffer from chaos; [MAAC](maac.md), which replaces real-trajectory evaluation with H-step model rollouts and a terminal Q; [Dreamer](dreamer.md), which applies the same pathwise actor objective in a latent model; [DMO](amigo-decoupled.md), which revives SVG's decoupling.

## 6. Relevance to TGI

TGI's policy update is an SVG-family estimator: the actor is reparameterised (`a = μ + ε` in the imagination step), the return is differentiated through a learned dynamics model, and a critic terminates the recursion. TGI sits closer to SVG(H) than to SVG(∞): it differentiates through 32 imagined steps and bootstraps with $V$, the structure SVG(1) introduced with a one-step expansion. Unlike SVG it generates the differentiated states with the model (as MAAC and [SuperMimic](supermimic.md) do), accepting compounding error in exchange for simulator-free rollouts; it mitigates that error by starting rollouts from simulator states and by fitting the model on autoregressive windows ([RWM](rwm.md)). SVG's warning that model-gradient error compounds is the earliest statement of the problem TGI's event-gated gradient cut addresses at poorly supported touchdowns.
