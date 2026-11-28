# Dreamer: Dream to Control: Learning Behaviors by Latent Imagination

| | |
|---|---|
| Authors | Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi |
| Venue / year | ICLR 2020 |
| Link | https://arxiv.org/abs/1912.01603 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[PlaNet](planet.md) planned in a learned latent space with CEM and a 12-step horizon, which made it myopic and expensive. Dreamer keeps PlaNet's world model (the RSSM) but replaces planning by an actor and a critic trained purely on imagined latent trajectories, and trains the actor by *backpropagating analytic gradients of multi-step value estimates through the learned dynamics*. The critic extends credit assignment beyond the imagination horizon; the pathwise gradient replaces the sample-hungry score-function estimator. Dreamer is the closest ancestor of [TGI](tgi.md)'s policy objective among world-model methods, and this report analyses that connection in detail in §6.

## 2. Method

**World model.** Representation $p_\theta(s_t\mid s_{t-1},a_{t-1},o_t)$, transition $q_\theta(s_t\mid s_{t-1},a_{t-1})$ and reward $q_\theta(r_t\mid s_t)$, implemented as an RSSM with 30-dimensional diagonal-Gaussian stochastic states and trained by the reconstruction ELBO ([m-rssm-elbo](../math-cookbook.md#m-rssm-elbo)) on batches of 50 sequences of length 50; the paper also tests contrastive and reward-only objectives. In tasks with early termination a discount predictor is learned. Latent overshooting was found unnecessary.

**Imagination.** From every posterior state of a training batch (2,500 start states), latent trajectories are imagined for $H=15$ steps with the transition model and the actor $a_\tau=\tanh(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\epsilon)$, $\epsilon\sim\mathcal N(0,I)$, a reparameterised tanh-Gaussian.

**Value estimates.** With critic $v_\psi$,

$$V^k_N(s_\tau)=\mathbb E\Big[\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\Big],\ h=\min(\tau+k,t+H),\qquad V_\lambda(s_\tau)=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V^n_N(s_\tau)+\lambda^{H-1}V^H_N(s_\tau),$$

the λ-return ([m-lambda-return](../math-cookbook.md#m-lambda-return)), with $\gamma=0.99$ and $\lambda=0.95$. Recursively, $V_\lambda(s_\tau)=r_\tau+\gamma[(1-\lambda)v_\psi(s_{\tau+1})+\lambda V_\lambda(s_{\tau+1})]$, closed by $v_\psi(s_{t+H})$.

**Objectives.**

$$\max_\phi\ \mathbb E\Big[\sum_{\tau=t}^{t+H}V_\lambda(s_\tau)\Big],\qquad \min_\psi\ \mathbb E\Big[\sum_{\tau=t}^{t+H}\tfrac12\big(v_\psi(s_\tau)-\mathrm{sg}(V_\lambda(s_\tau))\big)^2\Big].$$

The actor gradient is computed by stochastic backpropagation through rewards, values, imagined states and actions ([m-reparam](../math-cookbook.md#m-reparam), [m-bptt](../math-cookbook.md#m-bptt)); the world model is frozen during behaviour learning. With early termination, the terms are weighted by the cumulative product of predicted discounts. No entropy bonus and no target network were needed.

## 3. Evidence

On 20 DeepMind Control tasks from 64×64 pixels, Dreamer reaches an average score of 823 after $5\times10^6$ environment steps, against 332 for PlaNet at the same budget and 786 for D4PG after $10^8$ steps; training takes about 3 hours per $10^6$ steps on a V100, against 11 hours for PlaNet. The horizon study (Figure 4) is the most important result for this lineage: with a learned value, final performance is robust to the imagination horizon over 5–45 steps, whereas an actor trained on the reward sum $V_R$ without a value, and PlaNet's planner, degrade at short horizons and fail on tasks with long-horizon credit assignment (acrobot, hopper). With $H=20$, Dreamer beats both alternatives on 16 of 20 tasks with 4 ties. Pixel reconstruction outperforms contrastive learning on most tasks; reward prediction alone is insufficient. Appendix experiments cover discrete actions and early termination (Atari, DeepMind Lab) using straight-through gradients.

## 4. Critical analysis

The paper's evidence for the pathwise estimator is indirect: it compares Dreamer with planning and with a value-free actor, not with the same actor–critic trained by Reinforce through the same model. The benefit of the critic is therefore established; the benefit of differentiating through the dynamics, as opposed to merely evaluating with them, is argued by analogy with DDPG and SVG but not isolated. [DreamerV2](dreamer-v2.md) later found Reinforce substantially better on Atari and dynamics backpropagation substantially better on continuous control, and [DreamerV3](dreamer-v3.md), in the version read here, uses Reinforce for both discrete and continuous actions. The Dreamer line thus moved *away* from the pathwise gradient that TGI moved toward, which makes the conditions under which the pathwise gradient works the central question (§6).

The control tasks have smooth dense rewards and moderate contact; the gradient flows through a stochastic latent model trained one step at a time by the ELBO, through a learned reward head and through a learned critic, any of which the actor can exploit. The paper does not measure model exploitation or gradient quality. Its robustness to horizon is shown up to 45 steps at action repeat 2; the decay of gradient quality with horizon, central to SHAC's later analysis ([m-short-horizon](../math-cookbook.md#m-short-horizon)), is not studied.

## 5. Cross-references

**Builds on.**
[PlaNet](planet.md): the RSSM world model and the replay-seeded latent rollouts.
[SVG](svg.md) and [DDPG](ddpg.md): reparameterised value gradients, here through multi-step learned dynamics.
[PILCO](pilco.md): analytic return gradients through a learned model.

**Extended by.**
[DreamerV2](dreamer-v2.md): discrete latents, KL balancing, mixed Reinforce/pathwise actor loss.
[DreamerV3](dreamer-v3.md): fixed hyperparameters across domains.
[DayDreamer](daydreamer.md): physical robots.
[SHAC](shac.md): the same short-horizon actor–critic structure with a differentiable simulator in place of a learned model.

**Contrasts with.**
[TD-MPC](td-mpc.md): uses its model for planning with a terminal value, and trains its policy through Q only.
[MBPO](mbpo.md): samples from the model rather than differentiating it.

## 6. Relevance to TGI

Dreamer and TGI share the same skeleton: a world model fit to replayed experience, imagined rollouts started at replayed states, an actor updated by backpropagating a bootstrapped multi-step return through the frozen model, and a critic regressed on λ-return targets computed on the imagined states, without a target network, with $\gamma=0.99$, $\lambda=0.95$. The differences are precise and each is a response to a failure mode Dreamer leaves open.

**The return differentiated.** TGI's actor objective (`trainer.py:345-347`) is

$$G_t=r_t+\gamma(1-d_t)\big[(1-\ell_t)V_\psi(s_{t+1})+\ell_t\,G_{t+1}\big],\qquad J=\overline{G_0},$$

which is Dreamer's λ-return recursion with the constant $\lambda$ replaced by a *state-dependent* $\ell_t\in\{0,1\}$, computed without gradient, that is 0 at an imagined touchdown on poorly supported terrain and 1 elsewhere. Away from such events TGI therefore differentiates the $\lambda=1$ return, the SHAC objective: the full 32-step reward sum through the model with the critic only at the end. Dreamer's $\lambda=0.95$ instead routes a share $(1-\lambda)\lambda^{n-1}$ of the gradient through the critic at every intermediate step, which reduces reliance on long model Jacobian products but also makes the actor follow critic gradients at states where the critic may be poor. TGI's choice is binary: trust the model completely until the first event where its Jacobian is known to be unreliable (a foot landing on a stone edge, [m-contact-gradients](../math-cookbook.md#m-contact-gradients)), then trust the critic completely. Dreamer's λ is a global bias–variance knob; TGI's $\ell_t$ is a local, physically motivated stopping time.

**Where the objective is evaluated.** Dreamer maximises $\sum_\tau V_\lambda(s_\tau)$, the return from every imagined state; TGI maximises $G_0$ only, the return from the replayed start state, as SHAC does. TGI's critic, like Dreamer's, is trained on λ-returns at every imagined state (`trainer.py:351-365`), so the critic sees the imagined distribution, which [MVE](mve.md) showed is necessary when it is queried on imagined states.

**The model.** Dreamer's model is a stochastic latent RSSM trained one step ahead and differentiated through reparameterised latent samples. TGI's is a deterministic explicit-state MLP, conditioned on the simulator's randomisation vector, and trained on 8-step autoregressive windows ([RWM](rwm.md)). The gradient therefore passes through physically meaningful Jacobians and through a model trained on its own compounding errors, but there is no latent belief to absorb partial observability.

**Rewards and terminations.** Dreamer learns its reward and discount heads; TGI computes rewards and soft terminations analytically from the imagined state and the simulator's terrain, and evaluates the critic on exact terrain features. The actor cannot exploit a learned reward, and terrain discontinuities are not smoothed by regression.

**Estimator.** Dreamer's actor is purely pathwise. TGI adds a score-function term weighted 0.1 on standardised λ-return advantages, with fixed imagination noise 0.2, which is DreamerV2's ρ-mixture ([DreamerV2](dreamer-v2.md)) with a small fixed ρ rather than ρ = 0 for continuous control.

In short, TGI takes Dreamer's imagination actor–critic, removes the latent and learned-reward components that make sense from pixels but not in a simulator with known state and terrain, and replaces Dreamer's uniform λ-mixing by event-gated truncation. The gradient side of this inheritance is analysed in [g3](../groups/g3-analytic-gradients.md).
