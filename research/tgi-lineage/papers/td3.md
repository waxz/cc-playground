# TD3: Addressing Function Approximation Error in Actor-Critic Methods

| | |
|---|---|
| Authors | Scott Fujimoto, Herke van Hoof, David Meger |
| Venue / year | ICML 2018 (PMLR 80) |
| Link | https://arxiv.org/abs/1802.09477 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Overestimation bias in Q-learning, caused by maximising over noisy estimates, was well studied for discrete actions. Fujimoto et al. ask whether it also afflicts deterministic actor–critics, where the maximisation is an actor's gradient ascent on the critic, and whether variance accumulated through bootstrapping explains [DDPG](ddpg.md)'s brittleness. They show both effects empirically and theoretically and propose three modifications, which together form Twin Delayed DDPG (TD3). TD3 is the mature form of the deterministic branch; its clipped double-Q target was adopted by [SAC](sac.md) and by most later off-policy and model-based critics.

## 2. Method

*Overestimation under DPG.* Comparing the actor step along the approximate critic's gradient with the hypothetical step along the true $Q^\pi$'s gradient, the paper shows that for small enough step size $\mathbb{E}[Q_\theta(s,\pi_{\rm approx}(s))] \ge \mathbb{E}[Q^\pi(s,\pi_{\rm approx}(s))]$ whenever the critic is at least unbiased at the true update, i.e. the critic overestimates the policy it induces. Plots of DDPG's value estimates on Hopper and Walker2d confirm a large positive bias over 1 M steps. Double DQN's fix (evaluate with the target network) fails because the slowly changing actor keeps current and target critics nearly identical.

*Clipped double Q-learning.* Two critics are trained to a shared target that takes their minimum:

$$y = r + \gamma\min_{i=1,2}Q_{\theta'_i}\big(s',\ \pi_{\phi'}(s') + \epsilon\big),\qquad \epsilon\sim\mathrm{clip}(\mathcal N(0,\tilde\sigma),-c,c).$$

This can underestimate, but underestimated actions are not propagated by the actor, unlike overestimated ones; the minimum also favours states with low-variance estimates.

*Delayed policy updates.* Writing a critic with residual TD errors $\delta$ as $Q_\theta(s_t,a_t) = \mathbb{E}[\sum_i\gamma^{i-t}(r_i - \delta_i)]$ shows that variance accumulates with the effective horizon. Experiments without target networks show values diverging only when the policy is learned from them. TD3 therefore updates actor and target networks once every $d = 2$ critic updates, a two-timescale scheme.

*Target policy smoothing.* The noise $\epsilon$ in the target ($\tilde\sigma = 0.2$, $c = 0.5$) regularises the critic towards smoothness in action, an off-policy Expected-SARSA-like update that prevents the deterministic actor from exploiting narrow peaks.

Implementation: 400–300 ReLU MLPs with state and action at the critic's first layer, Adam $10^{-3}$, minibatch 100, $\tau = 0.005$, uncorrelated Gaussian exploration $\mathcal N(0,0.1)$ (OU noise gave no benefit), and purely random actions for the first $10^4$ or $10^3$ steps. See [DPG/TD3](../math-cookbook.md#m-dpg).

## 3. Evidence

On seven Gym MuJoCo v1 tasks, 1 M steps, 10 seeds, evaluation every 5000 steps, the maximum average returns were: HalfCheetah 9636.95 ± 859.07 (DDPG 3305.60, re-tuned DDPG 8577.29, PPO 1795.43, TRPO −15.57, ACKTR 1450.46, SAC 2347.19); Hopper 3564.07 ± 114.74 (PPO 2164.70, SAC 2996.66); Walker2d 4682.82 ± 539.64 (PPO 3317.69, SAC 1283.67); Ant 4372.44 ± 1000.33 (PPO 1083.20, SAC 655.35). The ablation over the last 10 evaluations shows that removing clipped double-Q hurts most on Walker2d and Ant (TD3 4565/4185 vs TD3−CDQ 2579/850), that single components added to the re-tuned DDPG help little, and that their combination is required. Actor–critic versions of Double Q-learning and Double DQN reduced overestimation less than CDQ in the bias plots.

## 4. Critical analysis

The theoretical result is local (small step, normalised gradients) and assumes an unbiased critic at the true update; it establishes a direction of bias, not its magnitude. The empirical case is strong for its benchmark and unusually careful about reproducibility (10 seeds, released code, ablations). The baseline numbers, however, should be read with care: PPO, TRPO and ACKTR are the OpenAI Baselines implementations at 1 M steps, a regime that favours off-policy methods, and the SAC numbers are far below those in the SAC paper on the same tasks, a discrepancy the authors acknowledge in their supplement. TD3's design also inherits DDPG's reliance on a single-environment, one-update-per-step loop; nothing in the paper tests the regime of thousands of parallel simulators, where on-policy PPO dominates in wall-clock time. The pessimistic target is a heuristic whose bias depends on critic correlation, a point later sharpened by ensemble methods ([ensembles](../math-cookbook.md#m-ensembles)).

## 5. Cross-references

- Builds on: [DDPG](ddpg.md), [DPG](dpg.md); Double Q-learning (van Hasselt, 2010).
- Extended by: [SAC](sac.md), which adopts the minimum of two Q-functions; [TD-MPC](td-mpc.md) and [MBPO](mbpo.md)-style critics that use ensembles or minima for pessimism.
- Contrasts with: [PPO](ppo.md) and [TRPO](trpo.md), which avoid bootstrapped Q-targets and with them overestimation, at the cost of on-policy data.

## 6. Relevance to TGI

TGI's critic is a state-value function trained on on-policy λ-returns from imagined segments, so the maximisation bias TD3 analyses does not arise in its critic update. The analysis is nonetheless relevant to TGI's policy update, which, like TD3's actor, ascends a *learned* differentiable function. In TGI the exploited function is the composition of the world model and the critic; an actor following its gradient will seek regions where the model or critic is optimistically wrong, the model-exploitation problem addressed in [g2](../groups/g2-world-models.md) by [uncertainty penalties](../math-cookbook.md#m-ensembles). TGI's countermeasures are different in kind: terrain is looked up from the simulator rather than learned, so the model cannot hallucinate support, and the gradient is cut at unsupported touchdowns, the event at which model gradients are least reliable. TD3's clipped Gaussian target noise also has a counterpart in TGI's reparameterised imagination noise ($\sigma = 0.2$), which smooths the pathwise objective over nearby actions. See [TGI](tgi.md).
