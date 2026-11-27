# PPO: Proximal Policy Optimization Algorithms

| | |
|---|---|
| Authors | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov |
| Venue / year | arXiv preprint, 2017 (never formally published) |
| Link | https://arxiv.org/abs/1707.06347 |
| Group | [g1-rl-foundations](../groups/g1-rl-foundations.md) |
| Source | full text read |

## 1. Problem and position in the lineage

[TRPO](trpo.md) delivered reliable policy improvement but required conjugate-gradient solves, could not share parameters between policy and value networks, and was awkward with stochastic layers. Vanilla policy gradients and [A3C](a3c.md) were simple but could not safely take several optimisation steps on one batch, since repeated maximisation of $\mathbb{E}[\log\pi\,\hat A]$ "often leads to destructively large policy updates". PPO asks for TRPO's reliability using only first-order optimisation, and answers with a clipped importance-ratio surrogate that removes the incentive to move the policy far from the data-collecting policy, allowing several epochs of minibatch SGD per batch. It is the culmination of the on-policy, score-function branch of this group and the default optimiser of the legged-robot literature reviewed in [g5](../groups/g5-learned-locomotion.md), [g6](../groups/g6-character-imitation.md) and [g7](../groups/g7-humanoid-tracking.md).

## 2. Method

With probability ratio $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\rm old}}(a_t\mid s_t)$ and the conservative-policy-iteration surrogate $L^{CPI} = \hat{\mathbb{E}}_t[r_t\hat A_t]$, PPO maximises

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big],\qquad \epsilon = 0.2.$$

The minimum makes the objective a pessimistic bound on $L^{CPI}$: ratio changes are ignored when they would improve the objective beyond the clip range and counted when they worsen it; to first order around $\theta_{\rm old}$ the two coincide. An alternative adaptive-KL penalty, $L^{KLPEN} = \hat{\mathbb{E}}[r_t\hat A_t - \beta\,\mathrm{KL}]$ with $\beta$ halved or doubled when the measured KL leaves $[d_{\rm targ}/1.5,\ 1.5\,d_{\rm targ}]$, is included as a baseline. With shared networks the full loss is $L^{CLIP} - c_1(V_\theta(s_t)-V^{\rm targ}_t)^2 + c_2 S[\pi_\theta](s_t)$. Algorithm 1 ("actor–critic style"): $N$ parallel actors each run $T$ steps; advantages are computed with truncated [GAE](gae.md), $\hat A_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}$; the surrogate is optimised for $K$ epochs with minibatches of size $M \le NT$ using Adam. See [PPO clip](../math-cookbook.md#m-ppo-clip).

## 3. Evidence

*Surrogate comparison.* Seven MuJoCo Gym tasks, 1 M steps, 3 seeds each, scores normalised so that random = 0 and best = 1, policy a 64–64 tanh MLP with state-independent log-std, no parameter sharing or entropy bonus, $T = 2048$, 10 epochs, minibatch 64, $\gamma = 0.99$, $\lambda = 0.95$. No clipping or penalty: −0.39 (HalfCheetah collapsed below random). Clipping with $\epsilon = 0.1/0.2/0.3$: 0.76/0.82/0.70. Adaptive KL with $d_{\rm targ} = 0.003/0.01/0.03$: 0.68/0.74/0.71. Fixed KL with $\beta = 0.3$–$10$: 0.62–0.72.

*Against other algorithms*, PPO-clip outperformed TRPO, CEM, adaptive-step vanilla PG, A2C and A2C with trust region on most of the seven tasks in learning curves (no table). The authors note A2C matched or beat asynchronous A3C. *Humanoid showcase*: Roboschool humanoid running, flag-run and flag-run with cube pelting and get-up were learned with 32–128 actors, horizon 512, minibatch 4096, 15 epochs and KL-adapted step size, over 50–100 M timesteps. *Atari* (49 games, 3 seeds, 8 actors): by average reward over all of training PPO won 30 games, ACER 18, A2C 1; by last-100-episode reward ACER won 28, PPO 19, A2C 1, with one tie.

## 4. Critical analysis

The paper's claim is empirical and modest: clipping performs best among the surrogates tried, on cheap benchmarks, at 1 M steps. Several things it does not show matter for later use. First, the clip does not enforce a trust region: once a sample's ratio leaves the interval its gradient vanishes, but nothing prevents other samples or shared parameters from pushing it further, and later analyses (Engstrom et al., 2020; Wang et al., 2019) found that the KL between successive policies is controlled mainly by learning rate, epochs and implementation details (value clipping, advantage and observation normalisation, orthogonal initialisation, learning-rate annealing). The large-scale study of Andrychowicz et al. (2021) similarly found many such choices as influential as the objective. Second, the "lower bound" is a bound on $L^{CPI}$, not on the true return; the TRPO theorem does not transfer. Third, the humanoid showcase itself switched to a KL-adapted step size, indicating that clipping alone was not sufficient at scale. Fourth, the Atari results show PPO winning on learning speed but not on final performance against ACER.

What the paper does establish is that a few lines of change to an A2C implementation give an optimiser whose hyperparameters transfer across tasks and which scales with the number of parallel actors. That property, rather than sample efficiency, explains its adoption in robotics. In [legged_gym](legged-gym.md) and its successors, thousands of GPU-simulated robots produce batches of $10^5$ transitions per iteration in seconds, so the $1/\sqrt{N}$ reduction of score-function variance is cheap and the on-policy requirement costs little. The estimator remains zeroth-order: it uses rewards of sampled actions, not $\partial r/\partial a$ or dynamics Jacobians. Where those derivatives are available and well-behaved, first-order methods such as [SHAC](shac.md) and [SuperMimic](supermimic.md) report large reductions in samples and compute; SuperMimic states that its policy reaches PPO's best return on the same world model with about 64× fewer imagined steps, and that a matched PPO-Sim baseline with 8192 environments per GPU reached 0.72 normalised return on eight GPUs against SuperMimic's 0.82 within 1.2 h on one GPU.

## 5. Cross-references

- Builds on: [TRPO](trpo.md) (surrogate, KL control), [GAE](gae.md) (advantages), [A3C](a3c.md) (parallel actors, shared loss with entropy).
- Extended by: [Heess et al. DPPO](heess-emergence.md), distributed KL-penalty PPO for terrain locomotion; [legged_gym](legged-gym.md) and [RMA](rma.md) in [g5](../groups/g5-learned-locomotion.md); [DeepMimic](deepmimic.md) and [AMP](amp.md) in g6 (DeepMimic used PPO); [RWM](rwm.md) uses PPO on imagined rollouts (MBPO-PPO).
- Contrasts with: [SAC](sac.md) and [TD3](td3.md) (off-policy, replay-based, critic action-gradients); [SHAC](shac.md), [SuperMimic](supermimic.md), [TGI](tgi.md) (pathwise gradients through simulator or model).

## 6. Relevance to TGI

PPO is TGI's principal point of comparison and, in the upstream pipeline, a direct ingredient: [Watch Your Step](watch-your-step.md), from which TGI takes its G1 environments, foothold perception and validation suites, trains its locomotion policy with PPO across 8192 Isaac Lab environments. TGI replaces that optimiser with a pathwise gradient through a learned world model, but keeps PPO-era conventions: $\gamma = 0.99$, $\lambda = 0.95$, a critic fitted for 5 epochs over 4 minibatches, gradient-norm clipping at 1.0, and a normalised-advantage score-function term (weight 0.1, no ratio or clip because the term is evaluated on the policy that generated the imagined actions). The [g1 report](../groups/g1-rl-foundations.md) discusses what PPO's zeroth-order nature costs relative to this design and why TGI nevertheless retains a likelihood-ratio component.
