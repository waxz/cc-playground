# SGS: A Balanced Data Diet — Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

| | |
|---|---|
| Authors | Octi Zhang, Mateo Guaman Castro, Patrick Yin, Ignacio Dagnino, Abhishek Gupta, Rosario Scalise, Byron Boots (University of Washington, NVIDIA) |
| Venue / year | CoRL 2026 |
| Link | https://arxiv.org/abs/2610.12465 |
| Group | [g0-anchor](../groups/g0-anchor.md) (depth-3 addition) |
| Source | full text read (main text; appendices skimmed) |

## 1. Problem and position in the lineage

Massively parallel on-policy RL ([Legged Gym](legged-gym.md)) is usually given a reset distribution over task configurations: terrain cells, start poses, goals. Sampling those configurations uniformly wastes a growing share of environments on configurations the policy has already mastered or cannot yet attempt. The waste worsens as the number of environments grows, so extra parallelism stops paying. Locomotion has traditionally used a hand-ordered "linear" terrain curriculum with promotion and demotion rules (Rudin et al.). Unsupervised environment design offers learned alternatives such as Prioritized Level Replay (PLR) and Sampling for Learnability (SFL). SGS is a deliberately simple success-rate-based sampler. The authors state that they "do not claim a fundamentally new curriculum principle". It enters this lineage because [TGI](tgi.md)'s terrain curriculum is an implementation of it (`tgi/env/sgs.py`, which cites this paper).

## 2. Method

A fixed set of $N$ configurations $\tau_i=(s_0,g,e)$ is drawn before training ($N=104{,}000$ for locomotion). For each, the last $H$ Boolean outcomes are kept and $p_i$ is their mean. Each configuration is scored by a Beta-shaped kernel in mode–concentration form, Beta$(1+\kappa t, 1+\kappa(1-t))$ (Eq. 1),
$$w_i=(p_i+\epsilon)^{\kappa t}(1-p_i+\epsilon)^{\kappa(1-t)},\qquad \ell_i=\log(w_i+\epsilon),$$
and the next configuration is drawn i.i.d. from $p(i)=\mathrm{softmax}_i(\ell_i/T)$ (Eq. 2). For locomotion, $t=0.66$, $\kappa=5$, $T=2$, $\epsilon=10^{-8}$ and $H=100$. The target $t$ sets the mode, so configurations near the chosen success rate dominate, while $\epsilon$ keeps every configuration's probability non-zero so that estimates for unsolved ones continue to be updated. Compared with SFL's $p(1-p)$ weighting, SGS adds an adjustable target, rolling estimates and a guaranteed floor. The training recipe pairs the sampler with one shared reward per domain: sparse task success plus small regularisers ([curriculum](../math-cookbook.md#m-curriculum)).

## 3. Evidence

On ANYmal-D multi-terrain locomotion over 13 terrain types, which include floating islands joined by narrow stones or jump boxes, a single policy is trained from scratch with PPO for 5k iterations. At $2^{20}$ (about one million) parallel environments, SGS reaches 73 % mean success against 54 % for PLR (Fig. 5a, three seeds). Only SGS improves monotonically with scale; the other samplers plateau or collapse. At 4k and 32k environments it beats SFL's final performance (0.456 vs 0.255 and 0.494 vs 0.051; Table 1), with SFL degrading late in training. On Franka nut-and-bolt assembly at 1M environments it reaches 70 % against 6 % (uniform) and 5 % (PLR). Locomotion training at 1M environments takes about 36 h on L40S/H200 GPUs.

## 4. Critical analysis

The evidence supports a scaling claim, that a success-based sampler lets on-policy RL use very large batches, more than a claim about sample efficiency at ordinary scales. The configuration set is fixed in advance and treated as exchangeable, so no notion of difficulty ordering or similarity between configurations is used. Unvisited or rarely visited configurations start with $p_i=0$ and sit at low weight until sampled by chance; with 104k configurations and $H=100$, estimates are noisy. Success is binary, so the sampler cannot see partial progress. The hyperparameters are reported as insensitive, but the target success rate differs between domains (0.66 vs 0.5), which suggests some tuning.

TGI's adaptation addresses two of these points. It adds a neighbour bound, $p^{\text{eff}}_\ell=\max(p_\ell,\,0.25\,p^{\text{eff}}_{\ell-1})$, that transfers evidence along the difficulty axis, and fixed per-terrain-type shares that prevent one type from absorbing the budget. It also uses a much shorter window (16) and a higher target (0.8), consistent with a policy that is updated every iteration from imagination rather than once per batch of on-policy rollouts.

## 5. Cross-references

**Builds on.**
[Legged Gym](legged-gym.md): parallel PPO and the linear terrain curriculum it replaces.
[PPO](ppo.md): the optimiser in all experiments.

**Contrasts with.**
PLR and SFL from unsupervised environment design; neither is in this corpus.
The return-based reset sampling of [SuperMimic](supermimic.md), which weights motion bins by low return rather than by moderate success.

**Extended by.**
[TGI](tgi.md).

## 6. Relevance to TGI

TGI's `SuccessGuidedSampler` implements Eqs. (1)–(2) verbatim ($\kappa=5$, $T=2$) over terrain cell × 4 spawn patches × 4 travel-direction bins. It adds the neighbour bound and per-type priors (and, for ANYmal, a mechanism by which solved types cede share), with target 0.8 and window 16. Because TGI's policy improves on imagined rollouts seeded from replayed simulator states, the curriculum shapes the training distribution twice: it decides which terrain the simulator collects on, and hence which states seed imagination and which transitions the world model is fit to.
