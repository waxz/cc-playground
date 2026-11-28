# DreamerV2: Mastering Atari with Discrete World Models

| | |
|---|---|
| Authors | Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba |
| Venue / year | ICLR 2021 |
| Link | https://arxiv.org/abs/2010.02193 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Before DreamerV2, no agent that learned its behaviour purely inside a world model had matched strong model-free agents on the Atari benchmark. DreamerV2 changes the world model of [Dreamer](dreamer.md) (categorical latents, KL balancing) and its actor loss (a mixture of Reinforce and dynamics backpropagation), and reaches human-level performance on 55 Atari games with sticky actions at 200 million frames on one GPU. For this study its importance is twofold: it introduced the loss-balancing tools that later robotics world models reuse, and it reported a domain-dependent verdict on the pathwise versus score-function estimator that bears directly on TGI.

## 2. Method

**World model.** An RSSM whose stochastic state is a vector of 32 categorical variables with 32 classes, sampled with straight-through gradients. The loss is

$$\mathcal L(\phi)=\mathbb E\Big[\sum_t-\ln p_\phi(x_t\mid h_t,z_t)-\ln p_\phi(r_t\mid h_t,z_t)-\ln p_\phi(\gamma_t\mid h_t,z_t)+\beta\,\mathrm{KL}\big(q_\phi(z_t\mid h_t,x_t)\Vert p_\phi(z_t\mid h_t)\big)\Big].$$

**KL balancing.** The KL is split as $\alpha\,\mathrm{KL}(\mathrm{sg}(q)\Vert p)+(1-\alpha)\,\mathrm{KL}(q\Vert\mathrm{sg}(p))$ with $\alpha=0.8$, so the prior (the transition used in imagination) is trained toward the posterior faster than the posterior is regularised toward a poor prior ([m-rssm-elbo](../math-cookbook.md#m-rssm-elbo)).

**Behaviour.** Imagination of $H=15$ steps from posterior states; the critic regresses λ-targets $V^\lambda_t=\hat r_t+\hat\gamma_t[(1-\lambda)v_\xi(\hat z_{t+1})+\lambda V^\lambda_{t+1}]$ with $\lambda=0.95$ and a target network updated every 100 steps; losses are weighted by cumulative predicted discounts. The actor loss is

$$\mathcal L(\psi)=\mathbb E\Big[\sum_{t}-\rho\ln p_\psi(\hat a_t\mid\hat z_t)\,\mathrm{sg}\big(V^\lambda_t-v_\xi(\hat z_t)\big)-(1-\rho)V^\lambda_t-\eta\,\mathrm H[a_t\mid\hat z_t]\Big],$$

a score-function term with baseline ([m-pg-theorem](../math-cookbook.md#m-pg-theorem)), a pathwise term through the dynamics, and an entropy bonus. Atari uses $\rho=1$, $\eta=10^{-3}$; continuous control uses $\rho=0$, $\eta=10^{-4}$.

## 3. Evidence

At 200M frames DreamerV2 reaches a gamer-normalised median of 2.15 (2.64 with annealed schedules), against 1.29 for IQN, 1.47 for Rainbow, 1.09 for C51 and 0.65 for DQN, and a clipped record-normalised mean of 0.28 against 0.21 for IQN; it learns from 468 billion imagined compact states, about 10,000 times the 50M real inputs. Ablations (earlier version, gamer median): full agent 1.64; without discrete latents 1.08; without KL balancing 0.84; without the Reinforce term 0.69; without image gradients 0.04. Categorical latents win on 42 of 55 games; KL balancing on 44. The authors report that for Atari Reinforce gradients worked "substantially better" and for continuous control dynamics backpropagation worked "substantially better".

## 4. Critical analysis

The estimator verdict is reported but not shown in the paper's figures for continuous control, and the explanation offered (biased low-variance pathwise versus unbiased high-variance score function) is generic. A more specific reading is that with discrete actions and discrete latents the straight-through "pathwise" gradient is a crude surrogate, while in smooth continuous control the true pathwise gradient is informative. Contact-rich locomotion on sparse terrain sits between these cases: continuous actions, but outcomes that change discontinuously with foot placement. The paper's hypotheses for why categorical latents help (multimodality, sparsity, easier optimisation, better fit to non-smooth game events) are listed, not tested. KL balancing is a principled fix for a real problem in latent models (the prior being dragged toward an uninformative posterior), but it has no counterpart in explicit-state models.

## 5. Cross-references

**Builds on.**
[Dreamer](dreamer.md): imagination actor–critic and RSSM.

**Extended by.**
[DreamerV3](dreamer-v3.md): free bits, symlog, return normalisation; Reinforce for all action types in the version read.
[DayDreamer](daydreamer.md): DreamerV2 implementation on physical robots.

**Contrasts with.**
[TD-MPC](td-mpc.md): decoder-free deterministic latents, planning.
[RWM](rwm.md): uses an RSSM with categorical latents only as a baseline and finds autoregressive training more important than the latent design.

## 6. Relevance to TGI

TGI's policy loss is a DreamerV2-style mixture with the weights reversed in emphasis: $\mathcal L_\pi=-\overline{G_0}-0.1\,\overline{\bar c_t\log\pi(a_t\mid o_t)\hat A_t}$, where $\hat A_t$ are standardised λ-return advantages and the imagination noise is fixed at 0.2 (`trainer.py:367-371`). This is the $\rho$-mixture with a pathwise term dominant and a small score-function correction, a middle ground between DreamerV2's continuous-control setting ($\rho=0$) and its Atari setting ($\rho=1$). The motivation is the one DreamerV2 states: the pathwise term is low-variance but biased, and at contact events in TGI it is biased badly enough that TGI also cuts it; the score-function term supplies an unbiased (if noisy) signal precisely where the pathwise term has been removed. Like DreamerV2, TGI weights losses by a soft survival product ($\bar c_t$), which descends from Dreamer's predicted discount, but TGI's terminations are analytic sigmoids rather than a learned discount head. TGI does not use a critic target network, following [Dreamer](dreamer.md) v1 and SuperMimic rather than DreamerV2.
