# MBPO: When to Trust Your Model: Model-Based Policy Optimization

| | |
|---|---|
| Authors | Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine |
| Venue / year | NeurIPS 2019 |
| Link | https://arxiv.org/abs/1906.08253 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

Model-generated data can make policy optimisation far more sample-efficient, but the policy exploits model errors, and errors compound along long rollouts. MBPO asks how much, and how, the model should be used. It combines a theoretical analysis of the gap between true and model returns with a practical algorithm: a probabilistic ensemble ([PETS](pets.md)) generates many *short* rollouts branched from real replay-buffer states, and SAC ([SAC](sac.md)) is trained on them. MBPO is the reference Dyna-style deep RL algorithm and the template that [MOPO](mopo.md) and [RWM](rwm.md) (as "MBPO-PPO") adapt.

## 2. Method

**Bound for full model rollouts.** With model error $\epsilon_m$ (total-variation distance on the data distribution) and policy shift $\epsilon_\pi$, Theorem 4.1 gives

$$\eta[\pi]\ \ge\ \hat\eta[\pi]-\Big[\frac{2\gamma r_{\max}(\epsilon_m+2\epsilon_\pi)}{(1-\gamma)^2}+\frac{4r_{\max}\epsilon_\pi}{1-\gamma}\Big],$$

whose quadratic dependence on the effective horizon $1/(1-\gamma)$ expresses compounding error.

**Branched rollouts.** Rollouts start from states of the data-collecting policy and run $k$ steps under the current policy in the model. Theorem 4.2 bounds the return of this scheme; taken literally it is maximised at $k=0$. The authors then measure empirically that model error grows with policy divergence at a rate $\mathrm d\epsilon_{m'}/\mathrm d\epsilon_\pi$ that shrinks as data increases, and with that linear approximation (Theorem 4.3) the bound admits an optimal $k>0$ when model error is small ([m-dyna-mbpo](../math-cookbook.md#m-dyna-mbpo)).

**Algorithm.** A bootstrap ensemble of probabilistic networks predicts $(s',r)$; each rollout step uses a randomly chosen member. Many $k$-step rollouts are started from states sampled uniformly from the environment buffer and added to a model buffer; SAC takes $G$ = 20–40 gradient steps per environment step on model data.

## 3. Evidence

On MuJoCo benchmarks at full 1,000-step episode length, MBPO matches the asymptotic performance of SAC with an order of magnitude less data on several tasks: its return on Ant at 300,000 steps equals SAC's at 3 million; Hopper and Walker2d are learned in the equivalent of 14 and 40 minutes of real-time simulation. It also learns Ant and Humanoid, where PETS fails. Ablations show that SAC with the same high update ratio but no model data cannot match it, so the gain is due to the model. The rollout-length study is the paper's key empirical message: on Hopper the best schedule increases $k$ from 1 to 15, but $k=1$ throughout retains much of the benefit; the model is accurate enough for 200-step rollouts yet these train worse, and 500-step rollouts are too inaccurate to learn from. Single-step model rollouts are also competitive in an H-step value-expansion comparison ([MVE](mve.md)).

## 4. Critical analysis

The theory is honest about its own weakness: the bound only justifies model use after an empirical assumption about model generalisation is substituted for a worst-case one, and the constants make it uninformative as a quantitative guide. The practical recommendation (short rollouts from real states) rests on experiments rather than the bound. Those experiments also show the price: with $k\approx1$ the model contributes little beyond augmenting one-step transitions, and the long-horizon credit assignment must still come from SAC's critic. MBPO therefore protects against compounding error by largely not compounding.

MBPO uses the model purely as a sampler for an off-policy algorithm; gradients of the model are not used. The protection afforded by ensembles (random member per step) is also partly implicit: it injects disagreement as noise rather than penalising it, which [MOPO](mopo.md) makes explicit for the offline setting.

## 5. Cross-references

**Builds on.**
[Dyna](dyna.md): MBPO's branched rollouts generalise Dyna's length-one simulated transitions.
[PETS](pets.md): the probabilistic ensemble.
[SAC](sac.md): the policy optimiser.

**Extended by.**
[MOPO](mopo.md) adds an uncertainty penalty for offline RL.
[RWM](rwm.md) replaces SAC by PPO and short rollouts by 100-step autoregressive rollouts (MBPO-PPO).

**Contrasts with.**
[ME-TRPO](me-trpo.md) and SLBO roll out from the initial-state distribution, coupling rollout length to the task horizon.
[Dreamer](dreamer.md) and [MAAC](maac.md) differentiate through short model rollouts instead of sampling from them.

## 6. Relevance to TGI

TGI inherits MBPO's start-state distribution: every imagined rollout begins at a state drawn from the replay buffer of simulator transitions, so model queries start on the data distribution, and the buffer is refreshed continuously by the current policy so that policy shift between data and imagination stays small. TGI departs from MBPO's two other choices. Its rollouts are 32 steps, longer than MBPO recommends, which it can afford because the model is trained autoregressively on 8-step windows ([RWM](rwm.md)) and because the critic closes the horizon. And its model is not an ensemble and not a sampler: the 32-step rollout is differentiated end to end ([g3](../groups/g3-analytic-gradients.md)), so model *Jacobians*, which MBPO never needs, must be accurate as well as model predictions.
