# PlaNet: Learning Latent Dynamics for Planning from Pixels

| | |
|---|---|
| Authors | Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson |
| Venue / year | ICML 2019 |
| Link | https://arxiv.org/abs/1811.04551 |
| Group | [g2-world-models](../groups/g2-world-models.md) |
| Source | full text read |

## 1. Problem and position in the lineage

PlaNet (Deep Planning Network) asks whether a dynamics model learned from images can be accurate enough for online planning on continuous-control tasks with contacts, sparse rewards and partial observability. [World Models](world-models.md) had trained its model and controller in separate stages; [PETS](pets.md) planned in the true state space. PlaNet learns a latent dynamics model jointly from pixels and rewards and plans in that latent space with CEM, with no policy or value network. Its lasting contribution is the recurrent state-space model (RSSM), the architecture that every Dreamer version, [DayDreamer](daydreamer.md), and the RSSM baselines in [RWM](rwm.md) build on.

## 2. Method

**RSSM.** The latent state has a deterministic part $h_t=f(h_{t-1},s_{t-1},a_{t-1})$ (a GRU) and a stochastic part $s_t\sim p(s_t\mid h_t)$; observations and rewards are decoded from $(h_t,s_t)$; an encoder $q(s_t\mid h_t,o_t)$ provides filtering posteriors ([m-rssm-elbo](../math-cookbook.md#m-rssm-elbo)). The authors argue that purely deterministic transitions (an RNN) "prevent the model from capturing multiple futures and make it easy for the planner to exploit inaccuracies", while purely stochastic ones struggle to remember information over many steps.

**Objective.** The model maximises the ELBO

$$\ln p(o_{1:T}\mid a_{1:T})\ \ge\ \sum_t \mathbb E_q[\ln p(o_t\mid s_t)]-\mathbb E\big[\mathrm{KL}\big(q(s_t\mid o_{\le t},a_{<t})\,\Vert\,p(s_t\mid s_{t-1},a_{t-1})\big)\big],$$

plus reward log-likelihood. The KL term trains the transition only for one-step predictions.

**Latent overshooting.** To train multi-step predictions, the paper generalises the bound to $d$-step priors $p(s_t\mid s_{t-d})$ and averages over $d=1..D$, with gradients stopped through the posterior for $d>1$. This is a latent-space analogue of multi-step model training ([m-model-learning](../math-cookbook.md#m-model-learning)) that avoids decoding images at every horizon.

**Planning.** CEM over action sequences with horizon $H=12$, $J=1000$ candidates, $I=10$ iterations and the top $K=100$ refit; one sampled latent trajectory per candidate; replanning at every step with action repeat $R$. New episodes are collected by the planner with small Gaussian exploration noise.

## 3. Evidence

On six DeepMind Control tasks from 64×64 images (cartpole swing-up, reacher, cheetah run, finger spin, cup catch, walker walk), PlaNet after at most 1,000 episodes reaches final scores of 821, 832, 662, 700, 930 and 951, against D4PG from pixels after 100,000 episodes (862, 967, 524, 985, 980, 968) and A3C from proprioception after 100,000 episodes. The estimated data-efficiency gains over D4PG range from 40× to over 500× per task; the discussion summarises this as about 200× fewer episodes, with 10–20 hours of training on one V100. Ablations show that the stochastic path is essential (the agent "does not learn without it"), the deterministic path improves all tasks, online data collection is necessary on three tasks, and CEM beats picking the best of 1,000 random sequences. Notably, the final RSSM agent does not require latent overshooting (appendix D), although other model designs benefit from it.

## 4. Critical analysis

PlaNet shows that latent planning from pixels is feasible and data-efficient; it does not show that planning is the best way to use the model. The 12-step horizon without a value function makes the agent myopic on tasks whose reward lies beyond the horizon, and the paper's own future-work list names the two fixes that became [Dreamer](dreamer.md): "learn a value function to approximate the sum of rewards beyond the planning horizon" and "gradient-based planning". CEM at every step is expensive (10,000 latent rollouts of 12 steps per action), which limits the approach to low control rates.

The finding that stochastic latents are critical deserves a careful reading. The authors offer two explanations: partial observability makes the tasks stochastic from the agent's perspective, and noise may add a safety margin against planner exploitation. Neither is tested in isolation. The result is often cited as evidence that deterministic world models are inadequate; in TGI's setting, where the state is fully observed in simulation and the randomisation parameters are supplied to the model, neither explanation applies with the same force.

## 5. Cross-references

**Builds on.**
[World Models](world-models.md): latent dynamics from pixels, here learned jointly and used for planning.
[PETS](pets.md): CEM planning with learned models.

**Extended by.**
[Dreamer](dreamer.md) keeps the RSSM and replaces planning by an actor–critic trained through the model.
[DreamerV2](dreamer-v2.md) and [DreamerV3](dreamer-v3.md) change the latent distribution and loss balancing.

**Contrasts with.**
[TD-MPC](td-mpc.md) plans in a decoder-free latent space with a terminal value.
[RWM](rwm.md) finds that an RSSM trained with teacher forcing is less accurate over long autoregressive horizons than a GRU model trained autoregressively.

## 6. Relevance to TGI

TGI shares PlaNet's insistence on multi-step accuracy, but obtains it differently. Latent overshooting trains $d$-step priors toward posteriors in latent space; TGI, following [RWM](rwm.md) and SuperMimic, rolls its explicit-state model forward for 8 steps on logged actions and penalises the state error at every step, re-querying the simulator's terrain at each predicted state. TGI rejects PlaNet's other two choices: it learns no latent state and no observation decoder, because simulator states are available and exact; and it uses the model only for training, not for decision-time planning, so it avoids per-step optimisation at deployment.
